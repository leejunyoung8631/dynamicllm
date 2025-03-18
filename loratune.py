'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''

import os
import sys
import argparse
from typing import List
from pathlib import Path

import torch
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

try:
    from peft import prepare_model_for_int8_training
    prepare_model_for_training = prepare_model_for_int8_training
except ImportError:
    print("'prepare_model_for_int8_training' is deprecated 'peft version > 1.0.0', ")
    print("it will use 'prepare_model_for_kbit_training' intead of it")
    from peft import prepare_model_for_kbit_training
    prepare_model_for_training = prepare_model_for_kbit_training


from datahelper import DataHelper



# class KDTrainer(transformers.Trainer):
#     def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.teacher_model = teacher_model
#         self.temperature = temperature
#         self.alpha = alpha

#     def compute_loss(self, model, inputs, return_outputs=False):
#         # Compute student loss using the original Trainer's compute_loss
#         student_loss, outputs = super().compute_loss(
#                 model, inputs, return_outputs=True)
#         student_logits = outputs.logits

#         # Compute teacher logits
#         with torch.no_grad():
#             teacher_outputs = self.teacher_model(**inputs)
#             teacher_logits = teacher_outputs.logits

#         # Compute distillation loss
#         distillation_loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
#         distillation_loss = distillation_loss_fct(
#             torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1),
#             torch.nn.functional.softmax(teacher_logits / self.temperature, dim=-1),
#         )

#         # Combine student loss and distillation loss
#         loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss * (self.temperature**2)

#         return (loss, outputs) if return_outputs else loss


from datautil import set_seed
from modelutils import get_model
from modelutils import load_mask_weight, set_inference
from huggingface_hub import login



def main(args):
    login("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")  
    set_seed(args.seed)
    
    # Load Pruned Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = get_model(base_model=args.base_model, model_class=args.model_class, loss_term=args.loss_term)    
    model = load_mask_weight(model, args.mask_weight)
    model = set_inference(model, args)
    
    if args.use_bfloat:
        print("here")
        model = model.bfloat16()
    
    # model.half()
    # model = model.cuda()
        
    # Prepare For LoRA
    # LoraConfig is inherited PeftConfig
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # from peft.peft_model import PeftModelForCausalLM, PeftModel
    # PeftModelForCausalLM is inherited PeftModel
    # PeftModel will call LoraModel
    # LoraModel will call Basetune, which is the base class for tuning models and injects adaptors using inject_adaptor
    model = get_peft_model(model, config) 
    model.print_trainable_parameters()  
    
    def get_model_size(model):
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        # Optionally include buffers if needed:
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        return total_size
    
    def bytes_to_MB_GB(num_bytes):
        mb = num_bytes / (1024 ** 2)  # 1 MB = 1024^2 bytes
        gb = num_bytes / (1024 ** 3)  # 1 GB = 1024^3 bytes
        return mb, gb
    
    size_in_bytes = get_model_size(model)
    print(f"Estimated model size: {size_in_bytes} bytes")
    size_in_mb, size_in_gb = bytes_to_MB_GB(size_in_bytes)
    
    
    print(f"Estimated model size: {size_in_bytes} bytes")
    print(f"Estimated model size: {size_in_mb:.2f} MB")
    print(f"Estimated model size: {size_in_gb:.4f} GB")
    
    # Load Train Dataset
    dataset_helper = DataHelper(
            tokenizer,
            args.cutoff_len, args.add_eos_token, args.train_on_inputs,
            args.no_instruction, args.prompt_template_name,
            args.verbose,
            no_template_ratio=0.5)

    if args.data_path.endswith(".json"):
        train_data, val_data = dataset_helper.create_dataset_from_json(
                args.data_path)
    else:
        if args.cache_dataset_dir:
            os.makedirs(args.cache_dataset_dir, exist_ok=True)

        train_data, val_data = dataset_helper.create_dataset(
                args.data_path, args.val_set_size, None, 
                args.cache_dataset_dir, None)

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    
    
    fp16_flag = True
    bf16_flag = False
    if args.use_bfloat:
        # model = model.bfloat16()
        fp16_flag = False
        bf16_flag = True
    
    trainer_args = transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=fp16_flag,
            bf16=bf16_flag,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=100,
            save_steps=200,
            # save_steps=20,
            output_dir=args.output_dir,
            save_total_limit=20,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            report_to="none",
            run_name=args.output_dir.split('/')[-1],
            # metric_for_best_model="{}_loss".format("yahma/alpaca-cleaned"),
            )

    data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )


    if args.kd_teacher_model:
        # Assume the kd teacher uses the same tokenizer
        # teacher_model, _ = get_model(args.kd_teacher_model, args.cache_model_dir, device, args.use_bfloat)
        # trainer = KDTrainer(
        #     teacher_model,
        #     temperature=args.kd_temp, alpha=args.kd_alpha,
        #     model=model,
        #     train_dataset=train_data,
        #     eval_dataset=val_data,
        #     args=trainer_args,
        #     data_collator=data_collator,
        # )
        print("The current version does not support kd_teacher_model")
        exit()
    else:
        # Non-KD case
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=trainer_args,
            data_collator=data_collator,
        )
    

    model.config.use_cache = False
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # model.state_dict = old_state_dict
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca2", help='output directory')
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    parser.add_argument("--cache_model_dir", type=str, default="./model_cache", help="llm weights")
    parser.add_argument('--cache_dataset_dir', type=str, default="./cache_dataset", help='data cache path')

    # KD
    parser.add_argument('--kd_teacher_model', type=str, default=None, help='KD teacher model name')
    parser.add_argument('--kd_temp', type=float, default=2.0, help='KD temperature')
    parser.add_argument('--kd_alpha', type=float, default=0.5, help='KD alpha')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False, help="Whether to use the instruction template or not.")
    parser.add_argument('--use_deepspeed', action='store_true', default=False, help="Whether to use deepspeed")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_coga_collator", action='store_true', default=False, help="Whether to use coga's collator")

    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')

    # llm hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")
   
    # wandb params
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")
    
    # my params
    parser.add_argument(
        "--model_class",
        type=str,
        default="dyllm",
        help="chosse in [dyllm, ...]",
    )
    parser.add_argument("--mask_weight", default=None)
    parser.add_argument("--check_count", action="store_true", help="if True, check the number of skipped blocks")
    parser.add_argument("--is_generation", action="store_true", )
    parser.add_argument('--loss_term', type=str, default="none", help='loss function for training')
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose on model structure printing",
    )
   
    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    main(args)
