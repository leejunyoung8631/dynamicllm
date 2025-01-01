import os
import argparse

import torch
import torch.nn as nn

import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer
from transformers import TrainerCallback, TrainerState, TrainerControl

from datahelper import DataHelper




class LossCallback(TrainerCallback):
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print("Training has begun with a custom callback.")

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"Epoch {int(state.epoch)} finished.")
        
    
    # def on_init_end(self, args, state, control, **kwargs):
    #     return super().on_init_end(args, state, control, **kwargs)
    
    # def on_step_begin(self, args, state, control, **kwargs):
    #     return super().on_step_begin(args, state, control, **kwargs)
    
    # def on_step_end(self, args, state, control, **kwargs):
    #     return super().on_step_end(args, state, control, **kwargs)




class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        We override the compute_loss method to use a custom loss.
        """
        # Forward pass
        outputs = model(**inputs)
        
        # The model outputs typically include 'logits', 
        # but you can adjust this depending on your model
        logits = outputs.get('logits')
        
        # Our labels are usually in 'labels' (passed in `inputs`)
        labels = inputs.get('labels')
        
        # Example: CrossEntropyLoss, but you can use any custom logic here
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # If we want to return outputs (for logging or other use),
        # we return (loss, outputs). Otherwise we just return the loss.
        return (loss, outputs) if return_outputs else loss






class Predictor(nn.Moule):
    def __init__(self, ):
        super.__init__()
    
    
    def forward(self, ):
        
        
        return
    
    


def create_trainer(
        model, tokenizer, train_data, val_data,
        epochs, outdir,
        show_progress, load_best_model,
        args,
        custom_eval_save_step=None):
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    fp16_flag = True
    bf16_flag = False
    if args.use_bfloat:
        model = model.bfloat16()
        fp16_flag = False
        bf16_flag = True

    if load_best_model:
        eval_steps=200
        evaluation_strategy="steps"
        save_steps=200
        save_total_limit=20
        load_best_model_at_end=True
        save_strategy="steps"
    else:
        eval_steps=None
        evaluation_strategy="no"
        save_steps=None
        save_total_limit=None
        load_best_model_at_end=False
        save_strategy="no"

    callbacks = []
    eval_callback = LossCallback()
    callbacks.append(eval_callback)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=callbacks,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=epochs,
            learning_rate=args.learning_rate,
            fp16=fp16_flag,
            bf16=bf16_flag,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            output_dir=outdir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            # report_to="wandb",
            run_name=args.output_dir.split("/")[-1],
            # metric_for_best_model="{}_loss".format("yahma/alpaca-cleaned"),
            disable_tqdm=(not show_progress),
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    return trainer



        


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss



def get_model(base_model):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    model = LlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
    
    return tokenizer, model



def main(args):
    base_model = "baffo32/decapoda-research-llama-7B-hf"
    # base_model = "meta-llama/Llama-2-7b-hf"
    model, tokenizer = get_model(base_model)
    model.eval()
    
    model.print_trainable_parameters()
    
    
    
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
        train_data, val_data = dataset_helper.create_dataset(
                args.data_path, args.val_set_size, args.extra_val_dataset,
                args.cache_dir, args.partial_dir)
    

    # Create trainer
    trainer = create_trainer(
            model, tokenizer, train_data, val_data, 
            args.num_epochs, args.output_dir,
            show_progress=True, load_best_model=False,
            args=args,
            custom_eval_save_step=args.detailed_extra)
    
    trainer.train()
    
    
    
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model Type&Path
    parser.add_argument(
        "--base_model",
        type=str,
        default="output_tune/llama-1-7b/ppl_n10/rm_6_blocks_lora_merge_fp16",
        help="base model name",
    )
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--fix_decapoda_config",
        default=False,
        action="store_true",
        help="fix tokenizer config of baffo32/decapoda-research-llama-7B-hf",
    )
    parser.add_argument("--use_bfloat", default=False, action="store_true")

    parser.add_argument(
        "--data_path", type=str, default="yahma/alpaca-cleaned", help="data path"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="cache_dir for datasets module"
    )
    parser.add_argument(
        "--partial_dir", type=str, default=None, help="Directory to save/load the partial dataset for C4"
    )
    parser.add_argument(
        "--val_set_size", type=int, default=2000, help="validation set size"
    )
    parser.add_argument(
        "--extra_val_dataset",
        type=str,
        default=None,
        help='validation datasets. Split with ","',
    )
    
    
    
    parser.add_argument("--num_epochs", type=float, default=5, help="number of epochs")
    parser.add_argument(
        "--output_dir", type=str, default="./finetune", help="output directory"
    )
    parser.add_argument(
        "--detailed_extra",
        type=int,
        default=None,
        help="Step size used detailed evaluation and saving for extra stage",
    )
    parser.add_argument(
        "--group_by_length",
        default=False,
        action="store_true",
        help="faster, but produces an odd training loss curve",
    )
    
    
    parser.add_argument("--cutoff_len", type=int, default=256, help="cutoff length")
    parser.add_argument("--add_eos_token", default=False, action="store_true")
    parser.add_argument(
        "--train_on_inputs",
        default=False,
        action="store_true",
        help="Train on inputs. If False, masks out inputs in loss",
    )
    parser.add_argument(
        "--no_instruction",
        action="store_true",
        default=False,
        help="Whether to use the instruction template or not.",
    )
    parser.add_argument(
        "--prompt_template_name",
        type=str,
        default="alpaca",
        help="The prompt template to use, will default to alpaca.",
    )
    
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose on model structure printing",
    )
    
    
    args = parser.parse_args()

    main(args)