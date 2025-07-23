'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''

import os
import argparse
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import transformers
from transformers import AutoModelForCausalLM

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
from utils import get_model, set_model_device_evalmode, set_seed

from eval_ppl import get_loaders_trainer


from dyllm_model_rev import DyLM
from transformers.models.llama import LlamaConfig
transformers.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING[LlamaConfig] = DyLM



class SubModuleTrainer(transformers.Trainer):
    def __init__(
        self,
        tokenizer = None,
        skip_num = None,
        token_len = None,
        skip_index = None,
        *args,
        **kwargs
    ):
        """
        Args:
            teacher_model: Pretrained teacher model (torch.nn.Module)
            temperature (float): Softmax temperature for distillation
            alpha (float): Balances between student CE loss and KD loss
            beta (float): Exponent to emphasize teacher's high-probability tokens
            gamma (float): Exponent to emphasize teacher's near-zero probability tokens
            ul_alpha (float): Weight for token-level unlikelihood loss
            tokenizer: Tokenizer for handling input sequences
        """
        super().__init__(*args, **kwargs)
        
        self.tokenizer = tokenizer
        self.skip_num = skip_num
        self.token_len = token_len
        self.skip_index = skip_index
    
    
    # it will be used when giving answers
    def get_answer(self, logits, ):
        # logits : (batch) * (length-1) * (possible skip configuration) => e.g. (4, 127, 10)
        
        # label algo 1 : just minimum value
        skip_label, skip_label_idx = torch.min(logits, dim=-1)
        
        # label algo 2 : one-hot encoding 
        num_classes = logits.size(-1)
        target_one_hot = F.one_hot(skip_label_idx, num_classes)
        target_one_hot = target_one_hot.to(dtype=logits.dtype) # (batch) * (length-1) * (possible skip configuration)
        
        
        return skip_label, target_one_hot # (b, length-1)
    
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        
        # new logics for trainer
        criterion = nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=self.tokenizer.pad_token_id  # will skip pad positions
        )
            
        # data
        input_ids = inputs["input_ids"]
        mask= inputs["attention_mask"]
        
        # print(input_ids)
        # print(input_ids.shape)
            
        ppl = []
        for i in range(self.skip_num):
            outputs = model.forward_skip(**inputs, skip_layer=self.skip_index[:(i+1)])
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()    
            shift_labels = input_ids[..., 1:].contiguous()
            # shift_mask   = mask[..., 1:].contiguous()
            shift_mask   = mask[..., :-1].contiguous() # maybe this is right i think
            num_tokens = shift_mask.sum()
            
            B, Lm1, V = shift_logits.shape
            
            flat_logits = shift_logits.view(-1, V)
            flat_labels = shift_labels.view(-1)
            flat_loss   = criterion(flat_logits, flat_labels)  
            
            token_loss = flat_loss.view(B, Lm1) 
            # token_ppls = torch.exp(token_loss) # does not use ppl? unstable
            token_ppls = token_loss
            
            ppl.append(token_ppls.unsqueeze(-1))
        
        
        ppl = torch.cat(ppl, dim=-1)
        layer_prob = outputs["layer_prob"]
        shifted_layer_prob = layer_prob[:, :-1, :] # B, Lm1, skip_layers ([4, 255, 10])
        
        loss = shifted_layer_prob * ppl # shift mask -> remove padding value
        loss = loss.sum() / num_tokens
        

        # outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        # if labels is not None:
        #     unwrapped_model = self.accelerator.unwrap_model(model)
        #     if _is_peft_model(unwrapped_model):
        #         model_name = unwrapped_model.base_model.model._get_name()
        #     else:
        #         model_name = unwrapped_model._get_name()
        #     if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss



class SafeDataCollatorForSeq2Seq(transformers.DataCollatorForSeq2Seq):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        required_keys = {"input_ids", "attention_mask"}
        if any("labels" in f for f in features):
            required_keys.add("labels")

        base_features = [
            {k: f[k] for k in required_keys if k in f and f[k] is not None}
            for f in features
        ]
        batch = super().__call__(base_features)

        all_keys = set().union(*(f.keys() for f in features))
        extra_keys = all_keys - required_keys

        for key in extra_keys:
            values = [f.get(key, None) for f in features]

            if all(v is None for v in values):
                batch[key] = None
                continue

            # Convert tensors to lists
            as_lists = []
            is_sequence = False
            for v in values:
                if isinstance(v, torch.Tensor):
                    if v.ndim == 0 and torch.isnan(v).item():
                        v = 0
                    else:
                        v = v.tolist()
                if isinstance(v, list):
                    is_sequence = True
                as_lists.append(v)

            if not is_sequence:
                # Scalar field (e.g., all int/float or None)
                scalar_values = [
                    v if v is not None else 0  # You can change default if needed
                    for v in as_lists
                ]
                batch[key] = torch.tensor(scalar_values)
            else:
                # Sequence field: pad to same length
                max_len = max(len(v) for v in as_lists if v is not None)
                padded_values = []
                for v in as_lists:
                    if v is None:
                        padded_values.append([self.tokenizer.pad_token_id] * max_len)
                    else:
                        padded_values.append(
                            v + [self.tokenizer.pad_token_id] * (max_len - len(v))
                        )
                batch[key] = torch.tensor(padded_values)


        return batch

from transformers.trainer import _is_peft_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        hidden_states = outputs.hidden_states
        
        print(hidden_states.shape)
        
        exit()
        
        




def main(args):
    #  Set WanDB
    os.environ["WANDB_PROJECT"] = args.wandb_project
    set_seed(args.seed)
    
    import wandb
    wandb.login(key="50d1622956d86d0c79e8a2c666215e41c34f7aa3")
    wandb.init(entity="ljy32051-daegu-gyeongbuk-institute-of-science-technology", project=args.wb_proj)
    
    
    torch.set_printoptions(profile='full')
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, tokenizer = get_model(args.base_model, args.cache_model_dir, device, args.use_bfloat, args.use_8bit_model)
    if args.model_class == "dylm":
        model_class = DyLM
    else:
        model_class = AutoModelForCausalLM
    
    model, tokenizer = get_model(base_model=args.base_model, cache_dir=args.cache_model_dir, device=device, use_bfloat=args.use_bfloat, load_in_8bit=args.use_8bit_model, model_class=model_class)
    
    for name, param in model.named_parameters():
        if "predictor" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
        
    
        
    # Prepare For LoRA
    # model = prepare_model_for_training(model)
    # config = LoraConfig(
    #     r=args.lora_r,
    #     lora_alpha=args.lora_alpha,
    #     target_modules=args.lora_target_modules.split(","),
    #     lora_dropout=args.lora_dropout,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(model, config)
    # if args.resume_from_checkpoint is not None:
    #     model.load_adapter(args.resume_from_checkpoint, adapter_name="default", is_trainable=True)
    # model.print_trainable_parameters()
    
    
    # additional model configuration
    model.skip_block = args.skip_block
    model.skip_order = args.skip_order
    model.get_hidden = args.get_hidden
    
    
    print(f"run with skip configuration : {args.skip_block}")
    
    
    # Load Train Dataset
    if args.custom_dataset:
        # batch_size is dummy now
        train_data, val_data = get_loaders_trainer(
            "wikitext2", tokenizer, seq_len=128, batch_size=-1, add_bos_to_every=False
        )
        train_data = train_data["train"]
        val_data = val_data["train"]
    else:
        dataset_helper = DataHelper(
                tokenizer,
                args.cutoff_len, args.add_eos_token, args.train_on_inputs,
                args.no_instruction, args.prompt_template_name,
                True,
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
        
        
        if args.data_path == "wikitext":
            train_data = train_data.remove_columns(['text'])
            # val_data = val_data.remove_columns('text')
        
        if args.data_path == "yahma/alpaca-cleaned":
            train_data = train_data.remove_columns(['output', "input", "instruction"])        
    
    
    # train_data = None
    # val_data = None
    
    
    # val_name = "eval_yahma/alpaca-cleaned"
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    trainer_args = transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            #gradient_checkpointing=True,  # Testing - not working ... (due to grad_fn = 0)
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            # bf16=False, # added
            # fp16=False,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            # evaluation_strategy="steps",
            evaluation_strategy="no",
            save_strategy="steps",
            # eval_steps=100,
            save_steps=200,
            output_dir=args.output_dir,
            save_total_limit=20,
            # load_best_model_at_end=True,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            report_to="wandb",        
            # report_to="none",        
            run_name=args.output_dir.split('/')[-1],
            # metric_for_best_model="{}_loss".format("yahma/alpaca-cleaned"),
            # metric_for_best_model="{}_loss".format("eval_yahma/alpaca-cleaned"),
            # metric_for_best_model="{}_loss".format(val_name),
            remove_unused_columns=False,  # For HybridMLEULTrainer using is_negative
            #deepspeed=ds_config,
            )
    
    # data_collator = SafeDataCollatorForSeq2Seq(
    #             tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #         )
    
    data_collator = transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            )

    trainer = SubModuleTrainer(
        # new arguments
        tokenizer = tokenizer,
        skip_num = args.skip_block,
        token_len = args.cutoff_len,
        skip_index = args.skip_order,
        # previous arguments
        model=model,
        train_dataset=train_data,
        # eval_dataset=val_data,
        args=trainer_args,
        data_collator=data_collator,
    )
    trainer.train()
    
    return







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    parser.add_argument("--cache_model_dir", type=str, default="./model_cache", help="llm weights")
    parser.add_argument('--cache_dataset_dir', type=str, default="./cache_dataset", help='data cache path')

    # CT unlikelihood
    parser.add_argument('--ct_alpha', type=float, default=1.0, help='CT alpha (relative portion when original CE loss is 1)')
    parser.add_argument('--ct_fraction', type=float, default=0.4, help='CT fraction in the answer tokens')
    parser.add_argument('--ct_max_len', type=int, default=150, help='Upperbound length of the tokens considerd to compute CT loss')
    parser.add_argument('--ct_neg_fraction', type=float, default=0.4, help='Fraction of negative tokens within the the tokens of the CT length computed by ct_fraction')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=float, default=5, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False, help="Whether to use the instruction template or not.")
    parser.add_argument('--use_deepspeed', action='store_true', default=False, help="Whether to use deepspeed")
    parser.add_argument("--seed", type=int, default=1234)

    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')
    
    # datarun
    parser.add_argument("--skip_block", default=0, type=int, help="the number of block skipped")
    parser.add_argument("--get_hidden", action='store_true', default=False, help="return hidden states or not")
    parser.add_argument("--custom_dataset", action='store_true', default=False, help="use datahelper or not")
    
    # skip info
    parser.add_argument('--skip_order', 
                        nargs='+', 
                        type=int, 
                        default=[24, 26, 25, 10, 27, 13, 22, 14, 9, 29, 12], 
                        help='skip order index of block')
    parser.add_argument('--model_class', type=str, default="dylm")
    
    # dummy
    parser.add_argument("--use_8bit_model", action='store_true', default=False, help="return hidden states or not")
    parser.add_argument('--resume_from_checkpoint', default=None, type=str, help="either training checkpoint or final adapter")
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")
    
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--wb_proj', type=str, default="test")
    
    
   
    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
