import os
import argparse

import torch
import torch.nn as nn

import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, AutoModelForCausalLM, LlamaTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl, AutoConfig

from datahelper import DataHelper


from model import CustomLlamaForCausalLM

import csv


class LossCallback(TrainerCallback):
    def __init__(self, file_path="loss_log.csv", write_interval=50):
        """
        Initialize the callback with a buffer and write interval.
        """
        self.file_path = file_path
        self.write_interval = write_interval
        self.loss_buffer = []  # Temporary storage for loss data
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Write the header to the CSV file
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["step", "loss"])  # CSV Header

    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Log loss every step in memory and write to the file every `write_interval` steps.
        """
        if state.log_history and "loss" in state.log_history[-1]:
            loss = state.log_history[-1]["loss"]
            step = state.global_step
            
            # Add to buffer
            self.loss_buffer.append((step, loss))
            
            # Write to file if buffer reaches the interval
            if len(self.loss_buffer) >= self.write_interval:
                self._flush_buffer()
                
    
    def _flush_buffer(self):
        """
        Write buffered loss data to the CSV file and clear the buffer.
        """
        if not self.loss_buffer:
            return  # Nothing to write
        
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.loss_buffer)
        
        print(f"Flushed {len(self.loss_buffer)} steps of loss data to {self.file_path}")
        self.loss_buffer = []  # Clear the buffer
        

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Ensure all remaining data is written when training ends.
        """
        self._flush_buffer()
        print("Training finished. All buffered data has been written.")





class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        We override the compute_loss method to use a custom loss.
        """
        # Forward pass
        outputs = model(**inputs)
        
        # target
        labels = inputs.get('labels') 
        b, n = labels.shape
        origin_loss = outputs.get("loss")
        origin_loss = origin_loss.reshape(b, -1).mean(dim=-1)
        
        # first layer feature
        # this will predict the loss 
        hidden = outputs.get('hidden_states')[0]
        
        pred = model.predictor(hidden)
        loss = LossPredLoss(pred, origin_loss)
        
        # If we want to return outputs (for logging or other use),
        # we return (loss, outputs). Otherwise we just return the loss.
        return (loss, outputs) if return_outputs else loss
    
    
    

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

    trainer = CustomTrainer(
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
            report_to="none",
            run_name=args.output_dir.split("/")[-1],
            # metric_for_best_model="{}_loss".format("yahma/alpaca-cleaned"),
            disable_tqdm=(not show_progress),
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    return trainer



def get_model(base_model):
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = CustomLlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
    
    model.config.output_hidden_states = True
    model.config.return_dict = True
    
    return model, tokenizer



def main(args):
    from huggingface_hub import login
    login("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")  
    base_model = "baffo32/decapoda-research-llama-7B-hf"
    # base_model = "meta-llama/Llama-2-7b-hf"
    model, tokenizer = get_model(base_model)
    
    # Freeze all base model parameters
    for param in model.base_model.parameters():
        param.requires_grad = False
    
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
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--micro_batch_size", type=int, default=4, help="micro batch size"
    )
    parser.add_argument("--num_epochs", type=float, default=5, help="number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="learning rate"
    )
    parser.add_argument(
        "--prompt_template_name",
        type=str,
        default="alpaca",
        help="The prompt template to use, will default to alpaca.",
    )
    parser.add_argument(
        "--no_instruction",
        action="store_true",
        default=False,
        help="Whether to use the instruction template or not.",
    )
    
    
    
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
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose on model structure printing",
    )
    
    
    args = parser.parse_args()

    main(args)