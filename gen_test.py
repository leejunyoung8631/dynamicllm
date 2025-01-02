import os
import argparse

import torch
import torch.nn as nn

import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, AutoModelForCausalLM, LlamaTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl, AutoConfig

from datahelper import DataHelper


from model_generate import CustomLlamaForCausalLM





def get_model(base_model):
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = CustomLlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
    
    model.config.output_hidden_states = True
    model.config.return_dict = True
    
    return model, tokenizer



from transformers import GenerationMixin 


def main(args):
    from huggingface_hub import login
    login("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")  
    base_model = "baffo32/decapoda-research-llama-7B-hf"
    # base_model = "meta-llama/Llama-2-7b-hf"
    model, tokenizer = get_model(base_model)
    
    # # Freeze all base model parameters
    # for param in model.base_model.parameters():
    #     param.requires_grad = False
    
    # dataset_helper = DataHelper(
    #         tokenizer,
    #         args.cutoff_len, args.add_eos_token, args.train_on_inputs,
    #         args.no_instruction, args.prompt_template_name,
    #         args.verbose,
    #         no_template_ratio=0.5)
    # if args.data_path.endswith(".json"):
    #     train_data, val_data = dataset_helper.create_dataset_from_json(
    #             args.data_path)
    # else:
    #     train_data, val_data = dataset_helper.create_dataset(
    #             args.data_path, args.val_set_size, args.extra_val_dataset,
    #             args.cache_dir, args.partial_dir)
    
    
    # Prompt for sentence generation
    prompt = "Once upon a time in a distant land,"

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate text with exact length
    output = model.generate(
        **inputs,
        max_length=64,      # Total length of tokens including prompt
        min_length=64,      # Ensures exactly 64 tokens
        do_sample=True,     # Enable sampling for diverse output
        temperature=0.7,    # Controls randomness
        pad_token_id=tokenizer.eos_token_id  # Avoids padding-related errors
    )

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated Sentence:\n", generated_text)
        
        
        
    








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