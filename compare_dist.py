import os
import argparse

import torch
import torch.nn as nn

import numpy as np
import random



import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, AutoModelForCausalLM, LlamaTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl, AutoConfig

from datahelper import DataHelper


from model_generate import CustomLlamaForCausalLM

from modelutils import get_model

from datautil import tokenlayer_table



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    




def main(args):
    from huggingface_hub import login
    login("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")  
    
    set_seed(444)

    # Load the LLaMA2 model and tokenizer
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Or other LLaMA2 variant
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = CustomLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    model.config.output_hidden_states = True
    model.config.return_dict = True



    # if tokeninfo file is specified, then plot and exit
    if args.tokeninfo:
        tokenlayer_table(tokenizer, args.tokeninfo,)
        exit()

    
    model.to("cuda")

    # Set your prompt
    prompt = "Once upon a time in a distant galaxy,"

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate text
    output = model.generate(
        **inputs,
        max_length=50,  # Adjust the output length
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(output)
    print(decoded_text)
    
    
    
    
    
    '''
    i dont know why below llama1 does not properly works..
    '''
    
    # base_model = "baffo32/decapoda-research-llama-7B-hf"
    # model, tokenizer = get_model(base_model, model_name=CustomLlamaForCausalLM, is_decapoda=True)
    # model.cuda()
    
    # # set_seed(42)
    
    # # Prompt for sentence generation
    # prompt = "the weather is so good and "

    # # Tokenize the input prompt
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # # Generate text with exact length
    # output = model.generate(
    #     **inputs,
    #     max_length=40,      # Total length of tokens including prompt
    #     min_length=40,      # Ensures exactly 64 tokens
    #     do_sample=True,     # Enable sampling for diverse output
    #     temperature=0.7,    # Controls randomness
    #     pad_token_id=tokenizer.eos_token_id  # Avoids padding-related errors
    # )

    # # Decode and print the generated text
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    # print("Generated Sentence:\n", generated_text)
    
    
    # # tokenlayer_table(tokenizer)





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
    
    
    
    # token inputfile
    parser.add_argument(
        "--tokeninfo",
        default=None,
        type=str,
        help="path for tokeninfo csv file."

    )
    
    
    args = parser.parse_args()

    main(args)