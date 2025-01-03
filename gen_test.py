import os
import argparse

import torch
import torch.nn as nn

import numpy as np
import random

import pandas as pd
import matplotlib.pyplot as plt


import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, AutoModelForCausalLM, LlamaTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl, AutoConfig

from datahelper import DataHelper


from model_generate import CustomLlamaForCausalLM



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_model(base_model):
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = CustomLlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=False)
    # model = CustomLlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
    # model = LlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
    
    model.config.output_hidden_states = True
    model.config.return_dict = True
    
    return model, tokenizer



def tokenlayer_table(tokenizer, file="tokeninfo.csv",):
    # Access the underlying SentencePiece model
    sp = tokenizer.sp_model

    # Get the total vocabulary size
    vocab_size = sp.get_piece_size()

    # Build index → token dictionary
    idx_to_token = {i: sp.id_to_piece(i) for i in range(vocab_size)}
    
    # read data
    df = pd.read_csv(file)  # shape = (length, 32)
    df_tokens = df.applymap(lambda x: idx_to_token[x] if x in idx_to_token else "[UNK]")
    
    
    
    fig, ax = plt.subplots(figsize=(20, 15))

    # Turn off default axis
    ax.set_axis_off()

    # Create a table at the center
    table = ax.table(
        cellText    = df_tokens.values,  # 2D array of strings
        rowLabels   = df_tokens.index,   # row labels
        colLabels   = df_tokens.columns, # column labels
        cellLoc     = 'center',
        loc         = 'center'
    )

    # Make it look nicer
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    # table.scale(1.2, 1.2)  # Adjust table size if needed

    # plt.title("2D table-like figure with numeric values (no color)")
    plt.tight_layout()
    plt.savefig("./lll.png")

    




def main(args):
    
    from huggingface_hub import login
    login("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")  
    base_model = "baffo32/decapoda-research-llama-7B-hf"
    # base_model = "meta-llama/Llama-2-7b-hf"
    model, tokenizer = get_model(base_model)
    model.cuda()
    
    tokenlayer_table(tokenizer)
    exit()
    
    set_seed(42)
    
    # Prompt for sentence generation
    prompt = "the weather is so good and "

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
    
    
    tokenlayer_table(tokenizer)

        
        
        
    








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