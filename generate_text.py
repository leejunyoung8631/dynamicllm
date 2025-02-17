import argparse
import csv
import os
import time

import numpy as np
import torch
from dataset import get_loaders
from tqdm import tqdm

from dyllm_model import DyLLM
from datautil import set_seed
from modelutils import get_model
from modelutils import load_mask_weight, set_inference




def generate_txt(
    output_dir,
    model,
    tokenizer,
    input_prompt="The Leaning Tower of Pisa is known for",
    num_output=5,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    max_seq_len=128,
    device="cuda",
):
    # generate a few samples
    txt_path = os.path.join(output_dir, "gen_text.txt")
    inputs = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to(device)
    input_len = inputs[0].size(0)

    with open(txt_path, "w", encoding="utf8") as f:
        f.write("=== input ===\n")
        f.write(f"{input_prompt}\n")

    for i in range(num_output):
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_length=(input_len + max_seq_len),
                min_length=(
                    input_len + max_seq_len
                ),  # forced output length (to avoid <EOS> sampling)
                return_dict_in_generate=True,
            )
        s = generation_output.sequences[0]
        output_len = len(s)
        output = tokenizer.decode(s)

        print(f"=== output {i} | leng gen {output_len-input_len} + input {input_len}\n")
        print(output)

        with open(txt_path, "a", encoding="utf8") as f:
            f.write(
                f"=== output {i} | leng gen {output_len-input_len} + input {input_len}\n"
            )
            f.write(f"{output}\n")
            




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="baffo32/decapoda-research-llama-7B-hf",
        help="base model name",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="if None, base model name is used"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="pretrain",
        choices=["pretrain", "pruneLLM", "tune_pruneLLM"],
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lora_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--input_prompt", type=str, default="The Leaning Tower of Pisa is known for"
    )
    parser.add_argument("--num_output", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/llama-7b-hf/ppl")
    parser.add_argument(
        "--fix_decapoda_config",
        default=False,
        action="store_true",
        help="fix tokenizer config of baffo32/decapoda-research-llama-7B-hf",
    )
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    
    parser.add_argument("--loss_term", default="mask")
    parser.add_argument("--mask_weight", default="./baffo32/decapoda-research-llama-7B-hf")
    parser.add_argument("--is_generation", action="store_true")
    
    args = parser.parse_args()
    
    from huggingface_hub import login
    login("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")  
    
    set_seed(args.seed)
    model, tokenizer = get_model(base_model=args.base_model, model_class=DyLLM, loss_term=args.loss_term)    
    model = load_mask_weight(model, args.mask_weight)
    model = set_inference(model, )
    model = model.cuda()
    
    
    generate_txt(
       output_dir=args.output_dir,
       model=model,
       tokenizer=tokenizer,
       input_prompt=args.input_prompt,
       num_output=args.num_output,
       top_k=args.top_k,
       top_p=args.top_p,
       temperature=args.temperature,
       max_seq_len=args.max_seq_len,
       device=args.device,
    )
    
    
    