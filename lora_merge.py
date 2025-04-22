import argparse
import csv
import os
import time

import numpy as np
import torch
from dataset import get_loaders
from tqdm import tqdm

from datautil import set_seed
from modelutils import get_model
from modelutils import load_mask_weight, set_inference

from peft import PeftModel



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="baffo32/decapoda-research-llama-7B-hf",
        help="base model name",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="dyllm",
        help="chosse in [dyllm, ...]",
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
    parser.add_argument(
        "--fix_decapoda_config",
        default=False,
        action="store_true",
        help="fix tokenizer config of baffo32/decapoda-research-llama-7B-hf",
    )
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    
    parser.add_argument("--loss_term", default=None)
    parser.add_argument("--mask_weight", default="./baffo32/decapoda-research-llama-7B-hf")
    
    # callback for checking the average number of blocks skipped
    parser.add_argument("--check_count", action="store_true", help="if True, check the number of skipped blocks")
    parser.add_argument("--is_generation", action="store_true", )
    
    # For peft model
    parser.add_argument("--lora_model", action="store_true", default=False)
    parser.add_argument("--lora_weight", default=None)    
    parser.add_argument("--output_dir", type=str, default="dsadsa")
    parser.add_argument("--save_bin_model", action="store_true", default=False)
    
    
    args = parser.parse_args()
    # from peft.tuners.lora.model
    
    
    # for test
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM("meta-llama/Llama-2-7b-chat-hf")
    model = PeftModel.from_pretrained(model, "./lora-alpaca2/checkpoint-2200")
    model.merge_and_unload()
    print(model)
    
    
    
    
    
    exit()
    
    
    
    
    
    
    # internally, it create model using "transformers.models.auto.modeling_auto.AutoModelForCausalLM"
    # by hijacking the config class, init the model
    from transformers import AutoModelForCausalLM
    from transformers.models.llama.configuration_llama import LlamaConfig
    from dyllm_model import DyLLM
    AutoModelForCausalLM.register(LlamaConfig, DyLLM, exist_ok=True)
    
    
    from huggingface_hub import login
    login("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")  
    
    # init default model
    model, tokenizer = get_model(base_model=args.base_model, model_class=args.model_class, loss_term=args.loss_term)    
    model = load_mask_weight(model, args.mask_weight)
    # model = set_inference(model, args)
    
    
    # init peft model
    if args.lora_model == False:
        print("set lora weight path")
        exit()
    
    model = PeftModel.from_pretrained(model, args.lora_weight, torch_dtype=torch.float16,)
    model.train()
    model.merge_and_unload()
    
    if args.save_bin_model:
        model.half()

        os.makedirs(args.output_dir)
        out_filename = args.output_dir + "/pytorch_model.bin"
        torch.save({
            'model': model, 
            'tokenizer': tokenizer,
        }, out_filename)

        print("Merged model stored in {}".format(out_filename))
        exit()
    
    

    model.save_pretrained(args.output_dir, max_shard_size="16GB",)