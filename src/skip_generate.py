'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''

import wandb
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

from dotenv import load_dotenv

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




PREFIXES = [
    "Once upon a time, ",
    "In a distant future, ",
    "Imagine a world where ",
    "Earlier today, ",
    "Deep in the heart of the forest, ",
    "According to legend, ",
    "On a stormy night, ",
    "Long before anyone knew, ",
    "In an alternate reality, ",
    "As the sun was setting, "
]

import random


def main(args):
    #  Set WanDB
    os.environ["WANDB_PROJECT"] = args.wandb_project
    set_seed(args.seed)
    
    load_dotenv()
    wandb.login(key="WANDB_API_KEY")
    wandb.init(entity="ljy32051-daegu-gyeongbuk-institute-of-science-technology", project=args.wb_proj)
    # torch.set_printoptions(profile='full')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, tokenizer = get_model(args.base_model, args.cache_model_dir, device, args.use_bfloat, args.use_8bit_model)
    if args.model_class == "dylm":
        model_class = DyLM
    else:
        model_class = AutoModelForCausalLM
    
    model, tokenizer = get_model(base_model=args.base_model, cache_dir=args.cache_model_dir, device=device, use_bfloat=args.use_bfloat, load_in_8bit=args.use_8bit_model, model_class=model_class)
    # model = load_weight(model, args.weight_path, )
    
    model.skip_block = args.skip_block
    model.skip_order = args.skip_order
    model.get_hidden = args.get_hidden
    
    
    
    
    ### test for just llm architecture
    prompt = random.choice(PREFIXES)
    prompt = tokenizer(prompt, return_tensors="pt").to(device)
    
    generate_ids = model.generate(
        **prompt,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    output = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    print("🔹 Model output:")
    print(output)
    


if __name__ == "__main__":
    main()
