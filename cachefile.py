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





def get_model(base_model):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)    
    model = AutoModelForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=False)
    
    model.config.output_hidden_states = True
    model.config.return_dict = True
    
    # for decapoda-research-llama-7B-hf
    model.config.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    tokenizer.pad_token_id = 0
    
    return model, tokenizer



import torch
from transformers import DynamicCache


from huggingface_hub import login
login("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")  

base_model = "baffo32/decapoda-research-llama-7B-hf"
model, tokenizer = get_model(base_model)
model.cuda()

past_key_values = DynamicCache()
messages = "the weather is so good and "
inputs = tokenizer(messages, return_tensors="pt", return_dict=True).to("cuda")

input_ids_list = inputs["input_ids"].tolist()[0]
for token_id in input_ids_list:
    print(token_id, tokenizer.decode([token_id]))

generated_ids = inputs.input_ids
cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device="cuda")
max_new_tokens = 10

for _ in range(max_new_tokens):
    outputs = model(**inputs, cache_position=cache_position, past_key_values=past_key_values, use_cache=True)
    # Greedily sample one next token
    next_token_ids = outputs.logits[:, -1:].argmax(-1)
    generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
    # Prepare inputs for the next generation step by leaaving unprocessed tokens, in our case we have only one new token
    # and expanding attn mask for the new token, as explained above
    attention_mask = inputs["attention_mask"]
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
    cache_position = cache_position[-1:] + 1 # add one more position for the next token
    for layer_idx, (k, v) in enumerate(outputs.past_key_values):
        if layer_idx == 5:
            print(f"Layer {layer_idx} - k.shape: {k.shape}, v.shape: {v.shape}")


print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])