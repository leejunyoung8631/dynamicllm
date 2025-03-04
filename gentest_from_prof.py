import os
import sys
import argparse
import torch
import json
import transformers
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

import torch.nn as nn

# # Custom model config - by Junyoung
# from transformers.models.llama.configuration_llama import LlamaConfig
# from custom_llama import CustomLlamaForCausalLM
# AutoModelForCausalLM.register(LlamaConfig, CustomLlamaForCausalLM, exist_ok=True)
# # End of custom config


from typing import Optional, Union

from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.generation.streamers import BaseStreamer


# from transformers.models.llama.configuration_llama import LlamaConfig
# from custom_llama import CustomLlamaForCausalLM
# AutoModelForCausalLM.register(LlamaConfig, CustomLlamaForCausalLM, exist_ok=True)



# from utils import get_model, set_model_device_evalmode, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from generation_utils import generate_with_instruction, generate_with_instruction_deepseek


from transformers.models.llama import LlamaForCausalLM



import numpy as np
import random


def set_seed(random_seed=1234):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)



def get_model(
    base_model=None,
    cache_dir=None,
    device="cuda",
    use_bfloat=False,
):
    if base_model.endswith(".bin"):
        # Pruning model
        pruned_dict = torch.load(base_model, map_location='cpu')
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        fix_decapoda_config = False
    else:
        # Default HF model
        # config = AutoConfig.from_pretrained(base_model)
        #model = LlamaForCausalLM.from_pretrained(
        #        base_model, low_cpu_mem_usage=True,
        #        cache_dir=cache_dir, 
        #        device_map=device,
        #        torch_dtype=torch.float16,
        #        )
        #tokenizer = LlamaTokenizer.from_pretrained(
        #        base_model,
        #        cache_dir=cache_dir, 
        #        )

        # Recent llama model can be loaded as follows
        model = AutoModelForCausalLM.from_pretrained(
                base_model, low_cpu_mem_usage=True,
                cache_dir=cache_dir, 
                device_map="cuda",
                torch_dtype=torch.float16,
                )
        tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                cache_dir=cache_dir, 
                )
        
        fix_decapoda_config = "decapoda-research/llama-7b-hf" == base_model.lower()
        if fix_decapoda_config:
            tokenizer.pad_token_id = 0
            tokenizer.padding_side = "left"

    # The token to check - https://github.com/unslothai/unsloth/issues/416
    special_token = "<|reserved_special_token_0|>"
    if tokenizer.convert_tokens_to_ids(special_token) == tokenizer.unk_token_id:
        print(f"The token '{special_token}' is not in the tokenizer. Adding it...")
        tokenizer.add_special_tokens({"pad_token": special_token})
    else:
        print(f"The token '{special_token}' already exists in the tokenizer.")
    model.config.pad_token_id = tokenizer.pad_token_id
    #tokenizer.padding_side = 'right'  # padding to the right

    # model = set_model_device_evalmode(model, device, fix_decapoda_config, use_bfloat)
    return model, tokenizer







def load_model(model_name, device="cuda"):
    if model_name.endswith(".bin"):
        pruned_dict = torch.load(model_name)
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        # generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        
        # for attention analysis
        model.config._attn_implementation = "eager"
        model.config.output_attentions = True
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    return model, tokenizer



def main(args):
    set_seed(args.seed)
    
    # Load Pruned Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.base_model}")
    
    model, tokenizer = load_model(args.base_model, device="cuda")
    
    # model, tokenizer = get_model(args.base_model, args.cache_model_dir, device, args.use_bfloat)
    model = model.half()
    model = model.cuda()

    if args.input_file.endswith(".json"):
        with open(args.input_file, 'r') as file:
            data = json.load(file)
    elif args.input_file.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=args.input_file)
        default_key = "train"
        print(f"Dataset: {args.input_file}, Keys: {dataset.keys()}, # of samples [{default_key}]: {len(dataset[default_key])}")
        data = dataset[default_key]
    else:
        raise NotImplementedError

    new_data = []
    

    for idx, item in tqdm(list(enumerate(data))):
        if item["input"] != "":
            text = "Below is a question paired with an input that provides further context. Write an appropriate response.\n### Question:\n{}\n\n### Input:\n{}".format(
                    item["instruction"], item["input"])
        else:
            text = item["instruction"]

        if args.use_deepseek:
            output_text = generate_with_instruction_deepseek(model, tokenizer, text, device, args.temperature)
        else:
            output_text = generate_with_instruction(model, tokenizer, text, device, args.temperature)
        new_data.append({
                    "instruction": text,
                    "input": "",
                    "output": output_text
                    })
        # if idx < 10:
            # print(text)
            # print(output_text)

        # if idx % 10 == 0:
            # print(f"Checkpoint {idx}: Generated data saved to {args.output_file}")
            # with open(args.output_file, "w", encoding="utf-8") as output_file:
                # json.dump(new_data, output_file, ensure_ascii=False, indent=4)

    if args.output_file != "n":
        with open(args.output_file, "w", encoding="utf-8") as output_file:
            json.dump(new_data, output_file, ensure_ascii=False, indent=4)
        print(f"Generated data saved to {args.output_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing with Lora-Tuned LLaMA')
    parser.add_argument('--base_model', type=str, required=True, help='base model name')
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--output_file', required=False, type=str, default="n")
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    parser.add_argument("--use_deepseek", default=False, action="store_true")
    parser.add_argument("--skip_template", default=False, action="store_true")
    parser.add_argument("--cache_model_dir", type=str, default="./model_cache", help="llm weights")
    parser.add_argument("--temperature", default=0.7, type=float, help="Temperature")
    parser.add_argument("--seed", default=1234, type=int, help="seed")

    args = parser.parse_args()
    
    
    import json

    # # Load your original JSON data from 'input.json'
    # with open(args.input_file, "r", encoding="utf-8") as f:
    #     data = json.load(f)

    # # Repeat the content 100 times
    # repeated_data = data * 100

    # # Save the repeated data into a new file 'output.json'
    # with open("repeat.json.json", "w", encoding="utf-8") as f:
    #     json.dump(repeated_data, f, ensure_ascii=False, indent=4)

    # print("Data successfully repeated and saved to output.json.")
    
    
    # exit()
    
    
    
    
    main(args)