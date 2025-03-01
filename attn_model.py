import os
import sys
import argparse
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel

# Custom model config - by Junyoung
import json
from safetensors import safe_open



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
    # special_token = "<|reserved_special_token_0|>"
    # if tokenizer.convert_tokens_to_ids(special_token) == tokenizer.unk_token_id:
    #     print(f"The token '{special_token}' is not in the tokenizer. Adding it...")
    #     tokenizer.add_special_tokens({"pad_token": special_token})
    # else:
    #     print(f"The token '{special_token}' already exists in the tokenizer.")
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = 'right'  # padding to the right

    # model = set_model_device_evalmode(model, device, fix_decapoda_config, use_bfloat)
    return model, tokenizer






def main(args):
    # Load Pruned Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = get_model(args.base_model, args.cache_model_dir, device, args.use_bfloat)
    print(model)
    
    
    exit()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merging Lora-Tuned LLaMA')

    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    parser.add_argument("--cache_model_dir", type=str, default="./model_cache", help="llm weights")
    parser.add_argument("--update_layer_config", default=False, action="store_true", help="Update config for the pruned-and-tuned model")
    parser.add_argument("--save_bin_model", default=False, action="store_true", help="Save model into a bin format")

    args = parser.parse_args()
    main(args)




