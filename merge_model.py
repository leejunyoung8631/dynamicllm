import os
import sys
import argparse
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM

from peft import PeftModel

# Custom model config - by Junyoung
import json
from safetensors import safe_open
'''
it will add layer configuration to config.json to init model
if already added -> automatically skip this process
'''
def add_layer_configuration_to_config(merged_path):
    '''
    merged_path : path to the merged weight
    '''
    # get layer info from weight
    layer_config = {}
    each_layer_dict = {}

    # TODO: "model.safetensors" could be split into multiple files, better way to handle the multiple files?, not hardcoded filename?
    with safe_open(os.path.join(merged_path, "model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            if "layers" not in key.lower():
                layer_config[key] = list(f.get_tensor(key).shape)
            else:
                idx = int(key.split(".")[2])
                if idx not in each_layer_dict:
                    each_layer_dict[idx] = dict()
                each_layer_dict[idx][key] = list(f.get_tensor(key).shape)[::-1]

    each_layer = []  # Dict to list
    for idx in range(max(each_layer_dict.keys())+1):
        each_layer.append(each_layer_dict[idx])

    layer_config["layers"] = each_layer

    # add to config.json
    config_path = os.path.join(merged_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    if "layers" not in config:
        print("Adding 'layers' to config.json...")
        config.update(layer_config)
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print("'layers' added successfully.")
    else:
        print("'layers' already exists in config.json.")
# End of custom config









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
    #tokenizer.padding_side = 'right'  # padding to the right

    # model = set_model_device_evalmode(model, device, fix_decapoda_config, use_bfloat)
    return model, tokenizer








def main(args):
    # Load Pruned Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = get_model(args.base_model, args.cache_model_dir, device, args.use_bfloat)
    
    model = PeftModel.from_pretrained(
        model,
        args.lora_ckpt,
        torch_dtype=torch.float16,
    )
    
    print(model)
    exit()
    
    description = "Model: {}\n LORA ckpt: {}".format(args.base_model, args.lora_ckpt)
    
    # for name, param in model.named_parameters():
        # print(name)

    model = model.merge_and_unload()

    # Temporal patch - Junyoung needs to check
    # Saving a merged model with the following way looks working. Why? What's the difference?
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_bin_model:
        model.half()

        out_filename = args.output_dir + "/pytorch_model.bin"
        torch.save({
            'model': model, 
            'tokenizer': tokenizer,
        }, out_filename)

        print("Merged model stored in {}".format(out_filename))
        exit()

    model.save_pretrained(
        args.output_dir,
        max_shard_size="16GB",
    )
    tokenizer.save_pretrained(args.output_dir)

    if args.update_layer_config:
        add_layer_configuration_to_config(args.output_dir)

    print("Merged model stored in {}".format(args.output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merging Lora-Tuned LLaMA')

    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--lora_ckpt', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    parser.add_argument("--cache_model_dir", type=str, default="./model_cache", help="llm weights")
    parser.add_argument("--update_layer_config", default=False, action="store_true", help="Update config for the pruned-and-tuned model")
    parser.add_argument("--save_bin_model", default=False, action="store_true", help="Save model into a bin format")

    args = parser.parse_args()
    main(args)