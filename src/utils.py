import torch
import numpy as np
import random
import gc
# import bitsandbytes as bnb

from transformers.integrations.bitsandbytes import replace_with_bnb_linear
from transformers.utils.quantization_config import BitsAndBytesConfig


from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)



def get_tokenizer(
    base_model=None,
    cache_dir=None,
):
    if base_model.endswith(".bin"):
        # Pruning model
        pruned_dict = torch.load(base_model, map_location='cpu')
        tokenizer = pruned_dict['tokenizer']
        fix_decapoda_config = False
    else:
        tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                cache_dir=cache_dir, 
                )
        fix_decapoda_config = "decapoda-research/llama-7b-hf" == base_model.lower()
        if fix_decapoda_config:
            tokenizer.pad_token_id = 0
            tokenizer.padding_side = "left"

    return tokenizer




def get_model(
    base_model=None,
    cache_dir=None,
    device="cuda",
    use_bfloat=False,
    load_in_8bit=False,  # Only supported for HF model that uses AutoModelForCausalLM
    model_class=AutoModelForCausalLM,
):
    if base_model.endswith(".bin"):
        # Pruning model
        pruned_dict = torch.load(base_model, map_location='cpu')
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        fix_decapoda_config = False

        if load_in_8bit:
            # Define quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,  # Enables 8-bit quantization
                llm_int8_threshold=6.0,  # Default threshold for LLM.int8() method
            )

            # Apply 8-bit quantization
            quant_model = replace_with_bnb_linear(model, quantization_config=quantization_config)
            pruned_dict = torch.load(base_model, map_location='cpu')
            _, model = pruned_dict['tokenizer'], pruned_dict['model']
            quant_model.load_state_dict(model.state_dict(), strict=False)
            quant_model.to('cuda')
            model = quant_model
    else:
        # Default HF model
        config = AutoConfig.from_pretrained(base_model)
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
        if not load_in_8bit:
            model = model_class.from_pretrained(
                    base_model, low_cpu_mem_usage=True,
                    cache_dir=cache_dir, 
                    device_map="cuda",
                    torch_dtype=torch.float16,
                    )
        else:
            model = model_class.from_pretrained(
                    base_model, low_cpu_mem_usage=True,
                    cache_dir=cache_dir, 
                    device_map="cuda",
                    load_in_8bit=True
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

    model = set_model_device_evalmode(model, device, fix_decapoda_config, use_bfloat, load_in_8bit)
    return model, tokenizer


def set_model_device_evalmode(
    model, device, fix_decapoda_config=False, use_bfloat=False, load_in_8bit=False
):
    if "cuda" in device:
        if not load_in_8bit:
            model.half()
            model = model.to(device)

    if fix_decapoda_config:
        # unwind broken decapoda-research config
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    model.eval()

    if use_bfloat and not load_in_8bit:
        model = model.bfloat16()

    gc.collect()
    torch.cuda.empty_cache()
    return model


def set_seed(random_seed=1234):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)



def version_warning():
    torch_version = torch.__version__
    major, minor, _ = torch_version.split('.')
    major = int(major)
    minor = int(minor)
    
    if (major, minor, ) < (2, 1, ):
        print("Torch version below 2.1.0 may not support tri() calcuation for torch.bfloat")
        print("This may raise error in attention process")
        print("In this case, please refer below")
        print("https://github.com/meta-llama/llama3/issues/110")

