import os 
import torch

from safetensors.torch import load_file

import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, AutoModelForCausalLM, LlamaTokenizer

from dyllm_model import DyLLM


def get_model(base_model, model_class=None, tokenize_name=None, is_decapoda=False, loss_term=None):
    if model_class == None:
        model_class = AutoModelForCausalLM
    if tokenize_name == None:
        tokenize_name = LlamaTokenizer
    
    tokenizer = tokenize_name.from_pretrained(base_model)
    model = model_class.from_pretrained(base_model, low_cpu_mem_usage=False)
    
    model.config.output_hidden_states = False
    model.config.return_dict = True
    
    
    if is_decapoda == True:
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        tokenizer.pad_token_id = 0
        
      
    # for calculate loss
    if loss_term is not None:
        setattr(model, "loss_term", loss_term.split(","))
    
    return model, tokenizer


def load_mask_weight(model, weight_file):
    # mask weight is saved at "model-00006-of-00006.safetensors"    
    weight_file = os.path.join(weight_file, "model-00006-of-00006.safetensors")
    model_dict = load_file(weight_file)
    
    # filter only mask weight
    mask_weight = {k: v for k, v in model_dict.items() if "diff_mask" in k}
    print(f"loaded weight names: {mask_weight.keys()}")
    
    model.load_state_dict(mask_weight, strict=False)
    
    
    return model



def set_inference(model, is_generation=False):
    # set inference mode for mask
    model.diff_mask.training = False
    
    if is_generation:
        model.generation_skip = True
    
    return model
    