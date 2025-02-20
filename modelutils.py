import os 
import torch

from safetensors.torch import load_file

import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, AutoModelForCausalLM, LlamaTokenizer

from dyllm_model import DyLLM

MODEL_MAPPING = {
    "dyllm" : DyLLM,
}


def get_model(base_model, model_class="normal", tokenize_name=None, is_decapoda=False, loss_term=None):
    if model_class == "normal":
        model_class = AutoModelForCausalLM
    else: 
        model_class = MODEL_MAPPING[model_class]
        
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
    if weight_file is None:
        print("no weight file")
        return model
    
    # mask weight is saved at "model-00006-of-00006.safetensors"    
    weight_file = os.path.join(weight_file, "model-00006-of-00006.safetensors")
    model_dict = load_file(weight_file)
    
    # filter only mask weight
    mask_weight = {k: v for k, v in model_dict.items() if "diff_mask" in k}
    print(f"loaded weight names: {mask_weight.keys()}")
    
    model.load_state_dict(mask_weight, strict=False)
    
    
    return model



def set_inference(model, args):
    # set inference mode for mask
    if hasattr(model, "diff_mask"):
        model.diff_mask.training = False
    
    if args.is_generation:
        model.generation_skip = True
    
    if args.check_count:
        model.check_count = True
    
    model.eval()
    
    return model


def set_training(model, args):
    # set inference mode for mask
    if hasattr(model, "diff_mask"):
        model.diff_mask.training = True
        model.diff_mask.hard = True
    
    
    
    return model
    


def debug_info(model, output_file):
    
    
    
    
    
    
    
    return





