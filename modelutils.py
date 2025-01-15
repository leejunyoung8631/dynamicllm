import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, AutoModelForCausalLM, LlamaTokenizer




def get_model(base_model, model_class=None, tokenize_name=None, is_decapoda=False):
    if model_class == None:
        model_class = AutoModelForCausalLM
    if tokenize_name == None:
        tokenize_name = LlamaTokenizer
    
    tokenizer = tokenize_name.from_pretrained(base_model)
    model = model_class.from_pretrained(base_model, low_cpu_mem_usage=False)
    
    # model.config.output_hidden_states = True
    model.config.output_hidden_states = False
    model.config.return_dict = True
    
    if is_decapoda == True:
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        tokenizer.pad_token_id = 0
    
    
    return model, tokenizer