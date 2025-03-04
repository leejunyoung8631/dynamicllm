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

from typing import Optional, Union

from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.generation.streamers import BaseStreamer


import matplotlib.pyplot as plt
import seaborn as sns


# # Custom model config -> for analysis activation
# from transformers import LlamaForCausalLM
# class LlamaForAttn(LlamaForCausalLM):
#     def __init__(self, config):
#         super().__init__(config)
    
    
#     def _sample(
#         self,
#         input_ids: torch.LongTensor,
#         logits_processor: LogitsProcessorList,
#         stopping_criteria: StoppingCriteriaList,
#         generation_config: GenerationConfig,
#         synced_gpus: bool,
#         streamer: Optional["BaseStreamer"],
#         **model_kwargs,
#     ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
#         r"""
#         Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
#         can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

#         Parameters:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 The sequence used as a prompt for the generation.
#             logits_processor (`LogitsProcessorList`):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
#                 used to modify the prediction scores of the language modeling head applied at each generation step.
#             stopping_criteria (`StoppingCriteriaList`):
#                 An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
#                 used to tell if the generation loop should stop.
#             generation_config ([`~generation.GenerationConfig`]):
#                 The generation configuration to be used as parametrization of the decoding method.
#             synced_gpus (`bool`):
#                 Whether to continue running the while loop until max_length (needed to avoid deadlocking with
#                 `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
#             streamer (`BaseStreamer`, *optional*):
#                 Streamer object that will be used to stream the generated sequences. Generated tokens are passed
#                 through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
#             model_kwargs:
#                 Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
#                 an encoder-decoder model the kwargs should include `encoder_outputs`.

#         Return:
#             [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
#             A `torch.LongTensor` containing the generated tokens (default behaviour) or a
#             [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
#             `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
#             `model.config.is_encoder_decoder=True`.
#         """
#         # init values
#         pad_token_id = generation_config._pad_token_tensor
#         output_attentions = generation_config.output_attentions
#         output_hidden_states = generation_config.output_hidden_states
#         output_scores = generation_config.output_scores
#         output_logits = generation_config.output_logits
#         return_dict_in_generate = generation_config.return_dict_in_generate
#         max_length = generation_config.max_length
#         has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
#         do_sample = generation_config.do_sample

#         # init attention / hidden states / scores tuples
#         scores = () if (return_dict_in_generate and output_scores) else None
#         raw_logits = () if (return_dict_in_generate and output_logits) else None
#         decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
#         cross_attentions = () if (return_dict_in_generate and output_attentions) else None
#         decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

#         # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
#             encoder_hidden_states = (
#                 model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
#             )

#         # keep track of which sequences are already finished
#         batch_size, cur_len = input_ids.shape
#         this_peer_finished = False
#         unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
#         model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

#         while self._has_unfinished_sequences(
#             this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
#         ):
#             # prepare model inputs
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

#             # prepare variable output controls (note: some models won't accept all output controls)
#             model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
#             model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

#             # forward pass to get next token
#             outputs = self(**model_inputs, return_dict=True)

#             # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
#             model_kwargs = self._update_model_kwargs_for_generation(
#                 outputs,
#                 model_kwargs,
#                 is_encoder_decoder=self.config.is_encoder_decoder,
#             )
#             if synced_gpus and this_peer_finished:
#                 continue

#             # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
#             # (the clone itself is always small)
#             next_token_logits = outputs.logits.clone()[:, -1, :].float()
#             next_token_logits = next_token_logits.to(input_ids.device)

#             # pre-process distribution
#             next_token_scores = logits_processor(input_ids, next_token_logits)

#             # Store scores, attentions and hidden_states when required
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += (next_token_scores,)
#                 if output_logits:
#                     raw_logits += (next_token_logits,)
#                 if output_attentions:
#                     decoder_attentions += (
#                         (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
#                     )
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += (outputs.cross_attentions,)

#                 if output_hidden_states:
#                     decoder_hidden_states += (
#                         (outputs.decoder_hidden_states,)
#                         if self.config.is_encoder_decoder
#                         else (outputs.hidden_states,)
#                     )

#             # token selection
#             if do_sample:
#                 probs = nn.functional.softmax(next_token_scores, dim=-1)
#                 # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
#                 next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
#             else:
#                 next_tokens = torch.argmax(next_token_scores, dim=-1)

#             # finished sentences should have their next token be a padding token
#             if has_eos_stopping_criteria:
#                 next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

#             # update generated ids, model inputs, and length for next step
#             input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
#             if streamer is not None:
#                 streamer.put(next_tokens.cpu())

#             unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
#             this_peer_finished = unfinished_sequences.max() == 0
#             cur_len += 1

#             # This is needed to properly delete outputs.logits which may be very large for first iteration
#             # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
#             del outputs

#         if streamer is not None:
#             streamer.end()

#         if return_dict_in_generate:
#             if self.config.is_encoder_decoder:
#                 return GenerateEncoderDecoderOutput(
#                     sequences=input_ids,
#                     scores=scores,
#                     logits=raw_logits,
#                     encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions,
#                     cross_attentions=cross_attentions,
#                     decoder_hidden_states=decoder_hidden_states,
#                     past_key_values=model_kwargs.get("past_key_values"),
#                 )
#             else:
#                 return GenerateDecoderOnlyOutput(
#                     sequences=input_ids,
#                     scores=scores,
#                     logits=raw_logits,
#                     attentions=decoder_attentions,
#                     hidden_states=decoder_hidden_states,
#                     past_key_values=model_kwargs.get("past_key_values"),
#                 )
#         else:
#             return input_ids



# Custom model config - by Junyoung
# from transformers.models.llama.configuration_llama import LlamaConfig
# AutoModelForCausalLM.register(LlamaConfig, LlamaForAttn, exist_ok=True)
# End of custom confi


# from utils import get_model, set_model_device_evalmode, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from generation_utils import generate_with_instruction, generate_with_instruction_deepseek
from transformers.models.llama import LlamaForCausalLM
import numpy as np

import re




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





def analysis1(whole_data, layer_index):
    layer_heads_value = [] 
    for i in range(32):
        layer_heads_value.append([])
    
    
    for i in range(149, len(whole_data), 12):
        
        # 2 + 12 tokens
        p_data = whole_data[i][:, :, :2]        
        b_data = whole_data[i][:, :, -12:] # last 12 tokens
        data = np.concatenate([p_data, b_data], axis=-1)
        
        # last 12 tokens
        # data = whole_data[i][:, :, -12:] # last 12 tokens
        
        for j, id in enumerate(layer_index):
            s = layer_index[j-1]
            e = layer_index[j]
            if j == 0:
                s = 0
            heads = data[s:e]
            heads_value = np.sum(heads, axis=-1).squeeze(axis=-1)
            layer_heads_value[j].append(heads_value.mean())

    
    for k, head_data in enumerate(layer_heads_value):
        plt.figure(figsize=(15, 10))
        plt.hist(head_data, bins=50, range=(0, 1), edgecolor='black', alpha=0.7)
        plt.xlabel("Value Range (0 to 1)")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {k}_layer_head")
        plt.grid(axis='y', linestyle='--', alpha=0.7) 
        
        plt.rc('font', size=20)        # 기본 폰트 크기
        plt.rc('axes', labelsize=20)   # x,y축 label 폰트 크기
        plt.rc('xtick', labelsize=20)  # x축 눈금 폰트 크기 
        plt.rc('ytick', labelsize=20)  # y축 눈금 폰트 크기
        plt.rc('legend', fontsize=20)  # 범례 폰트 크기
        plt.rc('figure', titlesize=50) # figure title 폰트 크기
        
        plt.savefig(f"./attnmap/attn_token2_fig_14/{k}_layer_headvalue.png")
        plt.close()
        


def analysis2(whole_data, layer_index):
    concat_data = []
    target_data = whole_data[1:25]
    _, _, max_length = target_data[-1].shape
    
    
    for data in target_data:
        _, _, current_length = data.shape
        if current_length < max_length:
            data = np.pad(data, ((0, 0), (0, 0), (0, max_length - current_length)), mode='constant', constant_values=0)
        concat_data.append(data)
    
    concat_data = np.concatenate(concat_data, axis=1)
    
    for j in range(len(layer_index)):
        s = layer_index[j-1]
        e = layer_index[j]
        if j == 0:
            s = 0
        
        layer_heads = concat_data[s:e]
        mean_layer_heads = layer_heads.mean(axis=0)
        mean_layer_heads = mean_layer_heads[:, :10] # last 12
        
        # Create the heatmap
        plt.figure(figsize=(30, 15))
        sns.heatmap(mean_layer_heads, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5)

        # Display the heatmap
        plt.savefig(f"./attnmap/attn_token4_heatmap/{j}_layer_attn.png")
        plt.close()
    
    
        
        
    
    
    
    




def main():    
    
    # parsing existing result
    outputfile = "./result_text/out_seed_rrrx.json" ######################################################3
    with open(outputfile, "r") as file:
        data = json.load(file)
    
    data = data[0]
    inst = data["instruction"]
    inp = data["input"]
    output = data["output"]
    
    modelfile = "/disk/yskim/LLM-Pruner/cellprune_results/llama31_25pruned_part75/merged/pytorch_model.bin" ######################################################3
    _, tokenizer = get_model(base_model=modelfile)
    
    inst_ids = tokenizer(inst, add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]
    
    decoded_inst = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in inst_ids]
    # print("Decoded Inst Tokens:", len(decoded_inst))
    
    decoded_output = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in output_ids]
    # print("Decoded Output Tokens:", len(decoded_output)) # 2048
    
    
    # from output index 137, it repeat the patterns
    # 149, 161, 173, 185, 197
    
    
    # get attentions
    layer_index = []
    layer_index.extend([32, ] * 4) # 1 ~ 4 layers
    layer_index.extend([24, ] * 26) # 5 ~ 28 layers
    layer_index.extend([32, ] * 2) # 29 ~ 32 layers
    layer_index = np.cumsum(layer_index)
    
    
        
    path = "./attnmap/attn_token3" ######################################################3
    files = sorted(os.listdir(path))
    
    files_sorted = sorted(files, key=lambda f: int(re.match(r"(\d+)_token.npy", f).group(1)))
    _, _, max_length = np.load(os.path.join(path, files_sorted[-1])).shape
    
    whole_data = []
    for i, file in enumerate(files_sorted):
        filepath = os.path.join(path, file)
        data = np.load(filepath)
        whole_data.append(data)
    
    '''
    analysis 1
    : sum of the probability of context length repeated length
    '''
    # analysis1(whole_data=whole_data, layer_index=layer_index)
    
    
    '''
    analysis 2
    : see the final output
    '''
    analysis2(whole_data=whole_data, layer_index=layer_index)

    

    
    
    
        # _, _, current_length = data.shape
        
        # if current_length < max_length:
        #     data = np.pad(data, ((0, 0), (0, 0), (0, max_length - current_length)), mode='constant', constant_values=0)
        
        
    
    
    
    
    

if __name__ == "__main__":   
    main()