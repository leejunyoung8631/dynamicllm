import os
import argparse
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax

import numpy as np
import random

from transformers import LlamaForCausalLM, LlamaModel
from transformers import  Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LLAMA_ATTENTION_CLASSES, LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding

from loss import distillation_loss




class DynLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.layer_idx = layer_idx

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        skip_mask = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        
        
        '''
        for generate process, add skip code will be needed
        '''        
        # do not maak attention 
        # in generation process, skiped layer KV cache will be copied from somewhere
        
        init_residual = hidden_states
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        
        # mask FF process to remove the effect of token
        if skip_mask is not None:
            hidden_states = hidden_states * skip_mask[:, :, self.layer_idx].unsqueeze(-1)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        if skip_mask is not None:
            hidden_states += init_residual * (1.0 - skip_mask[:, :, self.layer_idx]).unsqueeze(-1)
        

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



class DyLLMLlama(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DynLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        get_first = False,
        skip_mask = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            print(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                print(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            # temp
            if skip_mask is not None and i == 0:
                continue
            if output_hidden_states and get_first == False:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    skip_mask = skip_mask
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    skip_mask = skip_mask
                )

            hidden_states = layer_outputs[0]
            
            # this will return features from the first layer
            if get_first:
                return BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=None,
                    hidden_states=all_hidden_states,
                    attentions=all_self_attns,
                )

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


    

class DyLLM(LlamaForCausalLM):
    def __init__(self, config, ):
        super().__init__(config)
        
        # new model
        self.model = DyLLMLlama(config)
        
        # re-init weight as a new model is initialized
        self.post_init()
        
        # for running with mask
        self.run_mask = True
        self.skip_level = 11 # later change it with model init
        self.diff_mask = DifferentiableMask(self.skip_level)
        self.loss_term = "mask"
        
        # for callback function
        self.mask_loss = 0
        self.distill_loss = 0
        self.ppl_loss = 0
        
    
    def forward_with_mask(self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
            **loss_kwargs,
            ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
        # 1. predict layers using n-th features
        # 2. generate mask for token skipping
        predict_feature = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            get_first = True,
            skip_mask = None
        )
        predict_state = predict_feature[0]
        skip_mask = self.diff_mask(predict_state) # b, n, 32
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # skip the model
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=predict_state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            get_first = False,
            skip_mask = skip_mask,
        )
        hidden_states = outputs[0]
        
        
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = 0
        # if labels is not None:
            # loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)
            
            
        if "mask" in self.loss_term:
            mask_loss =  ( skip_mask - (self.config.num_hidden_layers) ) / (self.skip_level + 1)
            mask_loss = ( torch.sum(skip_mask, dim=-1) - (self.config.num_hidden_layers - self.skip_level) ) / self.skip_level        
            mask_loss = mask_loss.mean(dim=-1).mean(dim=-1)
            loss = loss + mask_loss
            self.mask_loss = mask_loss.item() # for callback function   
        
        
        if "distill" in self.loss_term:
            original_feature = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                get_first = False,
                skip_mask = None
            )
            original_hidden_state = original_feature[0]
            teacher_logits = self.lm_head(original_hidden_state[:, -num_logits_to_keep:, :])
            student_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)
            
            distill_loss = distillation_loss(student_loss=student_loss, student_logits=logits, teacher_logits=teacher_logits, labels=labels)
            loss = loss + distill_loss
            self.distill_loss = distill_loss.item()            
            
            
        # loss without ppl
        if "ppl" in self.loss_term:
            original_feature = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                get_first = False,
                skip_mask = None
            )
            original_hidden_state = original_feature[0]
            teacher_logits = self.lm_head(original_hidden_state[:, -num_logits_to_keep:, :])
            # have to fix it
            teacher_ppl = self.loss_function(logits=teacher_logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)
            student_ppl = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)
            ppl_loss = 0
            self.ppl_loss = ppl_loss
            
            
            

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
        
    
    # original implementation 
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        
        if self.run_mask:
            return self.forward_with_mask(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                cache_position,
                num_logits_to_keep,
                **loss_kwargs,
                )   
        
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    

'''
fix point (might)
- layer structure
'''
class Predictor(nn.Module):
    def __init__(self, d_feature=4096, skip_level=11):
        super().__init__()
        self.skip_level = 11 + 1 # 0 skip to 11 skip
        self.predictor = nn.Sequential(
            nn.Linear(d_feature, d_feature // 2),
            nn.Linear(d_feature // 2, self.skip_level),
        )
    
    def forward(self, x, ):
        b, n, d = x.shape
        x = self.predictor(x)
        
        return x



'''
input : feature from layers 
output : mask

- predictor
input : (b, n, d) -> (b, n, num of skipped)

- gumble softmax
(b, n, num of skipped) -> mask (continuos, approximated)

'''
class DifferentiableMask(nn.Module):
    def __init__(
            self,
            skip_level = 11,
            hard=False,
            temperature=3.0, 
            scale_multiplier=[1e3, 1e4], 
        ):
        '''
        Implementation of differantiable mask learner
        args:
            temperature: temperature of the gumbel distribution
            gate_param: the parameter to be masked
            init_prob: initial probability of the mask
            scale_multiplier: multiplies learned gates by this value, it is needed to make the gates more sensitive to small learning rates
            initialization: "none" means we start from 0.95 for all and dont bother with initialization, "initial_mask" means we start from the initial mask
            hard: sampling mask for full gumbel
            
        temperature parameter needs more attention, we should do annealing it during training
        '''
        super().__init__()
        
        
        # simple parameter setting
        hard=False 
        temperature=[4, 0.1]
        scale_multiplier=[1e3, 1e4]
    
        self.mask_difference = 1.0
        self.temperature = temperature
        self.scale_multiplier = scale_multiplier
        
        # mask setting
        mask_options = self.make_mask(skip_level=skip_level).cuda()
        self.register_buffer("mask_options", mask_options)
        self.hard = hard

        self.current_scale_multiplier = self.scale_multiplier[0]
        self.current_temperature = self.temperature[0]
        self.current_max_prob = 0.0
        
        # predictor 
        self.skip_level = skip_level
        self.mask_predictor = Predictor(skip_level=skip_level)
        self.iteration = None # This will be set through trainer
        self.whole_iter = None # This will be set through trainer
        
        # turn it false when inference
        self.training = True
    
    
    
    def make_mask(self, skip_level):
        token_mask = None
        order = [24, 26, 25, 10, 27, 13, 22, 14, 9, 29, 12]
        
        index = torch.arange(skip_level).unsqueeze(0)
        b, n = index.shape
        token_mask = torch.ones((b, skip_level+1, 32))
        for d in range(b):
            for t in range(n):
                n_zero = index[d, t]
                token_mask[d, t+1, order[:n_zero]] = 0
                
        # token_mask = token_mask.permute(0, 2, 1) # (b, n, 32) -> (b, 32, n)
        
        return token_mask
    
    

    def forward(self, x, ): 
        # predict the number of skipped using predictor
        x = self.mask_predictor(x)
        
        # predictor as Gumbel parameters
        gate = x 
        
        if self.training:
            start_temp, end_temp = self.temperature 
            self.current_temperature = start_temp + (end_temp - start_temp) * (self.iteration / self.whole_iter)
            start_scale, end_scale = self.scale_multiplier
            self.current_scale_multiplier = start_scale + (end_scale - start_scale) * (self.iteration / self.whole_iter)
            
            sampling_tensor = gate * self.current_scale_multiplier
            choices = gumbel_softmax(sampling_tensor, tau=self.current_temperature, hard=self.hard)
            self.current_max_prob = choices.max(-1)[0].mean().item()
            backprop_gate = (choices @ self.mask_options).squeeze(1)
        else:
            # just based on the maximum logit
            backprop_gate = self.mask_options[torch.arange(self.mask_options.shape[0]), self.gate.argmax(dim=-1)]
            
        self.sampled_gate = backprop_gate
        
        return backprop_gate