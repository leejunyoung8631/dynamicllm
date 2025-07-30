from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
import transformers.models.llama.modeling_llama as _llama_mod
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import (
    is_torchdynamo_compiling,
)
from transformers.models.llama import LlamaConfig


from predictor import DyPredictor

from attn import DyDynamicCache




@dataclass
class DyCausalLMOutputWithPast(CausalLMOutputWithPast):
    layer_prob: Any = None
@dataclass
class DyBaseModelOutputWithPast(BaseModelOutputWithPast):
    layer_feature: Any = None



# class DyLlamaDecoderLayer(LlamaDecoderLayer):
#     def __init__(self, config, layer_idx):
#         super().__init__(config, layer_idx)
        
        
#         self.self_attn = DyLLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        





class DyLlama(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
    
    
    def get_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        skip_block=1,
    ):
    
        hidden_states = self.embed_tokens(input_ids) if input_ids is not None else inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        past_key_values = past_key_values or [None] * len(self.layers)

        next_past_key_values = []
        for i, (block, past_key) in enumerate(zip(self.layers[:skip_block], past_key_values[:skip_block])):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            attn_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key,
                output_attentions=output_attentions,
            )
            hidden_states, _, past = attn_outputs
            next_past_key_values.append(past)

            if output_attentions:
                all_attentions = all_attentions + (attn_outputs[1],)
        
        
        return hidden_states, all_hidden_states, all_attentions, past_key_values

    
    
    
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
        skip_block = 0,
        layer_skip_index: List = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

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
                past_key_values = DyDynamicCache()
            else:
                past_key_values = DyDynamicCache.from_legacy_cache(past_key_values)
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
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if (layer_skip_index is not None) and  (i in layer_skip_index):
                # cache update to layer_index error by duplicating the last input
                # (for all-in-one inputs)
                past_key_values.update(
                    key_states = torch.tensor(0),
                    value_states = torch.tensor(0),
                    layer_idx = -100,
                )
                
                # skip computation: leave hidden_states unchanged
                layer_outputs = (hidden_states, None, past_key_values)
            else: 
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
                    )
                    
            

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            
            if skip_block == i:
                layer_feature = hidden_states

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
        
        return DyBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            layer_feature=layer_feature
        )



class DyLM(LlamaForCausalLM):
    def __init__(self, config, ):
        super().__init__(config)
        self.predictor = DyPredictor()
        self.model = DyLlama(config)
        
        # self.skip_block = None
        # self.skip_order = None
        # self.get_hidden = None
        
        self.layer_prob = None
        
        
        
    # for training skip predictor
    def forward_skip(
        self,
        input_ids:        torch.LongTensor = None,
        attention_mask:   Optional[torch.Tensor] = None,
        position_ids:     Optional[torch.LongTensor] = None,
        past_key_values:  Optional[Union[Tuple[torch.FloatTensor], List[torch.FloatTensor]]] = None,
        inputs_embeds:    Optional[torch.FloatTensor] = None,
        labels:           Optional[torch.LongTensor] = None,
        use_cache:        Optional[bool] = None,
        output_attentions:Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict:      Optional[bool] = None,
        cache_position:   Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        skip_layer : List = [],
    ) -> Union[Tuple, DyCausalLMOutputWithPast]:
        
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
            skip_block=0,
            layer_skip_index=skip_layer,
        )
        
        # predictor prob
        layer_features = outputs["layer_feature"]
        layer_prob = self.predictor(layer_features)

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if labels is None and not is_torchdynamo_compiling():
                print(
                    "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
                )
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            # TODO: remove the float() operation in v4.46
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()
        
        
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return DyCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            layer_prob=layer_prob
        )
        
    
    from transformers.generation.utils import GenerationMixin
    
    def skip_generate():
    
    
    
        
    
    def forward(
        self,
        input_ids:        torch.LongTensor = None,
        attention_mask:   Optional[torch.Tensor] = None,
        position_ids:     Optional[torch.LongTensor] = None,
        past_key_values:  Optional[Union[Tuple[torch.FloatTensor], List[torch.FloatTensor]]] = None,
        inputs_embeds:    Optional[torch.FloatTensor] = None,
        labels:           Optional[torch.LongTensor] = None,
        use_cache:        Optional[bool] = None,
        output_attentions:Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict:      Optional[bool] = None,
        cache_position:   Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, DyCausalLMOutputWithPast]:
        # default to config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # call original model
        if self.skip_block == 0:            
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
            )
            
        skip_block = self.skip_block # the number of block for first processing before predictor
        output_hidden_states = self.get_hidden
            
        # forward only some block to get features
        hidden_states, all_hidden_states, all_attentions, past_key_values = self.model.get_features(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            skip_block=skip_block,
        ) # hidden_states [b, L, 4096]
        
        # predictor
        self.layer_prob = self.predictor(hidden_states)
        
        # forward the rest of blocks with mask or not
        model_outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=hidden_states, # to hidden_states
            output_attentions=all_attentions, # to all_attentions
            output_hidden_states=all_hidden_states, # to all_hidden_states
            return_dict=return_dict,
            skip_block=skip_block,
            layer_skip_index = self.skip_order
        )
        
        
        # prediction
        hidden_states = model_outputs[0]
        logits = self.lm_head(hidden_states)
        

        # 2) compute loss if needed
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        if not return_dict:
            output = (logits,) + model_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return DyCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )
        


# monkey patch the model
_llama_mod.LlamaModel = DyLlama
_llama_mod.LlamaForCausalLM = DyLM