import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union, List, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.distributed as dist


from predictor import DyPredictor

from attn import DyDynamicCache

from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
import transformers.models.llama.modeling_llama as _llama_mod
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.utils import (
    is_torchdynamo_compiling,
)
from transformers.models.llama import LlamaConfig
from transformers.generation.utils import GenerationConfig, LogitsProcessorList, StoppingCriteriaList, GenerateOutput, GenerationMode
from transformers.generation.utils import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.utils import GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput, GenerateNonBeamOutput
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled



if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.generation.streamers import BaseStreamer




@dataclass
class DyCausalLMOutputWithPast(CausalLMOutputWithPast):
    layer_prob: Any = None
@dataclass
class DyBaseModelOutputWithPast(BaseModelOutputWithPast):
    layer_feature: Any = None



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
        layer_skip_index: List = [],
        generate_mode = True,
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
            
            
            
            if i in layer_skip_index:
                # First, fill the key and value states with dummy value and record
                # Later, check the skip_control to restore key, value 
                past_key_values.update(
                    key_states = torch.tensor(0),
                    value_states = torch.tensor(0),
                    layer_idx = -100,
                )
                past_key_values.add_to_buffer(i, hidden_states)
                layer_outputs = (hidden_states, None, past_key_values)
            else: 
                # if it is not skipped computation, it should check the cache and make input
                hidden_states = past_key_values.get_past(idx=i, hidden_state=hidden_states) # it will return same thing if there are no prev states
                
                
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
        
        self.skip_block = None
        self.skip_order = None
        self.get_hidden = None
        
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
    from transformers.generation.utils import DynamicCache
     
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        
        # cache : key_cache, value_cache
        # print("init")
        # print(model_kwargs['past_key_values'].key_cache) # 1, 8(heads), token_length, 128(dim)
        # print(model_kwargs['past_key_values'].value_cache)
        # exit()

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :].float()

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            ) # update cache when the whole layers are finish
            
            # print(f'{mk}th iteration')
            # print(model_kwargs['past_key_values'].key_cache[0].shape)
            # print(model_kwargs['past_key_values'].value_cache[0].shape)

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids    
    
        
    
    # from transformers.generation.utils import GenerationMixin    
    def skip_generate():
        213122
    
    
    
        
    
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