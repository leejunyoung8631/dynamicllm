import os
import sys
import argparse
import torch
import json
import transformers
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# # Custom model config - by Junyoung
# from transformers.models.llama.configuration_llama import LlamaConfig
# from custom_llama import CustomLlamaForCausalLM
# AutoModelForCausalLM.register(LlamaConfig, CustomLlamaForCausalLM, exist_ok=True)
# # End of custom config





# # Custom model config -> for analysis activation
from transformers import LlamaForCausalLM
class LlamaForAttn(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
    
    
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
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
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

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :].float()
            next_token_logits = next_token_logits.to(input_ids.device)

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




























# from transformers.models.llama.configuration_llama import LlamaConfig
# from custom_llama import CustomLlamaForCausalLM
# AutoModelForCausalLM.register(LlamaConfig, CustomLlamaForCausalLM, exist_ok=True)



# from utils import get_model, set_model_device_evalmode, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from generation_utils import generate_with_instruction, generate_with_instruction_deepseek


from transformers.models.llama import LlamaForCausalLM



import numpy as np
import random


def set_seed(random_seed=1234):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)



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







def load_model(model_name, device="cuda"):
    if model_name.endswith(".bin"):
        pruned_dict = torch.load(model_name)
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        # generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        
        # for attention analysis
        model.config._attn_implementation = "eager"
        model.config.output_attentions = True
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    return model, tokenizer



def main(args):
    set_seed(args.seed)
    
    # Load Pruned Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.base_model}")
    
    model, tokenizer = load_model(args.base_model, device="cuda")
    
    # model, tokenizer = get_model(args.base_model, args.cache_model_dir, device, args.use_bfloat)
    model = model.half()
    model = model.cuda()

    if args.input_file.endswith(".json"):
        with open(args.input_file, 'r') as file:
            data = json.load(file)
    elif args.input_file.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=args.input_file)
        default_key = "train"
        print(f"Dataset: {args.input_file}, Keys: {dataset.keys()}, # of samples [{default_key}]: {len(dataset[default_key])}")
        data = dataset[default_key]
    else:
        raise NotImplementedError

    new_data = []
    

    for idx, item in tqdm(list(enumerate(data))):
        if item["input"] != "":
            text = "Below is a question paired with an input that provides further context. Write an appropriate response.\n### Question:\n{}\n\n### Input:\n{}".format(
                    item["instruction"], item["input"])
        else:
            text = item["instruction"]

        if args.use_deepseek:
            output_text = generate_with_instruction_deepseek(model, tokenizer, text, device, args.temperature)
        else:
            output_text = generate_with_instruction(model, tokenizer, text, device, args.temperature)
        new_data.append({
                    "instruction": text,
                    "input": "",
                    "output": output_text
                    })
        # if idx < 10:
            # print(text)
            # print(output_text)

        # if idx % 10 == 0:
            # print(f"Checkpoint {idx}: Generated data saved to {args.output_file}")
            # with open(args.output_file, "w", encoding="utf-8") as output_file:
                # json.dump(new_data, output_file, ensure_ascii=False, indent=4)

    if args.output_file != "n":
        with open(args.output_file, "w", encoding="utf-8") as output_file:
            json.dump(new_data, output_file, ensure_ascii=False, indent=4)
        print(f"Generated data saved to {args.output_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing with Lora-Tuned LLaMA')
    parser.add_argument('--base_model', type=str, required=True, help='base model name')
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--output_file', required=False, type=str, default="n")
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    parser.add_argument("--use_deepseek", default=False, action="store_true")
    parser.add_argument("--skip_template", default=False, action="store_true")
    parser.add_argument("--cache_model_dir", type=str, default="./model_cache", help="llm weights")
    parser.add_argument("--temperature", default=0.7, type=float, help="Temperature")
    parser.add_argument("--seed", default=1234, type=int, help="seed")

    args = parser.parse_args()
    
    
    import json

    # # Load your original JSON data from 'input.json'
    # with open(args.input_file, "r", encoding="utf-8") as f:
    #     data = json.load(f)

    # # Repeat the content 100 times
    # repeated_data = data * 100

    # # Save the repeated data into a new file 'output.json'
    # with open("repeat.json.json", "w", encoding="utf-8") as f:
    #     json.dump(repeated_data, f, ensure_ascii=False, indent=4)

    # print("Data successfully repeated and saved to output.json.")
    
    
    # exit()
    
    
    
    
    main(args)