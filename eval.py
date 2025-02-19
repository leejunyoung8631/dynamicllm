import time
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import transformers
from tqdm import tqdm
from lm_eval import utils
from lm_eval.evaluator import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM, TemplateLM
from lm_eval.models.utils import pad_and_concat, Collator

from arguments import Arguments, simple_parse_args_string
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from benchmark import EvaluationMetrics


@dataclass
class EvalArguments:
    tasks: List[str] = None
    num_fewshot: Optional[int] = None
    device: Optional[str] = None
    use_cache: Optional[str] = None
    cache_requests: bool = False
    rewrite_requests_cache: bool = False
    delete_requests_cache: bool = False
    limit: Optional[int] = None
    bootstrap_iters: int = 100000
    check_integrity: bool = False
    write_out: bool = False
    log_samples: bool = True
    system_instruction: Optional[str] = None
    apply_chat_template: Union[bool, str] = False
    fewshot_as_multiturn: bool = False
    gen_kwargs: Optional[str] = None
    verbosity: str = "INFO"
    predict_only: bool = False
    random_seed: int = 0
    numpy_random_seed: int = 1234
    torch_random_seed: int = 1234
    fewshot_random_seed: int = 1234



@dataclass
class GenerationStrategyResult:
    predicted_tokens: List[int]
    acceptance_rate: Optional[float] = None


@dataclass
class GenerationResult:
    generation_strategy_result: GenerationStrategyResult
    decoded_prediction: str
    num_tokens_generated: int
    total_time: float
    time_per_token: float
    tokens_per_second: float


@dataclass
class GenerationConfig:
    max_steps: int = 512
    exit_layer: int = -1
    num_speculations: int = -1
    generation_strategy: str = "autoregressive"
    sample: bool = True
    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 0.9
    no_repeat_ngram_size: int = None
    stop_words: List[str] = None
    stop_token_ids: List[int] = None

    def __post_init__(self):
        if self.stop_token_ids is None:
            self.stop_token_ids = []


class GenerationStrategy:
    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_ids: List[int],
        generation_config: GenerationConfig,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,  
    ) -> GenerationStrategyResult:
        raise NotImplementedError()


class HuggingfaceLlamaGenerator:
    def __init__(
        self,
        tokenizer: transformers.LlamaTokenizer,
        model: transformers.LlamaForCausalLM,
        generation_strategy: GenerationStrategy,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.generation_strategy = generation_strategy

    def create_logits_processors(
            self,
            generation_config: GenerationConfig,
    ) -> transformers.generation.logits_process.LogitsProcessorList:
        logits_processors: transformers.generation.logits_process.LogitsProcessorList = transformers.generation.logits_process.LogitsProcessorList()
        if generation_config.no_repeat_ngram_size:
            logits_processors.append(transformers.generation.logits_process.NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))

        return logits_processors

    def create_stopping_criteria(
            self,
            generation_config: GenerationConfig,
    ) -> transformers.StoppingCriteriaList:
        stopping_criteria: transformers.StoppingCriteriaList = transformers.StoppingCriteriaList()
        if generation_config.stop_words:
            stopping_criteria.append(transformers.StopStringCriteria(self.tokenizer, generation_config.stop_words))

        return stopping_criteria

    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        streamer: Optional[transformers.TextStreamer] = None,
    ) -> GenerationResult:
        example = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        logits_processors = self.create_logits_processors(generation_config=generation_config)
        stopping_criteria = self.create_stopping_criteria(generation_config)
        eos_token_ids = generation_config.stop_token_ids + [self.tokenizer.eos_token_id]
        with torch.inference_mode():
            start = time.time()
            generation_strategy_result = self.generation_strategy.generate_token_ids(
                model=self.model,
                input_ids=example["input_ids"].tolist()[0],
                eos_token_ids=eos_token_ids,
                generation_config=generation_config,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
            total_time = time.time() - start
        decoded_prediction = self.tokenizer.decode(
            generation_strategy_result.predicted_tokens
        )
        num_tokens_generated = len(generation_strategy_result.predicted_tokens)
        return GenerationResult(
            generation_strategy_result=generation_strategy_result,
            decoded_prediction=decoded_prediction,
            num_tokens_generated=num_tokens_generated,
            total_time=total_time,
            time_per_token=total_time / num_tokens_generated if num_tokens_generated > 0 else None,
            tokens_per_second=num_tokens_generated / total_time,
        )





class AutoRegressiveGenerationStrategy(GenerationStrategy):
    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_ids: List[int],
        generation_config: GenerationConfig,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
    ) -> GenerationStrategyResult:
        """Variant of `generate` with inputs/outputs formatted as token_ids."""
        past_key_values = None

        input_ids: torch.Tensor = torch.tensor([input_ids]).to(model.device)
        output_ids: List[int] = []

        exit_query_cache = None
        for _ in range(generation_config.max_steps):
            if generation_config.exit_layer > 0:
                model_output = forward_early(
                    model,
                    input_ids,
                    past_key_values,
                    generation_config.exit_layer,
                    exit_query_cache,
                )
            else:
                model_output = forward(
                    model,
                    input_ids,
                    past_key_values,
                )
            logits = model_output.logits
            if logits_processors:
                logits = logits_processors(input_ids, logits)
            past_key_values = model_output.past_key_values
            next_token, _ = decode_next_token(logits=logits, token_idx=-1, sample=generation_config.sample, temperature=generation_config.temperature, top_k=generation_config.top_k, top_p=generation_config.top_p)
            if streamer:
                streamer.put(next_token)
            next_token = next_token.item()
            if next_token in eos_token_ids:
                break
            if stopping_criteria:
                # TODO: when implementing batch size > 1, stop each sample separately?
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break
            output_ids.append(next_token)
            # Don't concatenate `next_token` to original `input_ids` since we're using
            # the KV cache (`past_key_values`) to speed up generation.
            input_ids = torch.tensor([[next_token]]).to(input_ids)

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=None,
        )











def main(args: Arguments, eval_arguments: EvalArguments, generation_config: GenerationConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # setup(args, device=device)
    transformers.utils.logging.set_verbosity_error()
    model, tokenizer = load_model_and_tokenizer(args, device=device)


    if generation_config.generation_strategy == "autoregressive":
        generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
    elif generation_config.generation_strategy == "self_speculative":
        generation_strategy: GenerationStrategy = SelfSpeculativeGenerationStrategy()
    else:
        raise Exception(
            f"Unsupported generation strategy: {generation_config.generation_strategy}"
        )

    # initialize generator
    generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer, model=model, generation_strategy=generation_strategy
    )

    # create evaluator
    wrap = EvalHarnessLM(generator, generation_config, device)

    # Warmup
    warmup = 1
    for _ in range(warmup):
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generate(**tokenizer("This is a warmup prompt", return_tensors="pt").to(device), max_new_tokens=10)

    # Evaluate
    results = simple_evaluate(wrap, **asdict(eval_arguments))

    # TODO: log results, generation samples, etc.
    print(results["results"])
    wrap.metric_result.pop("predicted_text")
    print(wrap.metric_result)