# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import transformers
from tqdm import tqdm

from typing import List

from torchmetrics.text import BLEUScore, ROUGEScore, EditDistance
# TODO: create ExactMatch torchmetrics.text

from torcheval.metrics.aggregation.mean import Mean
from torcheval.metrics.metric import Metric

from torchmetrics.metric import Metric
from torchmetrics.wrappers.abstract import WrapperMetric
from torchmetrics.text import ROUGEScore

from typing import Any
from torch import Tensor


from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
)

from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy

import arguments
from arguments import Arguments, simple_parse_args_string



import argparse
import json
import random

import numpy as np
import torch
from tqdm import tqdm

from datautil import set_seed
from modelutils import get_model
from modelutils import load_mask_weight, set_inference


from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset
import pandas as pd




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


def LowercaseProcessingFunction(input: str) -> str:
    return input.lower()

def get_valid_dataset_formats():
    # Extract the values of class attributes, excluding internal dunder methods
    return [value for key, value in DatasetFormat.__dict__.items() if not key.startswith('__')]


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



class ROUGEScoreWrapper(WrapperMetric):
    def __init__(
        self,
        base_metric: ROUGEScore,
        score: str = "fmeasure",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(base_metric, ROUGEScore):
            raise ValueError(
                f"Expected base metric to be an instance of `torchmetrics.Metric` but received {type(base_metric)}"
            )
        if len(base_metric.rouge_keys) != 1:
            raise NotImplementedError(
                f"ROUGEScoreWrapper is only implemented to wrap a ROUGEScore with 1 rouge key but instead got {len(base_metric.rouge_keys)} keys."
            )
        self._base_metric = base_metric
        self._score = score

    def compute(self) -> Tensor:
        return self._base_metric.compute()[f"{self._base_metric.rouge_keys[0]}_{self._score}"]

    def update(
        self, 
        *args: Any,
        **kwargs: Any,
    ) -> None:
        return self._base_metric.update(*args, **kwargs)


@dataclass
class BenchmarkArguments:
    dataset: Optional[str] = None
    data_path: Optional[str] = None
    random_shuffle: bool = True
    num_samples: Optional[int] = None
    n_shot: Optional[int] = 0
    template: Optional[str] = None


@dataclass
class EvaluationExample:
    input: str
    output: str


@dataclass
class EvaluationMetrics:
    predicted_text: Dict[str, Metric]
    acceptance_rate: Dict[str, Metric]
    total_time: Dict[str, Metric]
    time_per_token: Dict[str, Metric]
    tokens_per_second: Dict[str, Metric]

    def update(
        self,
        evaluation_example: EvaluationExample,
        generation_result: GenerationResult,
    ) -> None:
        if evaluation_example is not None:
            for metric in self.predicted_text.values():
                metric.update(
                    evaluation_example.output, generation_result.decoded_prediction
                )

        for metric in self.acceptance_rate.values():
            if generation_result.generation_strategy_result.acceptance_rate is None:
                acceptance_rate = torch.tensor(0)
            else:
                acceptance_rate = torch.tensor(
                    generation_result.generation_strategy_result.acceptance_rate
                )
            metric.update(acceptance_rate)

        for metric in self.total_time.values():
            metric.update(torch.tensor(generation_result.total_time))

        for metric in self.time_per_token.values():
            metric.update(torch.tensor(generation_result.time_per_token))

        for metric in self.tokens_per_second.values():
            metric.update(torch.tensor(generation_result.tokens_per_second))

    def compute(self) -> Dict[str, torch.Tensor]:
        return {
            "predicted_text": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.predicted_text.items()
            },
            "acceptance_rate": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.acceptance_rate.items()
            },
            "total_time": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.total_time.items()
            },
            "time_per_token": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.time_per_token.items()
            },
            "tokens_per_second": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.tokens_per_second.items()
            },
        }

    @classmethod
    def build_metrics(cls) -> "EvaluationMetrics":
        return cls(
            predicted_text={
                "rouge-l": ROUGEScoreWrapper(
                    ROUGEScore(
                        rouge_keys="rougeL",
                        normalizer=LowercaseProcessingFunction,
                    )
                ),
                "rouge-1": ROUGEScoreWrapper(
                    ROUGEScore(
                        rouge_keys="rouge1", normalizer=LowercaseProcessingFunction
                    )
                ),
                "rouge-2": ROUGEScoreWrapper(
                    ROUGEScore(
                        rouge_keys="rouge2", normalizer=LowercaseProcessingFunction
                    )
                ),
                "rouge-3": ROUGEScoreWrapper(
                    ROUGEScore(
                        rouge_keys="rouge3", normalizer=LowercaseProcessingFunction
                    )
                ),
                "bleu_score": BLEUScore(
                    n_gram=4,
                ),
                "exact_match": EditDistance(),
            },
            acceptance_rate={"mean": Mean()},
            total_time={"mean": Mean()},
            time_per_token={"mean": Mean()},
            tokens_per_second={"mean": Mean()},
        )

def benchmark(
        model: torch.nn.Module, 
        tokenizer: transformers.PreTrainedTokenizerBase, 
        benchmark_arguments: BenchmarkArguments, 
        generation_config: GenerationConfig,
        seed = None,
    ):
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

    # input, output
    evaluation_set = get_data(
        random_shuffle=benchmark_arguments.random_shuffle,
        num_samples=benchmark_arguments.num_samples,
        dataset=benchmark_arguments.dataset,
        n_shot=benchmark_arguments.n_shot,
        seed=seed,
        data_path=benchmark_arguments.data_path,
        template=benchmark_arguments.template,
    )
    
    
    metrics = EvaluationMetrics.build_metrics()
    for i, example in enumerate(tqdm(evaluation_set)):
        response: GenerationResult = generator.generate(
            prompt=example.input,
            generation_config=generation_config,
        )
        print(f"[Prompt]:\n{example.input}")
        print(f"[Reference Response]:\n{example.output}")
        print(f"[Model Response]:\n{response.decoded_prediction}")
        if response.generation_strategy_result.acceptance_rate is not None:
            print(f"[Acceptance Rate]: {response.generation_strategy_result.acceptance_rate}")
        if response.num_tokens_generated == 0:
            print("Skipping metrics of empty generation")
            # TBD: print stats of emprty generations
            continue
        metrics.update(example, response)

    metric_result = metrics.compute()

    return metric_result




@dataclass
class EvaluationExample:
    input: str
    output: str


class DatasetFormat:
    CHAT_FORMAT: str = "chat_format"
    CNN_DM_SUMMARIZATION: str = "cnn_dm_summarization"
    CNN_DM_LM: str = "cnn_dm_lm"
    XSUM_SUMMARIZATION: str = "xsum_summarization"
    HUMAN_EVAL: str = "human_eval"
    CUSTOM_JSONL: str = "custom_jsonl"
    TOP_V2: str = "top_v2"


PREFIX_LENGTH = 100


def get_valid_dataset_formats():
    # Extract the values of class attributes, excluding internal dunder methods
    return [value for key, value in DatasetFormat.__dict__.items() if not key.startswith('__')]

def apply_template(message:str, template:str) -> str:
    """
    Applies a template to a given message.
    
    Parameters:
        message (str): The message to insert into the template.
        template (str): The template with a placeholder for the message in `{message}`.
        
    Returns:
        str: The formatted message with the template applied.
    """
    if template is None:
        return message
    return template.format(message=message) 


def LowercaseProcessingFunction(input: str) -> str:
    return input.lower()


# TODO: fix or remove TOPv2 benchmarking
def prepare_evaluation_examples_chat_format(data_path: str, template: str = None) -> List[EvaluationExample]:
    SINGLE_TURN_TEMPLATE: str = "\n[{role}]\n{message}\n[/{role}]"
    evaluation_data_points = []

    def stringify_conversation(conversation: List[Dict[str, str]]) -> str:
        return "".join(
            [
                SINGLE_TURN_TEMPLATE.format(role=x["role"], message=x["message"])
                for x in conversation
            ]
        )

    for line in open(data_path):
        json_line = json.loads(line)
        i: int = 0
        while i < len(json_line["data"]):
            if json_line["data"][i]["role"] == "PARSER":
                prompt = apply_template(message=stringify_conversation(json_line["data"][1:i]) + "\n[PARSER]\n", 
                                        template=template) 
                evaluation_data_points.append(
                    EvaluationExample(
                        input=prompt,
                        output=stringify_conversation([json_line["data"][i]]),
                    )
                )
            i += 1
    return evaluation_data_points


def prepare_cnn_dm_lm_format(template: str = None) -> List[EvaluationExample]:
    evaluation_data_points = []
    for data_point in load_dataset("cnn_dailymail", "3.0.0")["test"]:
        words = data_point["article"].split()
        prompt = apply_template(message=" ".join(words[:PREFIX_LENGTH]), template=template)
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=" ".join(words[PREFIX_LENGTH:]),
            )
        )
    return evaluation_data_points


def prepare_cnn_dm_summarization_format(n_shot: int = 0, seed: int = 42, template: str = None) -> List[EvaluationExample]:
    prompt_shots = ""
    if n_shot > 0:
        prompt_keys=["article", "highlights"]
        shots = load_dataset("cnn_dailymail", name="3.0.0", split="train").shuffle(seed=seed).select(range(n_shot))
        for i in range(n_shot):
            prompt = "Article: " + shots[i][prompt_keys[0]] + "\nSummary: " + shots[i][prompt_keys[1]].replace("\n", "") + "\n"
            prompt_shots += prompt
        prompt_shots += "\n"

    evaluation_data_points = []
    for data_point in load_dataset("cnn_dailymail", name="3.0.0", split="test"):
        article = data_point["article"]
        highlights = data_point["highlights"]
        prompt = apply_template(message=prompt_shots + f"Article: {article}\nSummary:", template=template) 
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=f" {highlights}",
            )
        )
    return evaluation_data_points

def prepare_xsum_summarization_format(n_shot: int = 0, seed: int = 42, template: str = None) -> List[EvaluationExample]:
    prompt_shots = ""
    if n_shot > 0:
        prompt_keys=["document", "summary"]
        shots = load_dataset("xsum", split="train").shuffle(seed=seed).select(range(n_shot))
        for i in range(n_shot):
            prompt = "Article: " + shots[i][prompt_keys[0]] + "\nSummary: " + shots[i][prompt_keys[1]].replace("\n", "") + "\n"
            prompt_shots += prompt
        prompt_shots += "\n"

    evaluation_data_points = []
    for data_point in load_dataset('xsum', split='test'):
        article = data_point["document"]
        highlights = data_point["summary"]
        prompt = apply_template(message=prompt_shots + f"Article: {article}\nSummary:", template=template) 
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=f" {highlights}",
            )
        )
    return evaluation_data_points

def prepare_human_eval(template: str = None) -> List[EvaluationExample]:
    evaluation_data_points = []
    for data_point in load_dataset('openai_humaneval', split='test'):
        prompt = apply_template(message=data_point["prompt"], template=template) 
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=data_point["canonical_solution"],
            )
        )
    return evaluation_data_points

def prepare_top_v2(template: str = None) -> List[EvaluationExample]:
    evaluation_data_points = []
    for data_point in load_dataset('WillHeld/top_v2', split='test'):
        # apply template if it exists
        prompt = apply_template(message=data_point["utterance"], template=template)
        evaluation_data_points.append(
            EvaluationExample(
               input= prompt,
                output=data_point["semantic_parse"],
            )
        )
    return evaluation_data_points

def prepare_custom(data_path: str, prompt_field: str = "prompt", response_field: str = "response", template: str = None) -> List[EvaluationExample]:
    evaluation_data_points = []
    for _, data_point in pd.read_json(data_path, lines=True).iterrows():
        prompt = apply_template(message=data_point[prompt_field], template=template)  
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=data_point[response_field],
            )
        )
    return evaluation_data_points



def get_data(
    random_shuffle: bool,
    num_samples: int,
    dataset: str,
    data_path: Optional[str] = None,
    n_shot: int = 0,
    seed: int = 42,
    prompt_field: str = "prompt",
    response_field: str = "response",
    template: str = None
) -> List[EvaluationExample]:
    if dataset == DatasetFormat.CHAT_FORMAT:
        evaluation_data_points = prepare_evaluation_examples_chat_format(data_path, template=template)
    elif dataset == DatasetFormat.CNN_DM_SUMMARIZATION:
        evaluation_data_points = prepare_cnn_dm_summarization_format(n_shot=n_shot, seed=seed, template=template)
    elif dataset == DatasetFormat.XSUM_SUMMARIZATION:
        evaluation_data_points = prepare_xsum_summarization_format(n_shot=n_shot, seed=seed, template=template)
    elif dataset == DatasetFormat.CNN_DM_LM:
        evaluation_data_points = prepare_cnn_dm_lm_format(template)
    elif dataset == DatasetFormat.HUMAN_EVAL:
        evaluation_data_points = prepare_human_eval(template)
    elif dataset == DatasetFormat.CUSTOM_JSONL:
        evaluation_data_points = prepare_custom(data_path, prompt_field=prompt_field, 
                                                response_field=response_field, template=template)
    elif dataset == DatasetFormat.TOP_V2:
        evaluation_data_points = prepare_top_v2(template)
    else:
        raise NotImplementedError(f"Unknown dataset format {dataset}")

    if random_shuffle:
        random.shuffle(evaluation_data_points)

    if num_samples:
        evaluation_data_points = evaluation_data_points[:num_samples]

    return evaluation_data_points









def process_cli_arguments() -> Tuple[arguments.Arguments, BenchmarkArguments, GenerationConfig]:
    parser = transformers.HfArgumentParser((arguments.Arguments, BenchmarkArguments, GenerationConfig))
    general_arguments, benchmark_arguments, generation_config, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    assert benchmark_arguments.dataset in get_valid_dataset_formats(), f"{benchmark_arguments.dataset} is not a supported dataset!"
    
    if general_arguments.model_args:
        general_arguments.model_args = simple_parse_args_string(general_arguments.model_args)
    else:
        general_arguments.model_arg = {}
        

    return general_arguments, benchmark_arguments, generation_config








    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="baffo32/decapoda-research-llama-7B-hf",
        help="base model name",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="dyllm",
        help="chosse in [dyllm, ...]",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="if None, base model name is used"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="pretrain",
        choices=["pretrain", "pruneLLM", "tune_pruneLLM"],
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lora_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--input_prompt", type=str, default="The Leaning Tower of Pisa is known for"
    )
    parser.add_argument("--num_output", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/llama-7b-hf/ppl")
    parser.add_argument(
        "--fix_decapoda_config",
        default=False,
        action="store_true",
        help="fix tokenizer config of baffo32/decapoda-research-llama-7B-hf",
    )
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    
    parser.add_argument("--loss_term", default="mask")
    parser.add_argument("--mask_weight", default="./baffo32/decapoda-research-llama-7B-hf")
    parser.add_argument("--is_generation", action="store_true", default=False)
    
    parser.add_argument("--check_count", action="store_true", help="if True, check the number of skipped blocks")
    
    # for benchmark dataset
    parser.add_argument("--dataset", default="cnn_dm_summarization",)
    parser.add_argument("--num_samples", default=100, type=int)
    
    
    # for general args, not used but for code compatibility
    parser.add_argument("--model", default="dummy_text",)
    
    args = parser.parse_args()
    
    
    
    from huggingface_hub import login
    login("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")  
    
    
    set_seed(args.seed)
    model, tokenizer = get_model(base_model=args.base_model, model_class="dyllm", loss_term=args.loss_term)    
    model = load_mask_weight(model, args.mask_weight)
    model = set_inference(model, args)
    model = model.cuda()
    
    _, benchmark_arguments, generation_config = process_cli_arguments()
    
    metric_result = benchmark(model, tokenizer, benchmark_arguments, generation_config)
    print(metric_result)

    if args.check_count:
        print(f"the number of skipped blocks : {np.mean(model.skip_count)}")