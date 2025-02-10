# Yeseong's refactored evaluation helper
import os
import json

from eval_ppl import eval_ppl
from lm_eval import evaluator, tasks, utils
from lm_eval.models.huggingface import AutoCausalLM_EDIT
from lm_eval.tasks import get_task_dict
from utils import convert_json2csv_zeroshot

import logging
logging.getLogger("openai").setLevel(logging.WARNING)


def evaluate_ppl(
        model, tokenizer, output_dir,
        datasets=["wikitext2", "ptb"], bos=[True, False],
        n_partial_batch=None):
    # ppl evaluation
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for add_bos_to_every in bos:
        results = eval_ppl(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            datasets=datasets,
            max_seq_len=128,
            device=model.device,
            add_bos_to_every=add_bos_to_every,
            n_partial_batch=n_partial_batch,
            )
        all_results += results
    return all_results


DEFAULT_TASKS = [
        "openbookqa",
        "arc_easy",
        "winogrande",
        "hellaswag",
        "arc_challenge",
        "piqa",
        "boolq",
        "qnli",
        "rte",
        "cola",
        "copa",
        "lambada_openai"
        ]


def eval_accuracy(
        model, tokenizer, output_dir,
        task_list=DEFAULT_TASKS, limit=None, task_limit_dict=None):
    # Zeroshot acc
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    task_names = utils.pattern_match(task_list, tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    task_dict = get_task_dict(task_names)
    description_dict = {}

    lm = AutoCausalLM_EDIT(
            model=model,
            tokenizer=tokenizer,
            batch_size="auto",
            max_batch_size=32,
            )

    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=0,
        limit=limit,
        bootstrap_iters=100000,
        description_dict=description_dict,
        decontamination_ngrams_path=None,
        write_out=False,
        output_base_path=None,
        task_limit_dict=task_limit_dict,
    )

    if output_dir is not None:
        dumped = json.dumps(results, indent=2)
        print(dumped)

        output_json = os.path.join(output_dir, "zeroshot_acc.json")
        if output_json:
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
            with open(output_json, "w") as f:
                f.write(dumped)

        csv_path = output_json.replace(".json", ".csv")
        convert_json2csv_zeroshot(output_json, csv_path)

    return results
