import argparse
import json
import logging
import os
import csv

from lm_eval import evaluator, tasks, utils


def convert_json2csv_zeroshot(json_path, csv_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    select_key = {
        "boolq": "acc",
        "piqa": "acc",
        "hellaswag": "acc_norm",
        "winogrande": "acc",
        "arc_easy": "acc",
        "arc_challenge": "acc_norm",
        "openbookqa": "acc_norm",
    }

    list_task = []
    list_metric = []
    list_score = []

    ave_score = 0
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for name, key in select_key.items():
            if name not in data["results"]:
                continue

            list_task.append(name)
            list_metric.append(key)

            score = data["results"][name][key] * 100
            list_score.append(score)
            ave_score += score

        ave_score /= len(list_score)

        list_task.append("AVE")
        list_metric.append("n/a")
        list_score.append(ave_score)

        writer.writerow(list_task)
        writer.writerow(list_metric)
        writer.writerow(list_score)

    print(csv_path)



logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    
    
    
    parser.add_argument("--mask_weight", default="./baffo32/decapoda-research-llama-7B-hf")
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
    parser.add_argument("--check_count", action="store_true", help="if True, check the number of skipped blocks")
    parser.add_argument("--is_generation", action="store_true", )
    
    

    return parser.parse_args()


def main():
    args = parse_args()
    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)
    
    
    # internally, it create model using "transformers.models.auto.modeling_auto.AutoModelForCausalLM"
    # by hijacking the config class, init the model
    from transformers import AutoModelForCausalLM
    from transformers.models.llama.configuration_llama import LlamaConfig
    from dyllm_model import DyLLM
    AutoModelForCausalLM.register(LlamaConfig, DyLLM, exist_ok=True)
    
    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))

    csv_path = args.output_json.replace(".json", ".csv")
    convert_json2csv_zeroshot(args.output_json, csv_path)


if __name__ == "__main__":
    main()
