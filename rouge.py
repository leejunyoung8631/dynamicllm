import argparse
import os
import json
import numpy as np

import torch

from datasets import load_dataset
from rouge_score import rouge_scorer
from statistics import mean

from benchmark import get_data, process_cli_arguments
from datautil import set_seed
from modelutils import get_model
from modelutils import load_mask_weight, set_inference


from dyllm_model import DyLLM


def main():
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
    parser.add_argument("--output_dir", type=str)
    parser.add_argument(
        "--fix_decapoda_config",
        default=False,
        action="store_true",
        help="fix tokenizer config of baffo32/decapoda-research-llama-7B-hf",
    )
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    
    parser.add_argument("--loss_term", default="mask")
    parser.add_argument("--mask_weight", default=None)
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
    model, tokenizer = get_model(base_model=args.base_model, model_class="normal", loss_term=args.loss_term)   
    model = load_mask_weight(model, args.mask_weight)
    model = set_inference(model, args)
    model = model.cuda()
    
    _, benchmark_arguments, generation_config = process_cli_arguments()
    
    # input, output
    dataset = get_data(
        random_shuffle=benchmark_arguments.random_shuffle,
        num_samples=benchmark_arguments.num_samples,
        dataset=benchmark_arguments.dataset,
        n_shot=benchmark_arguments.n_shot,
        seed=args.seed,
        data_path=benchmark_arguments.data_path,
        template=benchmark_arguments.template,
    )
    
    
    # Suppose you have a function that given an article returns a predicted summary.
    # Replace "generate_summary(article)" with your own LLaMA generation call.
    def generate_summary(model, tokenizer, prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        
        # Generate output
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                temperature=0.9,
                do_sample=True,              # or False if you prefer greedy
                top_k=0,                    # you can tune these 
                top_p=0.9,                  # parameters as well
                # no_repeat_ngram_size=3       # helps reduce repetition
            )
        
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return summary
    
    
    # have to fix
    # maximum token generation
    
    output_data = []
    predictions = []
    references = []

    for example in dataset[:20]:
        article_text = example.input
        reference_summary = example.output  # This is the gold reference

        # Generate your model’s summary
        generated_summary = generate_summary(model, tokenizer, prompt=article_text)
        
        predictions.append(generated_summary)
        references.append(reference_summary)
        output_data.append({"article" : article_text,
                            "reference" : reference_summary,
                            "prediction" : generated_summary}
                           )
        
        
        if args.check_count and isinstance(model, DyLLM):
            print(f"the number of skipped blocks : {np.mean(model.skip_count)}")
            model.skip_count = []
    
    
    # save output
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    
    
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rouge3", "rougeL"], 
        use_stemmer=True
    )
    
    
    
    rouge1_scores = []
    rouge2_scores = []
    rouge3_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        # Each entry in `scores` is a namedtuple with precision, recall, and fmeasure
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rouge3_scores.append(scores["rouge3"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    print("Average ROUGE-1 F1:", mean(rouge1_scores))
    print("Average ROUGE-2 F1:", mean(rouge2_scores))
    print("Average ROUGE-3 F1:", mean(rouge3_scores))
    print("Average ROUGE-L F1:", mean(rougeL_scores))
    


if __name__ == "__main__":
    main()