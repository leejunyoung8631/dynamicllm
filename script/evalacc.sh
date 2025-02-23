#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

run_command () {
    python eval_zeroshot_acc.py --model hf-causal-experimental --model_args pretrained=weight/dyllm_test_10/checkpoint-200,tokenizer=meta-llama/Llama-2-7b-chat-hf \
    --base_model meta-llama/Llama-2-7b-chat-hf --model_class dyllm --mask_weight weight/dyllm_test_10/checkpoint-200 --check_count --tasks openbookqa --device cuda --output_json results/test/zeroshot_acc.json | tee results/test/zeroshot_acc.txt
}


run_command