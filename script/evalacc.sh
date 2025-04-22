#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

# --model_args pretrained=weight/dyllm_test_11/checkpoint-1200,tokenizer=meta-llama/Llama-2-7b-chat-hf \
# we should input merged model 


run_command () {
    python eval_zeroshot_acc.py --model hf-causal-experimental \
    --model_args pretrained=./lora-alpaca2/,tokenizer=meta-llama/Llama-2-7b-chat-hf \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --model_class dyllm \
    --mask_weight weight/dyllm_test_11/checkpoint-1200 --check_count --tasks openbookqa --device cuda \
    --lora_model --lora_weight ./lora-alpaca2/checkpoint-2200 \
    --output_json results/test/zeroshot_acc.json | tee results/test/zeroshot_acc.txt 
}

run_command_bin () {
    python eval_zeroshot_acc.py --model /disk/ljy/llmexp/lora-alpaca2/merged/pytorch_model.bin \
    --model_args pretrained=./lora-alpaca2/,tokenizer=meta-llama/Llama-2-7b-chat-hf \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --model_class dyllm \
    --mask_weight weight/dyllm_test_11/checkpoint-1200 --check_count --tasks openbookqa --device cuda \
    --lora_model --lora_weight ./lora-alpaca2/checkpoint-2200 \
    --output_json results/test/zeroshot_acc.json | tee results/test/zeroshot_acc.txt 
}


# run_command
run_command_bin