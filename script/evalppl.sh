#!/bin/bash

# normal llama
# CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class normal


# for mask model
# CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class dyllm --mask_weight weight/dyllm_test_11/checkpoint-1200 --check_count 

# for mask finetuned model
CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class dyllm --mask_weight weight/dyllm_test_11/checkpoint-1200 --check_count --lora_model --lora_weight ./lora-alpaca2/checkpoint-2000
# CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class dyllm --mask_weight weight/dyllm_test_11/checkpoint-1200 --check_count --lora_model --lora_weight ./lora-alpaca


or i in $(seq 600 200 4400)
do
    echo "Running with value $i"
    python your_script.py --param $i
done

