#!/bin/bash

# normal llama
# CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class normal


# for mask model
# CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class dyllm --mask_weight weight/dyllm_test_11/checkpoint-1200 --check_count 

# for mask finetuned model
# CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class dyllm --mask_weight weight/dyllm_test_11/checkpoint-1200 --check_count --lora_model --lora_weight ./lora-alpaca2/checkpoint-2000
# CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class dyllm --mask_weight weight/dyllm_test_11/checkpoint-1200 --check_count --lora_model --lora_weight ./lora-alpaca


# for i in $(seq 200 200 2000)
# do
#     echo "Running with value $i"
#     echo "Running with value $i"
#     CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class dyllm --mask_weight weight/dyllm_test_13/checkpoint-1000 --check_count --lora_model --lora_weight ./dyllm_test13_lora/checkpoint-$i
#     echo ""
#     echo ""
# done


# see just mask weight (not finetuned)
# for i in $(seq 200 200 800)
# do
#     echo "Running with value $i"
#     CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class dyllm --mask_weight ./data/ljy_data/dyllm_test_17/checkpoint-$i --check_count 
#     echo ""
# done

CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class dyllm --mask_weight ./data/ljy_data/dyllm_test_19/checkpoint-400 --check_count 



# see just mask weight (not finetuned)
# for i in 200 400 600 1200
# do
#     echo "Running with value $i"
#     CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class dyllm --mask_weight weight/dyllm_test_11/checkpoint-$i --check_count 
#     echo ""
# done

