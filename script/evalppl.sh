# normal llama
# CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class normal



CUDA_VISIBLE_DEVICES=2 python eval_ppl.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class dyllm --mask_weight weight/dyllm_test_7/checkpoint-800 --check_count 
