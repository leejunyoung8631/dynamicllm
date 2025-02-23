# original model
# CUDA_VISIBLE_DEVICES=3 python rouge.py --base_model meta-llama/Llama-2-7b-chat-hf --check_count --dataset cnn_dm_summarization --model dummy_text --num_samples 100 --output_dir results/test

# dyllm model
CUDA_VISIBLE_DEVICES=3 python rouge.py --base_model meta-llama/Llama-2-7b-chat-hf --model_class dyllm --mask_weight weight/dyllm_test_9/checkpoint-200 --check_count --dataset cnn_dm_summarization --model dummy_text --num_samples 100 --output_dir results/test --check_count