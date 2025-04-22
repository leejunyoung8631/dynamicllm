CUDA_VISIBLE_DEVICES=0 python lora_merge.py \
                            --base_model meta-llama/Llama-2-7b-chat-hf \
                            --model_class dyllm \
                            --mask_weight ./weight/dyllm_test_11/checkpoint-1200 \
                            --lora_model \
                            --lora_weight ./lora-alpaca2/checkpoint-2200 \
                            --output_dir ./lora-alpaca2//merged2 \
                            