CUDA_VISIBLE_DEVICES=1 python loratune.py \
                    --base_model meta-llama/Llama-2-7b-chat-hf \
                    --model_class dyllm \
                    --mask_weight weight/dyllm_test_13/checkpoint-1000 \
                    --output_dir dyllm_test13_lora/ \
                    --data_path yahma/alpaca-cleaned \
                    --use_bfloat \
                    --loss_term none \
