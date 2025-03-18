CUDA_VISIBLE_DEVICES=1 python loratune.py \
                    --base_model meta-llama/Llama-2-7b-chat-hf \
                    --model_class dyllm \
                    --mask_weight weight/dyllm_test_11/checkpoint-1200 \
                    --data_path yahma/alpaca-cleaned \
                    --use_bfloat \
                    --loss_term none \
