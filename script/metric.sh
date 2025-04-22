CUDA_VISIBLE_DEVICES=2 python metrics.py \
                        --base_model meta-llama/Llama-2-7b-chat-hf \
                        --model_class dyllm \
                        --mask_weight weight/dyllm_test_13/checkpoint-1000 \
                        # --savedata met_save
                        # --check_count --lora_model --lora_weight ./dyllm_test13_lora/checkpoint-$i