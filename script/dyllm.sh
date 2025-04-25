# wiki text (partial)
CUDA_VISIBLE_DEVICES=1 python dyllm.py \
                    --base_model meta-llama/Llama-2-7b-chat-hf \
                    --data_path wikitext \
                    --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
                    --output_dir ./data/ljy_data/dyllm_test_20 \
                    --num_epochs 3 --loss_term ppl_token_count \
                    --model_class dyllm --use_bfloat


# wiki_103
# CUDA_VISIBLE_DEVICES=2 python dyllm.py \
#                     --base_model meta-llama/Llama-2-7b-chat-hf \
#                     --data_path wiki_103 \
#                     --train_set_size 200000 \
#                     --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
#                     --output_dir ./data/ljy_data/dyllm_test_14 \
#                     --num_epochs 3 --loss_term mask,ppl --model_class dyllm --use_bfloat



