CACHE_DIR=/home/tako/ljy/data
MODEL_PATH=/home/tako/ljy/data/skip_ppl/only_ppl/checkpoint-2328
MODEL_PATH=/disk/ljy/data/ljy_data/skip_ppl/only_ppl/checkpoint-2328


CUDA_VISIBLE_DEVICES=3 python src/skip_generate.py \
                    --base_model $MODEL_PATH \
                    --use_bfloat \
                    --cache_model_dir $CACHE_DIR \
                    --cache_dataset_dir $CACHE_DIR \



