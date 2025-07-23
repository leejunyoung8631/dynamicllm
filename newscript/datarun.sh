MY_DIR=/disk/ljy/data/ljy_data
MODEL=unsloth/Meta-Llama-3.1-8B-Instruct
CACHE_DIR=/disk/ljy/data/ljy_data


# CUDA_VISIBLE_DEVICES=3 python src/lora_train.py \
#         --base_model $MODEL \
#         --data_path wikitext \
#         --output_dir /disk/ljy/data/ljy_data/dummy \
#         --use_bfloat \
#         --cache_model_dir $CACHE_DIR \
#         --micro_batch_size 4 \
#         --num_epochs 3 \
#         --cutoff_len 256 \
#         --lora_r 16 --learning_rate 1e-4 --batch_size 64 \
#         --skip_block 10 \
#         --get_hidden \
#         --custom_dataset \

# should i add below?
# export ACCELERATE_MIXED_PRECISION=fp16

CUDA_VISIBLE_DEVICES=3 python src/lora_train.py \
        --base_model $MODEL \
        --data_path yahma/alpaca-cleaned \
        --output_dir /disk/ljy/data/ljy_data/skip_ppl/only_ppl \
        --use_bfloat \
        --cache_model_dir $CACHE_DIR \
        --micro_batch_size 4 \
        --num_epochs 3 \
        --cutoff_len 256 \
        --lora_r 16 --learning_rate 1e-4 --batch_size 64 \
        --skip_block 10 \
        --get_hidden \
        --wb_proj skipllm \