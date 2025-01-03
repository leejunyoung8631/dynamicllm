#!/bin/bash

# rtx 40xx does not support P2P, IB for fast communication
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if echo "$GPU_MODEL" | grep -q "RTX 40"; then
    echo "RTX 40xx GPU detected. Setting NCCL options."
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
else
    echo "Non-RTX 40xx GPU detected. Skipping NCCL options."
fi



# sentence task
# CUDA_VISIBLE_DEVICES=0 python test.py --data_path yahma/alpaca-cleaned --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 
# CUDA_VISIBLE_DEVICES=2 python test.py --data_path wikitext --learning_rate 1e-4 --batch_size 64 --micro_batch_size 8

CUDA_VISIBLE_DEVICES=2 python sentence_infer.py --batch_size 8 


# token task
# CUDA_VISIBLE_DEVICES=1 python gen_test.py  