export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# CUDA_VISIBLE_DEVICES=0 python test.py --data_path yahma/alpaca-cleaned --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 

CUDA_VISIBLE_DEVICES=0 python test.py --data_path wikitext --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 