CUDA_VISIBLE_DEVICES=2 python dyllm.py --base_model meta-llama/Llama-2-7b-chat-hf --data_path wikitext --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 --output_dir ./weight/dyllm_test --num_epochs 3

