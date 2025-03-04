# CUDA_VISIBLE_DEVICES=0 python attn_model.py --base_model ./weight/llama31_25pruned_part75/merged2

# CUDA_VISIBLE_DEVICES=0 python attn_model.py --base_model meta-llama/Llama-3.1-8B-Instruct

# CUDA_VISIBLE_DEVICES=0 python attn_model.py \
#     --base_model ./weight/llama31_25pruned_part75/merged/pytorch_model.bin \
#     --input_file rag_distill_cp.json \
#     --output_file "result_text/out_seed_rrrr.json" \
#     --seed 1089





CUDA_VISIBLE_DEVICES=0 python attn_model.py \
    --base_model /disk/yskim/LLM-Pruner/cellprune_results/llama31_25pruned_part75/merged/pytorch_model.bin \
    --input_file rag_distill_cp.json \
    --output_file "result_text/out_seed_rrrr.json" \
    --seed 1089