# CUDA_VISIBLE_DEVICES=1 python gentest_from_prof.py --base_model ./weight/merged/pytorch_model.bin --input_file repeat.json --output_file out.json

# CUDA_VISIBLE_DEVICES=1 python gentest_from_prof.py --base_model /disk/yskim/LLM-Pruner/cellprune_results/llama31_25pruned_part75/merged/pytorch_model.bin --input_file rag_distill_cp.json --output_file out3.json



# for attention analysis
# for seed in {1000..1100}; do
#   CUDA_VISIBLE_DEVICES=1 python gentest_from_prof.py \
#     --base_model /disk/yskim/LLM-Pruner/cellprune_results/llama31_25pruned_part75/merged/pytorch_model.bin \
#     --input_file rag_distill_cp.json \
#     --output_file "result_text/out_seed${seed}.json" \
#     --seed "${seed}"
# done




CUDA_VISIBLE_DEVICES=0 python gentest_from_prof.py \
    --base_model /disk/yskim/LLM-Pruner/cellprune_results/llama31_25pruned_part75/merged/pytorch_model.bin \
    --input_file rag_distill_cp.json \
    --output_file "result_text/out_seed_rrrx.json" \
    --seed 28