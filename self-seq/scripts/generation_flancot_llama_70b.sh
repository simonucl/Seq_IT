INPUT_FILE=self-seq/data/flancot_extract/final_15k_data_origin.jsonl

# python3 self-seq/extract_input.py \
#     --input_file $INPUT_FILE \
#     --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
#     --output_file self-seq/data/flancot_extract_refine/final_15k_data_origin.jsonl \
#     --batch_size 1 \
#     --use_vllm \
#     --use_instruct

# python3 self-seq/gpt-query.py \
#     --input_file self-seq/data/flancot_extract_refine/final_15k_data_origin-extracted-input.jsonl \
#     --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
#     --output_file self-seq/data/flancot_extract_refine/flancot_15k_Meta-Llama-3-70B-Instruct_iter_0.jsonl \
#     --batch_size 4 \
#     --use_instruct \
#     --ignore_cache \
#     --use_vllm \
#     --no_refinement \
#     --direct_response 
# 
# python3 self-seq/gpt-query.py \
#     --input_file self-seq/data/flancot_extract_refine/final_15k_data_origin-extracted-input.jsonl \
#     --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
#     --output_file self-seq/data/flancot_extract_refine/flancot_15k_Meta-Llama-3-70B-Instruct_iter_0.jsonl \
#     --batch_size 4 \
#     --use_instruct \
#     --ignore_cache \
#     --use_vllm \
#     --no_refinement

python3 self-seq/data/process_multi_it.py \
	--file_path self-seq/data/flancot_extract_refine/flancot_15k_Meta-Llama-3-70B-Instruct_iter_0-generate_instruct-refine-response-final.jsonl \
	--output_file self-seq/data/flancot_extract_refine/flancot_llama70b_iteration_1.jsonl

# python3 self-seq/gpt-query.py \
#     --input_file self-seq/data/flancot_extract_refine/flancot_llama70b_iteration_1-iter.jsonl \
# 	--query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
#     --output_file self-seq/data/flancot_extract_refine/flancot_15k_Meta-Llama-3-70B-Instruct_iter_1.jsonl \
#     --batch_size 4 \
#     --use_instruct \
#     --use_vllm \
#     --no_refinement \
#     --iteration

python3 self-seq/data/process_multi_it.py \
        --file_path self-seq/data/flancot_extract_refine/flancot_15k_Meta-Llama-3-70B-Instruct_iter_1-generate_instruct-refine-response-final.jsonl \
        --output_file self-seq/data/flancot_extract_refine/flancot_llama70b_iteration_2.jsonl
# 
# python3 self-seq/gpt-query.py \
#     --input_file self-seq/data/flancot_extract/flancot_llama70b_iteration_2-iter.jsonl \
# 	--query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
#     --output_file self-seq/data/flancot_extract/flancot_15k_Meta-Llama-3-70B-Instruct_iter_2.jsonl \
#     --batch_size 4 \
#     --use_instruct \
#     --ignore_cache \
#     --use_vllm \
#     --no_refinement \
#     --iteration
# 
# python3 self-seq/data/process_multi_it.py \
#         --file_path self-seq/data/flancot_extract/flancot_15k_Meta-Llama-3-70B-Instruct_iter_2-generate_instruct-refine-response-final.jsonl \
#         --output_file self-seq/data/flancot_extract/flancot_llama70b_iteration_3.jsonl
