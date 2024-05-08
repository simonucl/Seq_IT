INPUT_FILE=self-seq/data/flancot_extract/final_15k_data_origin.jsonl

python3 self-seq/extract_input.py \
    --input_file $INPUT_FILE \
    --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --output_file self-seq/data/flancot_extract/final_15k_data.jsonl \
    --batch_size 1 \
    --use_vllm \
    --use_instruct

python3 self-seq/gpt-query.py \
    --input_file self-seq/data/flancot_extract/final_15k_data-extracted-input.jsonl \
    --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --output_file self-seq/data/flancot_extract/flancot_15k_Meta-Llama-3-70B-Instruct_iter_0.jsonl \
    --batch_size 4 \
    --use_instruct \
    --ignore_cache \
    --use_vllm \
    --no_refinement

python3 self-seq/gpt-query.py \
    --input_file self-seq/data/flancot_extract/flancot_15k_Meta-Llama-3-70B-Instruct_iter_0-generate_instruct-refine-response-final.jsonl \
    --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --output_file self-seq/data/flancot_extract/flancot_15k_Meta-Llama-3-70B-Instruct_iter_1.jsonl \
    --batch_size 4 \
    --use_instruct \
    --ignore_cache \
    --use_vllm \
    --no_refinement \
    --iteration

python3 self-seq/gpt-query.py \
    --input_file self-seq/data/flancot_extract/flancot_15k_Meta-Llama-3-70B-Instruct_iter_1-generate_instruct-refine-response-final.jsonl \
    --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --output_file self-seq/data/flancot_extract/flancot_15k_Meta-Llama-3-70B-Instruct_iter_2.jsonl \
    --batch_size 4 \
    --use_instruct \
    --ignore_cache \
    --use_vllm \
    --no_refinement \
    --iteration