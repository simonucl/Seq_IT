INPUT_FILE=self-seq/data/metamath/metamath

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE.jsonl \
    --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --output_file $INPUT_FILE-direct_response.jsonl \
    --batch_size 4 \
    --use_instruct \
    --direct_response \
    --use_vllm \
    --no_refinement \
    --iteration


python3 self-seq/data/process_multi_it.py \
	--file_path $INPUT_FILE-direct_response-generate_instruct-refine-response-final.jsonl \
	--output_file $INPUT_FILE-direct_response.jsonl