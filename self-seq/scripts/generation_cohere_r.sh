
# python3 self-seq/gpt-query.py \
#     --input_file self-seq/data/flancot/flancot_filtered_15k.jsonl \
#     --sample 50 \
#     --query gpt-3.5-turbo \
#     --batch_size 16 \
# 
# python3 self-seq/gpt-query.py \
#     --input_file self-seq/data/flancot/flancot_filtered_15k.jsonl \
#     --sample 50 \
#     --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
#     --batch_size 1 \
#     --use_vllm

python3 self-seq/gpt-query.py \
    --input_file self-seq/data/flancot/flancot_filtered_15k.jsonl \
    --sample 200 \
    --query /mnt/nfs/public/hf/models/CohereForAI/c4ai-command-r-v01 \
    --batch_size 4 \
    --use_instruct

python3 self-seq/gpt-query.py \
    --input_file self-seq/data/alpaca/alpaca-cleaned.jsonl \
    --sample 200 \
    --query /mnt/nfs/public/hf/models/CohereForAI/c4ai-command-r-v01 \
    --batch_size 4 \
    --use_instruct

cd /mnt/data/models/c4ai-command-r-plus 
git restore --source=HEAD :/

cd /mnt/data/Seq_IT/

python3 self-seq/gpt-query.py \
    --input_file self-seq/data/flancot/flancot_filtered_15k.jsonl \
    --sample 200 \
    --query /mnt/data/models/c4ai-command-r-plus \
    --batch_size 1 \
    --load_8bit \
    --use_instruct

python3 self-seq/gpt-query.py \
    --input_file self-seq/data/alpaca/alpaca-cleaned.jsonl \
    --sample 200 \
    --query /mnt/data/models/c4ai-command-r-plus \
    --batch_size 1 \
    --load_8bit \
    --use_instruct