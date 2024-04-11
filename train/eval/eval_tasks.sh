# export CUDA_VISIBLE_DEVICES=0,1
CHECKPOINT_PATH=$1
MODEL_NAME=$2

# CHECKPOINT_PATH=/mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged
# CHECKPOINT_PATH=meta-llama/Llama-2-7b-hf
# CHECKPOINT_PATH=/mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000
# CHECKPOINT_PATH=/mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_3000

# GSM8K 5 shot
python3 -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/gsm/${MODEL_NAME}-cot-8shot \
    --model $CHECKPOINT_PATH \
    --tokenizer $CHECKPOINT_PATH \
    --n_shot 5 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

# ToxiGen
# python3 -m eval.toxigen.run_eval \
#     --data_dir data/eval/toxigen/ \
#     --save_dir results/toxigen/${MODEL_NAME} \
#     --model_name_or_path $CHECKPOINT_PATH \
#     --use_vllm \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# TyDiQA (Multi-lingual)
python3 -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/${MODEL_NAME} \
    --model $CHECKPOINT_PATH \
    --tokenizer $CHECKPOINT_PATH \
    --eval_batch_size 20 \
    --load_in_8bit

# # MMLU 5 shot
python3 -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/${MODEL_NAME}-5shot \
    --model_name_or_path $CHECKPOINT_PATH \
    --tokenizer_name_or_path $CHECKPOINT_PATH \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# ## Alpaca-Eval
# python3 -m eval.alpaca_farm.run_eval \
#     --save_dir results/alpaca_farm/${MODEL_NAME} \
#     --model_name_or_path $CHECKPOINT_PATH \
#     --tokenizer_name_or_path $CHECKPOINT_PATH \
#     --eval_batch_size 32 \
#     --use_vllm \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# TruthfulQA
python3 -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/trutufulqa/${MODEL_NAME} \
    --model_name_or_path $CHECKPOINT_PATH \
    --tokenizer_name_or_path $CHECKPOINT_PATH \
    --metrics judge info mc \
    --preset qa \
    --gpt_judge_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
    --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
    --eval_batch_size 20 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# Codex Humaneval
python3 -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/{$MODEL_NAME}_temp_0_8 \
    --model $CHECKPOINT_PATH \
    --tokenizer $CHECKPOINT_PATH \
    --use_vllm

# ARC Challenge
# python3 -m eval.arc.run_eval \
#     --ntrain 25 \
#     --save_dir results/arc/${MODEL_NAME} \
#     --model_name_or_path $CHECKPOINT_PATH \
#     --tokenizer_name_or_path $CHECKPOINT_PATH \
#     --eval_batch_size 8 \
#     --n_instances 100
    # --use_chat_format \
    # --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# HellaSwag
# python3 -m eval.hellaswag.run_eval \
#     --ntrain 10 \
#     --save_dir results/hellaswag/${MODEL_NAME} \
#     --model_name_or_path $CHECKPOINT_PATH \
#     --tokenizer_name_or_path $CHECKPOINT_PATH \
#     --eval_batch_size 8 \
#     --n_instances 200 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format