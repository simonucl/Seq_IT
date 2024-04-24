export CUDA_VISIBLE_DEVICES=0

for CHECKPOINT_PATH in simonycl/self-seq-7b-1-3-new simonycl/self-seq-7b-baseline simonycl/self-seq-Llama-2-7b-hf simonycl/self-seq-alpaca-replaced-wizardlm
do
    MODEL_NAME=$(basename $CHECKPOINT_PATH)
    # AlpacaEval
    python3 -m eval.alpaca_farm.run_eval \
        --model_name_or_path $CHECKPOINT_PATH \
        --save_dir results/alpaca_farm/${MODEL_NAME} \
        --eval_batch_size 20 \
        --max_new_tokens 1024 \
        --use_vllm
done
