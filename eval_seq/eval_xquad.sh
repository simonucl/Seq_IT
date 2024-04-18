#!/bin/bash
TYPES=(fewshot fewshot_en fewshot_multi)
FILES=(self-seq-7b-1-3-new self-seq-7B-baseline self-seq-alpaca-cleaned_repeat self-seq-alpaca-cleaned_wizardlm_replaced self-seq-wizardlm)
FILES=(Llama-2-7b-hf)

for TYPE in ${TYPES[@]}
do
    for FILE in ${FILES[@]}
    do
        for LANG in en es ar el hi th de ru zh tr vi
        do
            echo "Evaluating ${FILE}_${TYPE} on XQuAD ${LANG}"
            python3 eval_seq/eval_xquad.py \
                --test_file eval_results/${FILE}_${TYPE}/${FILE}_${TYPE}_base_xquad_${LANG}.jsonl \
                --ref_file data/xquad/${TYPE}/xquad_en.jsonl
        done

    done
done