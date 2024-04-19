import pandas as pd
from datasets import load_dataset

import argparse
import json
import random
import sys
sys.path.append('..')
from token_store import HF_TOKEN

def extract_instruction_output(p):
    if 'completions' in p:
        completion = p['completions'].split('\n', 1)
        if len(completion) != 2:
            print(completion)
            return None
        instruction, output = completion[0], completion[1]
        if 'Instruction: ' in instruction:
            instruction = instruction.split('Instruction: ')[1]
        else:
            print(f'Missing instruction: {instruction}')
        output = output.lstrip('\n')
    else:
        instruction = p['instruction']
        output = p['output']
    return {'instruction': instruction, 'output': output, 'idx': p['idx']}

def replace(p, instruction_output):
    if p['idx'] in instruction_output:
        p['instruction'] = instruction_output[p['idx']]['instruction']
        p['output'] = instruction_output[p['idx']]['output']
    return p

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, default='yahma/alpaca-cleaned')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--replace_data', type=str, default=None)

    args = args.parse_args()
    if args.data.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=args.data, split='train', token=HF_TOKEN)
    else:
        dataset = load_dataset(args.data, split='train', token=HF_TOKEN)
    dataset_name = args.data.split('/')[-1].split('.')[0]
    random.seed(args.seed)
    if 'idx' not in dataset.column_names:
        idx = list(range(len(dataset)))
        dataset = dataset.add_column('idx', idx)

    with open(args.replace_data, 'r', encoding='utf-8') as file:
        replace_data = []
        for line in file:
            replace_data.append(json.loads(line))

    replace_instructions = {
        p['idx']: extract_instruction_output(p) for p in replace_data
        # p['idx']: p for p in replace_data
    }

    # remove None values
    replace_instructions = {k: v for k, v in replace_instructions.items() if v is not None}
    replaced_data = dataset.map(lambda p: replace(p, replace_instructions))
    print(f'Replaced {len(replace_instructions)} instructions')

    replaced_data = replaced_data.shuffle()
    replaced_data.to_json(f'data/{dataset_name}_replaced.jsonl', orient='records', lines=True)