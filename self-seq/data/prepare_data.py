import pandas as pd
from datasets import load_dataset

import argparse
import json
import random
from token_store import HF_TOKEN

def extract_conversations(p):
    instruction, output = p['conversations'][0], p['conversations'][1]
    if isinstance(instruction, dict):
        if 'content' in instruction:
            instruction, output = instruction['content'], output['content']
        elif 'value' in instruction:
            instruction, output = instruction['value'], output['value']
    return {'instruction': instruction, 'output': output, 'input': ''}

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, default='yahma/alpaca-cleaned')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--sample', type=int, default=None)

    args = args.parse_args()
    dataset = load_dataset(args.data, split='train', token=HF_TOKEN)
    dataset_name = args.data.split('/')[-1]
    random.seed(args.seed)
    if 'idx' not in dataset.column_names:
        idx = list(range(len(dataset)))
        dataset = dataset.add_column('idx', idx)
    else:
        if 'WizardLM' in dataset_name:
            # keep only those data that contain alpaca in 'idx'
            dataset = dataset.filter(lambda x: 'alpaca' in x['idx'])
            dataset = dataset.map(lambda x: {'idx': int(x['idx'].split('_')[-1])}, remove_columns=['idx'])
    dataset = dataset.shuffle()

    if 'conversations' in dataset.column_names:
        dataset = dataset.map(extract_conversations, remove_columns=['conversations'])
        
    if args.sample:
        dataset = dataset.select(range(args.sample))
        dataset.to_json(f'data/{dataset_name}_{args.sample}.jsonl', orient='records', lines=True)
    else:
        dataset.to_json(f'data/{dataset_name}.jsonl', orient='records', lines=True)
    
    # with open(input_file, 'r', encoding='utf-8') as file:
    #     data = []
    #     for line in file:
    #         data.append(json.loads(line))

    # df = pd.DataFrame(data)
    # df.to_csv(output_file, index=False)