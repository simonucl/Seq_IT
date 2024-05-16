import json
import os.path
import random

from tqdm import tqdm, trange
import argparse
import os
from extract_input_template import *
from rouge import Rouge

rouge = Rouge()

def check_position(entry):
    if ('position' in entry) or (entry['input'] == ''):
        return entry
    
    ori = entry['ori_instruction']
    input = entry['input']
    instruction = entry['instruction']

    if (input not in instruction) and (rouge.get_scores(input, instruction)[0]['rouge-1']['f'] <= 0.3):
        entry['instruction'] = ori
        entry['input'] = ""
        entry['position'] = "random"
    else:
        if "following" in instruction:
            entry['position'] = "right"
        else:
            entry['position'] = "left"
    return entry

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_file', type=str, default='data/alpaca.jsonl')

    args = args.parse_args()
    input_file = args.input_file

    input_data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            input_data.append(json.loads(line))

    new_input_data = [check_position(data) for data in tqdm(input_data, desc='Checking position')]
    
    with open(input_file, 'w', encoding='utf-8') as file:
        for data in new_input_data:
            file.write(json.dumps(data) + '\n')