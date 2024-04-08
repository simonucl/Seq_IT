import json
import os.path
import random

from openai import OpenAI
from tqdm import tqdm
from template import *
import argparse
from token import API_KEYs

global false, null, true
false = null = true = ''


# openai.organization = 'org-2pMBKWAfO0ZLtQhJwhG6wEAf'

def make_requests_GPT_turbo(prompt):
    print('Querying GPT-3.5-turbo...')
    client = OpenAI(
                # This is the default and can be omitted
                api_key=random.choice(API_KEYs),
                max_retries=3,
            )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{prompt['prompt']}",
            }
        ],
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(chat_completion)
    result = chat_completion.choices[0].message.content
    return result.strip()
        # except openai.error.OpenAIError as e:
        #     print(f"OpenAIError: {e}.")

def get_prompt(p):
    e = FEW_SHOTS_EXAMPLE.copy()
    random.shuffle(e)
    prompt = '\n\n'.join(e)

    if 'conversations' in p: # cases for lima
        instruction, output = p['conversations'][0], p['conversations'][1]
        prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, '')
        return {'prompt': prompt, 'instruction': instruction, 'output': output}
    else:
        # cases for alpaca
        if p['input'] != '':
            input = INPUT_TEMPLATE.format(p['input'])
            prompt += '\n\n' + PROMPT_TEMPLATE.format(p['instruction'], input)
        else:
            prompt += '\n\n' + PROMPT_TEMPLATE.format(p['instruction'], '')
    return {'prompt': prompt}

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_file', type=str, default='data/alpaca.jsonl')
    args.add_argument('--output_file', type=str, default=None)
    args.add_argument('--seed', type=int, default=42)


    args = args.parse_args()
    input_file = args.input_file
    if args.output_file is None:
        output_file = input_file.replace('.jsonl', '-gpt.jsonl')
    else:
        output_file = args.output_file

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as initial_file:
            for line in initial_file:
                json_data = json.loads(line)
    else:
        with open(output_file, 'w', encoding='utf-8') as initial_file:
            json_data = []

    input_data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            input_data.append(json.loads(line))

    for index in tqdm(range(len(json_data), len(input_data))):
        question_info = input_data[index]
        prompt = get_prompt(question_info)

        gpt_answer = make_requests_GPT_turbo(prompt)
        json_data.append({
            'idx': index,
            'prompt': prompt['prompt'],
            'completions': gpt_answer,
        })

        with open(output_file, 'a', encoding='utf-8') as json_file:
            for data in json_data:
                json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
        print(f'↑ has been stored.')