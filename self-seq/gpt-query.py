import json
import os.path
import random

from openai import OpenAI
from tqdm import tqdm
from template import *
import argparse
from token_store import API_KEYs
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import vllm

def make_requests_GPT_turbo(prompt):
    print('Querying GPT-3.5-turbo...')
    client = OpenAI(
                # This is the default and can be omitted
                api_key=random.choice(API_KEYs),
                max_retries=3,
            )

    chat_completion = client.chat.completions.create(
        messages=prompt['messages'],
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
    prompt = PROMPT_PREFIX + '\n\n' + '\n\n'.join(e)

    if 'conversations' in p: # cases for lima
        instruction, output = p['conversations'][0], p['conversations'][1]
        prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, '')
        input = ''
        return {'prompt': prompt, 'instruction': instruction}
    elif 'question' in p: # cases for flancot
        instruction = p['question']
        prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction)
        input = ''
        return {'prompt': prompt, 'instruction': instruction}
    else:
        # cases for alpaca
        if p['input'] != '':
            instruction = p['instruction']
            input = INPUT_TEMPLATE.format(p['input'])
            prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, input)
        else:
            prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, '')
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
    return {'prompt': prompt, 'instruction': instruction, 'input': input, 'messages': messages}

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_file', type=str, default='data/alpaca.jsonl')
    args.add_argument('--output_file', type=str, default=None)
    args.add_argument('--sample', type=int, default=-1)
    args.add_argument('--query', type=str, default='gpt-3.5-turbo')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--load_8bit', action='store_true')
    args.add_argument('--use_vllm', action='store_true')

    args = args.parse_args()
    input_file = args.input_file
    if args.output_file is None:
        output_file = input_file.replace('.jsonl', f'-{os.path.basename(args.query)}.jsonl')
    else:
        output_file = args.output_file
    json_data = []

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as initial_file:
            for line in initial_file:
                json_data.append(json.loads(line))
    else:
        with open(output_file, 'w', encoding='utf-8') as initial_file:
            json_data = []

    input_data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            input_data.append(json.loads(line))

    if args.sample > 0:
        random.seed(args.seed)
        input_data = random.sample(input_data, args.sample)

    prompts = [get_prompt(p) for p in input_data]

    if args.query in ['gpt-3.5-turbo']:
        with open(output_file, 'a', encoding='utf-8') as json_file:
            
            for index in tqdm(range(len(json_data), len(input_data))):
                prompt = prompts[index]
                gpt_answer = make_requests_GPT_turbo(prompt)
                json_data.append({
                    'idx': index,
                    'input': prompt['input'],
                    'prompt': prompt['prompt'],
                    'completions': gpt_answer,
                })
                json_file.write(json.dumps(json_data[-1], ensure_ascii=False) + '\n')
                print(f'↑ has been stored.')
    elif args.use_vllm:
        tokenizer = AutoTokenizer.from_pretrained(args.query)
        model = vllm.LLM(
            model=args.query,
            tokenizer=tokenizer,
            tokenizer_mode="slow",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.97
        )
        tokenize_prompts = [tokenizer.apply_chat_template(p['messages'], add_generation_prompt=True, tokenize=False) for p in prompts]
        sampling_params = vllm.SamplingParams(
                temperature=0,
                top_p=1,
                top_k=50,
                max_tokens=4096,
                # stop=["\n\n"],
            )
        
        generations = model.generate(prompts, sampling_params)
        prompt_to_output = {
                g.prompt: g.outputs[0].text for g in generations
            }
        outputs = [prompt_to_output[p] if prompt in prompt_to_output else "" for p in prompts]
        for i, p in enumerate(prompts):
            json_data.append({
                'idx': i,
                'input': p['input'],
                'prompt': p['prompt'],
                'completions': outputs[i],
            })
        with open(output_file, 'w', encoding='utf-8') as json_file:
            for g in json_data:
                json_file.write(json.dumps(g, ensure_ascii=False) + '\n')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.query)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.query,
            load_in_8bit=args.load_8bit,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            device_map="auto",
        )
        generation_config = GenerationConfig(
                temperature=0,
                top_p=1,
                top_k=50,
                num_beams=1,
                max_new_tokens=4096,
                use_cache=True,
            )
        generations = []
        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = [p['messages'] for p in prompts[i:i+args.batch_size]]
            tokenized = [tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in batch_prompts]
            tokenized_prompts = tokenizer(tokenized, padding="longest", return_tensors="pt", add_special_tokens=True)
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask

            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                **generation_config
            )

            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            for prompt, output in zip(batch_prompts, batch_outputs):
                generations.append({
                    'idx': i,
                    'input': prompt['input'], # 'input': '
                    'prompt': prompt,
                    'completions': output
                })
            with open(output_file, 'a', encoding='utf-8') as json_file:
                for g in generations:
                    json_file.write(json.dumps(g, ensure_ascii=False) + '\n')