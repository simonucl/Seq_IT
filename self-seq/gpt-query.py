import json
import os.path
import random

from openai import OpenAI
from tqdm import tqdm, trange
from template import *
from instruct_template import *
import argparse
from token_store import API_KEYs
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import vllm
import cohere
import time
from agent import HfAgent, VllmAgent, GptAgent, CohereAgent
import refine_template

def get_prompt(p, is_chat=False):
    few_shot_example = FEW_SHOTS_EXAMPLE if not is_chat else FEW_SHOTS_EXAMPLE_CHAT
    prompt_prefix = PROMPT_PREFIX if not is_chat else PROMPT_PREFIX_CHAT

    e = few_shot_example.copy()
    random.shuffle(e)
    prompt = prompt_prefix + '\n\n' + '\n\n'.join(e)

    if 'conversations' in p: # cases for lima
        instruction, output = p['conversations'][0], p['conversations'][1]
        prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, '')
        input = ''
    elif 'question' in p: # cases for flancot
        instruction = p['question']
        prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, '')
        input = ''
    else:
        # cases for alpaca
        instruction = p['instruction']
        input = ''
        if p['input'] != '':
            input = INPUT_TEMPLATE.format(p['input'])
            prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, input)
        else:
            prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, '')
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
    return {'prompt': prompt, 'instruction': instruction, 'input': input, 'messages': messages}

def get_gen_instruction_prompt(p):
    if (p['option'] is None) or (p['option'] == 'D'):
        return {
            **p,
            'new_instruction': p['instruction'],
        }
    
    if p['option'] == 'A':
        prompt_prefix = PROMPT_PREFIX_A
        few_shot_examples = FEW_SHOTS_EXAMPLE_A
        prompt_template = PROMPT_TEMPLATE_A
    elif p['option'] == 'B':
        prompt_prefix = PROMPT_PREFIX_B
        few_shot_examples = FEW_SHOTS_EXAMPLE_B
        prompt_template = PROMPT_TEMPLATE_B
    elif p['option'] == 'C':
        prompt_prefix = PROMPT_PREFIX_C
        few_shot_examples = FEW_SHOTS_EXAMPLE_C
        prompt_template = PROMPT_TEMPLATE_C

    e = few_shot_examples.copy()
    random.shuffle(e)
    prompt = prompt_prefix + '\n\n' + '\n\n'.join(e)
    prompt += '\n\n' + prompt_template.format(p['instruction'])
    messages = [{'role': 'user', 'content': prompt}]
    return {**p, 'messages': messages}

def get_refine_prompt(p, is_chat=False):
    if p['extracted_instruction'] is None:
        return p

    original_instruction = p['instruction']
    new_instruction = p['extracted_instruction']

    prompt_prefix = refine_template.PROMPT_PREFIX
    few_shot_example = refine_template.FEW_SHOTS_EXAMPLE
    prompt_template = refine_template.PROMPT_TEMPLATE

    e = few_shot_example.copy()
    random.shuffle(e)
    prompt = prompt_prefix + '\n\n' + '\n\n'.join(e)
    prompt += '\n\n' + prompt_template.format(p['instruction'], p['extracted_instruction'])
    messages = [{'role': 'user', 'content': prompt}]
    return {**p, 'messages': messages}

def extract_classification(o):
    # check if 'option: a', 'option: b', 'option: c', 'option: d' in the completion
    # if so, extract the option and the explanation
    # if not, return None
    if ('option a' in o.lower()) or ('option: a' in o.lower()):
        return 'A'
    elif ('option b' in o.lower()) or ('option: b' in o.lower()):
        return 'B'
    elif ('option c' in o.lower()) or ('option: c' in o.lower()):
        return 'C'
    elif ('option d' in o.lower()) or ('option: d' in o.lower()):
        return 'D'
    else:
        if 'A.' in o:
            return 'A'
        elif 'B.' in o:
            return 'B'
        elif 'C.' in o:
            return 'C'
        elif 'D.' in o:
            return 'D'
        else:
            if 'prefix task' in o.lower():
                return 'B'
            elif 'suffix task' in o.lower():
                return 'C'
            elif 'decompose' in o.lower():
                return 'A'
            else:
                return 'D'

def extract_instruction(o):
    # check if the completion contains 'new instruction' or 'new task'
    # if so, extract the new instruction
    # if not, return None
    if '#new instruction#' in o.lower():
        return o[o.lower().index('new instruction') + len('new instruction'):].strip(":# \n")
    elif 'new instruction' in o.lower():
        return o[o.lower().index('new instruction') + len('new instruction'):].strip(":# \n")
    else:
        return None
    
def extracted_refined_instruction(o):
    if "no" in " ".join(o.split()[:10]).lower():
        if "#new instruction#" in o.lower():
            o = o[o.lower().index('#new instruction#') + len('#new instruction#'):]
        elif "new instruction" in o.lower():
            o = o[o.lower().index('new instruction') + len('new instruction'):]

        if "###" in o.lower():
            o = o[:o.lower().index("###")]
        return o.strip(":# \n")
    else:
        return None
        
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_file', type=str, default='data/alpaca.jsonl')
    args.add_argument('--output_file', type=str, default=None)
    args.add_argument('--sample', type=int, default=-1)
    args.add_argument('--query', type=str, default='gpt-3.5-turbo')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--load_8bit', action='store_true')
    args.add_argument('--load_4bit', action='store_true')
    args.add_argument('--use_vllm', action='store_true')
    args.add_argument('--use_instruct', action='store_true')
    args.add_argument('--do_refine', action='store_true')

    args = args.parse_args()
    assert not (args.load_8bit and args.load_4bit)
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

    prompts = [get_prompt(p, is_chat=args.use_instruct) for p in input_data]

    if 'gpt' in args.query:
        agent = GptAgent(api_key=random.choice(API_KEYs), model_name=args.query)

        json_data = []
        for index in trange(0, len(input_data)):
            prompt = prompts[index]
            gpt_answer = agent.generate(prompt)
            json_data.append({
                'idx': index,
                'input': prompt['input'],
                'instruction': prompt['instruction'],
                'completions': gpt_answer,
                'option': extract_classification(gpt_answer)
            })
        with open(output_file, 'w', encoding='utf-8') as json_file:
            for g in json_data:
                json_file.write(json.dumps(g, ensure_ascii=False) + '\n')
                
        get_gen_instruction_prompts = [get_gen_instruction_prompt(p) for p in json_data]
        gen_instruction = [p for p in get_gen_instruction_prompts if 'messages' in p]
        new_generations = [p for p in get_gen_instruction_prompts if 'messages' not in p]
        for index in trange(0, len(gen_instruction)):
            batch_prompts = gen_instruction[index]
            outputs = agent.generate(batch_prompts)
            batch_prompts.pop('messages')
            new_generations.append({
                **batch_prompts,
                'new_instruction': outputs,
                'extracted_instruction': extract_instruction(outputs)
            })
        output_file = output_file.replace('.jsonl', '-generate_instruct.jsonl')
        with open(output_file, 'w', encoding='utf-8') as json_file:
            for g in new_generations:
                json_file.write(json.dumps(g, ensure_ascii=False) + '\n')

        # Step 3 (Optional): Add refinement
        refining_generations = [p for p in new_generations if ('extracted_instruction' in p) and (p['extracted_instruction'] is not None)]
        refined_generations = [p for p in new_generations if ('extracted_instruction' not in p) or (p['extracted_instruction'] is None)]
        
        refineing_prompts = [get_refine_prompt(p, is_chat=args.use_instruct) for p in refining_generations]
        for i in trange(0, len(refineing_prompts)):
            batch_prompts = refineing_prompts[i]
            outputs = agent.generate(batch_prompts)
            batch_prompts.pop('messages')
            refined_generations.append({
                **batch_prompts,
                'refine_instruction': outputs,
                'extracted_refined_instruction': extracted_refined_instruction(outputs)
            })
        output_file = output_file.replace('.jsonl', '-refine.jsonl')
        with open(output_file, 'w', encoding='utf-8') as json_file:
            for g in refined_generations:
                json_file.write(json.dumps(g, ensure_ascii=False) + '\n')

        # Step 4: Return the final output
        instruction_prompts = []
        for p in refined_generations:
            if "extracted_instruction" in p and p["extracted_instruction"] is not None:
                instruction_prompts.append(p["extracted_instruction"])
            else:
                instruction_prompts.append(p["instruction"])
        instruction_prompts = [{'messages': [{'role': 'user', 'content': p}]} for p in instruction_prompts]
        for i in trange(0, len(instruction_prompts)):
            batch_prompts = instruction_prompts[i]
            outputs = agent.generate(batch_prompts)
            refined_generations[i] = {
                **refined_generations[i],
                # 'final_instruction': batch_prompts,
                'final_instruction_reponse': outputs,
            }

        refined_generations = [p for p in refined_generations if ('extracted_refined_instruction' in p) and (p['extracted_refined_instruction'] is not None)]
        remaining_generations = [p for p in refined_generations if ('final_instruction_reponse' in p) and (p['final_instruction_reponse'] is not None)]
        prompts = []
        for p in refined_generations:
            prompts.append(p['extracted_refined_instruction'])
        instruction_prompts = [{'messages': [{'role': 'user', 'content': p}]} for p in prompts]

        for i in trange(0, len(instruction_prompts)):
            batch_prompts = instruction_prompts[i]
            outputs = agent.generate(batch_prompts)
            remaining_generations.append({
                **refined_generations[i],
                'final_refined_instruction_reponse': outputs,
            })

        output_file = output_file.replace('.jsonl', '-final.jsonl')
        with open(output_file, 'w', encoding='utf-8') as json_file:
            for g in remaining_generations:
                json_file.write(json.dumps(g, ensure_ascii=False) + '\n')

    elif args.query in ['command-r']:
        Agent = CohereAgent(api_key=random.choice(API_KEYs))

        with open(output_file, 'a', encoding='utf-8') as json_file:
            for index in tqdm(range(len(json_data), len(input_data))):
                prompt = prompts[index]
                cohere_answer = Agent.generate(prompt)
                json_data.append({
                    'idx': index,
                    'input': prompt['input'],
                    'prompt': prompt['prompt'],
                    'completions': cohere_answer,
                })
                json_file.write(json.dumps(json_data[-1], ensure_ascii=False) + '\n')
                print(f'↑ has been stored.')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.query)
        if args.use_instruct:
            tokenize_prompts = [tokenizer.apply_chat_template(p['messages'], add_generation_prompt=True, tokenize=False) for p in prompts]
            stop = None
            stop_id_sequences = None
        else:
            tokenize_prompts = [p['prompt'] for p in prompts]
            stop = ["###", "###\n", "###\n\n"]
            stop_id_sequences = [tokenizer.encode(s, add_special_tokens=False) for s in stop]

        generation_kwargs = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 50,
            "max_new_tokens": 1024,
        }
        if args.use_vllm:
            vllm_kwargs = {
            "tokenizer_mode": "slow",
            "tensor_parallel_size": torch.cuda.device_count(),
            "gpu_memory_utilization": 0.97,
        }
            generation_kwargs["stop"] = stop
            # replace max_new_tokens with max_tokens
            generation_kwargs["max_tokens"] = generation_kwargs.pop("max_new_tokens")
            agent = VllmAgent(args.query, vllm_kwargs, generation_kwargs)
            outputs = agent.generate(tokenize_prompts)
            for i, p in enumerate(prompts):
                option = extract_classification(outputs[i])
                json_data.append({
                    'idx': i,
                    'input': p['input'],
                    'instruction': p['instruction'],
                    'completions': outputs[i],
                    'option': option
                })
            with open(output_file, 'w', encoding='utf-8') as json_file:
                for g in json_data:
                    json_file.write(json.dumps(g, ensure_ascii=False) + '\n')
        else:
            model_kwargs = {
            "load_in_8bit": args.load_8bit,
            "load_in_4bit": args.load_4bit,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": 'flash_attention_2',
            "device_map": "auto",
        }
            
            generation_kwargs['use_cache'] = True
            agent = HfAgent(args.query, model_kwargs, generation_kwargs)

            generations = []
            for i in trange(0, len(tokenize_prompts), args.batch_size):
                batch_prompts = tokenize_prompts[i:i + args.batch_size]
                outputs = agent.generate(batch_prompts, stop_id_sequences=stop_id_sequences)
                for idx, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
                    generations.append({
                        'idx': i + idx,
                        "input": prompts[i + idx]['input'],
                        "instruction": prompts[i + idx]['instruction'],
                        'completions': output,
                        'option': extract_classification(output)
                    })
            with open(output_file, 'w', encoding='utf-8') as json_file:
                for g in generations:
                    json_file.write(json.dumps(g, ensure_ascii=False) + '\n')

            # Step 2: Add sequential instruction generation
            get_gen_instruction_prompts = [get_gen_instruction_prompt(p) for p in generations]
            gen_instruction = [p for p in get_gen_instruction_prompts if 'messages' in p]
            new_generations = [p for p in get_gen_instruction_prompts if 'messages' not in p]
            for i in trange(0, len(gen_instruction), args.batch_size):
                batch_prompts = [p for p in gen_instruction[i:i + args.batch_size]]
                batch_tokenized_prompts = [tokenizer.apply_chat_template(p['messages'], add_generation_prompt=True, tokenize=False) for p in batch_prompts]

                outputs = agent.generate(batch_tokenized_prompts, stop_id_sequences=stop_id_sequences)
                for prompt, output in zip(batch_prompts, outputs):
                    # remove messages from the prompt
                    prompt.pop('messages')
                    new_generations.append({
                        **prompt,
                        'new_instruction': output,
                        'extracted_instruction': extract_instruction(output)
                    })
            output_file = output_file.replace('.jsonl', '-generate_instruct.jsonl')
            with open(output_file, 'w', encoding='utf-8') as json_file:
                for g in new_generations:
                    json_file.write(json.dumps(g, ensure_ascii=False) + '\n')

            # Step 3 (Optional): Add refinement
            refining_generations = [p for p in new_generations if ('extracted_instruction' in p) and (p['extracted_instruction'] is not None)]
            refined_generations = [p for p in new_generations if ('extracted_instruction' not in p) or (p['extracted_instruction'] is None)]
            
            refineing_prompts = [get_refine_prompt(p, is_chat=args.use_instruct) for p in refining_generations]
            for i in trange(0, len(refineing_prompts), args.batch_size):
                batch_prompts = [p for p in refineing_prompts[i:i + args.batch_size]]
                batch_tokenized_prompts = [tokenizer.apply_chat_template(p['messages'], add_generation_prompt=True, tokenize=False) for p in batch_prompts]

                outputs = agent.generate(batch_tokenized_prompts, stop_id_sequences=stop_id_sequences)
                for prompt, output in zip(batch_prompts, outputs):
                    # remove messages from the prompt
                    prompt.pop('messages')
                    refined_generations.append({
                        **prompt,
                        'refine_instruction': output,
                        'extracted_refined_instruction': extracted_refined_instruction(output)
                    })
            output_file = output_file.replace('.jsonl', '-refine.jsonl')
            with open(output_file, 'w', encoding='utf-8') as json_file:
                for g in refined_generations:
                    json_file.write(json.dumps(g, ensure_ascii=False) + '\n')

            # Step 4: Return the final output
            instruction_prompts = []
            for p in refined_generations:
                if "extracted_instruction" in p and p["extracted_instruction"] is not None:
                    instruction_prompts.append(p["extracted_instruction"])
                else:
                    instruction_prompts.append(p["instruction"])
            instruction_prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True, tokenize=False) for p in instruction_prompts]
            
            for i in trange(0, len(instruction_prompts), args.batch_size):
                batch_prompts = [p for p in instruction_prompts[i:i + args.batch_size]]
                outputs = agent.generate(batch_prompts, stop_id_sequences=stop_id_sequences)
                for idx, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
                    # remove messages from the prompt
                    refined_generations.append({
                        **refined_generations[i + idx],
                        'final_instruction': prompt,
                        'final_instruction_response': output,
                    })
            output_file = output_file.replace('.jsonl', '-response.jsonl')
            with open(output_file, 'w', encoding='utf-8') as json_file:
                for g in refined_generations:
                    json_file.write(json.dumps(g, ensure_ascii=False) + '\n')

            extracted_refined_generations = [p for p in refined_generations if ('extracted_refined_instruction' in p) and (p['extracted_refined_instruction'] is not None)]
            remaining_generations = [p for p in refined_generations if ('extracted_refined_instruction' not in p) or (p['extracted_refined_instruction'] is None)]
            prompts = []
            for p in extracted_refined_generations:
                prompts.append(p['extracted_refined_instruction'])
            instruction_prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True, tokenize=False) for p in prompts]
            for i in trange(0, len(instruction_prompts), args.batch_size):
                batch_prompts = [p for p in instruction_prompts[i:i + args.batch_size]]
                outputs = agent.generate(batch_prompts, stop_id_sequences=stop_id_sequences)
                for idx, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
                    # remove messages from the prompt
                    remaining_generations.append({
                        **refined_generations[i + idx],
                        'final_refined_instruction_reponse': output,
                    })

            output_file = output_file.replace('.jsonl', '-final.jsonl')
            with open(output_file, 'w', encoding='utf-8') as json_file:
                for g in remaining_generations:
                    json_file.write(json.dumps(g, ensure_ascii=False) + '\n')