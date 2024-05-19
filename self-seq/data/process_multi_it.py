import json
from argparse import ArgumentParser
from rouge import Rouge
import random

rouge = Rouge()
def process_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        data = json.loads(line)
        if ('extracted_instruction' in data) and (data['extracted_instruction'] is not None):
            final_instruction = data['extracted_instruction']
        else:
            final_instruction = data['instruction']
        if 'explanation:' in final_instruction.lower():
            final_instruction = final_instruction[:final_instruction.lower().index('explanation:')]
        # if 'Input: ' in final_instruction:
        #     final_instruction = final_instruction[:final_instruction.index('Input: ')]
        
        final_instruction = final_instruction.strip('\'\"“” \n')

        input = data['input'].replace('Input: ', '').replace('###', '').strip("\'\"")
        final_instruction = final_instruction.strip("\'\"“”")
        new_data = {
            'idx': data['idx'] if 'idx' in data else i,
            'instruction': final_instruction,
            'output': data['final_instruction_response'] if 'final_instruction_response' in data else data['output'],
            'system_prompt':data['system_prompt'],
            'input': input,
            'option': data['option'] if 'option' in data else None,
            'position': data['position'] if 'position' in data else "random"
        }
        results.append(new_data)
    return results

def filter_input(instructions):
    filtered_instructions = []
    count = 0
    for instruction in instructions:
        if instruction['input'] == '':
            filtered_instructions.append(instruction)
        else:
            if (instruction['input'] in instruction['instruction']):
                count += 1
            else:
                if (len(instruction['input']) > 0) and (len(instruction['instruction']) > 0):
                    rouge_scores = rouge.get_scores(instruction['input'], instruction['instruction'], ignore_empty=True)
                else:
                    rouge_scores = []
                if (len(rouge_scores) == 0) or (rouge_scores[0]['rouge-1']['f'] > 0.3):
                    count += 1
                else:
                    delimiter = random.choice([' ', '\n', '\n\n'])
                    if 'position' in instruction:
                        if instruction['position'] == "left":
                            instruction['instruction'] = f"{instruction['input']}{delimiter}{instruction['instruction']}"
                        elif instruction['position'] == "right":
                            instruction['instruction'] = f"{instruction['instruction']}{delimiter}{instruction['input']}"
                        else:
                            if random.choice([True, False]):
                                instruction['instruction'] = f"{instruction['input']}{delimiter}{instruction['instruction']}"
                            else:
                                instruction['instruction'] = f"{instruction['instruction']}{delimiter}{instruction['input']}"
                    else:
                        instruction['instruction'] = f"{instruction['input']}{delimiter}{instruction['instruction']}"
            instruction['input'] = ""

            filtered_instructions.append(instruction)
    print(f'Filtered {count} instructions')
    return filtered_instructions

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_path', type=str, default='flancot/flancot_llama70b_iteration2-Meta-Llama-3-70B-Instruct-generate_instruct-refine-response-final.jsonl')
    parser.add_argument('--output_file', type=str, default='flancot/flancot_llama70b_iteration3.jsonl')
    args = parser.parse_args()

    new_data = process_jsonl_file(args.file_path)
    with open(args.output_file.replace('.jsonl', '-iter.jsonl'), 'w', encoding='utf-8') as file:
        for item in new_data:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')
    if ('flancot' in args.file_path) or ('slimorca' in args.file_path):
        new_data = filter_input(new_data)

    with open(args.output_file, 'w', encoding='utf-8') as file:
        for item in new_data:
            item.pop('option')
            item.pop('position')
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')
