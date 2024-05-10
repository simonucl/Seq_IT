import json
from argparse import ArgumentParser

def process_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        data = json.loads(line)
        if 'extracted_instruction' in data:
            final_instruction = data['extracted_instruction']
        else:
            final_instruction = data['instruction']
        if 'Input: ' in final_instruction:
            final_instruction = final_instruction[:final_instruction.index('Input: ')]
        final_instruction = final_instruction.strip('\'\"“” ')

        input = data['input'].replace('Input: ', '').replace('###', '').strip("\'\"")
        final_instruction = final_instruction.strip("\'\"“”")
        new_data = {
            'idx': data['idx'] if 'idx' in data else i,
            'instruction': final_instruction,
            'output': data['final_instruction_response'],
            'system_prompt':data['system_prompt'],
            'input': input,
            'option': data['option'],
        }
        results.append(new_data)
    return results

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
    with open(args.output_file, 'w', encoding='utf-8') as file:
        for item in new_data:
            item.pop('option')
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')
