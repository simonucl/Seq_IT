import json
def process_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            final_instruction = data['final_instruction']
            instruction_part = final_instruction.split('<|eot_id|><|start_header_id|>user<|end_header_id|>')[-1]
            start_l = 3
            length = 59 #len('�~@~]<|eot_id|><|start_header_id|>assistant<|end_header_id|>')
            new_data = {
                    'instruction': instruction_part[start_l:-length],
                'output': data['final_instruction_response'],
                'system_prompt':data['system_prompt'],
                'input': '',
            }
            results.append(new_data)
    return results

file_path = 'flancot/flancot_llama70b_iteration2-Meta-Llama-3-70B-Instruct-generate_instruct-refine-response-final.jsonl'
new_data_list = process_jsonl_file(file_path)
output_file_path = 'flancot/flancot_llama70b_iteration3.jsonl'
idx = 0
for item in new_data_list:
    item['idx'] = idx
    idx+=1
with open(output_file_path, 'w', encoding='utf-8') as file:
        for item in new_data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')

