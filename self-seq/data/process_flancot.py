import json
import jsonlines
data = []
for num in ['0', '1', '2']:
    input_file = f'flancot_split/flancot_split_{num}-c4ai-command-r-plus-GPTQ-generate_instruct-refine-response-final.jsonl'
    with open(input_file, 'r', encoding='utf-8') as file:
       for line in file:
         data.append(json.loads(line))
ref_file = 'flancot/flancot_filtered_15k.jsonl'
final_data = []
ref_file_dict = {}
input_file = 'flancot_split/flancot_filtered_15k-Meta-Llama-3-70B-Instruct-generate_instruct-refine-response-final.jsonl'
llama3_data = []
with open(input_file, 'r', encoding='utf-8') as file:
       for line in file:
         llama3_data.append(json.loads(line))
with open(ref_file, 'r', encoding='utf-8') as file:
       ref_data = []
       for line in file:
         ref_data.append(json.loads(line))
for item in ref_data:
    ref_file_dict[item['question']] = item
origin_data = []
idx = 0
for item in ref_data:
    new_item = {}
    new_item['idx'] = idx
    idx += 1
    new_item['system_prompt'] = item['system_prompt']
    new_item['instruction'] = item['question']
    new_item['output'] = item['response']
    origin_data.append(new_item)
with jsonlines.open('flancot/final_15k_data_origin.jsonl', mode='w') as writer:
    writer.write_all(origin_data)

idx = 0
for item in data:
    new_item = {}
    new_item['system_prompt'] = item["system_prompt"]
    new_item['idx'] = idx
    idx+=1
    new_item['input'] = ""
    if len(item['new_instruction']) < len(item['instruction']):
        new_item['instruction'] = item['instruction']
        new_item['output'] = ref_file_dict[item['instruction']]['response']
    else:
        new_item['instruction'] = item['new_instruction']
        new_item['output'] = item['final_instruction_response']
    final_data.append(new_item)
print(len(final_data))
with jsonlines.open('flancot/final_15k_data_cmd_rplus.jsonl', mode='w') as writer:
    writer.write_all(final_data)
llama3_final_data = []
for item in llama3_data:
    new_item = {}
    new_item['system_prompt'] = item["system_prompt"]
    new_item['idx'] = item['idx']
    new_item['input'] = ""
    if len(item['new_instruction']) < len(item['instruction']):
        new_item['instruction'] = item['instruction']
        new_item['output'] = ref_file_dict[item['instruction']]['response']
    else:
        new_item['instruction'] = item['new_instruction']
        new_item['output'] = item['final_instruction_response']
    llama3_final_data.append(new_item)
with jsonlines.open('flancot/final_15k_data_llama3_70binstruct.jsonl', mode='w') as writer:
    writer.write_all(llama3_final_data)
