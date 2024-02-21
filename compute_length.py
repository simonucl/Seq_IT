
from transformers import GPT2Tokenizer, LlamaTokenizer
import jsonlines
from transformers import GPT2Tokenizer

def calculate_token_lengths(jsonl_file_path):
    # 初始化 tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")

    # 存储每个 input 的 token 数量
    token_lengths = []

    # 读取 JSONL 文件
    with jsonlines.open(jsonl_file_path) as reader:
        for obj in reader:
            # 提取 input 字段
            input_text = obj['input']

            # Tokenize input 文本
            tokens = tokenizer.tokenize(input_text)

            # 计算 token 数量并添加到列表
            token_lengths.append(len(tokens))
    
    return token_lengths

# 调用函数并传入文件路径
file_path = "../multilingual-alpaca/llama-7b_5lang_xquad_es_base.jsonl"
lengths = calculate_token_lengths(file_path[:100])

# 输出结果
for length in lengths:
    print(length)
sorted_lengths = sorted(lengths)
print(sorted_lengths)

'''# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Your text
#text = "Your input text here"

# Tokenize the text
tokens = tokenizer.tokenize(text)

# Number of tokens
num_tokens = len(tokens)

print(f"Number of tokens: {num_tokens}")
print(f"Tokens: {tokens}")
'''
