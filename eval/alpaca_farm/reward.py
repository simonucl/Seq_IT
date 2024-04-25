from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, pipeline, Pipeline, PreTrainedModel
import torch
import pandas as pd
import random

pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 1
    }
def win_rate(rewards, ref_rewards):
    # return 1 if larger, when tie, randomly return 1 or 0, otherwise return 0
    winrates = []
    for reward, ref_reward in zip(rewards, ref_rewards):
        if reward > ref_reward:
            winrates.append(1)
        elif reward < ref_reward:
            winrates.append(0)
        else:
            winrates.append(random.choice([0, 1]))
    return sum(winrates) / len(winrates)
    
def main(args):
    reward_tokenizer = AutoTokenizer.from_pretrained(args.rm, cache_dir=args.cache_dir)

    reward_model = pipeline(
        "sentiment-analysis",
        model=args.rm,
        device_map='auto',
        tokenizer=reward_tokenizer,
        model_kwargs={"torch_dtype": torch.float16, "cache_dir": args.cache_dir},
    )

    # load the input file which is a jsonl file
    df = pd.read_json(args.input_file, lines=True)
    # turn the df into list of list of dict, which [{'role': 'user', 'content': df['instruction']}, {'role': 'assistant', 'content': df['output']}]
    dialogues = df.apply(lambda x: [{'role': 'user', 'content': x['instruction']}, {'role': 'assistant', 'content': x['output']}], axis=1).tolist()
    tokenized_dialogues = [reward_tokenizer.apply_chat_template(text, tokenize=False).replace(reward_tokenizer.bos_token, "") for text in dialogues]
    with torch.no_grad():
        rewards = reward_model(tokenized_dialogues, **pipe_kwargs)
        df['reward'] = [reward[0]['score'] for reward in rewards]

    # save the reward to the input file
    output_file = args.input_file.replace('.json', '_reward.json')
    df.to_json(output_file, lines=True, orient='records')
    
    if args.ref_file:
        ref_df = pd.read_json(args.ref_file, lines=True)
        ref_rewards = [r['reward'] for r in ref_df]
        winrate = win_rate(df['reward'], ref_rewards)
        # save as metric json file
        with open('/'.join(output_file.split('/')[:-1]) + '/metrics.json', 'w') as f:
            f.write(f'{{"win_rate": {winrate}}}')
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Alpaca Farm Reward')
    parser.add_argument('--input_file', type=str, help='Input file')
    parser.add_argument('--ref_file', type=str, help='Output file')
    parser.add_argument('--rm', type=str, default='sfairXC/FsfairX-LLaMA3-RM-v0.1', help='Model name')
    parser.add_argument('--cache_dir', type=str, default='.cache', help='Cache directory')


    args = parser.parse_args()
    main(args)