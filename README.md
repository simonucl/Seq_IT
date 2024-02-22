# Fine-tuning Large Language Models with Sequential Instructions

This is the code to replicate the instruction tuning experiments in the paper [*Fine-tuning Large Language Models with Sequential Instructions*]. [[cite]](#citation)

Our implementation is based on the [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) and [LAVIS](https://github.com/salesforce/LAVIS/tree/main) repository.

# Setup

For text-only experiments 
```bash
conda create -n seq_ins python=3.8
conda activate seq_ins
pip install -r requirements.txt
```

For vision-language experiments
```bash
cd LAVIS
conda create -n seq_ins_vl python=3.8
conda activate seq_ins_vl
pip install -e .
```

Next, prepare train and eval data:
for text-only data:
```bash
cd construct_data
bash download_all_data.sh
```
for vision-langauge data:
```bash
cd LAVIS
bash download_vqa.sh
```

# 




