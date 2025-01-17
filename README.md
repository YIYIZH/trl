<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/trl_banner_dark.png">
</div>

# Official Impliment of LibraAlign


## Installation
Our code is built upon TRL. Please follow the https://github.com/huggingface/trl.git for pre installation.

## Quick Start

**SFT:**

```bash
trl sft --model_name_or_path facebook/opt-125m --dataset_name imdb --output_dir opt-sft-imdb
```

**DPO:**

```bash
trl dpo --model_name_or_path facebook/opt-125m --dataset_name trl-internal-testing/hh-rlhf-helpful-base-trl-style --output_dir opt-sft-hh-rlhf 
```

**Merge:**

```bash
trl chat --model_name_or_path Qwen/Qwen1.5-0.5B-Chat
```

## References

### TRL

### Proximal Policy Optimisation
The PPO implementation largely follows the structure introduced in the paper **"Fine-Tuning Language Models from Human Preferences"** by D. Ziegler et al. \[[paper](https://huggingface.co/papers/1909.08593), [code](https://github.com/openai/lm-human-preferences)].

### Direct Preference Optimization
DPO is based on the original implementation of **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"** by E. Mitchell et al. \[[paper](https://huggingface.co/papers/2305.18290), [code](https://github.com/eric-mitchell/direct-preference-optimization)]


## Citation

```bibtex

```
