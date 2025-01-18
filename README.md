<div style="text-align: center">
<img src="https://github.com/YIYIZH/trl/blob/main/frame.png">
</div>

# Official Implement of LibraAlign
# The Dual-use Dilemma in LLMs: Do Empowering Ethical Capacities Make a Degraded Utility?

## Installation
Our code is built upon TRL. Please follow the https://github.com/huggingface/trl.git for pre-installation.

## Quick Start

**SFT:**

```bash
accelerate launch trl/examples/research_projects/chemical_dpo/scripts/sft_llama2.py \
--output_dir="./sft"  --max_steps=300  --logging_steps=10   \
--save_steps=100    \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1  --gradient_accumulation_steps=2    \
--gradient_checkpointing=False  --group_by_length=False     \
--learning_rate=1e-4    --lr_scheduler_type="cosine"  \
--warmup_steps=100      --weight_decay=0.05 \
--optim="paged_adamw_32bit"     --bf16=True   \
--remove_unused_columns=False  \
--run_name="sft_llama2"     \
--report_to="wandb"
```

**DPO:**

```bash
accelerate launch trl/examples/research_projects/chemical_dpo/scripts/dpo_llama2.py \
	--model_name_or_path="sft/final_checkpoint" 
```

**Merge:**

```bash
python trl/examples/research_projects/stack_llama/scripts/merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-7b-hf" --adapter_model_name="dpo_results/final_checkpoint/" --output_name="stack-llama-2-smiles"
```

**Inference:**

```bash
python trl/examples/research_projects/chemical_dpo/scripts/dpo_test.py # you may need to change the related paths in the file
```

**Evaluation:**

```bash
python trl/examples/research_projects/chemical_dpo/scripts/dpo_evaluate.py # you may need to change the related paths in the file
```

## References

### Proximal Policy Optimisation
The PPO implementation largely follows the structure introduced in the paper **"Fine-Tuning Language Models from Human Preferences"** by D. Ziegler et al. \[[paper](https://huggingface.co/papers/1909.08593), [code](https://github.com/openai/lm-human-preferences)].

### Direct Preference Optimization
DPO is based on the original implementation of **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"** by E. Mitchell et al. \[[paper](https://huggingface.co/papers/2305.18290), [code](https://github.com/eric-mitchell/direct-preference-optimization)]


## Citation

```bibtex

```
