# Local Checkpoint Loading

This note documents the optional local-loading path for the included checkpoints.

Minimal optional Python packages:

- `transformers>=4.48.0`
- `accelerate>=0.34.0`
- `safetensors>=0.4.3`

They are listed in `requirements_optional.txt` at the repository root.

Example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "models/task_bank_reward"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
```

Available local folders:

- `models/base_model/`
- `models/task_bank_reward/`
- `models/end2end_reward/`
- `models/choice_only_reward/`
- `models/end2end_sft_without_rl/`

Backbone and training-data-generator ablations (RQ4):

- `models/llama31_8b_task_bank_reward/`: Llama-3.1-8B-Instruct post-trained with the same task bank, RL step 28.
- `models/gptoss120b_generator_sft/`: Qwen3-4B-Instruct-2507 trained on GPT-OSS-120B-generated data, SFT.
- `models/gptoss120b_generator_rl_step40/`, `models/gptoss120b_generator_rl_step80/`, `models/gptoss120b_generator_rl_step100/`: the same student after SFT and 40, 80, and 100 RL steps.

`models/llama31_8b_task_bank_reward/` is released as the original VERL FSDP checkpoint (world size 4) together with its
Hugging Face config and tokenizer under `huggingface/`. Merge it into a standard Hugging Face checkpoint before loading
it with `transformers`:

```bash
python -m verl.model_merger merge --backend fsdp \
    --local_dir models/llama31_8b_task_bank_reward \
    --target_dir models/llama31_8b_task_bank_reward_hf
```

This loading path is optional. The minimal demo in the repository root README relies on the API-based inference path and does not require local checkpoint loading.
