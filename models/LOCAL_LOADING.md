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

This loading path is optional. The minimal demo in the repository root README relies on the API-based inference path and does not require local checkpoint loading.
