# RQ2: Post-Training Variants in the GRID Framework

## Research Question

How do the principal post-training designs differ in effectiveness and engineering cost?

## Artifacts

- `post_training_variants.csv`: main comparison table across the five variants

## Paper Table

| Post-training Design | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Task-bank Reward | 84.62 | 64.91 | 68.53 |
| End2End Reward | 76.91 | 53.85 | 58.06 |
| Choice-only Reward | 72.03 | 33.73 | 40.56 |
| End2End SFT without RL | 69.75 | 30.48 | 37.13 |
| No post-training | 73.99 | 27.21 | 35.44 |

## Main Observations

- **Task-bank Reward** is the strongest post-training variant in the repository.
- **End2End Reward** improves substantially over the base model, but its online judge-driven training protocol is materially more expensive.
- **Choice-only Reward** and **End2End SFT without RL** both underperform the full Task-bank Reward setting, highlighting the importance of the complete reward design and RL optimization.

## Variant-to-Checkpoint Mapping

- `Task-bank Reward` -> `models/task_bank_reward/ref-to-hf-link.txt`
- `End2End Reward` -> `models/end2end_reward/ref-to-hf-link.txt`
- `Choice-only Reward` -> `models/choice_only_reward/ref-to-hf-link.txt`
- `End2End SFT without RL` -> `models/end2end_sft_without_rl/ref-to-hf-link.txt`
- `No post-training` -> `models/base_model/ref-to-hf-link.txt`

## Reading Guide

- `post_training_variants.csv` is the machine-readable version of the table shown above.
