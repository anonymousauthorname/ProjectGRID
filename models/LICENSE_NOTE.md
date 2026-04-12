# Checkpoint License Note

This directory groups one base checkpoint and four derived post-trained variants used in the paper.

- `base_model/` retains the upstream license file included with the original backbone checkpoint.
- `task_bank_reward/`, `end2end_reward/`, `choice_only_reward/`, and `end2end_sft_without_rl/` are local derived variants built on the same backbone family for the paper experiments.

For reviewer convenience, the retained upstream license text is stored at:

- `models/base_model/LICENSE`

The repository preserves the bundled checkpoint-level license material for the base model and keeps the derived variants in the same local directory layout used by the paper artifact.
