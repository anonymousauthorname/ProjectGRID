# RQ3: Ablation of Article Rewriting and Article-Complexity-Ordered Training

## Research Question

How much do article rewriting and article-complexity-ordered training contribute within the primary Task-bank Reward setting?

## Artifacts

- `ablation_table.csv`: main ablation table
- `rq3_curve.png`: figure aligned with the paper

## Paper Table

| Setting | Train Reward | Test Score (P + R) / 2 |
| --- | ---: | ---: |
| Full setting | 0.7917 | 0.6641 |
| w/o article rewriting | 0.6265 | 0.6371 |
| w/o article-complexity ordering | 0.3906 | 0.5025 |

## Main Observations

- The **full setting** yields the strongest training reward and the best test score.
- Removing **article rewriting** leads to a measurable drop in downstream performance.
- Removing **complexity ordering** causes the largest degradation, indicating that curriculum design is important in this setup.

## Reading Guide

- `ablation_table.csv` is the machine-readable version of the table shown above.
- `rq3_curve.png` provides the complementary figure view of the same ablation trend.
