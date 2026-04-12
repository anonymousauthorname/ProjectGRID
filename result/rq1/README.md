# RQ1: Comparison with Representative Baselines

## Research Question

How does GRID compare with representative CTI knowledge-graph construction baselines on the unified five-source benchmark?

## Artifacts

- `rq1_main_table.csv`: main comparison table
- `../../baseline/registry.csv`: baseline artifact index
- `../../baseline/<method>/generated/`: canonical generated graphs for each representative baseline
- `../../baseline/<method>/source_metrics.csv`: per-source metric summary for each baseline
- `../../src/baseline/Approach_*.py`: restored baseline implementations corresponding to the compared methods

## Paper Table

| Method | CASIE P | CASIE R | CASIE F1 | CTINexus P | CTINexus R | CTINexus F1 | GRID P | GRID R | GRID F1 | MALKG P | MALKG R | MALKG F1 | SecureNLP P | SecureNLP R | SecureNLP F1 | Avg P | Avg R | Avg F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRID_Ours (Task-bank) | 81.80 | 77.68 | 77.22 | 86.69 | 81.10 | 82.58 | 84.18 | 78.11 | 79.62 | 84.43 | 38.55 | 44.66 | 85.97 | 49.10 | 58.58 | 84.62 | 64.91 | 68.53 |
| GRID_Ours (End2End) | 80.88 | 66.98 | 70.76 | 80.30 | 76.67 | 76.83 | 78.71 | 64.93 | 67.35 | 74.94 | 24.87 | 31.61 | 69.71 | 35.81 | 43.77 | 76.91 | 53.85 | 58.06 |
| CTINexus | 83.44 | 61.41 | 67.49 | 86.75 | 91.02 | 87.83 | 83.62 | 71.64 | 75.91 | 88.71 | 39.42 | 48.24 | 85.07 | 53.88 | 63.80 | 85.52 | 63.47 | 68.66 |
| Cognee | 48.87 | 43.75 | 39.79 | 63.98 | 64.58 | 61.80 | 57.13 | 52.01 | 50.24 | 63.74 | 27.73 | 30.33 | 68.04 | 48.74 | 52.54 | 60.35 | 47.36 | 46.94 |
| LLM-CAKG | 85.39 | 52.76 | 58.81 | 79.90 | 55.63 | 63.41 | 80.65 | 69.61 | 72.60 | 78.49 | 38.27 | 46.70 | 80.35 | 65.16 | 68.70 | 80.96 | 56.29 | 62.04 |
| Graphiti | 70.50 | 24.39 | 32.40 | 70.87 | 35.05 | 44.39 | 69.97 | 46.13 | 51.50 | 74.12 | 32.14 | 39.02 | 71.78 | 33.88 | 43.35 | 71.45 | 34.32 | 42.13 |
| CTIKG | 87.41 | 34.51 | 44.57 | 82.67 | 40.82 | 50.25 | 84.28 | 43.06 | 52.30 | 79.20 | 18.91 | 26.29 | 84.12 | 40.61 | 51.12 | 83.54 | 35.58 | 44.91 |
| GraphRAG | 90.30 | 12.43 | 16.69 | 87.09 | 33.79 | 43.81 | 77.28 | 35.70 | 44.07 | 88.10 | 21.32 | 30.25 | 84.93 | 20.26 | 25.62 | 85.54 | 24.70 | 32.09 |
| AttacKG+ | 26.35 | 19.54 | 16.67 | 25.65 | 36.27 | 26.55 | 26.50 | 25.66 | 20.06 | 48.22 | 13.64 | 15.04 | 48.47 | 44.01 | 42.95 | 35.04 | 27.82 | 24.25 |
| KnowGL | 25.83 | 0.40 | 0.53 | 27.34 | 1.42 | 2.35 | 23.71 | 1.93 | 2.20 | 28.94 | 2.05 | 2.85 | 28.80 | 5.58 | 6.89 | 26.93 | 2.28 | 2.96 |

## Main Observations

- **Task-bank Reward + GRID_Ours inference** reaches **84.62%** precision, **64.91%** recall, and **68.53%** Avg F1.
- GRID_Ours attains the **strongest source-averaged recall** among the compared systems.
- **CTINexus** is near-tied on Avg F1 (**68.66%**) but with lower recall and a more elaborate multi-stage pipeline.
- The remaining baselines exhibit a more pronounced precision-recall imbalance, especially on longer and denser articles.

## Reading Guide

- `rq1_main_table.csv` is the machine-readable version of the table shown above.
- `../../baseline/registry.csv` provides the method-to-artifact mapping for the baselines discussed in this section.
- Each `../../baseline/<method>/` directory contains the public generated outputs and source-level summaries for that method.
- `../../src/baseline/Approach_*.py` provides the restored method implementations underlying these baselines.
- The benchmark inputs used for these baseline experiments are stored under `../../benchmark/runtime_input/`.
- The corresponding baseline outputs used in the paper are stored under `../../baseline/<method>/generated/`.
