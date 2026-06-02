# GRID Docker Artifact Entrypoint

Build once:

```bash
docker build -t grid-artifact:latest .
```

The examples below pass required files at startup. Host inputs are mounted at
`/input`, outputs at `/workspace/grid/docker_output`, and local models at
`/models`.

## 0. Configuration Check

Input: optional LLM endpoint, key, and model.

```bash
docker run --rm \
  -e GRID_LLM_ENDPOINT="https://your-openai-compatible-endpoint/v1" \
  -e GRID_LLM_KEY="..." \
  -e GRID_LLM_MODEL="your-model" \
  grid-artifact:latest check
```

## 1. Training Data Generation

Input: CTI article file. For CSV/Parquet, also pass text/id column names when
needed.

```bash
docker run --rm \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/docker_output:/workspace/grid/docker_output" \
  grid-artifact:latest make-parquet \
  --input-file /input/articles.jsonl \
  --train-parquet /workspace/grid/docker_output/train_task_bank.parquet \
  --eval-parquet /workspace/grid/docker_output/eval_input.parquet
```

## 2. Training and Export

Input: task-bank training Parquet and base model.

```bash
docker run --rm --gpus '"device=0,1"' --ipc=host --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -v "$PWD/input:/input:ro" \
  -v "$HOME/llmmodel:/models:ro" \
  -v "$PWD/docker_output:/workspace/grid/docker_output" \
  grid-artifact:latest verl-sft-rl-export \
  --source-parquet /input/GRID-train-task_bank.parquet \
  --base-model /models/Qwen3-4B-Instruct-2507 \
  --gpus 2
```

## 3. Generation

Input: article file plus LLM endpoint, key, and model.

```bash
docker run --rm \
  -e GRID_LLM_ENDPOINT="https://your-openai-compatible-endpoint/v1" \
  -e GRID_LLM_KEY="..." \
  -e GRID_LLM_MODEL="your-model" \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/docker_output:/workspace/grid/docker_output" \
  grid-artifact:latest generate-kg \
  --input-file /input/articles.jsonl \
  --output-file /workspace/grid/docker_output/predictions.jsonl \
  --backend llm
```

## 4. Evaluation

Input: article/gold file and prediction JSONL.

```bash
docker run --rm \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/docker_output:/workspace/grid/docker_output" \
  grid-artifact:latest evaluate \
  --input-file /input/articles.jsonl \
  --predictions-file /workspace/grid/docker_output/predictions.jsonl \
  --output-file /workspace/grid/docker_output/evaluation.json
```
