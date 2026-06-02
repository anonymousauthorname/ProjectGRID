# GRID Docker Artifact Entrypoint

This directory provides the reviewer-facing Docker path for the packaged GRID
artifact. It is designed to run without local Dropbox paths or API keys. The
default command starts a real 2-GPU VERL chain: SFT, SFT export, RL/GRPO with the
local task-bank reward, and RL export.

## Build

```bash
docker build -t grid-artifact:latest .
```

The Dockerfile uses a VERL runtime image by default and clones VERL source during
build. You can override either value:

```bash
docker build \
  --build-arg GRID_VERL_BASE_IMAGE=verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 \
  --build-arg GRID_VERL_REF=v0.7.0 \
  -t grid-artifact:latest .
```

## One-command VERL SFT/RL Export Test

```bash
docker run --rm --gpus '"device=0,1"' --ipc=host --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e GRID_VERL_GPUS=2 \
  -v "$PWD/docker_output:/workspace/grid/docker_output" \
  grid-artifact:latest
```

If the base model is already present on the host:

```bash
docker run --rm --gpus '"device=0,1"' --ipc=host --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e GRID_VERL_GPUS=2 \
  -e GRID_VERL_BASE_MODEL=/models/Qwen3-4B-Instruct-2507 \
  -v "$HOME/llmmodel:/models:ro" \
  -v "$PWD/docker_output:/workspace/grid/docker_output" \
  grid-artifact:latest
```

The command writes data, checkpoints, and exported models under
`docker_output/verl_sft_rl_export/`.

## Environment Variables

- `GRID_INPUT_FILE`: input article file (`.txt`, `.jsonl`, `.json`, `.csv`, or `.parquet`)
- `GRID_CONTENT_COL`: text column for CSV/Parquet input
- `GRID_ID_COL`: article id column for CSV/Parquet input
- `GRID_OUTPUT_DIR`: output directory inside the container
- `GRID_VERL_BASE_MODEL`: Hugging Face model id or mounted model path, default `Qwen/Qwen3-4B-Instruct-2507`
- `GRID_VERL_GPUS`: number of visible GPUs for VERL, default `2`
- `GRID_VERL_SOURCE_PARQUET`: source task-bank Parquet used to prepare the compact VERL training files
- `GRID_VERL_SFT_STEPS`: SFT steps for the smoke chain, default `1`
- `GRID_VERL_RL_STEPS`: RL/GRPO steps for the smoke chain, default `1`
- `GRID_TRAIN_PARQUET`: training parquet path
- `GRID_MODEL_DIR`: exported portable model directory
- `GRID_RL_STEPS`: local RL smoke steps, default `1`
- `GRID_RL_LEARNING_RATE`: local RL smoke learning rate, default `0.2`
- `GRID_RL_SEED`: local RL smoke random seed, default `7`
- `GRID_PREDICTIONS_FILE`: generated KG JSONL path
- `GRID_EVAL_FILE`: evaluation JSON path
- `GRID_LLM_ENDPOINT` or `GRID_LLM_BASE_URL`: OpenAI-compatible endpoint
- `GRID_LLM_KEY` or `GRID_LLM_API_KEY`: API key
- `GRID_LLM_MODEL`: model name

## Individual Commands

```bash
docker run --rm grid-artifact:latest env-check
docker run --rm --gpus '"device=0,1"' --ipc=host --shm-size=16g -v "$PWD/docker_output:/workspace/grid/docker_output" grid-artifact:latest verl-sft-rl-export
docker run --rm -v "$PWD/docker_output:/workspace/grid/docker_output" grid-artifact:latest make-parquet
docker run --rm -v "$PWD/docker_output:/workspace/grid/docker_output" grid-artifact:latest train-export
docker run --rm -v "$PWD/docker_output:/workspace/grid/docker_output" grid-artifact:latest generate-kg --backend model
docker run --rm -v "$PWD/docker_output:/workspace/grid/docker_output" grid-artifact:latest evaluate
```

To use an external LLM endpoint for KG generation:

```bash
docker run --rm \
  -e GRID_LLM_ENDPOINT="https://your-openai-compatible-endpoint/v1" \
  -e GRID_LLM_KEY="..." \
  -e GRID_LLM_MODEL="your-model" \
  -e GRID_GENERATE_BACKEND="llm" \
  -v "$PWD/docker_output:/workspace/grid/docker_output" \
  grid-artifact:latest generate-kg
```

For non-GPU environments, `smoke` remains available as a CPU-only artifact check:

```bash
docker run --rm -v "$PWD/docker_output:/workspace/grid/docker_output" grid-artifact:latest smoke
```

This auxiliary command uses `docker/sample_articles.jsonl`, creates small
Parquet files, runs a one-step local categorical RL update, exports
`model.json`, generates KG predictions, and computes exact-match precision,
recall, and F1. It is not the default reviewer-facing VERL training path.
