# GRID Docker Artifact Entrypoint

This directory provides a minimal executable Docker path for the packaged GRID
artifact. It is designed for reviewer smoke tests and for running the artifact
without local Dropbox paths, vLLM services, or a VERL cluster.

## Build

```bash
docker build -t grid-artifact:latest .
```

## One-command Smoke Test

```bash
docker run --rm -v "$PWD/docker_output:/workspace/grid/docker_output" grid-artifact:latest smoke
```

The smoke test runs:

1. `make-parquet`
2. `train-export`
3. `generate-kg`
4. `evaluate`

It uses `docker/sample_articles.jsonl` and writes results to `docker_output/`.

## Environment Variables

- `GRID_INPUT_FILE`: input article file (`.txt`, `.jsonl`, `.json`, `.csv`, or `.parquet`)
- `GRID_CONTENT_COL`: text column for CSV/Parquet input
- `GRID_ID_COL`: article id column for CSV/Parquet input
- `GRID_OUTPUT_DIR`: output directory inside the container
- `GRID_TRAIN_PARQUET`: training parquet path
- `GRID_MODEL_DIR`: exported portable model directory
- `GRID_PREDICTIONS_FILE`: generated KG JSONL path
- `GRID_EVAL_FILE`: evaluation JSON path
- `GRID_LLM_ENDPOINT` or `GRID_LLM_BASE_URL`: OpenAI-compatible endpoint
- `GRID_LLM_KEY` or `GRID_LLM_API_KEY`: API key
- `GRID_LLM_MODEL`: model name

## Individual Commands

```bash
docker run --rm grid-artifact:latest env-check
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

The default `train-export` backend is a deterministic portable model that learns
article-id/content-hash to gold-KG mappings from the generated training Parquet.
It is intentionally small so the artifact can run anywhere. Full paper-scale
post-training remains a GPU/VERL workflow; use `GRID_TRAIN_BACKEND=external`
with `GRID_TRAIN_COMMAND` to route the Docker entrypoint to an external training
command when that infrastructure is available.
