#!/usr/bin/env bash
# Filename: verl_sft_rl_export.sh
# Description: Run GRID's Docker VERL smoke chain: SFT, SFT export, RL, RL export.
# Keywords: GRID, VERL, Docker, SFT, GRPO, export

set -euo pipefail

log() {
  printf '[GRID-VERL] %s\n' "$*"
}

die() {
  printf '[GRID-VERL][ERROR] %s\n' "$*" >&2
  exit 1
}

REPO_ROOT="${GRID_REPO_ROOT:-/workspace/grid}"
OUTPUT_DIR="${GRID_OUTPUT_DIR:-${REPO_ROOT}/docker_output}"
VERL_ROOT="${GRID_VERL_ROOT:-/workspace/verl}"
DATA_SOURCE="${GRID_VERL_SOURCE_PARQUET:-${REPO_ROOT}/train-data/data/generated_pipeline/parquet_outputs/GRID-train-task_bank.parquet}"
BASE_MODEL="${GRID_VERL_BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
GPU_COUNT="${GRID_VERL_GPUS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
RUN_DIR="${GRID_VERL_RUN_DIR:-${OUTPUT_DIR}/verl_sft_rl_export}"
DATA_DIR="${RUN_DIR}/data"
SFT_CKPT_DIR="${RUN_DIR}/sft_checkpoints"
SFT_EXPORT_DIR="${RUN_DIR}/sft_export"
RL_CKPT_DIR="${RUN_DIR}/rl_checkpoints"
RL_EXPORT_DIR="${RUN_DIR}/rl_export"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_HOME="${HF_HOME:-${OUTPUT_DIR}/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${REPO_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"

SFT_STEPS="${GRID_VERL_SFT_STEPS:-1}"
RL_STEPS="${GRID_VERL_RL_STEPS:-1}"
TRAIN_ROWS="${GRID_VERL_TRAIN_ROWS:-4}"
VAL_ROWS="${GRID_VERL_VAL_ROWS:-2}"
MAX_PROMPT_CHARS="${GRID_VERL_MAX_PROMPT_CHARS:-1200}"
SFT_MAX_LENGTH="${GRID_VERL_SFT_MAX_LENGTH:-1024}"
RL_MAX_PROMPT_LENGTH="${GRID_VERL_RL_MAX_PROMPT_LENGTH:-768}"
RL_MAX_RESPONSE_LENGTH="${GRID_VERL_RL_MAX_RESPONSE_LENGTH:-64}"
RL_MAX_MODEL_LEN="${GRID_VERL_RL_MAX_MODEL_LEN:-$((RL_MAX_PROMPT_LENGTH + RL_MAX_RESPONSE_LENGTH))}"
RL_TRAIN_BATCH_SIZE="${GRID_VERL_RL_TRAIN_BATCH_SIZE:-4}"
RL_PPO_MINI_BATCH_SIZE="${GRID_VERL_RL_PPO_MINI_BATCH_SIZE:-4}"
RL_ROLLOUT_N="${GRID_VERL_RL_ROLLOUT_N:-2}"

[[ -d "${VERL_ROOT}/verl" ]] || die "VERL source not found at ${VERL_ROOT}. Rebuild the Docker image or set GRID_VERL_ROOT."
[[ -f "${DATA_SOURCE}" ]] || die "GRID task-bank Parquet not found: ${DATA_SOURCE}"

mkdir -p "${RUN_DIR}" "${DATA_DIR}" "${SFT_CKPT_DIR}" "${SFT_EXPORT_DIR}" "${RL_CKPT_DIR}" "${RL_EXPORT_DIR}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}"

python3 - <<'PY'
import json, os, torch
payload = {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_cuda_device_count": torch.cuda.device_count(),
    "grid_verl_gpus": os.environ.get("GRID_VERL_GPUS", "2"),
}
print(json.dumps(payload, indent=2))
if torch.cuda.device_count() < int(os.environ.get("GRID_VERL_GPUS", "2")):
    raise SystemExit("Not enough visible CUDA devices for GRID_VERL_GPUS")
PY

log "Preparing compact task-bank Parquet files from ${DATA_SOURCE}"
python3 -m src.grid.prepare_verl_smoke_data \
  --source-parquet "${DATA_SOURCE}" \
  --output-dir "${DATA_DIR}" \
  --train-rows "${TRAIN_ROWS}" \
  --val-rows "${VAL_ROWS}" \
  --max-prompt-chars "${MAX_PROMPT_CHARS}"

find_latest_step() {
  local root="$1"
  find "${root}" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -1
}

copy_or_merge_sft_export() {
  local step_dir target_dir
  step_dir="$(find_latest_step "${SFT_CKPT_DIR}")"
  [[ -n "${step_dir}" ]] || die "No SFT checkpoint found in ${SFT_CKPT_DIR}"
  target_dir="${SFT_EXPORT_DIR}"
  rm -rf "${target_dir}" "${target_dir}.merge_tmp"
  if [[ -f "${step_dir}/huggingface/config.json" ]]; then
    cp -a "${step_dir}/huggingface" "${target_dir}"
  else
    CUDA_VISIBLE_DEVICES="" python3 -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "${step_dir}" \
      --target_dir "${target_dir}" \
      --trust-remote-code
  fi
  [[ -f "${target_dir}/config.json" ]] || die "SFT export missing config.json: ${target_dir}"
}

copy_or_merge_rl_export() {
  local step_dir actor_dir target_dir
  step_dir="$(find_latest_step "${RL_CKPT_DIR}")"
  [[ -n "${step_dir}" ]] || die "No RL checkpoint found in ${RL_CKPT_DIR}"
  actor_dir="${step_dir}/actor"
  target_dir="${RL_EXPORT_DIR}"
  rm -rf "${target_dir}" "${target_dir}.merge_tmp"
  if [[ -f "${actor_dir}/huggingface/config.json" ]]; then
    cp -a "${actor_dir}/huggingface" "${target_dir}"
  else
    CUDA_VISIBLE_DEVICES="" python3 -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "${actor_dir}" \
      --target_dir "${target_dir}" \
      --trust-remote-code
  fi
  [[ -f "${target_dir}/config.json" ]] || die "RL export missing config.json: ${target_dir}"
}

log "Running SFT for ${SFT_STEPS} step(s) on ${GPU_COUNT} GPU(s)"
torchrun --standalone --nnodes=1 --nproc_per_node="${GPU_COUNT}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.train_batch_size=2 \
  data.micro_batch_size_per_gpu=1 \
  data.train_max_samples="${TRAIN_ROWS}" \
  data.val_max_samples="${VAL_ROWS}" \
  data.max_length="${SFT_MAX_LENGTH}" \
  data.truncation=right \
  model.path="${BASE_MODEL}" \
  model.trust_remote_code=True \
  model.enable_gradient_checkpointing=True \
  model.use_remove_padding=False \
  engine=fsdp \
  engine.dtype=bfloat16 \
  engine.ulysses_sequence_parallel_size=1 \
  optim.lr=1e-5 \
  optim.lr_warmup_steps_ratio=0.0 \
  trainer.project_name=grid_docker_verl \
  trainer.experiment_name=sft_smoke \
  trainer.default_local_dir="${SFT_CKPT_DIR}" \
  trainer.total_training_steps="${SFT_STEPS}" \
  trainer.total_epochs=1 \
  trainer.save_freq="${SFT_STEPS}" \
  trainer.test_freq="${SFT_STEPS}" \
  trainer.resume_mode=disable \
  trainer.max_ckpt_to_keep=1 \
  trainer.logger='["console"]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${GPU_COUNT}" \
  checkpoint.save_contents='["model","hf_model"]'

log "Exporting SFT checkpoint"
copy_or_merge_sft_export

log "Running RL/GRPO for ${RL_STEPS} step(s) on ${GPU_COUNT} GPU(s)"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.project_name=grid_docker_verl \
  trainer.experiment_name=rl_smoke \
  trainer.default_local_dir="${RL_CKPT_DIR}" \
  trainer.logger='["console"]' \
  trainer.val_before_train=False \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${GPU_COUNT}" \
  trainer.total_training_steps="${RL_STEPS}" \
  trainer.total_epochs=1 \
  trainer.save_freq="${RL_STEPS}" \
  trainer.test_freq="${RL_STEPS}" \
  trainer.resume_mode=disable \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=0 \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.train_batch_size="${RL_TRAIN_BATCH_SIZE}" \
  data.val_batch_size=2 \
  data.max_prompt_length="${RL_MAX_PROMPT_LENGTH}" \
  data.max_response_length="${RL_MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=right \
  data.shuffle=False \
  custom_reward_function.path="${REPO_ROOT}/src/grid/verl_taskbank_reward.py" \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="${SFT_EXPORT_DIR}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${RL_PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","hf_model"]' \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_model_len="${RL_MAX_MODEL_LEN}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len="${RL_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.n="${RL_ROLLOUT_N}" \
  actor_rollout_ref.rollout.prompt_length="${RL_MAX_PROMPT_LENGTH}" \
  actor_rollout_ref.rollout.response_length="${RL_MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.fsdp_size=-1

log "Exporting RL checkpoint"
copy_or_merge_rl_export

cat > "${RUN_DIR}/summary.json" <<EOF
{
  "status": "complete",
  "base_model": "${BASE_MODEL}",
  "gpus": "${GPU_COUNT}",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
  "data_dir": "${DATA_DIR}",
  "sft_export_dir": "${SFT_EXPORT_DIR}",
  "rl_export_dir": "${RL_EXPORT_DIR}"
}
EOF

log "VERL SFT + export + RL + export complete: ${RUN_DIR}"
