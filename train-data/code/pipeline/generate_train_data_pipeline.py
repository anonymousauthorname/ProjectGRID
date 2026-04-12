# -*- coding: utf-8 -*-

import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DROPBOX_ROOT = os.path.dirname(REPO_ROOT)
TRAIN_DATA_ROOT = os.path.join(REPO_ROOT, "train-data")
BENCHMARK_ROOT = os.path.join(REPO_ROOT, "benchmark")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# UV environment switching is optional in the packaged repository.
target_uv_env = str(os.environ.get("GRID_TRAIN_DATA_PYTHON_ENV", "") or "").strip()
if target_uv_env:
    target_uv_env = os.path.abspath(target_uv_env)

current_venv = os.environ.get("VIRTUAL_ENV", "")
if current_venv:
    current_venv = os.path.abspath(current_venv)

if target_uv_env and current_venv != target_uv_env and not sys.prefix.startswith(target_uv_env):
    print(f"🚀 Switching to UV env: {target_uv_env}")
    python_exe = os.path.join(target_uv_env, "bin/python")
    os.environ["VIRTUAL_ENV"] = target_uv_env
    if os.path.exists(python_exe):
        cmd = [python_exe, "-u"] + sys.argv
    else:
        cmd = ["uv", "run", "--python", target_uv_env, "python", "-u"] + sys.argv
    os.execvp(cmd[0], cmd)
    exit()
import pandas as pd
import numpy as np
import random
import json
import importlib.util
import re
import time
import hashlib
import argparse
import queue
import threading
import traceback
import copy
import yaml
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from collections import defaultdict, deque
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm

try:
    if DROPBOX_ROOT not in sys.path:
        sys.path.append(DROPBOX_ROOT)
    import tools
    from src import tools_prompt_nano as tools_prompt
    import json_repair
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("Error: 'tools' / 'src.tools_prompt_nano' / 'json_repair' / 'pyarrow' module not found.")
    print(f"Please ensure '{DROPBOX_ROOT}' is in your PYTHONPATH.")
    sys.exit(1)


# Prompt-name compatibility for the packaged artifact.
LEGACY_TOOLS_PROMPT_NAME_FALLBACKS = {
    "grid_kg_single_prompt_maker_tracerawtext_251231": "grid_kg_single_prompt_maker_tracerawtext",
    "grid_kg_reverse_prompt_maker_251231": "grid_kg_reverse_prompt_maker",
    "grid_kg_single_prompt_maker_very_simple_251231": "grid_kg_single_prompt_maker_very_simple",
    "grid_kg_sft_reasoning_reconstruction_prompt_251231": "grid_kg_sft_reasoning_reconstruction_prompt",
}
PROMPT_FUNC_NAME_ALIASES = {
    "prompt_maker_只生成精度题目": "prompt_maker_precision_only",
}


def resolve_tools_prompt_callable(prompt_func_name):
    """Resolve a prompt function name against the packaged local prompt bundle."""
    candidate_names = [prompt_func_name]
    alias_name = PROMPT_FUNC_NAME_ALIASES.get(prompt_func_name)
    if alias_name and alias_name not in candidate_names:
        candidate_names.append(alias_name)
    fallback_name = LEGACY_TOOLS_PROMPT_NAME_FALLBACKS.get(prompt_func_name)
    if fallback_name and fallback_name not in candidate_names:
        candidate_names.append(fallback_name)

    for name in candidate_names:
        if hasattr(tools_prompt, name):
            if name != prompt_func_name:
                print(f"⚠️ tools_prompt.{prompt_func_name} not found; falling back to tools_prompt.{name}")
            return getattr(tools_prompt, name)

    raise AttributeError(
        f"Prompt function '{prompt_func_name}' was not found in the packaged prompt bundle. "
        f"Tried: {candidate_names}"
    )


DEFAULT_BASE_VERY_SIMPLE_PROMPT_FUNC = "grid_kg_single_prompt_maker_very_simple_20260303"
STEP6_EXPORTER_PATH = os.path.join(TRAIN_DATA_ROOT, "code", "pipeline", "export_step6_cached_to_verl_parquet.py")
STEP6_REWARD_MODULE_PATH = os.path.join(TRAIN_DATA_ROOT, "code", "reward", "kg_reward.py")
STEP6_ROLLOUT_DEFAULT_TARGET_SERVERS = ["super", "normal", "ultra"]
STEP6_ROLLOUT_DEFAULT_REPEAT = 8
STEP6_ROLLOUT_DEFAULT_KEEP_MIN = 0.33
STEP6_ROLLOUT_DEFAULT_KEEP_MAX = 0.66
STEP6_ROLLOUT_DEFAULT_TOKEN = 16384
STEP6_ROLLOUT_DEFAULT_TEMP = 0.7
STEP6_ROLLOUT_DEFAULT_VLLM_WORKERS = 1024


def get_active_base_very_simple_prompt_name():
    explicit_name = str(globals().get("BASE_VERY_SIMPLE_PROMPT_FUNC", "") or "").strip()
    if explicit_name:
        return explicit_name

    stage1_prompt_name = str(STAGE1_CONFIG.get("prompt_func", "") or "").strip()
    if "tracerawtext" in stage1_prompt_name:
        derived_name = stage1_prompt_name.replace("tracerawtext", "very_simple")
        fallback_name = LEGACY_TOOLS_PROMPT_NAME_FALLBACKS.get(derived_name)
        if hasattr(tools_prompt, derived_name) or fallback_name:
            return derived_name

    return DEFAULT_BASE_VERY_SIMPLE_PROMPT_FUNC


def resolve_current_base_very_simple_prompt_callable():
    return resolve_tools_prompt_callable(get_active_base_very_simple_prompt_name())


STABLE_SAMPLE_NAMESPACE = "grid_stable_sample_v1"
STABLE_SPLIT_NAMESPACE = "grid_stable_split_v1"
QA_BACKGROUND_TEXT_260303 = """
[A. Core Task: Text-Provable KG Extraction]
Extract a cybersecurity knowledge graph from the CTI text.
All entities and relations must be grounded in the text itself.

1. Entity Rule:
- Extract only entities that are explicitly mentioned or can be resolved by clear co-reference in the text.
- If an entity does not participate in any text-supported relation, remove it from the final KG.

2. Relation Rule:
- A relation is correct only if it is directly supported by the text.
- The `rel` can be the literal action phrase or the normalized GRID relation category, but it must still be text-provable.

3. Forbidden Inference Rule:
- No external knowledge completion.
- No subject elevation: do not attribute a tool/component behavior to its operator unless the text explicitly says so.
- No behavioral-to-structural conversion: do not convert `uses` into `is-part-of` unless the text explicitly states a structural relation.
- No chain deduction with subject change: if text says `A uses B` and `B does C`, do not conclude `A does C` unless the text explicitly says so.

4. MCQ Judging Principle:
- Correct options must follow the same Text-Provable Truth rule.
- If an option depends mainly on optimistic guessing, external domain knowledge, or indirect chain completion, treat it as incorrect.
""".strip()


def infer_input_type_from_path(path):
    ext = os.path.splitext(str(path))[1].lower()
    if ext in [".xlsx", ".xls"]:
        return "xlsx"
    if ext == ".csv":
        return "csv"
    if ext == ".parquet":
        return "parquet"
    return ""


def normalize_source_type(source_type, source_path):
    if source_type:
        return str(source_type).strip().lower()
    inferred = infer_input_type_from_path(source_path)
    if inferred:
        return inferred
    raise ValueError(f"无法根据路径推断输入文件类型: {source_path}")


def stable_hash_int(text, namespace):
    raw = f"{namespace}::{text}".encode("utf-8", errors="ignore")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def make_stable_article_id(source_path, original_id, row_idx, content):
    source_tag = os.path.basename(str(source_path))
    if original_id and str(original_id).strip():
        return f"{source_tag}::{original_id}"
    content_hash = hashlib.sha256(str(content).encode("utf-8", errors="ignore")).hexdigest()[:24]
    return f"{source_tag}::row_{row_idx}::content_{content_hash}"


def load_priority_stable_article_ids_from_parquets(parquet_paths):
    normalized_paths = normalize_compat_yaml_paths(parquet_paths)
    if not normalized_paths:
        return set()

    stable_ids = set()
    total_rows = 0
    for parquet_path in normalized_paths:
        if not os.path.exists(parquet_path):
            print(f"⚠️ [Priority Tier] 参考 parquet 不存在，跳过: {parquet_path}")
            continue
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            print(f"⚠️ [Priority Tier] 读取参考 parquet 失败，跳过: {parquet_path} | {exc}")
            continue

        total_rows += len(df)
        if "stable_article_id" in df.columns:
            stable_ids.update(
                str(x).strip()
                for x in df["stable_article_id"].tolist()
                if str(x).strip()
            )
        if "extra_info" in df.columns:
            for extra in df["extra_info"].tolist():
                if not isinstance(extra, dict):
                    continue
                stable_article_id = str(extra.get("stable_article_id", "") or "").strip()
                if stable_article_id:
                    stable_ids.add(stable_article_id)

        print(
            f"📦 [Priority Tier] 已读取参考 parquet: {parquet_path} "
            f"(rows={len(df)}, stable_ids_now={len(stable_ids)})"
        )

    print(
        f"🎯 [Priority Tier] 参考 parquet 总行数={total_rows}，"
        f"去重后 stable_article_id={len(stable_ids)}"
    )
    return stable_ids


def get_sample_rank(stable_article_id, seed):
    return stable_hash_int(f"{seed}::{stable_article_id}", STABLE_SAMPLE_NAMESPACE)


def get_split_score(stable_article_id):
    return stable_hash_int(stable_article_id, STABLE_SPLIT_NAMESPACE) / float(16 ** 16)


def is_train_split(sample_order=None, stable_article_id=None, split_ratio=0.8):
    if stable_article_id:
        return get_split_score(stable_article_id) < float(split_ratio)
    return False


def is_valid_cached_result(value):
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "none":
            return False
    return True


def normalize_text_payload(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def make_content_identity(text):
    normalized = normalize_text_payload(text)
    normalized = "" if normalized is None else normalized
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def is_invalid_text_payload(value):
    normalized = normalize_text_payload(value)
    if normalized is None:
        return True
    if normalized == "":
        return True
    if normalized.lower() == "none":
        return True
    return False


def serialize_prompt_for_validation(prompt):
    if isinstance(prompt, str):
        return prompt
    try:
        return json.dumps(prompt, ensure_ascii=False, default=str)
    except Exception:
        return repr(prompt)


def prompt_contains_invalid_input_text(prompt):
    prompt_text = serialize_prompt_for_validation(prompt)
    return "Input Text: None" in prompt_text or "Input Text: none" in prompt_text


def is_effectively_empty_graph_payload(graph):
    entity_list, relationship_list = parse_graph_response_to_lists(graph)
    return len(entity_list) == 0 and len(relationship_list) == 0


def is_valid_graph_payload(graph):
    return not is_effectively_empty_graph_payload(graph)


def remove_file_if_exists(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def filter_aligned_records_by_mask(*aligned_lists, valid_mask, stage_name=""):
    filtered = []
    for seq in aligned_lists:
        filtered.append([item for item, keep in zip(seq, valid_mask) if keep])
    kept = sum(1 for keep in valid_mask if keep)
    dropped = len(valid_mask) - kept
    if stage_name:
        print(f"🧹 [{stage_name}] 仅保留缓存完整样本: kept={kept}, dropped={dropped}")
    return filtered


def iter_chunks(seq, chunk_size):
    if chunk_size is None or int(chunk_size) <= 0:
        yield 0, len(seq), seq
        return
    chunk_size = int(chunk_size)
    for start in range(0, len(seq), chunk_size):
        end = min(start + chunk_size, len(seq))
        yield start, end, seq[start:end]


def concat_nonempty_dataframes(frames):
    valid_frames = [df for df in frames if df is not None and not df.empty]
    if not valid_frames:
        return pd.DataFrame()
    return pd.concat(valid_frames, ignore_index=True)


def collect_queue_batch(task_queue, batch_size, upstream_done_event, stop_event, poll_timeout=0.5):
    batch = []
    while len(batch) < batch_size and not stop_event.is_set():
        timeout = poll_timeout if not batch else 0.05
        try:
            batch.append(task_queue.get(timeout=timeout))
        except queue.Empty:
            if batch:
                break
            if upstream_done_event.is_set() and task_queue.empty():
                return []
    return batch


class SmoothRateLimiter:
    def __init__(self, rate_per_sec, stage_priority=None, stage_active_getter=None):
        self.rate_per_sec = max(1e-6, float(rate_per_sec))
        self.interval_sec = 1.0 / self.rate_per_sec
        self._next_allowed_ts = time.monotonic()
        self._condition = threading.Condition()
        self._waiters = defaultdict(int)
        self._stage_active_getter = stage_active_getter
        self._stage_priority = {
            "step1": 2,
            "step2": 1,
            "step5a": 0,
            "step5b": 0,
        }
        if stage_priority:
            self._stage_priority.update(stage_priority)

    def _get_stage_active_snapshot(self):
        if self._stage_active_getter is None:
            return {}
        try:
            snapshot = self._stage_active_getter() or {}
            if isinstance(snapshot, dict):
                return snapshot
        except Exception:
            pass
        return {}

    def _pick_best_waiter_stage(self):
        active_snapshot = self._get_stage_active_snapshot()
        best_stage = None
        best_choice = None
        for stage_name, waiting_count in self._waiters.items():
            if waiting_count <= 0:
                continue
            active_count = max(0, int(active_snapshot.get(stage_name, 0)))
            inverse_active_score = 1.0 / float(active_count + 1)
            choice = (
                inverse_active_score,
                -int(self._stage_priority.get(stage_name, 99)),
                stage_name,
            )
            if best_choice is None or choice > best_choice:
                best_choice = choice
                best_stage = stage_name
        return best_stage

    def wait(self, stage_name="step1"):
        stage_name = stage_name or "step1"
        with self._condition:
            registered = False
            try:
                self._waiters[stage_name] += 1
                registered = True
                while True:
                    best_stage = self._pick_best_waiter_stage()
                    if best_stage is not None and best_stage != stage_name:
                        self._condition.wait(timeout=0.05)
                        continue

                    now_ts = time.monotonic()
                    sleep_sec = self._next_allowed_ts - now_ts
                    if sleep_sec > 0:
                        self._condition.wait(timeout=min(max(sleep_sec, 0.01), 0.25))
                        continue

                    self._waiters[stage_name] -= 1
                    registered = False
                    self._next_allowed_ts = max(self._next_allowed_ts, time.monotonic()) + self.interval_sec
                    self._condition.notify_all()
                    return
            finally:
                if registered:
                    self._waiters[stage_name] -= 1
                    self._condition.notify_all()


class StageBatchCacheChecker:
    def __init__(self, stage_name, stage_config, cache_signature, max_workers_vllm, batch_size=256, flush_wait_sec=0.08):
        self.stage_name = stage_name
        self.stage_config = stage_config
        self.cache_signature = cache_signature
        self.max_workers_vllm = max_workers_vllm
        self.batch_size = max(1, int(batch_size))
        self.flush_wait_sec = max(0.01, float(flush_wait_sec))
        self.request_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run, name=f"{stage_name}-batch-cache", daemon=True)
        self.worker.start()

    def submit(self, prompt):
        result_q = queue.Queue(maxsize=1)
        self.request_queue.put({"prompt": prompt, "result_q": result_q})
        return result_q.get()

    def shutdown(self):
        self.stop_event.set()
        self.worker.join(timeout=2.0)

    def _collect_batch(self):
        batch = []
        while len(batch) < self.batch_size and not self.stop_event.is_set():
            timeout = self.flush_wait_sec if not batch else 0.01
            try:
                batch.append(self.request_queue.get(timeout=timeout))
            except queue.Empty:
                if batch:
                    break
                if self.stop_event.is_set():
                    return []
        return batch

    def _run(self):
        while not self.stop_event.is_set() or not self.request_queue.empty():
            batch = self._collect_batch()
            if not batch:
                continue

            prompt_list = [item["prompt"] for item in batch]
            think_val = self.stage_config.get("think")
            if think_val is None:
                think_val = self.stage_config.get("openai_reasoning", 2)

            try:
                cached_results, _, uncached_indices, _ = tools.check_cache_batch(
                    prompt_list=prompt_list,
                    model=self.stage_config["model"],
                    token=self.stage_config["token"],
                    temp=self.stage_config["temp"],
                    think=think_val,
                    max_workers_Vllm=self.max_workers_vllm,
                    openai_verbosity=self.stage_config.get("openai_verbosity"),
                    model_name_override=self.cache_signature["setmodel"],
                    top_p=self.cache_signature["top_p"],
                    top_k=self.cache_signature["top_k"],
                )
                uncached_index_set = set(uncached_indices)
                for idx, request in enumerate(batch):
                    if idx in cached_results:
                        request["result_q"].put({"hit": True, "response": cached_results[idx]})
                    elif idx in uncached_index_set:
                        request["result_q"].put({"hit": False, "response": None})
                    else:
                        request["result_q"].put({"hit": False, "response": None})
            except Exception as exc:
                print(f"⚠️ [{self.stage_name}] 批量缓存检查失败，回退单条检查: {exc!r}")
                for request in batch:
                    cached_response = check_stage_prompt_cache(
                        request["prompt"],
                        self.stage_config,
                        self.cache_signature,
                        showthink=False,
                    )
                    request["result_q"].put({
                        "hit": cached_response is not None,
                        "response": cached_response,
                    })


def resolve_vllm_server_name_from_workers(max_workers_vllm):
    if isinstance(max_workers_vllm, str):
        return max_workers_vllm
    if isinstance(max_workers_vllm, list) and max_workers_vllm and isinstance(max_workers_vllm[0], str):
        return max_workers_vllm[0]
    return "super"


def build_stage_cache_signature(stage_config, max_workers_vllm=None):
    model = stage_config["model"]
    server_to_resolve_on = resolve_vllm_server_name_from_workers(max_workers_vllm)
    think_val = stage_config.get("think")
    explicit_reasoning = stage_config.get("openai_reasoning")
    if think_val is None:
        think_val = explicit_reasoning if explicit_reasoning is not None else 2
    normalized_reasoning = tools.normalize_think_param(think_val)

    is_vllm_target = ('local' in str(model)) or (model in tools.VLLM_SERVERS)
    resolved_model_for_think = tools.resolve_model_name(model, vllm_server_name=server_to_resolve_on) or (model or "")
    supports_qwen35_request_level_thinking = (
        is_vllm_target and tools._ask_helper_is_qwen35_model(resolved_model_for_think)
    )
    if normalized_reasoning is not None:
        if not tools._ask_helper_use_input_reasoning_effort(model, vllm_server_name=server_to_resolve_on) and not supports_qwen35_request_level_thinking:
            normalized_reasoning = None

    setmodel = tools.resolve_model_name(model, vllm_server_name=server_to_resolve_on) or (model or "")
    is_special_chat = any(sm in setmodel for sm in tools.force_as_chat_model) or ("gpt-5" in setmodel and "chat" in setmodel)
    supports_reasoning = not is_special_chat and any(
        sm in setmodel for sm in tools.model_that_support_reasoning_setting
    )
    supports_verbosity = not is_special_chat and any(
        sm in setmodel for sm in tools.model_that_support_verbosity_setting
    )
    reasoning_for_db = tools._ask_helper_build_reasoning_cache_value(
        setmodel,
        normalized_reasoning,
        supports_reasoning=supports_reasoning,
        is_vllm_request=is_vllm_target,
    )
    openai_verbosity = stage_config.get("openai_verbosity")
    verbosity_for_db = (
        str(openai_verbosity)
        if supports_verbosity and openai_verbosity is not None
        else "None"
    )
    return {
        "setmodel": setmodel,
        "reasoning_for_db": reasoning_for_db,
        "verbosity_for_db": verbosity_for_db,
        "top_p": stage_config.get("top_p", "NotSet"),
        "top_k": stage_config.get("top_k", "NotSet"),
    }


def check_stage_prompt_cache(prompt, stage_config, cache_signature, showthink=False):
    cached_data = tools.check_cache(
        prompt,
        cache_signature["setmodel"],
        stage_config["temp"],
        stage_config["token"],
        cache_signature["reasoning_for_db"],
        cache_signature["verbosity_for_db"],
        top_p=cache_signature["top_p"],
        top_k=cache_signature["top_k"],
    )
    if not cached_data:
        return None
    return tools._ask_helper_prepare_cached_response(
        cached_data["response"],
        showthink,
        cache_signature["reasoning_for_db"],
    )


def parse_graph_response_to_lists(graph):
    if isinstance(graph, dict):
        entity_list = graph.get("entity", [])
        relationship_list = graph.get("relationship", [])
        return entity_list if isinstance(entity_list, list) else [], relationship_list if isinstance(relationship_list, list) else []
    try:
        graph_str = str(graph)
        e_start = graph_str.find("#Entity_List_Start#") + len("#Entity_List_Start#")
        e_end = graph_str.find("#Entity_List_End#")
        r_start = graph_str.find("#Relationship_List_Start#") + len("#Relationship_List_Start#")
        r_end = graph_str.find("#Relationship_List_End#")
        entity_list = json_repair.loads(graph_str[e_start:e_end]) if e_start != -1 else []
        relationship_list = json_repair.loads(graph_str[r_start:r_end]) if r_start != -1 else []
        if not isinstance(entity_list, list):
            entity_list = []
        if not isinstance(relationship_list, list):
            relationship_list = []
        return entity_list, relationship_list
    except Exception:
        return [], []


def run_async_multistage_pipeline(sampled_data, run_genqa=False, run_genkg=False, skip_reasoning=False):
    total_articles = len(sampled_data)
    max_in_flight = max(1, int(PIPELINE_MAX_IN_FLIGHT))
    llm_miss_rate = max(1e-6, float(PIPELINE_FILL_RATE_PER_SEC))
    priority_downstream_stages = bool(PIPELINE_PRIORITY_DOWNSTREAM_STAGES)
    step1_soft_limit = int(PIPELINE_STAGE1_SOFT_LIMIT) if int(PIPELINE_STAGE1_SOFT_LIMIT) > 0 else 0
    if step1_soft_limit <= 0:
        step1_soft_limit = max(128, min(max_in_flight, 512))
    step1_soft_limit = min(step1_soft_limit, max_in_flight)
    step1_hard_limit = int(PIPELINE_STAGE1_HARD_LIMIT) if int(PIPELINE_STAGE1_HARD_LIMIT) > 0 else 0
    if step1_hard_limit <= 0:
        step1_hard_limit = max(64, min(step1_soft_limit, max(128, step1_soft_limit // 2)))
    step1_hard_limit = min(step1_hard_limit, step1_soft_limit, max_in_flight)
    step1_headroom = max(16, min(128, step1_soft_limit // 4))
    stage_priority = {
        "step1": 2 if priority_downstream_stages else 0,
        "step2": 1 if priority_downstream_stages else 0,
        "step5a": 0,
        "step5b": 0,
    }
    print(
        f"🚦 两阶段流水线已启用: total_articles={total_articles}, "
        f"max_in_flight={max_in_flight}, llm_miss_rate={llm_miss_rate}/s"
    )
    print("🧊 Phase 1 = 启动时统一缓存检查；🚚 Phase 2 = 仅对 miss 开 worker 补缺")
    print(
        f"🎯 Worker Admission 策略: step1_soft_limit={step1_soft_limit}, "
        f"step1_hard_limit={step1_hard_limit}, step1_headroom={step1_headroom}, "
        f"downstream_priority={priority_downstream_stages}"
    )
    print("🎛️ Miss 放行策略: inverse_active=1/(active+1)，同分时按 step5 > step2 > step1")

    all_folder_paths = []
    qa_frames = []
    kg_frames = []
    lock = threading.Lock()
    active_count = 0
    errors = []
    miss_rate_limiter = SmoothRateLimiter(
        llm_miss_rate,
        stage_priority=stage_priority,
        stage_active_getter=lambda: get_worker_stage_active_snapshot(),
    )
    stage_counter_lock = threading.Lock()
    worker_stage_condition = threading.Condition()
    stage_counters = {
        "step1": {"hit": 0, "miss": 0},
        "step2": {"hit": 0, "miss": 0},
        "step5a": {"hit": 0, "miss": 0},
        "step5b": {"hit": 0, "miss": 0},
    }
    worker_stage_active = {
        "step1": 0,
        "step2": 0,
        "step5a": 0,
        "step5b": 0,
    }
    worker_stage_waiting = {
        "step1": 0,
        "step2": 0,
        "step5a": 0,
        "step5b": 0,
    }
    worker_stage_slot_holding = {
        "step1": 0,
        "step2": 0,
        "step5a": 0,
        "step5b": 0,
    }
    step1_send_semaphore = threading.Semaphore(step1_hard_limit) if step1_hard_limit > 0 else None

    stage1_prompt_func = resolve_tools_prompt_callable(STAGE1_CONFIG["prompt_func"])
    stage2_prompt_func = resolve_tools_prompt_callable(STAGE2_CONFIG["prompt_func"])
    stage5a_prompt_func = resolve_tools_prompt_callable(STAGE5A_CONFIG["prompt_func"])
    stage5b_prompt_func = resolve_tools_prompt_callable(STAGE5B_CONFIG["prompt_func"])
    base_very_simple_prompt_func = resolve_current_base_very_simple_prompt_callable()

    stage1_workers = [16, 0, 32]
    stage2_workers = ['local', 'ultra']
    stage5a_workers = ['local', 'ultra']
    stage5b_workers = ['local', 'ultra']

    
    stage_cache_scan_batch_sizes = {
        "step1": 4096,
        "step2": 4096,
        "step5a": 2048,
        "step5b": 4096,
    }

    def bump_stage_counter(stage_name, field, delta=1):
        with stage_counter_lock:
            stage_counters[stage_name][field] += delta

    def set_worker_stage_active_delta(stage_name, delta):
        with worker_stage_condition:
            worker_stage_active[stage_name] += delta
            worker_stage_condition.notify_all()
            return worker_stage_active[stage_name]

    def get_worker_stage_active_snapshot():
        with worker_stage_condition:
            return dict(worker_stage_active)

    def set_worker_stage_waiting_delta(stage_name, delta):
        with worker_stage_condition:
            worker_stage_waiting[stage_name] += delta
            worker_stage_condition.notify_all()
            return worker_stage_waiting[stage_name]

    def set_worker_stage_slot_holding_delta(stage_name, delta):
        with worker_stage_condition:
            worker_stage_slot_holding[stage_name] += delta
            worker_stage_condition.notify_all()
            return worker_stage_slot_holding[stage_name]

    def get_worker_stage_flow_snapshot():
        with worker_stage_condition:
            return {
                "active": dict(worker_stage_active),
                "waiting": dict(worker_stage_waiting),
                "slot_holding": dict(worker_stage_slot_holding),
            }

    def compute_step1_admission_limit():
        flow_snapshot = get_worker_stage_flow_snapshot()
        stage_active = flow_snapshot["active"]
        stage_waiting = flow_snapshot["waiting"]
        stage_slot_holding = flow_snapshot["slot_holding"]
        downstream_active = stage_active["step2"] + stage_active["step5a"] + stage_active["step5b"]
        step1_pressure = stage_slot_holding["step1"] + stage_waiting["step1"]
        dynamic_limit = min(step1_soft_limit, max(step1_headroom, downstream_active + step1_headroom))
        return dynamic_limit, stage_active, stage_waiting, stage_slot_holding, step1_pressure

    def get_active_count():
        with lock:
            return active_count

    def set_active_delta(delta):
        nonlocal active_count
        with lock:
            active_count += delta
            return active_count

    def acquire_stage_send_slot(stage_name):
        if stage_name != "step1" or step1_send_semaphore is None:
            return False

        if not step1_send_semaphore.acquire(blocking=False):
            set_worker_stage_waiting_delta(stage_name, 1)
            try:
                step1_send_semaphore.acquire()
            finally:
                set_worker_stage_waiting_delta(stage_name, -1)

        set_worker_stage_slot_holding_delta(stage_name, 1)
        return True

    def release_stage_send_slot(stage_name, slot_acquired):
        if stage_name != "step1" or step1_send_semaphore is None or not slot_acquired:
            return
        set_worker_stage_slot_holding_delta(stage_name, -1)
        step1_send_semaphore.release()

    states = []
    for global_idx, item in enumerate(sampled_data):
        states.append({
            "article_id": global_idx,
            "item": item,
            "article_dir": os.path.join(KG_BASE_DIR, f"文章顺序_{int(global_idx):06d}"),
            "graph": None,
            "entity_list": [],
            "relationship_list": [],
            "revised_text": None,
            "qa_done": not run_genqa,
            "kg_done": not run_genkg,
            "qa_df": pd.DataFrame(),
            "kg_df": pd.DataFrame(),
            "_collected": False,
        })

    def ensure_article_dir(state):
        os.makedirs(state["article_dir"], exist_ok=True)

    def save_step1_result(state, graph):
        entity_list, relationship_list = parse_graph_response_to_lists(graph)
        valid_graph = not (len(entity_list) == 0 and len(relationship_list) == 0)
        state["graph"] = graph if valid_graph else None
        state["entity_list"] = entity_list if valid_graph else []
        state["relationship_list"] = relationship_list if valid_graph else []
        ensure_article_dir(state)
        with open(os.path.join(state["article_dir"], "原始文章.txt"), 'w', encoding='utf-8') as f:
            f.write(str(state["item"]["content"]))
        if not valid_graph:
            print(f"⚠️ [Step1] article_id={state['article_id']} 解析到空图谱，跳过后续阶段。")
            remove_file_if_exists(os.path.join(state["article_dir"], "原始文章对应知识图谱.json"))
            remove_file_if_exists(os.path.join(state["article_dir"], "根据知识图谱修改后的文章.txt"))
            remove_file_if_exists(os.path.join(state["article_dir"], "知识图谱-修改后文章的实体出现信息.json"))
            return False
        with open(os.path.join(state["article_dir"], "原始文章对应知识图谱.json"), 'w', encoding='utf-8') as f:
            json.dump({"entity": entity_list, "relationship": relationship_list}, f, ensure_ascii=False, indent=2)
        return True

    def run_local_step3_silent(state):
        revised_text = str(state.get("revised_text") or "")
        entity_list = json.loads(json.dumps(state.get("entity_list", []), ensure_ascii=False))
        relationship_list = json.loads(json.dumps(state.get("relationship_list", []), ensure_ascii=False))
        entity_stats = analyze_entity_appearances(revised_text, entity_list)
        for entity in entity_list:
            if isinstance(entity, dict):
                entity_name = normalize_entity_text_value(entity.get("name", ""))
                if entity_name:
                    entity["name"] = entity_name
                stats = entity_stats.get(entity_name, {'appearance_count': 0, 'appearance_positions': []})
                entity.update(stats)
        update_relationships_with_raw_name_distance(revised_text, relationship_list)
        with open(os.path.join(state["article_dir"], "知识图谱-修改后文章的实体出现信息.json"), 'w', encoding='utf-8') as f:
            json.dump({"content": revised_text, "entity": entity_list, "relationship": relationship_list}, f, ensure_ascii=False, indent=2)

    def save_step2_result(state, revised_text):
        if is_invalid_text_payload(revised_text):
            state["revised_text"] = None
            ensure_article_dir(state)
            print(f"⚠️ [Step2] article_id={state['article_id']} 返回无效正文，跳过后续阶段。")
            remove_file_if_exists(os.path.join(state["article_dir"], "根据知识图谱修改后的文章.txt"))
            remove_file_if_exists(os.path.join(state["article_dir"], "知识图谱-修改后文章的实体出现信息.json"))
            return False
        state["revised_text"] = normalize_text_payload(revised_text)
        ensure_article_dir(state)
        with open(os.path.join(state["article_dir"], "根据知识图谱修改后的文章.txt"), 'w', encoding='utf-8') as f:
            f.write(state["revised_text"])
        run_local_step3_silent(state)
        return True

    def build_step5a_prompt(state):
        return stage5a_prompt_func(state["revised_text"], state["graph"])

    def build_step5b_prompt_and_graph_data(state):
        if is_invalid_text_payload(state.get("revised_text")):
            raise ValueError(f"article_id={state['article_id']} 的 revised_text 无效，不能构造 Step5B prompt")
        if not is_valid_graph_payload(state.get("graph")):
            raise ValueError(f"article_id={state['article_id']} 的 graph 为空，不能构造 Step5B prompt")
        base_prompt_msgs = base_very_simple_prompt_func(state["revised_text"])
        base_prompt_content = base_prompt_msgs[0]['content'] if base_prompt_msgs else ""
        graph_data = {
            "entity": json.loads(json.dumps(state["entity_list"], ensure_ascii=False)),
            "relationship": json.loads(json.dumps(state["relationship_list"], ensure_ascii=False)),
        }
        if "entity" in graph_data and isinstance(graph_data["entity"], list):
            for ent in graph_data["entity"]:
                if isinstance(ent, dict):
                    ent.pop("appearance_count", None)
                    ent.pop("appearance_positions", None)
        if "relationship" in graph_data and isinstance(graph_data["relationship"], list):
            for rel in graph_data["relationship"]:
                if isinstance(rel, dict):
                    rel.pop("distance_sub_obj", None)
                    rel.pop("raw_sub_name", None)
                    rel.pop("raw_obj_name", None)
                    rel.pop("raw_text_start", None)
                    rel.pop("raw_text_end", None)
        return stage5b_prompt_func(base_prompt_content, graph_data)

    def save_step5a_result(state, qa_raw):
        qa_records = []
        try:
            qa_data = json_repair.loads(str(qa_raw))
            if isinstance(qa_data, list):
                for q in qa_data:
                    if isinstance(q, dict):
                        qa_records.append({
                            "ArticleNUM": 0,
                            "Content": state["revised_text"],
                            "Question": q.get("question", ""),
                            "Options": " | ".join([f"{k}: {v}" for k, v in q.get("options", {}).items()]),
                            "Answers": ", ".join(q.get("answer", [])),
                            "how_to_get_answer_step_by_step": q.get("how_to_get_answer_step_by_step", ""),
                            "text_raw_from_file": state["item"].get('content', ""),
                            "source_file": state["item"].get('source', ""),
                            "original_id": state["item"].get('original_id', ""),
                            "stable_article_id": state["item"].get('stable_article_id', ""),
                            "sample_order": state["item"].get('sample_order', state["article_id"]),
                            "graph_from_text_raw_from_file": json.dumps(state["graph"], ensure_ascii=False) if isinstance(state["graph"], (dict, list)) else str(state["graph"]),
                            "text_fixed_by_revision": state["revised_text"],
                        })
        except Exception as exc:
            print(f"⚠️ QA 解析失败 article_id={state['article_id']}: {exc}")
        state["qa_df"] = pd.DataFrame(qa_records)
        state["qa_done"] = True

    def save_step5b_result(state, reasoning_response):
        if is_invalid_text_payload(reasoning_response):
            print(f"⚠️ [Step5B] article_id={state['article_id']} 推理链为空，跳过该样本。")
            state["kg_done"] = False
            state["kg_df"] = pd.DataFrame()
            return False
        if is_invalid_text_payload(state.get("revised_text")) or not is_valid_graph_payload(state.get("graph")):
            print(f"⚠️ [Step5B] article_id={state['article_id']} 输入正文/图谱无效，跳过该样本。")
            state["kg_done"] = False
            state["kg_df"] = pd.DataFrame()
            return False
        original_prompt_msgs = base_very_simple_prompt_func(state["revised_text"])
        clean_entities, clean_relationships = build_very_simple_ground_truth_lists(
            state["entity_list"],
            state["relationship_list"],
        )
        entity_json_str = json.dumps(clean_entities, ensure_ascii=False)
        rel_json_str = json.dumps(clean_relationships, ensure_ascii=False)
        full_ground_truth = (
            f"{str(reasoning_response).strip()}\n\n"
            f"#Entity_List_Start#\n{entity_json_str}\n#Entity_List_End#\n\n"
            f"#Relationship_List_Start#\n{rel_json_str}\n#Relationship_List_End#"
        )
        state["kg_df"] = pd.DataFrame([{
            "prompt": original_prompt_msgs,
            "ground_truth": full_ground_truth,
            "text_raw_from_file": state["item"].get('content', ""),
            "source_file": state["item"].get('source', ""),
            "original_id": state["item"].get('original_id', ""),
            "stable_article_id": state["item"].get('stable_article_id', ""),
            "sample_order": state["item"].get('sample_order', state["article_id"]),
            "graph_from_text_raw_from_file": json.dumps(state["graph"], ensure_ascii=False) if isinstance(state["graph"], (dict, list)) else str(state["graph"]),
            "text_fixed_by_revision": state["revised_text"],
            "index": 0,
            "totalnum": 1,
        }])
        state["kg_done"] = True
        return True

    def is_state_complete(state):
        if not is_valid_graph_payload(state.get("graph")):
            return False
        if is_invalid_text_payload(state.get("revised_text")):
            return False
        if run_genqa and not state.get("qa_done", False):
            return False
        if run_genkg and not state.get("kg_done", False):
            return False
        return True

    def collect_completed_state(state):
        if state.get("_collected"):
            return
        if not is_state_complete(state):
            return
        if run_genqa and state["qa_df"] is not None and not state["qa_df"].empty:
            qa_frames.append(state["qa_df"])
        if run_genkg and state["kg_df"] is not None and not state["kg_df"].empty:
            kg_frames.append(state["kg_df"])
        all_folder_paths.append(state["article_dir"])
        state["_collected"] = True

    def run_cache_scan(stage_name, target_states, prompt_builder, result_saver, stage_config, max_workers_vllm, cache_signature, chunk_size):
        if not target_states:
            print(f"🧊 [Startup Cache Scan:{stage_name}] 无待检查文章")
            return [], []

        print(f"🧊 [Startup Cache Scan:{stage_name}] 开始检查 {len(target_states)} 篇文章的现有缓存...")
        hit_states = []
        miss_states = []
        think_val = stage_config.get("think")
        if think_val is None:
            think_val = stage_config.get("openai_reasoning", 2)
        for start in range(0, len(target_states), chunk_size):
            chunk_states = target_states[start:start + chunk_size]
            with ThreadPoolExecutor(max_workers=min(64, len(chunk_states))) as executor:
                chunk_prompts = list(executor.map(prompt_builder, chunk_states))

            cached_results, _, uncached_indices, _ = tools.check_cache_batch(
                prompt_list=chunk_prompts,
                model=stage_config["model"],
                token=stage_config["token"],
                temp=stage_config["temp"],
                think=think_val,
                max_workers_Vllm=max_workers_vllm,
                openai_verbosity=stage_config.get("openai_verbosity"),
                showthink=False,
                model_name_override=cache_signature["setmodel"],
                top_p=cache_signature["top_p"],
                top_k=cache_signature["top_k"],
            )

            chunk_hit = 0
            chunk_miss = 0
            uncached_index_set = set(uncached_indices)
            for idx, state in enumerate(chunk_states):
                response = cached_results.get(idx)
                if idx in cached_results and is_valid_cached_result(response):
                    bump_stage_counter(stage_name, "hit")
                    stage_ok = result_saver(state, response)
                    if stage_ok:
                        hit_states.append(state)
                        chunk_hit += 1
                    else:
                        if idx not in uncached_index_set:
                            uncached_index_set.add(idx)
                        miss_states.append(state)
                        chunk_miss += 1
                else:
                    
                    if idx not in uncached_index_set:
                        uncached_index_set.add(idx)
                    bump_stage_counter(stage_name, "miss")
                    miss_states.append(state)
                    chunk_miss += 1
            print(
                f"   📦 [{stage_name}] chunk {start}:{start + len(chunk_states)} -> "
                f"hit={chunk_hit}, miss={chunk_miss}"
            )

        print(f"✅ [Startup Cache Scan:{stage_name}] 命中 {len(hit_states)}，未命中 {len(miss_states)}")
        return hit_states, miss_states

    def ask_single_prompt_no_cache(prompt, stage_name, stage_config, max_workers_vllm):
        stage_slot_acquired = acquire_stage_send_slot(stage_name)
        miss_rate_limiter.wait(stage_name=stage_name)
        cloud_kwargs = build_stage_cloud_kwargs(stage_config, max_workers_vllm=max_workers_vllm)
        
        
        cloud_kwargs["check_history_cache"] = True
        cloud_kwargs["only_check_cache"] = False
        set_worker_stage_active_delta(stage_name, 1)
        try:
            responses = tools.ask_group_link(
                prompt_list=[prompt],
                model=stage_config["model"],
                token=stage_config["token"],
                temp=stage_config["temp"],
                **cloud_kwargs,
            )
            responses = tools.cleanthinkans(responses)
            return responses[0] if responses else None
        finally:
            set_worker_stage_active_delta(stage_name, -1)
            release_stage_send_slot(stage_name, stage_slot_acquired)

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    step1_hit_states, step1_miss_states = run_cache_scan(
        "step1",
        states,
        lambda s: stage1_prompt_func(s["item"]["content"]),
        save_step1_result,
        STAGE1_CONFIG,
        stage1_workers,
        build_stage_cache_signature(STAGE1_CONFIG, max_workers_vllm=stage1_workers),
        stage_cache_scan_batch_sizes["step1"],
    )

    step2_hit_states, step2_miss_states = run_cache_scan(
        "step2",
        step1_hit_states,
        lambda s: stage2_prompt_func(s["item"]["content"], s["graph"]),
        save_step2_result,
        STAGE2_CONFIG,
        stage2_workers,
        build_stage_cache_signature(STAGE2_CONFIG, max_workers_vllm=stage2_workers),
        stage_cache_scan_batch_sizes["step2"],
    )

    if run_genqa:
        run_cache_scan(
            "step5a",
            step2_hit_states,
            build_step5a_prompt,
            save_step5a_result,
            STAGE5A_CONFIG,
            stage5a_workers,
            build_stage_cache_signature(STAGE5A_CONFIG, max_workers_vllm=stage5a_workers),
            stage_cache_scan_batch_sizes["step5a"],
        )
    else:
        print("⏭️ (Startup Cache Scan) 跳过 Step 5-A QA")

    if run_genkg:
        if skip_reasoning:
            placeholder_reasoning = "%这是SFT推理链的填充符因为打开了-skip-reason-trace-re-generate%"
            synthetic_step5b_ok = 0
            synthetic_step5b_drop = 0
            print(
                f"⏭️ (Startup Cache Scan) 跳过 Step 5-B SFT 推理复现，"
                f"直接为 {len(step2_hit_states)} 篇 Step2 命中文章填充占位推理链。"
            )
            for state in step2_hit_states:
                if save_step5b_result(state, placeholder_reasoning):
                    synthetic_step5b_ok += 1
                else:
                    synthetic_step5b_drop += 1
            print(
                f"✅ [Startup Skip Step5B] placeholder_kept={synthetic_step5b_ok}, "
                f"dropped={synthetic_step5b_drop}"
            )
        else:
            run_cache_scan(
                "step5b",
                step2_hit_states,
                build_step5b_prompt_and_graph_data,
                save_step5b_result,
                STAGE5B_CONFIG,
                stage5b_workers,
                build_stage_cache_signature(STAGE5B_CONFIG, max_workers_vllm=stage5b_workers),
                stage_cache_scan_batch_sizes["step5b"],
            )
    else:
        print("⏭️ (Startup Cache Scan) 跳过 Step 5-B KG")

    completed_articles = 0
    for state in states:
        if is_state_complete(state):
            collect_completed_state(state)
            completed_articles += 1

    pending_states = [state for state in states if not is_state_complete(state)]
    pending_high_states = deque([state for state in pending_states if state["item"].get("priority_tier") == "high"])
    pending_low_states = deque([state for state in pending_states if state["item"].get("priority_tier") != "high"])
    priority_unlock_threshold = max(0, int(PRIORITY_UNLOCK_LOW_PRIORITY_THRESHOLD))
    remaining_high_unfinished = len(pending_high_states)
    priority_dynamic_gate_enabled = bool(PRIORITY_REFERENCE_PARQUET_PATHS)
    with stage_counter_lock:
        startup_counter_snapshot = json.dumps(stage_counters, ensure_ascii=False)
    print(
        f"✅ [Startup Cache Scan Done] completed_from_cache={completed_articles}/{total_articles}, "
        f"pending_for_workers={len(pending_states)}, stage_cache={startup_counter_snapshot}"
    )
    if priority_dynamic_gate_enabled:
        print(
            f"🎯 [Priority Tier Worker Gate] pending_high={len(pending_high_states)}, "
            f"pending_low={len(pending_low_states)}, unlock_threshold={priority_unlock_threshold}, "
            f"low_unlocked={remaining_high_unfinished <= priority_unlock_threshold}"
        )

    if ONLY_CHECK_CACHE:
        print("🛑 ONLY_CHECK_CACHE=True，启动缓存检查阶段结束后不再进入发送阶段。")
        if all_folder_paths:
            step_4_generate_plots(all_folder_paths)
            rename_and_generate_statistics(all_folder_paths)
        return concat_nonempty_dataframes(qa_frames), concat_nonempty_dataframes(kg_frames)

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    print(f"🚚 [Worker Phase] 开始处理 {len(pending_states)} 篇缺缓存文章...")

    def run_pending_article_to_completion(state):
        article_id = state["article_id"]
        try:
            ensure_article_dir(state)

            if state.get("graph") is None:
                graph = ask_single_prompt_no_cache(
                    stage1_prompt_func(state["item"]["content"]),
                    "step1",
                    STAGE1_CONFIG,
                    stage1_workers,
                )
                if not save_step1_result(state, graph):
                    return {"status": "filtered", "article_id": article_id, "reason": "invalid_step1_graph"}

            if state.get("revised_text") is None:
                revised_text = ask_single_prompt_no_cache(
                    stage2_prompt_func(state["item"]["content"], state["graph"]),
                    "step2",
                    STAGE2_CONFIG,
                    stage2_workers,
                )
                if not save_step2_result(state, revised_text):
                    return {"status": "filtered", "article_id": article_id, "reason": "invalid_step2_text"}

            if run_genqa and not state.get("qa_done", False):
                qa_raw = ask_single_prompt_no_cache(
                    build_step5a_prompt(state),
                    "step5a",
                    STAGE5A_CONFIG,
                    stage5a_workers,
                )
                save_step5a_result(state, qa_raw)

            if run_genkg and not state.get("kg_done", False):
                if skip_reasoning:
                    reasoning_response = "%这是SFT推理链的填充符因为打开了-skip-reason-trace-re-generate%"
                else:
                    reasoning_response = ask_single_prompt_no_cache(
                        build_step5b_prompt_and_graph_data(state),
                        "step5b",
                        STAGE5B_CONFIG,
                        stage5b_workers,
                    )
                if not save_step5b_result(state, reasoning_response):
                    return {"status": "filtered", "article_id": article_id, "reason": "invalid_step5b_reasoning"}

            return {"status": "ok", "state": state}
        except Exception as exc:
            short_tb = traceback.format_exc(limit=6)
            with lock:
                errors.append({
                    "article_id": article_id,
                    "error": repr(exc),
                    "traceback": short_tb,
                })
            print(f"⚠️ Worker Article Error: article_id={article_id}, error={exc!r}")
            print(short_tb)
            return {"status": "error", "article_id": article_id, "error": repr(exc)}
        finally:
            set_active_delta(-1)

    def worker_wrapper(state):
        set_active_delta(1)
        return run_pending_article_to_completion(state)

    if pending_states:
        worker_monitor_stop = threading.Event()

        
        def worker_monitor_loop():
            while not worker_monitor_stop.wait(15):
                flow_snapshot = get_worker_stage_flow_snapshot()
                stage_active = flow_snapshot["active"]
                stage_waiting = flow_snapshot["waiting"]
                stage_slot_holding = flow_snapshot["slot_holding"]
                with lock:
                    current_remaining_high_unfinished = remaining_high_unfinished
                current_low_unlocked = (
                    (not priority_dynamic_gate_enabled)
                    or current_remaining_high_unfinished <= priority_unlock_threshold
                )
                print(
                    f"📡 Worker Active Snapshot: articles_in_flight={get_active_count()}, "
                    f"step1_active={stage_active['step1']}, "
                    f"step1_slots={stage_slot_holding['step1']}, "
                    f"step1_waiting={stage_waiting['step1']}, "
                    f"step2_active={stage_active['step2']}, "
                    f"step5a_active={stage_active['step5a']}, "
                    f"step5b_active={stage_active['step5b']}, "
                    f"priority_high_remaining={current_remaining_high_unfinished}, "
                    f"priority_high_queued={len(pending_high_states)}, "
                    f"priority_low_queued={len(pending_low_states)}, "
                    f"priority_low_unlocked={current_low_unlocked}"
                )

        worker_monitor_thread = threading.Thread(
            target=worker_monitor_loop,
            name="worker-active-monitor",
            daemon=True,
        )
        worker_monitor_thread.start()
        with ThreadPoolExecutor(max_workers=min(max_in_flight, len(pending_states))) as executor:
            future_to_state = {}
            worker_target = min(max_in_flight, len(pending_states))
            submit_batch_soft_cap = max(1, min(step1_headroom, 64))

            def pick_next_state_for_submission():
                nonlocal remaining_high_unfinished

                if not priority_dynamic_gate_enabled:
                    if pending_high_states:
                        return pending_high_states.popleft()
                    if pending_low_states:
                        return pending_low_states.popleft()
                    return None

                low_unlocked = remaining_high_unfinished <= priority_unlock_threshold
                if pending_high_states and not low_unlocked:
                    return pending_high_states.popleft()

                if pending_high_states and pending_low_states:
                    next_high = pending_high_states[0]
                    next_low = pending_low_states[0]
                    next_high_order = next_high["item"].get("sample_order", next_high["article_id"])
                    next_low_order = next_low["item"].get("sample_order", next_low["article_id"])
                    if next_high_order <= next_low_order:
                        return pending_high_states.popleft()
                    return pending_low_states.popleft()

                if pending_high_states:
                    return pending_high_states.popleft()

                if low_unlocked and pending_low_states:
                    return pending_low_states.popleft()

                return None

            def try_submit_more_workers():
                submitted_now = 0
                while len(future_to_state) < worker_target:
                    dynamic_step1_limit, stage_active, stage_waiting, stage_slot_holding, step1_pressure = compute_step1_admission_limit()
                    if future_to_state and step1_pressure >= dynamic_step1_limit:
                        break
                    state = pick_next_state_for_submission()
                    if state is None:
                        break
                    future = executor.submit(worker_wrapper, state)
                    future_to_state[future] = state
                    submitted_now += 1
                    if submitted_now >= submit_batch_soft_cap:
                        break
                return submitted_now

            try_submit_more_workers()
            while future_to_state:
                done, _ = wait(list(future_to_state.keys()), timeout=1.0, return_when=FIRST_COMPLETED)
                if not done:
                    try_submit_more_workers()
                    continue

                for future in done:
                    finished_state = future_to_state.pop(future, None)
                    result = future.result()
                    if finished_state and finished_state["item"].get("priority_tier") == "high":
                        with lock:
                            remaining_high_unfinished = max(0, remaining_high_unfinished - 1)
                    if result.get("status") == "ok":
                        state = result["state"]
                        with lock:
                            collect_completed_state(state)
                        completed_articles += 1
                        if completed_articles % 50 == 0:
                            flow_snapshot = get_worker_stage_flow_snapshot()
                            stage_active = flow_snapshot["active"]
                            stage_waiting = flow_snapshot["waiting"]
                            stage_slot_holding = flow_snapshot["slot_holding"]
                            print(
                                f"✅ Worker Progress: completed_articles={completed_articles}/{total_articles}, "
                                f"in_flight={get_active_count()}, "
                                f"step1_active={stage_active['step1']}, "
                                f"step1_slots={stage_slot_holding['step1']}, "
                                f"step1_waiting={stage_waiting['step1']}, "
                                f"step2_active={stage_active['step2']}, "
                                f"step5a_active={stage_active['step5a']}, "
                                f"step5b_active={stage_active['step5b']}, "
                                f"priority_high_remaining={remaining_high_unfinished}, "
                                f"priority_high_queued={len(pending_high_states)}, "
                                f"priority_low_queued={len(pending_low_states)}, "
                                f"priority_low_unlocked={remaining_high_unfinished <= priority_unlock_threshold}"
                            )
                    else:
                        completed_articles += 1
                        if completed_articles % 50 == 0:
                            stage_active = get_worker_stage_active_snapshot()
                            print(
                                f"⚠️ Worker Error Progress: completed_articles={completed_articles}/{total_articles}, "
                                f"errors={len(errors)}, in_flight={get_active_count()}, "
                                f"step1_active={stage_active['step1']}, "
                                f"step2_active={stage_active['step2']}, "
                                f"step5a_active={stage_active['step5a']}, "
                                f"step5b_active={stage_active['step5b']}, "
                                f"priority_high_remaining={remaining_high_unfinished}, "
                                f"priority_high_queued={len(pending_high_states)}, "
                                f"priority_low_queued={len(pending_low_states)}, "
                                f"priority_low_unlocked={remaining_high_unfinished <= priority_unlock_threshold}"
                            )
                try_submit_more_workers()
        worker_monitor_stop.set()
        worker_monitor_thread.join(timeout=1)

    if errors:
        print(f"⚠️ Worker Phase 期间共有 {len(errors)} 篇文章报错；已跳过这些文章并继续处理。")
        preview_count = min(3, len(errors))
        for idx in range(preview_count):
            err = errors[idx]
            print(f"   [{idx + 1}/{preview_count}] article_id={err['article_id']} error={err['error']}")

    if all_folder_paths:
        step_4_generate_plots(all_folder_paths)
        rename_and_generate_statistics(all_folder_paths)
    else:
        print("⚠️ 两阶段流水线完成，但没有可用于 Step 4-4.5 的有效文件夹。")

    if not run_genqa:
        print("⏭️ (Step 5-A) 跳过 QA Generation")
    if not run_genkg:
        print("⏭️ (Step 5-B) 跳过 KG Extraction Data Generation")

    return concat_nonempty_dataframes(qa_frames), concat_nonempty_dataframes(kg_frames)


def build_single_input_source(path, input_type="", content_col="content", id_col="id",
                              id_prefix="", name=None, filter_col=None, filter_value=None):
    source_type = normalize_source_type(input_type, path)
    source = {
        "path": path,
        "type": source_type,
        "content_col": content_col,
        "id_col": id_col,
        "id_prefix": id_prefix,
        "name": name or os.path.splitext(os.path.basename(path))[0],
    }
    if filter_col:
        source["filter_col"] = filter_col
    if filter_value is not None:
        source["filter_value"] = filter_value
    return source


# =============================================================================
# --- Helper Functions (Text, Entity & Distance Processing) ---
# =============================================================================

def find_all_occurrences(text, entity_name, case_insensitive=True):
    """Find all occurrences of an entity name in the text."""
    normalized_entity_name = normalize_entity_text_value(entity_name)
    if not normalized_entity_name or not text:
        return []
    escaped_entity = re.escape(normalized_entity_name)
    flags = re.IGNORECASE if case_insensitive else 0
    pattern = re.compile(escaped_entity, flags)
    matches = []
    for match in pattern.finditer(text):
        matches.append((match.start(), match.end()))
    return matches


def normalize_entity_text_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""
    if isinstance(value, dict):
        for key in ("name", "text", "value", "entity", "canonical_name", "label", "title"):
            normalized = normalize_entity_text_value(value.get(key))
            if normalized:
                return normalized
        return ""
    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = normalize_entity_text_value(item)
            if normalized:
                return normalized
        return ""
    return str(value).strip()

def analyze_entity_appearances(text, entity_list):
    """Analyze appearance statistics for all entities and their aliases in the text."""
    entity_groups = defaultdict(list)
    for entity in entity_list:
        if not isinstance(entity, dict): continue
        entity_name = normalize_entity_text_value(entity.get("name", ""))
        if not entity_name: continue
        aliases = entity.get("alias", [])
        if not isinstance(aliases, list):
            aliases = [aliases] if aliases else []
        valid_aliases = []
        seen_aliases = set()
        for alias in aliases:
            alias_text = normalize_entity_text_value(alias)
            if not alias_text or alias_text == "None" or alias_text in seen_aliases:
                continue
            seen_aliases.add(alias_text)
            valid_aliases.append(alias_text)
        entity_groups[entity_name].append(entity_name)
        entity_groups[entity_name].extend(valid_aliases)

    entity_stats = {}
    for canonical_name, names in entity_groups.items():
        all_positions = []
        total_count = 0
        for name in names:
            positions = find_all_occurrences(text, name)
            all_positions.extend(positions)
            total_count += len(positions)
        all_positions.sort(key=lambda x: x[0])
        
        if all_positions:
            filtered_positions = [all_positions[0]]
            for start, end in all_positions[1:]:
                last_start, last_end = filtered_positions[-1]
                if start >= last_end:
                    filtered_positions.append((start, end))
        else:
            filtered_positions = []

        entity_stats[canonical_name] = {
            'appearance_count': total_count,
            'appearance_positions': filtered_positions
        }
    return entity_stats


def flatten_anchor_candidates(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        results = []
        seen = set()
        for item in value:
            for candidate in flatten_anchor_candidates(item):
                if candidate not in seen:
                    seen.add(candidate)
                    results.append(candidate)
        return results
    text = str(value).strip()
    return [text] if text else []


def resolve_anchor_text(value, content_str=""):
    candidates = flatten_anchor_candidates(value)
    if not candidates:
        return ""
    for candidate in candidates:
        if candidate and candidate in content_str:
            return candidate
    return candidates[0]


def build_very_simple_ground_truth_lists(entity_list, relationship_list):
    clean_entities = json.loads(json.dumps(entity_list or [], ensure_ascii=False))
    clean_relationships = json.loads(json.dumps(relationship_list or [], ensure_ascii=False))

    def _normalize_optional_list_field(value):
        if value is None:
            return None
        if isinstance(value, list):
            normalized = []
            for item in value:
                text = str(item).strip()
                if not text or text == "None":
                    continue
                normalized.append(text)
            return normalized or None
        text = str(value).strip()
        if not text or text == "None":
            return None
        return [text]

    if isinstance(clean_entities, list):
        for ent in clean_entities:
            if isinstance(ent, dict):
                normalized_name = normalize_entity_text_value(ent.get("name"))
                if normalized_name:
                    ent["name"] = normalized_name
                ent.pop("appearance_count", None)
                ent.pop("appearance_positions", None)
                aliases = _normalize_optional_list_field(ent.get("alias"))
                if aliases is None:
                    ent.pop("alias", None)
                else:
                    ent["alias"] = aliases

                mother_entities = _normalize_optional_list_field(ent.get("mother entity"))
                if mother_entities is None:
                    ent.pop("mother entity", None)
                else:
                    ent["mother entity"] = mother_entities

    if isinstance(clean_relationships, list):
        for rel in clean_relationships:
            if isinstance(rel, dict):
                rel.pop("distance_sub_obj", None)
                rel.pop("raw_sub_name", None)
                rel.pop("raw_obj_name", None)
                rel.pop("raw_text_start", None)
                rel.pop("raw_text_end", None)
                factuality = _normalize_optional_list_field(rel.get("special_factuality"))
                if factuality is None:
                    rel.pop("special_factuality", None)
                else:
                    rel["special_factuality"] = factuality

    return clean_entities, clean_relationships

def update_relationships_with_raw_name_distance(content, relationship_list):
    content_str = str(content)
    
    for rel in relationship_list:
        if not isinstance(rel, dict): continue
        
        
        rel["distance_sub_obj"] = -1

        raw_sub = resolve_anchor_text(rel.get("raw_sub_name", ""), content_str)
        raw_obj = resolve_anchor_text(rel.get("raw_obj_name", ""), content_str)
        rel["raw_sub_name"] = raw_sub
        rel["raw_obj_name"] = raw_obj

        if not raw_sub or not raw_obj:
            continue

        
        sub_pos = content_str.find(raw_sub)
        obj_pos = content_str.find(raw_obj)

        
        if sub_pos != -1 and obj_pos != -1:
            
            dist = int(abs(sub_pos - obj_pos))
            rel["distance_sub_obj"] = dist
        else:
            
            pass

# =============================================================================
# --- Graph Plotting Functions ---
# =============================================================================

def get_entity_positions(entity_list):
    positions = {}
    for entity in entity_list:
        if not isinstance(entity, dict): continue
        entity_name = normalize_entity_text_value(entity.get("name", ""))
        if not entity_name: continue
        appearance_positions = entity.get("appearance_positions", [])
        if appearance_positions and isinstance(appearance_positions[0], (list, tuple)) and len(appearance_positions[0]) > 0:
            positions[entity_name] = appearance_positions[0][0]
    return positions

def get_all_entity_positions_map(entity_list):
    all_positions = {}
    for entity in entity_list:
        if not isinstance(entity, dict): continue
        entity_name = normalize_entity_text_value(entity.get("name", ""))
        if not entity_name: continue
        positions_list = entity.get("appearance_positions", [])
        if positions_list:
            all_positions[entity_name] = [
                pos[0] for pos in positions_list 
                if isinstance(pos, (list, tuple)) and len(pos) > 0
            ]
    return all_positions

def find_closest_position(positions_list, anchor_pos):
    if not positions_list: return None
    min_dist = float('inf')
    best_pos = positions_list[0]
    for pos in positions_list:
        dist = abs(pos - anchor_pos)
        if dist < min_dist:
            min_dist = dist
            best_pos = pos
    return best_pos

def get_relationship_pairs_and_endpoints_for_plot(content, relationship_list, all_entity_positions_map):
    pairs = []
    line_endpoints = set()
    content_str = str(content)

    for relationship in relationship_list:
        if not isinstance(relationship, dict): continue
        
        sub_name = relationship.get("sub", "")
        obj_name = relationship.get("obj", "")
        start_clue = resolve_anchor_text(relationship.get("raw_text_start", ""), content_str)
        end_clue = resolve_anchor_text(relationship.get("raw_text_end", ""), content_str)
        relationship["raw_text_start"] = start_clue
        relationship["raw_text_end"] = end_clue

        if not all([sub_name, obj_name, start_clue, end_clue]): continue 

        start_pos = content_str.find(start_clue)
        if start_pos == -1: continue
        end_pos_clue = content_str.find(end_clue, start_pos)
        if end_pos_clue == -1: continue
        end_pos = end_pos_clue + len(end_clue)
        
        anchor_pos = (start_pos + end_pos) / 2.0
        all_pos_sub = all_entity_positions_map.get(sub_name, [])
        all_pos_obj = all_entity_positions_map.get(obj_name, [])

        if not all_pos_sub or not all_pos_obj: continue
            
        best_sub_pos = find_closest_position(all_pos_sub, anchor_pos)
        best_obj_pos = find_closest_position(all_pos_obj, anchor_pos)
        
        rel_type = relationship.get("rel", "unknown")
        pairs.append((sub_name, obj_name, best_sub_pos, best_obj_pos, rel_type))
        line_endpoints.add(best_sub_pos)
        line_endpoints.add(best_obj_pos)
        
    return pairs, line_endpoints

def plot_knowledge_graph(content, entity_list, relationship_list, output_path):
    # 1. Build Alias Map (Name/Alias -> Canonical Name)
    alias_to_canonical = {}
    for ent in entity_list:
        if not isinstance(ent, dict): continue
        name = ent.get("name")
        if not name: continue
        alias_to_canonical[name] = name
        for alias in ent.get("alias", []):
            if alias and alias != "None":
                alias_to_canonical[alias] = name

    # 2. Normalize Relationships to Canonical Names & Identify Active Entities
    canonical_relationship_list = []
    active_canonical_entities = set()
    
    for rel in relationship_list:
        if not isinstance(rel, dict): continue
        new_rel = rel.copy()
        
        sub = rel.get("sub", "")
        obj = rel.get("obj", "")
        
        if sub:
            canon_sub = alias_to_canonical.get(sub, sub)
            new_rel["sub"] = canon_sub
            active_canonical_entities.add(canon_sub)
        
        if obj:
            canon_obj = alias_to_canonical.get(obj, obj)
            new_rel["obj"] = canon_obj
            active_canonical_entities.add(canon_obj)
            
        canonical_relationship_list.append(new_rel)

    all_entity_positions_map = get_all_entity_positions_map(entity_list)
    # Use canonical_relationship_list for plotting so aliases are resolved to canonical positions
    relationship_pairs, line_endpoints = get_relationship_pairs_and_endpoints_for_plot(content, canonical_relationship_list, all_entity_positions_map)
    first_occurrence_positions = get_entity_positions(entity_list)

    num_entities = len(active_canonical_entities)
    height = max(8, num_entities * 0.3 + 8)
    fig = Figure(figsize=(15, height))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    content_length = len(content)
    ax.set_ylim(content_length + 100, -100) 
    ax.set_xlim(-2.2, 2.2) 
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(True)
    ax.set_ylabel('Position in Article (characters)', fontsize=10)

    nodes_to_draw_map = {}
    # Filter: Only draw if canonical name is active
    for name, pos in first_occurrence_positions.items():
        if name in active_canonical_entities:
            nodes_to_draw_map[pos] = name
    for name, positions in all_entity_positions_map.items():
        if name in active_canonical_entities:
            for pos in positions:
                if pos in line_endpoints:
                    nodes_to_draw_map[pos] = name

    entity_x_pos = 0 
    try:
        from matplotlib import colormaps
        cmap = colormaps['tab10']
        unique_entity_names = list(active_canonical_entities)
        entity_colors = cmap(np.linspace(0, 1, max(len(unique_entity_names), 1)))
    except (AttributeError, ImportError):
        import matplotlib.cm as cm
        unique_entity_names = list(active_canonical_entities)
        entity_colors = cm.get_cmap('tab10')(np.linspace(0, 1, max(len(unique_entity_names), 1)))

    color_map = {name: entity_colors[i % len(entity_colors)] for i, name in enumerate(unique_entity_names)}

    # Data for JSON export
    export_nodes = []
    export_links = []

    for pos, entity_name in nodes_to_draw_map.items():
        color = color_map.get(entity_name, 'gray')
        ax.scatter(entity_x_pos, pos, s=100, c=[color], alpha=0.8, edgecolors='black', linewidth=2)
        
        
        
        ax.annotate(
            entity_name,
            (entity_x_pos, pos),
            xytext=(15, 0),
            textcoords='offset points',
            ha='left',
            va='center',
            fontsize=8,
            parse_math=False,
        )
        export_nodes.append({"name": entity_name, "position": pos})

    for i, (entity1, entity2, pos1, pos2, rel_type) in enumerate(relationship_pairs):
        # Ensure we only draw/export if both are active (should be guaranteed by get_relationship_pairs logic but good to be safe)
        # Actually get_relationship_pairs checks all_entity_positions_map, which contains all entities.
        # We should double check if they are in active_canonical_entities? 
        # The logic in get_relationship_pairs uses sub_name/obj_name from relationship directly.
        # If those are aliases, we might have issues if we don't map them.
        # But get_relationship_pairs uses all_entity_positions_map.get(sub_name). 
        # If sub_name is an alias, does all_entity_positions_map have it?
        # get_all_entity_positions_map keys are entity['name'] (canonical).
        # So if relationship uses alias, get_relationship_pairs might fail to find positions if not handled.
        # However, the prompt didn't ask to fix get_relationship_pairs, just filtering.
        # Assuming relationship uses canonical names OR aliases are not the issue for plotting lines (if they match keys).
        # But for filtering "isolated" ones, we used the map.
        
        span = abs(pos2 - pos1)
        max_span = content_length if content_length > 0 else 1
        base_height = 0.4
        scale_factor = 1.6
        curve_height = base_height + (span / max_span) * scale_factor
        mid_y = (pos1 + pos2) / 2
        mid_x = entity_x_pos - curve_height 
        t = np.linspace(0, 1, 100)
        curve_y = (1-t)**2 * pos1 + 2*(1-t)*t * mid_y + t**2 * pos2
        curve_x = (1-t)**2 * entity_x_pos + 2*(1-t)*t * mid_x + t**2 * entity_x_pos
        ax.plot(curve_x, curve_y, color='black', linewidth=2, alpha=0.7)
        
        export_links.append({
            "source": entity1,
            "target": entity2,
            "source_pos": pos1,
            "target_pos": pos2,
            "relation": rel_type
        })

    ax.set_title(
        f'Knowledge Graph Entity Distribution\nTotal entities drawn: {len(set(nodes_to_draw_map.values()))}, Relationships: {len(relationship_pairs)}',
        fontsize=12,
        pad=20,
        parse_math=False,
    )
    fig.tight_layout()
    canvas.print_figure(output_path, dpi=96, bbox_inches='tight')
    fig.clear()
    plt.close(fig)
    del fig, canvas, ax

    # Export Data to JSON
    json_output_path = output_path.replace("文章关系图.jpg", "文章关系图-数据.json")
    # If extension is different, handle it, but here we know it's .jpg
    
    export_data = {
        "nodes": export_nodes,
        "links": export_links
    }
    
    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save graph data JSON: {e}")

    return f"Generated: {os.path.basename(os.path.dirname(output_path))}"

def process_article_folder(folder_path):
    json_file = os.path.join(folder_path, "知识图谱-修改后文章的实体出现信息.json")
    output_file = os.path.join(folder_path, "文章关系图.jpg")
    if not os.path.exists(json_file): raise FileNotFoundError(f"⚠️ JSON file not found: {json_file}")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        content = data.get("content", "")
        entity_list = data.get("entity", [])
        relationship_list = data.get("relationship", [])
        if not content: raise ValueError(f"⚠️ No content found in {json_file}")
        return plot_knowledge_graph(content, entity_list, relationship_list, output_file)
    except Exception as e:
        
        
        print(f"⚠️ 跳过异常关系图: {json_file} | error={e}")
        return None

def generate_graph_plots(target_folder_paths):
    print(f"\n🚀 (2.5) 开始为当前批次的 {len(target_folder_paths)} 篇文章【生成关系图】...")
    
    if not target_folder_paths:
        print("⚠️ 没有文件夹需要处理.")
        return

    
    
    num_workers = min(64, (os.cpu_count() or 1) * 2, max(1, len(target_folder_paths))) 
    
    chunksize = max(1, len(target_folder_paths) // (num_workers * 4))
    success_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        results = list(tqdm(
            executor.map(process_article_folder, target_folder_paths, chunksize=chunksize),
            total=len(target_folder_paths),
            desc="Generating Plots",
            ncols=100
        ))
        success_count = sum(1 for item in results if item is not None)
        failed_count = len(target_folder_paths) - success_count

    print(f"✅ 关系图生成完成: success={success_count}, failed={failed_count}, total={len(target_folder_paths)}")

# =============================================================================
# --- Statistics & Renaming Functions ---
# =============================================================================

def analyze_folder_stats(folder_path, target_file_name, graph_data_file_name):
    """
    Analyze a single folder for token length and graph stats.
    Returns a dict with stats.
    """
    stats = {
        "Folder": os.path.basename(folder_path),
        "TokenLength": 0,
        "CharLength": 0,
        "EntityCount": 0,
        "RelationshipCount": 0,
        "IsolatedEntityCount": 0,
        "IsolatedRatio": 0.0,
        "AvgRelDistance": 0.0,
        "MaxRelDistance": 0,
        "MinRelDistance": 0
    }
    
    # 1. Token Length
    txt_path = os.path.join(folder_path, target_file_name)
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            stats["TokenLength"] = tools.tokenlen(content)
            stats["CharLength"] = len(content)
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")

    # 2. Graph Stats
    json_path = os.path.join(folder_path, graph_data_file_name)
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            nodes = data.get("nodes", [])
            links = data.get("links", [])
            
            stats["EntityCount"] = len(nodes)
            stats["RelationshipCount"] = len(links)
            
            # Isolated Entities
            entities_in_links = set()
            for link in links:
                entities_in_links.add(link.get("source", ""))
                entities_in_links.add(link.get("target", ""))
            
            isolated_count = 0
            for node in nodes:
                if node.get("name", "") not in entities_in_links:
                    isolated_count += 1
            stats["IsolatedEntityCount"] = isolated_count
            stats["IsolatedRatio"] = isolated_count / len(nodes) if len(nodes) > 0 else 0
            
            # Distances
            distances = []
            for link in links:
                s = link.get("source_pos", 0)
                t = link.get("target_pos", 0)
                distances.append(abs(t - s))
            
            if distances:
                stats["AvgRelDistance"] = np.mean(distances)
                stats["MaxRelDistance"] = np.max(distances)
                stats["MinRelDistance"] = np.min(distances)
                
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            
    return stats

def plot_distributions_integrated(df, output_path):
    """
    Plot distributions from the DataFrame.
    """
    if df.empty: return

    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle('Knowledge Graph Statistics Distribution', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Entity Count
    ax1 = axes[0]
    data = df['EntityCount']
    if len(data) > 0 and data.max() > data.min():
        bins = np.arange(data.min(), data.max() + 2, max(1, (data.max() - data.min()) // 30))
    else:
        bins = 10
    ax1.hist(data, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_title(f'Entity Count (Mean: {data.mean():.2f})')
    
    # 2. Relationship Count
    ax2 = axes[1]
    data = df['RelationshipCount']
    if len(data) > 0 and data.max() > data.min():
        bins = np.arange(data.min(), data.max() + 2, max(1, (data.max() - data.min()) // 30))
    else:
        bins = 10
    ax2.hist(data, bins=bins, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_title(f'Relationship Count (Mean: {data.mean():.2f})')

    # 3. Isolated Ratio
    ax3 = axes[2]
    data = df['IsolatedRatio'] * 100
    bins = np.linspace(0, 100, 51)
    ax3.hist(data, bins=bins, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax3.set_title(f'Isolated Entity Ratio % (Mean: {data.mean():.2f}%)')

    # 4. Avg Rel Distance
    ax4 = axes[3]
    data = df['AvgRelDistance']
    if len(data) > 0:
        bins = np.linspace(0, data.max() + 1, 51)
    else:
        bins = 10
    ax4.hist(data, bins=bins, color='seagreen', edgecolor='black', alpha=0.7)
    ax4.set_title(f'Avg Relationship Distance (Mean: {data.mean():.2f})')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def rename_and_generate_statistics(folder_paths):
    print(f"\n🚀 (Step 4.5) 正在重命名文件并生成统计信息...")
    
    rename_map = {
        "原始文章.txt": "1原始文章.txt",
        "原始文章对应知识图谱.json": "2原始文章对应知识图谱.json",
        "根据知识图谱修改后的文章.txt": "3根据知识图谱修改后的文章.txt",
        "文章关系图-数据.json": "4文章关系图-数据.json",
        "文章关系图.jpg": "5文章关系图.jpg",
        "知识图谱-修改后文章的实体出现信息.json": "6修改后文章-知识图谱-增强信息.json"
    }
    
    # 1. Rename & Generate Simplified JSON
    for folder in folder_paths:
        # Rename
        for old, new in rename_map.items():
            old_p = os.path.join(folder, old)
            new_p = os.path.join(folder, new)
            if os.path.exists(old_p):
                try:
                    os.rename(old_p, new_p)
                except OSError as e:
                    print(f"Error renaming {old} to {new} in {folder}: {e}")
        
        # Generate Simplified JSON (7)
        enhanced_json_path = os.path.join(folder, "6修改后文章-知识图谱-增强信息.json")
        simplified_json_path = os.path.join(folder, "7修改后文章-知识图谱-简化版.json")
        
        if os.path.exists(enhanced_json_path):
            try:
                with open(enhanced_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Remove keys
                if "entity" in data and isinstance(data["entity"], list):
                    for ent in data["entity"]:
                        if isinstance(ent, dict):
                            ent.pop("appearance_count", None)
                            ent.pop("appearance_positions", None)
                
                if "relationship" in data and isinstance(data["relationship"], list):
                    for rel in data["relationship"]:
                        if isinstance(rel, dict):
                            rel.pop("distance_sub_obj", None)
                            rel.pop("raw_sub_name", None)
                            rel.pop("raw_obj_name", None)
                            rel.pop("raw_text_start", None)
                            rel.pop("raw_text_end", None)
                
                with open(simplified_json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error generating simplified JSON in {folder}: {e}")

    # 2. Statistics
    # New filenames to look for
    target_txt = "3根据知识图谱修改后的文章.txt"
    target_json = "4文章关系图-数据.json"
    
    all_stats = []
    for folder in folder_paths:
        stats = analyze_folder_stats(folder, target_txt, target_json)
        all_stats.append(stats)
    
    if not all_stats:
        print("⚠️ No stats collected.")
        return

    df = pd.DataFrame(all_stats)
    
    # Save CSV
    stats_dir = os.path.join(os.path.dirname(KG_BASE_DIR), "临时知识图谱处理结果_统计结果")
    os.makedirs(stats_dir, exist_ok=True)

    csv_path = os.path.join(stats_dir, "数据分布.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 统计数据已保存至: {csv_path}")
    
    # Plot
    plot_path = os.path.join(stats_dir, "数据分布统计图.png")
    plot_distributions_integrated(df, plot_path)
    print(f"✅ 统计图表已保存至: {plot_path}")


# =============================================================================

# =============================================================================









# =============================================================================

INPUT_SOURCES = [
    build_single_input_source(
        path=os.path.join(TRAIN_DATA_ROOT, "data", "representative_articles", "representative_articles_20.parquet"),
        input_type="parquet",
        content_col="content",
        id_col="original_id",
        id_prefix="",
        name="representative_20_articles",
    )
]

OUTPUT_DIR = os.path.join(TRAIN_DATA_ROOT, "data", "generated_pipeline", "parquet_outputs")
OUTPUT_NAME = "representative20_runtime"
TARGET_COUNT = 1000 
MAX_TOKENS = 2100
SEED = 9999


SPLIT_RATIO = 0.8







PIPELINE_CHUNK_SIZE = 1
PIPELINE_MAX_IN_FLIGHT = 2048
PIPELINE_FILL_RATE_PER_SEC = 20.0
PIPELINE_STAGE1_SOFT_LIMIT = 0
PIPELINE_STAGE1_HARD_LIMIT = 0
PIPELINE_PRIORITY_DOWNSTREAM_STAGES = True
ONLY_CHECK_CACHE = False
GENERATED_DATASET_ONLY_CHECK_CACHE = False
GENERATED_DATASET_FAST_FINALIZE_FROM_DISK = False
REAL_TESTSET_ONLY_CHECK_CACHE = False
REAL_TESTSET_PRIORITY_FIRST = False
COMPAT_OLD_YAML_PATH = []
COMPAT_RUN_OLD_YAML_FIRST = False
COMPAT_USE_OLD_REAL_TEST_DF_FOR_CURRENT_OUTPUT = False
BASE_VERY_SIMPLE_PROMPT_FUNC = ""
PRIORITY_REFERENCE_PARQUET_PATHS = []
PRIORITY_UNLOCK_LOW_PRIORITY_THRESHOLD = 1024
KG_BASE_DIR = os.path.join(TRAIN_DATA_ROOT, "data", "generated_pipeline", "kg_base_dir")

# =============================================================================

# =============================================================================

STAGE1_CONFIG = {
    "model": "gpt-5.4-mini",
    "token": 64 * 1024,
    "temp": 0.7,
    "flex": True,
    "prompt_func": "grid_kg_single_prompt_maker_tracerawtext_20260303",  
    "retry": True,
    "max_retry_attempts": 8,
    "max_total_seconds": 20 * 60,
    "stream_stall_seconds": 8 * 60,
}


STAGE2_CONFIG = {
    "model": "gpt-5.4-mini",
    "token": 64 * 1024,
    "temp": 0.7,
    "flex": True,
    "prompt_func": "grid_kg_reverse_prompt_maker_20260303",  
    "retry": True,
    "max_retry_attempts": 8,
    "max_total_seconds": 25 * 60,
    "stream_stall_seconds": 10 * 60,
}


STAGE5A_CONFIG = {
    "model": "gpt-5.4-mini",
    "token": 64 * 1024,
    "temp": 0.7,
    "flex": True,
    "prompt_func": "prompt_maker_precision_only",  # resolved to the authoritative tools_prompt function
    "retry": True,
    "max_retry_attempts": 8,
    "max_total_seconds": 20 * 60,
    "stream_stall_seconds": 8 * 60,
}


STAGE5B_CONFIG = {
    "model": "gpt-5.4-mini",
    "token": 64 * 1024,
    "temp": 0.7,
    "flex": True,
    "prompt_func": "grid_kg_sft_reasoning_reconstruction_prompt_20260303",  
    "request_batch_size": 96,
    "retry": True,
    "max_retry_attempts": 10,
    "max_total_seconds": 30 * 60,
    "stream_stall_seconds": 12 * 60,
}






REAL_TESTSET_CONFIG = {
    "dataset_path": os.path.join(TRAIN_DATA_ROOT, "data", "real_testset", "benchmark_full249.json"),
    "sources": ["all"],
    "sample_size_per_source": 50,
    "seed": 42,
    "mode": "SFT",
}

REAL_TEST_STAGE5B_CONFIG = {
    "model": "gpt-5.4-mini",
    "token": 64 * 1024,
    "temp": 0.7,
    "flex": True,
    "prompt_func": "grid_kg_sft_reasoning_reconstruction_prompt_20260303",
    "request_batch_size": 96,
    "retry": True,
    "max_retry_attempts": 10,
    "max_total_seconds": 30 * 60,
    "stream_stall_seconds": 12 * 60,
}







VERL_ROUTING_CONFIG = {
    "train_reward_model": "gpt-5.4-mini",
    "train_reward_flex": True,
    "test_reward_model": "gpt-5.4-mini",
    "test_reward_flex": True,
}

EXPECTED_STAGE_POLICY = {
    "STAGE1_CONFIG": {
        "model": "gpt-5.4-mini",
        "prompt_func": "grid_kg_single_prompt_maker_tracerawtext_20260303",
    },
    "STAGE2_CONFIG": {
        "model": "gpt-5.4-mini",
        "prompt_func": "grid_kg_reverse_prompt_maker_20260303",
    },
    "STAGE5A_CONFIG": {
        "model": "gpt-5.4-mini",
        "prompt_func": "prompt_maker_precision_only",
    },
    "STAGE5B_CONFIG": {
        "model": "gpt-5.4-mini",
        "prompt_func": "grid_kg_sft_reasoning_reconstruction_prompt_20260303",
    },
}

RUNTIME_STATE_KEYS = [
    "INPUT_SOURCES",
    "OUTPUT_DIR",
    "OUTPUT_NAME",
    "TARGET_COUNT",
    "MAX_TOKENS",
    "SEED",
    "SPLIT_RATIO",
    "PIPELINE_CHUNK_SIZE",
    "PIPELINE_MAX_IN_FLIGHT",
    "PIPELINE_FILL_RATE_PER_SEC",
    "ONLY_CHECK_CACHE",
    "GENERATED_DATASET_ONLY_CHECK_CACHE",
    "GENERATED_DATASET_FAST_FINALIZE_FROM_DISK",
    "REAL_TESTSET_ONLY_CHECK_CACHE",
    "COMPAT_OLD_YAML_PATH",
    "COMPAT_RUN_OLD_YAML_FIRST",
    "COMPAT_USE_OLD_REAL_TEST_DF_FOR_CURRENT_OUTPUT",
    "BASE_VERY_SIMPLE_PROMPT_FUNC",
    "PRIORITY_REFERENCE_PARQUET_PATHS",
    "PRIORITY_UNLOCK_LOW_PRIORITY_THRESHOLD",
    "KG_BASE_DIR",
    "STAGE1_CONFIG",
    "STAGE2_CONFIG",
    "STAGE5A_CONFIG",
    "STAGE5B_CONFIG",
    "REAL_TESTSET_CONFIG",
    "REAL_TEST_STAGE5B_CONFIG",
    "VERL_ROUTING_CONFIG",
]


def snapshot_runtime_state():
    return {
        key: copy.deepcopy(globals()[key])
        for key in RUNTIME_STATE_KEYS
    }


def restore_runtime_state(state):
    for key, value in state.items():
        globals()[key] = copy.deepcopy(value)


DEFAULT_RUNTIME_STATE = snapshot_runtime_state()


def print_stage_policy_warnings():
    current_configs = {
        "STAGE1_CONFIG": STAGE1_CONFIG,
        "STAGE2_CONFIG": STAGE2_CONFIG,
        "STAGE5A_CONFIG": STAGE5A_CONFIG,
        "STAGE5B_CONFIG": STAGE5B_CONFIG,
    }
    for stage_name, expected in EXPECTED_STAGE_POLICY.items():
        current = current_configs[stage_name]
        mismatches = []
        for key in ["model", "prompt_func"]:
            if current.get(key) != expected.get(key):
                mismatches.append(f"{key}={current.get(key)} (expected {expected.get(key)})")
        if mismatches:
            print(f"⚠️ {stage_name} 偏离当前默认实验设置: " + "; ".join(mismatches))


def build_stage_cloud_kwargs(stage_config, max_workers_vllm=None, only_check_cache_override=None):
    model_name = str(stage_config.get("model", "") or "")
    ask_kwargs = {
        "flex_for_openaiOmodel": bool(stage_config.get("flex", True)),
        "streamprint": False,
        "check_history_cache": True,
        "only_check_cache": ONLY_CHECK_CACHE if only_check_cache_override is None else bool(only_check_cache_override),
        "VllmSmartMode": True,
        "retry": stage_config.get("retry", True),
        "force_api_do_huge_input_Cloud": True,
    }
    if max_workers_vllm is not None:
        ask_kwargs["max_workers_Vllm"] = max_workers_vllm

    
    
    
    
    
    has_max_retry_attempts = "max_retry_attempts" in stage_config
    has_max_total_seconds = "max_total_seconds" in stage_config
    has_stream_stall_seconds = "stream_stall_seconds" in stage_config
    max_retry_attempts = stage_config.get("max_retry_attempts")
    max_total_seconds = stage_config.get("max_total_seconds")
    stream_stall_seconds = stage_config.get("stream_stall_seconds")

    if has_max_retry_attempts:
        ask_kwargs["retry"] = max_retry_attempts

    return ask_kwargs


def get_stage_request_batch_size(stage_config, default_batch_size=96):
    try:
        batch_size = int(stage_config.get("request_batch_size", default_batch_size))
    except Exception:
        batch_size = default_batch_size
    if batch_size <= 0:
        return None
    return max(1, batch_size)


def ask_group_link_in_stage_batches(prompt_list, stage_config, stage_name, max_workers_vllm=None, only_check_cache_override=None):
    if not prompt_list:
        return []

    batch_size = get_stage_request_batch_size(stage_config)
    if batch_size is None or batch_size >= len(prompt_list):
        print(
            f"    🚀 [{stage_name}] 整批 asks 模式: "
            f"0:{len(prompt_list)} (共 {len(prompt_list)} 条)"
        )
        return tools.ask_group_link(
            prompt_list=prompt_list,
            model=stage_config["model"],
            token=stage_config["token"],
            temp=stage_config["temp"],
            note=f"{stage_name}-allinone",
            **build_stage_cloud_kwargs(
                stage_config,
                max_workers_vllm=max_workers_vllm,
                only_check_cache_override=only_check_cache_override,
            ),
        )

    total = len(prompt_list)
    total_batches = (total + batch_size - 1) // batch_size
    all_responses = []

    for batch_idx, (start, end, prompt_chunk) in enumerate(iter_chunks(prompt_list, batch_size), start=1):
        print(
            f"    📦 [{stage_name}] 批次 {batch_idx}/{total_batches}: "
            f"{start}:{end} (共 {len(prompt_chunk)} 条)"
        )
        chunk_responses = tools.ask_group_link(
            prompt_list=prompt_chunk,
            model=stage_config["model"],
            token=stage_config["token"],
            temp=stage_config["temp"],
            note=f"{stage_name}-batch{batch_idx}/{total_batches}",
            **build_stage_cloud_kwargs(
                stage_config,
                max_workers_vllm=max_workers_vllm,
                only_check_cache_override=only_check_cache_override,
            ),
        )
        all_responses.extend(chunk_responses)

    return all_responses


# =============================================================================
# --- Part 1: Load and Sample Data ---
# =============================================================================

def load_and_sample_articles(exclude_content_identities=None, return_article_lookup=False):
    print("--- Part 1: Loading and Sampling Articles ---")
    
    all_data = []
    exclude_content_identities = set(exclude_content_identities or [])
    article_lookup = {}
    priority_stable_ids = load_priority_stable_article_ids_from_parquets(PRIORITY_REFERENCE_PARQUET_PATHS)
    
    for idx, source in enumerate(INPUT_SOURCES):
        source_name = source.get('name', f'source_{idx}')
        source_path = source['path']
        source_type = normalize_source_type(source.get('type', ''), source_path)
        content_col = source['content_col']
        id_col = source['id_col']
        id_prefix = source.get('id_prefix', '')
        filter_col = source.get('filter_col', None)
        filter_value = source.get('filter_value', None)
        
        print(f"\n📂 正在加载数据源 [{idx+1}/{len(INPUT_SOURCES)}]: {source_name}")
        print(f"   路径: {source_path}")
        
        
        try:
            if source_type == 'xlsx':
                df = pd.read_excel(source_path, engine="openpyxl")
            elif source_type == 'csv':
                df = pd.read_csv(source_path, encoding="utf-8")
            elif source_type == 'parquet':
                df = pd.read_parquet(source_path)
            else:
                print(f"   ⚠️ 不支持的文件类型: {source_type}，跳过")
                continue
        except Exception as e:
            print(f"   ❌ 读取文件失败: {e}")
            continue
        
        print(f"   📊 原始记录数: {len(df)}")
        
        
        if filter_col and filter_value:
            if filter_col in df.columns:
                df = df[df[filter_col] == filter_value].copy()
                print(f"   🔍 筛选 {filter_col}='{filter_value}' 后: {len(df)} 条")
            else:
                print(f"   ⚠️ 筛选列 '{filter_col}' 不存在，跳过筛选")
        
        
        if content_col not in df.columns:
            print(f"   ❌ 内容列 '{content_col}' 不存在，跳过此数据源")
            continue
        if id_col not in df.columns:
            print(f"   ⚠️ ID列 '{id_col}' 不存在，使用索引作为ID")
            df[id_col] = df.index.astype(str)
        
        
        source_file_basename = os.path.basename(source_path)
        valid_count = 0
        for row_idx, row in df.iterrows():
            content = row[content_col]
            if pd.isna(content) or not str(content).strip():
                continue

            raw_id = row[id_col]
            if pd.isna(raw_id) or str(raw_id).strip() == "":
                raw_id = f"row_{row_idx}"
            original_id = f"{id_prefix}{raw_id}"
            stable_article_id = make_stable_article_id(source_path, original_id, row_idx, content)
            content_identity = make_content_identity(content)
            if content_identity not in article_lookup:
                article_lookup[content_identity] = {
                    "stable_article_id": stable_article_id,
                    "original_id": original_id,
                    "source_file": source_file_basename,
                    "source_name": source_name,
                }
            all_data.append({
                "content": content,  
                "source": source_file_basename,
                "original_id": original_id,
                "source_name": source_name,
                "stable_article_id": stable_article_id,
                "content_identity": content_identity,
                "sample_rank": get_sample_rank(stable_article_id, SEED),
                "priority_tier": ("high" if stable_article_id in priority_stable_ids else "low"),
            })
            valid_count += 1
        
        print(f"   ✅ 提取有效记录: {valid_count} 条")
    
    print(f"\n📊 合并后总样本数: {len(all_data)}")
    
    if len(all_data) == 0:
        print("❌ 没有加载到任何数据!")
        sys.exit(1)

    if exclude_content_identities:
        before_exclude = len(all_data)
        all_data = [
            item for item in all_data
            if item.get("content_identity") not in exclude_content_identities
        ]
        print(
            f"🧩 旧已完成池排除完成: excluded={before_exclude - len(all_data)}, "
            f"remaining_candidates={len(all_data)}"
        )
    
    print(f"⚙️ 开始筛选文章，token长度 <= {MAX_TOKENS}，目标数量: {TARGET_COUNT}...")
    eligible_items = []
    for item in all_data:
        text = item['content']
        if tools.tokenlen(str(text)) <= MAX_TOKENS:
            eligible_items.append(dict(item))

    
    
    
    sorted_data = sorted(eligible_items, key=lambda x: (x["sample_rank"], x["stable_article_id"]))
    if priority_stable_ids:
        high_priority_count = sum(1 for item in sorted_data if item.get("priority_tier") == "high")
        low_priority_count = len(sorted_data) - high_priority_count
        unlock_threshold = max(0, int(PRIORITY_UNLOCK_LOW_PRIORITY_THRESHOLD))
        print(
            f"🎯 [Priority Tier] token 过滤后 high={high_priority_count}, "
            f"low={low_priority_count}, dynamic_unlock_threshold={unlock_threshold}"
        )
        print(
            "🚦 [Priority Tier] 运行时动态解锁模式启用："
            f"当剩余未完成 high <= {unlock_threshold} 时，worker 阶段开始放行 low。"
        )

    sampled_data = []
    for item in sorted_data:
        sampled_item = dict(item)
        sampled_item["sample_order"] = len(sampled_data)
        sampled_data.append(sampled_item)
        if TARGET_COUNT > 0 and len(sampled_data) == TARGET_COUNT:
            break

    if len(sampled_data) < TARGET_COUNT:
        print(f"⚠️ 警告：只找到了 {len(sampled_data)} 篇符合条件的文章。")
        if len(sampled_data) == 0: sys.exit(1)
    else:
        print(f"✅ 成功筛选出 {len(sampled_data)} 篇长度符合要求的文章。")
    
    
    source_counts = {}
    for item in sampled_data:
        src = item.get('source_name', item['source'])
        source_counts[src] = source_counts.get(src, 0) + 1
    print("📈 采样来源分布:")
    for src, cnt in sorted(source_counts.items()):
        print(f"   - {src}: {cnt} 篇 ({100*cnt/len(sampled_data):.1f}%)")
    if sampled_data:
        train_cnt = sum(
            1 for item in sampled_data
            if is_train_split(
                stable_article_id=item.get("stable_article_id"),
                split_ratio=SPLIT_RATIO,
            )
        )
        test_cnt = len(sampled_data) - train_cnt
        print(f"🧪 稳定切分预览: train={train_cnt}, test={test_cnt}, split_ratio={SPLIT_RATIO}")
    
    if return_article_lookup:
        return sampled_data, article_lookup
    return sampled_data


# =============================================================================
# --- Part 2: Split Sub-Functions ---
# =============================================================================

def step_1_generate_kg(sampled_data, article_start_index=0, article_indices=None):
    if article_indices is None:
        article_indices = [article_start_index + idx for idx in range(len(sampled_data))]
    if article_indices:
        article_start_index = min(article_indices)
        article_end_index = max(article_indices)
    else:
        article_end_index = article_start_index
    print(f"🚀 (Step 1) 开始为 {len(sampled_data)} 篇文章【生成知识图谱】... [{article_start_index}:{article_end_index}]")
    print(f"   🤖 模型: {STAGE1_CONFIG['model']}, Token: {STAGE1_CONFIG['token']}, Temp: {STAGE1_CONFIG['temp']}")
    print(f"   📝 Prompt: {STAGE1_CONFIG['prompt_func']}")
    
    # Extract Texts
    texts_only = [item['content'] for item in sampled_data]
    
    
    prompt_func_name = STAGE1_CONFIG['prompt_func']
    try:
        prompt_func = resolve_tools_prompt_callable(prompt_func_name)
    except AttributeError:
        print(f"❌ Prompt 函数 '{prompt_func_name}' 在 tools_prompt 中不存在!")
        sys.exit(1)
    
    with ThreadPoolExecutor(max_workers=min(32, len(sampled_data))) as executor:
        kg_prompts = list(executor.map(prompt_func, texts_only))
    
    all_graphs = tools.ask_group_link(
        prompt_list=kg_prompts,
        model=STAGE1_CONFIG['model'],
        token=STAGE1_CONFIG['token'],
        temp=STAGE1_CONFIG['temp'],
        **build_stage_cloud_kwargs(STAGE1_CONFIG, max_workers_vllm=[16, 0, 32]),
    )
    all_graphs = tools.cleanthinkans(all_graphs)
    print(f"✅ 知识图谱提取完成: {len(all_graphs)} 个。")
    
    os.makedirs(KG_BASE_DIR, exist_ok=True)
    folder_paths = []

    
    for idx, (graph, item) in enumerate(zip(all_graphs, sampled_data)):
        original_text = item['content']
        article_dir = os.path.join(KG_BASE_DIR, f"文章顺序_{int(article_indices[idx]):06d}")
        os.makedirs(article_dir, exist_ok=True)
        folder_paths.append(article_dir)
        if ONLY_CHECK_CACHE and not is_valid_cached_result(graph):
            continue
        
        # Parse Graph
        try:
            graph_str = str(graph)
            e_start = graph_str.find("#Entity_List_Start#") + len("#Entity_List_Start#")
            e_end = graph_str.find("#Entity_List_End#")
            r_start = graph_str.find("#Relationship_List_Start#") + len("#Relationship_List_Start#")
            r_end = graph_str.find("#Relationship_List_End#")
            e_list = json_repair.loads(graph_str[e_start:e_end]) if e_start != -1 else []
            r_list = json_repair.loads(graph_str[r_start:r_end]) if r_start != -1 else []
        except:
            e_list, r_list = [], []
        
        with open(os.path.join(article_dir, "原始文章.txt"), 'w', encoding='utf-8') as f:
            f.write(str(original_text))
        if len(e_list) == 0 and len(r_list) == 0:
            print(f"⚠️ [Step1 Batch] article_idx={idx} 解析到空图谱，跳过图谱落盘。")
            continue
        with open(os.path.join(article_dir, "原始文章对应知识图谱.json"), 'w', encoding='utf-8') as f:
            json.dump({"entity": e_list, "relationship": r_list}, f, ensure_ascii=False, indent=2)

    return all_graphs, folder_paths

def step_2_revise_text(sampled_data, all_graphs, folder_paths):
    print("🚀 (Step 2) 开始【修正原文】...")
    print(f"   🤖 模型: {STAGE2_CONFIG['model']}, Token: {STAGE2_CONFIG['token']}, Temp: {STAGE2_CONFIG['temp']}")
    print(f"   📝 Prompt: {STAGE2_CONFIG['prompt_func']}")
    
    # Extract Texts
    texts_only = [item['content'] for item in sampled_data]
    
    
    prompt_func_name = STAGE2_CONFIG['prompt_func']
    try:
        prompt_func = resolve_tools_prompt_callable(prompt_func_name)
    except AttributeError:
        print(f"❌ Prompt 函数 '{prompt_func_name}' 在 tools_prompt 中不存在!")
        sys.exit(1)
    
    with ThreadPoolExecutor(max_workers=min(32, len(sampled_data))) as executor:
        rev_prompts = list(executor.map(lambda x: prompt_func(*x), zip(texts_only, all_graphs)))

    all_revised_texts = tools.ask_group_link(
        prompt_list=rev_prompts,
        model=STAGE2_CONFIG['model'],
        token=STAGE2_CONFIG['token'],
        temp=STAGE2_CONFIG['temp'],
        **build_stage_cloud_kwargs(STAGE2_CONFIG, max_workers_vllm=['local', 'ultra']),
    )
    all_revised_texts = tools.cleanthinkans(all_revised_texts)
    print(f"✅ 原文修正完成: {len(all_revised_texts)} 篇。")

    
    for idx, (revised_text, article_dir) in enumerate(zip(all_revised_texts, folder_paths)):
        if ONLY_CHECK_CACHE and not is_valid_cached_result(revised_text):
            continue
        if is_invalid_text_payload(revised_text):
            print(f"⚠️ [Step2 Batch] article_idx={idx} 返回无效正文，跳过落盘。")
            remove_file_if_exists(os.path.join(article_dir, "根据知识图谱修改后的文章.txt"))
            remove_file_if_exists(os.path.join(article_dir, "知识图谱-修改后文章的实体出现信息.json"))
            continue
        with open(os.path.join(article_dir, "根据知识图谱修改后的文章.txt"), 'w', encoding='utf-8') as f:
            f.write(normalize_text_payload(revised_text))
            
    return all_revised_texts

def step_3_generate_final_json(all_revised_texts, all_graphs, folder_paths):
    """
    Step 3: Create Final JSON.
    - Parses KG.
    - Calculates Entity Stats (Appearance Count/Positions) via string matching.
    - Calculates Relationship Distance via raw_sub_name/raw_obj_name matching.
    """
    print("🚀 (Step 3) 生成最终 JSON (计算实体统计 & 关系距离)...")
    
    for idx, (revised_text, graph, article_dir) in enumerate(zip(all_revised_texts, all_graphs, folder_paths)):
        if is_invalid_text_payload(revised_text) or not is_valid_graph_payload(graph):
            continue
        # Parse again
        try:
            graph_str = str(graph)
            e_start = graph_str.find("#Entity_List_Start#") + len("#Entity_List_Start#")
            e_end = graph_str.find("#Entity_List_End#")
            r_start = graph_str.find("#Relationship_List_Start#") + len("#Relationship_List_Start#")
            r_end = graph_str.find("#Relationship_List_End#")
            e_list = json_repair.loads(graph_str[e_start:e_end]) if e_start != -1 else []
            r_list = json_repair.loads(graph_str[r_start:r_end]) if r_start != -1 else []
        except: 
            e_list, r_list = [], []

        # 1. Entity Stats Calculation (String Matching)
        entity_stats = analyze_entity_appearances(str(revised_text), e_list)
        for entity in e_list:
            if isinstance(entity, dict):
                entity_name = normalize_entity_text_value(entity.get("name", ""))
                if entity_name:
                    entity["name"] = entity_name
                stats = entity_stats.get(entity_name, {'appearance_count':0, 'appearance_positions':[]})
                entity.update(stats)

        # 2. Relationship Distance Calculation (String Matching on Raw Names)
        
        update_relationships_with_raw_name_distance(str(revised_text), r_list)

        # Save Final JSON
        with open(os.path.join(article_dir, "知识图谱-修改后文章的实体出现信息.json"), 'w', encoding='utf-8') as f:
            json.dump({"content": str(revised_text), "entity": e_list, "relationship": r_list}, f, ensure_ascii=False, indent=2)
            
    print("✅ 最终 JSON 生成完毕。")

def step_4_generate_plots(folder_paths):
    """
    Step 4: Generate Visual Graphs from the saved JSONs.
    """
    print(f"🚀 (Step 4) 开始为 {len(folder_paths)} 篇文章【生成关系图】...")
    generate_graph_plots(folder_paths)

def step_5_generate_qa(all_revised_texts, all_graphs, sampled_data):
    print("🚀 (Step 5) 开始【生成QA题目】...")
    print(f"   🤖 模型: {STAGE5A_CONFIG['model']}, Token: {STAGE5A_CONFIG['token']}, Temp: {STAGE5A_CONFIG['temp']}")
    print(f"   📝 Prompt: {STAGE5A_CONFIG['prompt_func']}")
    
    
    prompt_func_name = STAGE5A_CONFIG['prompt_func']
    try:
        prompt_func = resolve_tools_prompt_callable(prompt_func_name)
    except AttributeError:
        print(f"❌ Prompt 函数 '{prompt_func_name}' 在本地 prompt 模块中不存在!")
        sys.exit(1)
    
    with ThreadPoolExecutor(max_workers=min(32, len(all_revised_texts))) as executor:
        qa_prompts = list(executor.map(lambda x: prompt_func(*x), zip(all_revised_texts, all_graphs)))

    all_qa_results = tools.ask_group_link(
        prompt_list=qa_prompts,
        model=STAGE5A_CONFIG['model'],
        token=STAGE5A_CONFIG['token'],
        temp=STAGE5A_CONFIG['temp'],
        **build_stage_cloud_kwargs(STAGE5A_CONFIG, max_workers_vllm=['local', 'ultra']),
    )
    all_qa_results = tools.cleanthinkans(all_qa_results)
    print(f"✅ QA题目生成完成: {len(all_qa_results)} 组。")

    if ONLY_CHECK_CACHE:
        valid_mask = [is_valid_cached_result(res) for res in all_qa_results]
        all_revised_texts, all_graphs, sampled_data, all_qa_results = filter_aligned_records_by_mask(
            all_revised_texts, all_graphs, sampled_data, all_qa_results,
            valid_mask=valid_mask,
            stage_name="Step 5A Cache Finalize"
        )

    # Build DataFrame
    print("\n📊 正在构建DataFrame...")
    all_records = []
    
    def process_qa_json(args):
        idx, res, txt, graph, item = args
        rows = []
        try:
            data = json_repair.loads(str(res))
            if isinstance(data, list):
                for q in data:
                    if isinstance(q, dict):
                        rows.append({
                            "ArticleNUM": idx,
                            "Content": txt,
                            "Question": q.get("question", ""),
                            "Options": " | ".join([f"{k}: {v}" for k, v in q.get("options", {}).items()]),
                            "Answers": ", ".join(q.get("answer", [])),
                            "how_to_get_answer_step_by_step": q.get("how_to_get_answer_step_by_step", ""),
                            
                            # Original Metadata
                            "text_raw_from_file": item.get('content', ""),
                            "source_file": item.get('source', ""),
                            "original_id": item.get('original_id', ""),
                            "stable_article_id": item.get('stable_article_id', ""),
                            "sample_order": item.get('sample_order', idx),
                            "graph_from_text_raw_from_file": json.dumps(graph, ensure_ascii=False) if isinstance(graph, (dict, list)) else str(graph),
                            "text_fixed_by_revision": txt
                        })
        except Exception as e:
            print(f"Parse error Article {idx}: {e}")
        return rows

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(process_qa_json, (i, r, t, g, d)) for i, (r, t, g, d) in enumerate(zip(all_qa_results, all_revised_texts, all_graphs, sampled_data))]
        for f in as_completed(futures):
            all_records.extend(f.result())

    if not all_records:
        print("❌ 错误：未能解析出QA记录。")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_records)
    
    # Shuffle & SFT Format
    np.random.seed(SEED)
    unique_nums = df['ArticleNUM'].unique()
    shuffled_nums = np.random.permutation(unique_nums)
    df = pd.concat([df[df['ArticleNUM'] == num] for num in shuffled_nums], ignore_index=True)
    
    df['sft_answer'] = df.apply(lambda r: (f"<mythink>{r['how_to_get_answer_step_by_step']}</mythink>" if r.get("how_to_get_answer_step_by_step") else "") + f"#### {r.get('Answers', '')}", axis=1)
    
    return df

def step_5ba_generate_testqa_data(all_revised_texts, all_graphs, sampled_data):
    """
    Step 5-B.a: Generate Test QA Data (KG Extraction).
    Input: all_revised_texts (Memory)
    Prompt: tools_prompt.grid_kg_single_prompt_maker_very_simple_20260303(text)
    Ground Truth: all_graphs (Memory) -> formatted as {"entity":..., "relationship":...}
    """
    print("🚀 (Step 5-B.a) 开始【生成 Test QA 数据 (KG Extraction)】...")
    
    all_records = []
    
    for text, graph, item in zip(all_revised_texts, all_graphs, sampled_data):
        if not text: continue
        
        # Handle graph (can be raw string or dict)
        if isinstance(graph, str):
             try:
                graph_str = graph
                e_start = graph_str.find("#Entity_List_Start#") + len("#Entity_List_Start#")
                e_end = graph_str.find("#Entity_List_End#")
                r_start = graph_str.find("#Relationship_List_Start#") + len("#Relationship_List_Start#")
                r_end = graph_str.find("#Relationship_List_End#")
                e_list = json_repair.loads(graph_str[e_start:e_end]) if e_start != -1 else []
                r_list = json_repair.loads(graph_str[r_start:r_end]) if r_start != -1 else []
                graph_data = {"entity": e_list, "relationship": r_list}
             except:
                graph_data = {"entity": [], "relationship": []}
        else:
             graph_data = graph

        # Clean graph_data for ground_truth
        try:
            graph_data = json.loads(json.dumps(graph_data))
        except:
            pass

        if "entity" in graph_data and isinstance(graph_data["entity"], list):
            for ent in graph_data["entity"]:
                if isinstance(ent, dict):
                    ent.pop("appearance_count", None)
                    ent.pop("appearance_positions", None)
        
        if "relationship" in graph_data and isinstance(graph_data["relationship"], list):
            for rel in graph_data["relationship"]:
                if isinstance(rel, dict):
                    rel.pop("distance_sub_obj", None)
                    rel.pop("raw_sub_name", None)
                    rel.pop("raw_obj_name", None)
                    rel.pop("raw_text_start", None)
                    rel.pop("raw_text_end", None)
        
        # Generate Prompt
        prompt_msgs = resolve_current_base_very_simple_prompt_callable()(text)
        
        # Ground Truth
        ground_truth = text + "###我是分割线###" + json.dumps(graph_data, ensure_ascii=False)
        
        all_records.append({
            "prompt": prompt_msgs,
            "ground_truth": ground_truth,
            "text_raw_from_file": item.get('content', ""),
            "source_file": item.get('source', ""),
            "original_id": item.get('original_id', ""),
            "stable_article_id": item.get('stable_article_id', ""),
            "sample_order": item.get('sample_order', len(all_records)),
            "graph_from_text_raw_from_file": json.dumps(graph, ensure_ascii=False) if isinstance(graph, (dict, list)) else str(graph),
            "text_fixed_by_revision": text
        })
        
    if not all_records:
        print("❌ 错误：未能生成 Test QA 数据。")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_records)
    
    # Add index and totalnum
    df['index'] = range(len(df))
    df['totalnum'] = len(df)
    
    print(f"✅ Test QA 数据生成完成: {len(df)} 条。")
    return df

def step_5bb_generate_sft_reasoning_data(all_revised_texts, all_graphs, sampled_data, skip_reasoning=False):
    print("🚀 (Step 5-B) 开始【生成知识图谱 SFT 数据】...")
    if skip_reasoning:
        print("   ⏭️ (Skip Reasoning) 已跳过 LLM 推理链生成，使用占位符。")
    else:
        print(f"   🤖 模型: {STAGE5B_CONFIG['model']}, Token: {STAGE5B_CONFIG['token']}, Temp: {STAGE5B_CONFIG['temp']}")
        print(f"   📝 Prompt: {STAGE5B_CONFIG['prompt_func']}")
    
    
    prompt_func_name = STAGE5B_CONFIG['prompt_func']
    try:
        prompt_func = resolve_tools_prompt_callable(prompt_func_name)
    except AttributeError:
        print(f"❌ Prompt 函数 '{prompt_func_name}' 在 tools_prompt 中不存在!")
        sys.exit(1)
    
    # 1. Construct Prompts
    sft_prompts = []
    valid_indices = []
    
    for i, (text, graph) in enumerate(zip(all_revised_texts, all_graphs)):
        if is_invalid_text_payload(text):
            continue
        if not is_valid_graph_payload(graph):
            continue
        
        # Normalize Graph
        if isinstance(graph, str):
             try:
                graph_str = str(graph)
                e_start = graph_str.find("#Entity_List_Start#") + len("#Entity_List_Start#")
                e_end = graph_str.find("#Entity_List_End#")
                r_start = graph_str.find("#Relationship_List_Start#") + len("#Relationship_List_Start#")
                r_end = graph_str.find("#Relationship_List_End#")
                e_list = json_repair.loads(graph_str[e_start:e_end]) if e_start != -1 else []
                r_list = json_repair.loads(graph_str[r_start:r_end]) if r_start != -1 else []
                graph_data = {"entity": e_list, "relationship": r_list}
             except:
                graph_data = {"entity": [], "relationship": []}
        else:
             graph_data = graph

        # Clean Graph
        try:
            graph_data = json.loads(json.dumps(graph_data))
        except:
            pass
        
        if "entity" in graph_data and isinstance(graph_data["entity"], list):
            for ent in graph_data["entity"]:
                if isinstance(ent, dict):
                    ent.pop("appearance_count", None)
                    ent.pop("appearance_positions", None)
        if "relationship" in graph_data and isinstance(graph_data["relationship"], list):
            for rel in graph_data["relationship"]:
                if isinstance(rel, dict):
                    rel.pop("distance_sub_obj", None)
                    rel.pop("raw_sub_name", None)
                    rel.pop("raw_obj_name", None)
                    rel.pop("raw_text_start", None)
                    rel.pop("raw_text_end", None)

        # Get Base Prompt Content
        base_prompt_msgs = resolve_current_base_very_simple_prompt_callable()(text)
        base_prompt_content = base_prompt_msgs[0]['content'] if base_prompt_msgs else ""
        
        # Construct Generation Prompt using configured function
        gen_prompt = prompt_func(base_prompt_content, graph_data)
        sft_prompts.append(gen_prompt)
        valid_indices.append(i)

    # 2. Call LLM or Use Placeholder
    all_responses = []
    if skip_reasoning:
        all_responses = ["%这是SFT推理链的填充符因为打开了-skip-reason-trace-re-generate%"] * len(sft_prompts)
    else:
        print(f"    正在调用 LLM 生成推理过程 (共 {len(sft_prompts)} 条)...")
        all_responses = ask_group_link_in_stage_batches(
            prompt_list=sft_prompts,
            stage_config=STAGE5B_CONFIG,
            stage_name="Step5B-SFT",
            max_workers_vllm=['local', 'ultra'],
        )
        all_responses = tools.cleanthinkans(all_responses)

    # 3. Build Records
    all_records = []
    filtered_pairs = [
        (idx, response)
        for idx, response in zip(valid_indices, all_responses)
        if is_valid_cached_result(response)
    ]
    dropped_reasoning = len(valid_indices) - len(filtered_pairs)
    if ONLY_CHECK_CACHE:
        print(f"🧹 [Step 5B Cache Finalize] 仅保留缓存完整推理链: kept={len(filtered_pairs)}, dropped={dropped_reasoning}")
    elif dropped_reasoning > 0:
        print(f"🧹 [Step 5B] 跳过空/失败推理链响应: kept={len(filtered_pairs)}, dropped={dropped_reasoning}")

    for idx, response in filtered_pairs:
        text = all_revised_texts[idx]
        graph = all_graphs[idx]
        item = sampled_data[idx]
        if is_invalid_text_payload(text):
            continue
        if not is_valid_graph_payload(graph):
            continue
        # The prompt column should be the ORIGINAL extraction prompt
        original_prompt_msgs = resolve_current_base_very_simple_prompt_callable()(text)
        
        # Reconstruct the full SFT ground truth: Reasoning + Entity List + Relationship List
        # response should contain #Reasoning_Start#...#Reasoning_End#
        
        # Normalize graph (may be raw string or dict)
        if isinstance(graph, str):
            try:
                graph_str = str(graph)
                e_start = graph_str.find("#Entity_List_Start#") + len("#Entity_List_Start#")
                e_end = graph_str.find("#Entity_List_End#")
                r_start = graph_str.find("#Relationship_List_Start#") + len("#Relationship_List_Start#")
                r_end = graph_str.find("#Relationship_List_End#")
                entity_list = json_repair.loads(graph_str[e_start:e_end]) if e_start != -1 else []
                rel_list = json_repair.loads(graph_str[r_start:r_end]) if r_start != -1 else []
            except:
                entity_list, rel_list = [], []
        else:
            entity_list = graph.get("entity", [])
            rel_list = graph.get("relationship", [])

        clean_entities, clean_relationships = build_very_simple_ground_truth_lists(
            entity_list,
            rel_list,
        )
        entity_json_str = json.dumps(clean_entities, ensure_ascii=False)
        rel_json_str = json.dumps(clean_relationships, ensure_ascii=False)
        
        full_ground_truth = (
            f"{normalize_text_payload(response)}\n\n"
            f"#Entity_List_Start#\n{entity_json_str}\n#Entity_List_End#\n\n"
            f"#Relationship_List_Start#\n{rel_json_str}\n#Relationship_List_End#"
        )

        all_records.append({
            "prompt": original_prompt_msgs,
            "ground_truth": full_ground_truth,
            "text_raw_from_file": item.get('content', ""),
            "source_file": item.get('source', ""),
            "original_id": item.get('original_id', ""),
            "stable_article_id": item.get('stable_article_id', ""),
            "sample_order": item.get('sample_order', idx),
            "graph_from_text_raw_from_file": json.dumps(graph, ensure_ascii=False) if isinstance(graph, (dict, list)) else str(graph),
            "text_fixed_by_revision": text
        })

    if not all_records:
        print("❌ 错误：未能生成 SFT Reasoning 数据。")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    
    # Add index and totalnum
    df['index'] = range(len(df))
    df['totalnum'] = len(df)
    
    print(f"✅ SFT Reasoning 数据生成完成: {len(df)} 条。")
    return df

def load_real_testset(folder_path):
    """
    Load real test set from JSON files in the specified folder.
    Returns a DataFrame with 'prompt' and 'ground_truth' columns.
    For SFT, ground_truth will have a placeholder for reasoning.
    """
    print(f"🚀 (Real Testset) Loading from {folder_path}...")
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found: {folder_path}")
        return pd.DataFrame()
        
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    if not json_files:
        print(f"⚠️ No JSON files found in {folder_path}")
        return pd.DataFrame()
        
    all_records = []
    
    for json_file in json_files:
        path = os.path.join(folder_path, json_file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            content = data.get("正文", "")
            entities = data.get("实体列表(更新别名后)", [])
            relationships = data.get("实体关系", [])
            
            if not content: continue
            
            # Format Graph Data
            graph_data = {
                "entity": entities,
                "relationship": relationships
            }
            
            # Generate Prompt
            prompt_msgs = resolve_current_base_very_simple_prompt_callable()(content)
            
            # Generate Ground Truth (SFT format with placeholder)
            # Note: For RL, we might need just the JSON, but the current pipeline seems to use string GT for both?
            # Let's check step_6. It uses: text + separator + json_str
            # Let's check step_7. It uses: Reasoning + EntityList + RelList
            
            # Since this function is generic for "real testset", we need to decide the format.
            
            
            
            # So we should probably return a structure that can be adapted, OR return the SFT format directly?
            # The user request implies we replace the *Generated Test Set*.
            
            # If we are in -genkg_plussft mode (SFT), the format is `#Reasoning_Start#...`
            
            # Let's construct the SFT format here as it's the most complex one mentioned.
            # For RL, we can reconstruct it or just use the SFT format if the RL reward model handles it?
            # Wait, step_6 produces `text + separator + json`.
            # step_7 produces `#Reasoning_Start#...`.
            
            # I will create two columns in the returned DF: 'ground_truth_rl' and 'ground_truth_sft'
            # and let the caller decide which one to use? 
            # Or better, just return the raw data and let the caller format it?
            # But the caller `process_and_save_dataset_locally` expects `ground_truth` column.
            
            # Let's look at `process_and_save_dataset_locally`. It uses `row.ground_truth`.
            # So I should probably return a DF with `prompt` and `ground_truth`.
            # But the format depends on whether it's RL or SFT.
            
            
            # I will implement logic to return the SFT format with placeholder.
            
            
            # Let's construct both and store in different columns, then rename based on usage?
            # Or just pass a `mode` to this function?
            # I'll pass `mode` to this function.
            
            
            # This implies I should handle both formats.
            
            # Let's make this function return a list of dicts with raw data, and then format it later?
            # No, `process_and_save_dataset_locally` takes a DF.
            
            # I'll add a `mode` argument to `load_real_testset`.
            pass 
            
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    # I will implement the loop properly in the replacement string.


def load_real_testset(folder_path, mode="RL", skip_reasoning=False):
    print(f"🚀 (Real Testset) Loading from {folder_path} (mode={mode}, skip_reasoning={skip_reasoning})...")
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found: {folder_path}")
        return pd.DataFrame()
        
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    if not json_files:
        print(f"⚠️ No JSON files found in {folder_path}")
        return pd.DataFrame()
        
    all_records = []
    
    for json_file in json_files:
        path = os.path.join(folder_path, json_file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            content = data.get("正文", "")
            entities = data.get("实体列表(更新别名后)", [])
            relationships = data.get("实体关系", [])
            
            if not content: continue
            
            # Format Graph Data
            graph_data = {
                "entity": entities,
                "relationship": relationships
            }
            
            # Generate Prompt
            prompt_msgs = resolve_current_base_very_simple_prompt_callable()(content)
            
            all_records.append({
                "prompt": prompt_msgs,
                "content": content,
                "graph_data": graph_data,
                "entities": entities,
                "relationships": relationships
            })
            
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    if not all_records:
        print("❌ Error: No valid records found in real testset.")
        return pd.DataFrame()

    
    if mode == "RL":
        
        for rec in all_records:
            rec["ground_truth"] = rec["content"] + "###我是分割线###" + json.dumps(rec["graph_data"], ensure_ascii=False)
    else:
        
        if skip_reasoning:
             print("   ⏭️ (Real Testset SFT) Skip Reasoning: Using placeholders.")
             all_responses = ["%这是SFT推理链的填充符因为打开了-skip-reason-trace-re-generate%"] * len(all_records)
        else:
            print(f"📝 (Real Testset SFT) 正在为 {len(all_records)} 条数据生成推理链...")
            print(f"   🤖 模型: {STAGE5B_CONFIG['model']}, Prompt: {STAGE5B_CONFIG['prompt_func']}")
            
            
            prompt_func_name = STAGE5B_CONFIG['prompt_func']
            try:
                prompt_func = resolve_tools_prompt_callable(prompt_func_name)
            except AttributeError:
                print(f"❌ Prompt 函数 '{prompt_func_name}' 在 tools_prompt 中不存在!")
                sys.exit(1)
            
            
            sft_prompts = []
            for rec in all_records:
                base_prompt_content = rec["prompt"][0]['content'] if rec["prompt"] else ""
                gen_prompt = prompt_func(base_prompt_content, rec["graph_data"])
                sft_prompts.append(gen_prompt)
            
            
            all_responses = tools.ask_group_link(
                prompt_list=sft_prompts,
                model=STAGE5B_CONFIG['model'],
                token=STAGE5B_CONFIG['token'],
                temp=STAGE5B_CONFIG['temp'],
                **build_stage_cloud_kwargs(STAGE5B_CONFIG, max_workers_vllm=['local', 'ultra']),
            )
            all_responses = tools.cleanthinkans(all_responses)
        
        
        if ONLY_CHECK_CACHE:
            valid_pairs = [
                (rec, reasoning)
                for rec, reasoning in zip(all_records, all_responses)
                if is_valid_cached_result(reasoning)
            ]
            print(f"🧹 [Real Testset SFT Cache Finalize] kept={len(valid_pairs)}, dropped={len(all_records) - len(valid_pairs)}")
        else:
            valid_pairs = list(zip(all_records, all_responses))

        for i, (rec, reasoning) in enumerate(valid_pairs):
            clean_entities, clean_relationships = build_very_simple_ground_truth_lists(
                rec["entities"],
                rec["relationships"],
            )
            entity_json_str = json.dumps(clean_entities, ensure_ascii=False)
            rel_json_str = json.dumps(clean_relationships, ensure_ascii=False)
            
            rec["ground_truth"] = (
                f"{str(reasoning).strip()}\n\n"
                f"#Entity_List_Start#\n{entity_json_str}\n#Entity_List_End#\n\n"
                f"#Relationship_List_Start#\n{rel_json_str}\n#Relationship_List_End#"
            )
        
        if not skip_reasoning:
            print(f"✅ (Real Testset SFT) 推理链生成完成!")

    
    final_records = []
    for rec in all_records:
        if "ground_truth" not in rec:
            continue
        final_records.append({
            "prompt": rec["prompt"],
            "ground_truth": rec["ground_truth"]
        })

    df = pd.DataFrame(final_records)
    # Add index and totalnum
    df['index'] = range(len(df))
    df['totalnum'] = len(df)
    
    print(f"✅ Real Testset Loaded: {len(df)} rows.")
    return df

def normalize_source_selector_for_ultimate_dataset(sources):
    if sources is None:
        return ["all"]
    if isinstance(sources, str):
        return [sources]
    return list(sources)


def sample_ultimate_dataset_like_full_eval(dataset_path, sources=None, sample_size_per_source=None, seed=42):
    if not os.path.exists(dataset_path):
        print(f"❌ 究极测试集 JSON 不存在: {dataset_path}")
        return []

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    by_source = defaultdict(list)
    for item in all_data:
        src = item.get("source_approach_provided_dataset", "unknown")
        by_source[src].append(item)

    sources = normalize_source_selector_for_ultimate_dataset(sources)
    if sources and sources != ["all"]:
        by_source = {k: v for k, v in by_source.items() if k in sources}

    random.seed(int(seed))
    sampled = []
    for src, items in by_source.items():
        if sample_size_per_source is not None and int(sample_size_per_source) < len(items):
            chosen = random.sample(items, int(sample_size_per_source))
            print(f"   🎯 [究极测试集同采样] {src}: {len(chosen)}/{len(items)}")
        else:
            chosen = items
            print(f"   📚 [究极测试集同采样] {src}: 全部 {len(items)}")
        sampled.extend(chosen)

    print(f"✅ [究极测试集同采样] 最终样本数: {len(sampled)}")
    return sampled


def build_minimal_entity_list_from_names(names):
    entity_list = []
    seen = set()
    for name in names:
        text = str(name or "").strip()
        if text and text not in seen:
            seen.add(text)
            entity_list.append({
                "name": text,
                "type": "unknown",
                "alias": ["None"],
                "mother entity": ["None"],
            })
    return entity_list


def normalize_ultimate_graph_data(item):
    extra = item.get("extra_info", {}) or {}
    raw_relationships = item.get("ground_truth", []) or []

    relationship_list = []
    entity_name_candidates = []
    for rel in raw_relationships:
        if not isinstance(rel, dict):
            continue
        sub = str(rel.get("sub", "") or "").strip()
        pred = str(rel.get("rel", "") or "").strip()
        obj = str(rel.get("obj", "") or "").strip()
        if not (sub and pred and obj):
            continue
        normalized = {"sub": sub, "rel": pred, "obj": obj}
        if "rel_type" in rel:
            normalized["rel_type"] = rel["rel_type"]
        relationship_list.append(normalized)
        entity_name_candidates.extend([sub, obj])

    entities = extra.get("实体列表(更新别名后)")
    if isinstance(entities, list) and entities:
        entity_list = json.loads(json.dumps(entities, ensure_ascii=False))
    else:
        entities = extra.get("entities")
        if isinstance(entities, dict) and entities:
            entity_list = build_minimal_entity_list_from_names(list(entities.values()))
        elif isinstance(entities, list) and entities:
            normalized_entities = []
            for ent in entities:
                if isinstance(ent, dict):
                    normalized_entities.append(json.loads(json.dumps(ent, ensure_ascii=False)))
                else:
                    normalized_entities.extend(build_minimal_entity_list_from_names([ent]))
            entity_list = normalized_entities
        else:
            entity_list = build_minimal_entity_list_from_names(entity_name_candidates)

    return {"entity": entity_list, "relationship": relationship_list}


def load_real_testset_records_from_legacy_folder(folder_path):
    records = []
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found: {folder_path}")
        return records

    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    for json_file in json_files:
        path = os.path.join(folder_path, json_file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            content = data.get("正文", "")
            if not content:
                continue
            graph_data = {
                "entity": data.get("实体列表(更新别名后)", []),
                "relationship": data.get("实体关系", []),
            }
            prompt_msgs = resolve_current_base_very_simple_prompt_callable()(content)
            records.append({
                "prompt": prompt_msgs,
                "content": content,
                "graph_data": graph_data,
                "entities": graph_data["entity"],
                "relationships": graph_data["relationship"],
                "source_file": json_file,
                "original_id": data.get("idorurl", json_file),
                "stable_article_id": f"legacy_real::{json_file}",
                "sample_order": len(records),
                "text_raw_from_file": content,
                "graph_from_text_raw_from_file": json.dumps(graph_data, ensure_ascii=False),
                "text_fixed_by_revision": content,
            })
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    return records


def load_real_testset_records_from_ultimate_dataset(dataset_path, sources=None, sample_size_per_source=50, seed=42):
    records = []
    sampled_items = sample_ultimate_dataset_like_full_eval(
        dataset_path=dataset_path,
        sources=sources,
        sample_size_per_source=sample_size_per_source,
        seed=seed,
    )

    for idx, item in enumerate(sampled_items):
        content = str(item.get("content", "") or "")
        if not content:
            continue
        extra = item.get("extra_info", {}) or {}
        source_name = str(item.get("source_approach_provided_dataset", "unknown") or "unknown")
        file_name = str(extra.get("file_name", f"{source_name}_{idx}.json") or f"{source_name}_{idx}.json")
        original_id = str(extra.get("idorurl", file_name) or file_name)
        graph_data = normalize_ultimate_graph_data(item)
        prompt_msgs = resolve_current_base_very_simple_prompt_callable()(content)
        records.append({
            "prompt": prompt_msgs,
            "content": content,
            "graph_data": graph_data,
            "entities": graph_data["entity"],
            "relationships": graph_data["relationship"],
            "source_file": f"{source_name}:{file_name}",
            "original_id": original_id,
            "stable_article_id": f"ultimate_real::{source_name}::{file_name}",
            "sample_order": idx,
            "text_raw_from_file": content,
            "graph_from_text_raw_from_file": json.dumps(graph_data, ensure_ascii=False),
            "text_fixed_by_revision": content,
        })
    return records


def merge_real_test_records(existing_records, new_records, label=""):
    merged = []
    seen = set()
    dropped = 0

    for rec in (existing_records or []) + (new_records or []):
        stable_article_id = str(rec.get("stable_article_id", "") or "").strip()
        if stable_article_id:
            dedup_key = ("stable", stable_article_id)
        else:
            dedup_key = (
                "content",
                make_content_identity(rec.get("text_raw_from_file", rec.get("content", ""))),
            )
        if dedup_key in seen:
            dropped += 1
            continue
        seen.add(dedup_key)
        merged.append(rec)

    if label:
        print(
            f"🧩 [{label}] real-test records 合并去重: "
            f"kept={len(merged)}, dropped={dropped}"
        )
    return merged


def build_real_testset_dataframe(records, mode="RL", skip_reasoning=False, stage5b_config=None, only_check_cache=False):
    if not records:
        print("❌ Error: No valid records found in real testset.")
        return pd.DataFrame()

    if mode == "RL":
        for rec in records:
            rec["ground_truth"] = rec["content"] + "###我是分割线###" + json.dumps(rec["graph_data"], ensure_ascii=False)
    else:
        active_stage5b_config = stage5b_config or STAGE5B_CONFIG
        if skip_reasoning:
            print("   ⏭️ (Real Testset SFT) Skip Reasoning: Using placeholders.")
            all_responses = ["%这是SFT推理链的填充符因为打开了-skip-reason-trace-re-generate%"] * len(records)
        else:
            print(f"📝 (Real Testset SFT) 正在为 {len(records)} 条数据生成推理链...")
            print(
                f"   🤖 模型: {active_stage5b_config['model']}, "
                f"Prompt: {active_stage5b_config['prompt_func']}, "
                f"Flex={active_stage5b_config.get('flex', True)}"
            )
            prompt_func = resolve_tools_prompt_callable(active_stage5b_config["prompt_func"])
            sft_prompts = []
            for rec in records:
                base_prompt_content = rec["prompt"][0]["content"] if rec["prompt"] else ""
                sft_prompts.append(prompt_func(base_prompt_content, rec["graph_data"]))
            all_responses = ask_group_link_in_stage_batches(
                prompt_list=sft_prompts,
                stage_config=active_stage5b_config,
                stage_name="RealTest-Step5B-SFT",
                max_workers_vllm=["local", "ultra"],
                only_check_cache_override=only_check_cache,
            )
            all_responses = tools.cleanthinkans(all_responses)

        if only_check_cache:
            valid_pairs = [
                (rec, reasoning)
                for rec, reasoning in zip(records, all_responses)
                if is_valid_cached_result(reasoning)
            ]
            print(f"🧹 [Real Testset SFT Cache Finalize] kept={len(valid_pairs)}, dropped={len(records) - len(valid_pairs)}")
        else:
            valid_pairs = list(zip(records, all_responses))

        finalized_records = []
        for rec, reasoning in valid_pairs:
            clean_entities, clean_relationships = build_very_simple_ground_truth_lists(
                rec["entities"],
                rec["relationships"],
            )
            entity_json_str = json.dumps(clean_entities, ensure_ascii=False)
            rel_json_str = json.dumps(clean_relationships, ensure_ascii=False)
            rec["ground_truth"] = (
                f"{str(reasoning).strip()}\n\n"
                f"#Entity_List_Start#\n{entity_json_str}\n#Entity_List_End#\n\n"
                f"#Relationship_List_Start#\n{rel_json_str}\n#Relationship_List_End#"
            )
            finalized_records.append(rec)
        records = finalized_records
        if not skip_reasoning:
            print("✅ (Real Testset SFT) 推理链生成完成!")

    df = pd.DataFrame([
        {
            "prompt": rec["prompt"],
            "ground_truth": rec["ground_truth"],
            "text_raw_from_file": rec.get("text_raw_from_file", rec.get("content", "")),
            "graph_from_text_raw_from_file": rec.get("graph_from_text_raw_from_file", json.dumps(rec.get("graph_data", {}), ensure_ascii=False)),
            "text_fixed_by_revision": rec.get("text_fixed_by_revision", rec.get("content", "")),
            "source_file": rec.get("source_file", ""),
            "original_id": rec.get("original_id", ""),
            "stable_article_id": rec.get("stable_article_id", ""),
            "sample_order": rec.get("sample_order", idx),
        }
        for idx, rec in enumerate(records)
        if "ground_truth" in rec
    ])
    df["index"] = range(len(df))
    df["totalnum"] = len(df)
    print(f"✅ Real Testset Loaded: {len(df)} rows.")
    return df


def load_real_testset(source_path, mode="RL", skip_reasoning=False, sources=None, sample_size_per_source=50, seed=42, stage5b_config=None, only_check_cache=False):
    source_paths = normalize_compat_yaml_paths(source_path)
    if not source_paths:
        print("❌ Error: Real Testset source_path 为空")
        return pd.DataFrame()

    print(
        f"🚀 (Real Testset) Loading from {source_paths} "
        f"(mode={mode}, skip_reasoning={skip_reasoning}, only_check_cache={only_check_cache})..."
    )

    records = []
    for idx, single_source_path in enumerate(source_paths, start=1):
        print(f"   📚 [Real Testset Source {idx}/{len(source_paths)}] {single_source_path}")
        if os.path.isdir(single_source_path):
            current_records = load_real_testset_records_from_legacy_folder(single_source_path)
        else:
            current_records = load_real_testset_records_from_ultimate_dataset(
                dataset_path=single_source_path,
                sources=sources,
                sample_size_per_source=sample_size_per_source,
                seed=seed,
            )
        records = merge_real_test_records(
            records,
            current_records,
            label=f"Real Test Source Merge[{idx}]",
        )

    return build_real_testset_dataframe(
        records=records,
        mode=mode,
        skip_reasoning=skip_reasoning,
        stage5b_config=stage5b_config,
        only_check_cache=only_check_cache,
    )


def generate_qa_dataframe_pipeline(sampled_data, run_kg=False, run_rewrite=False, run_genqa=False, run_genkg=False, skip_reasoning=False):
    """
    Orchestrator function connecting Step 1 -> 5.
    
    Args:
        sampled_data: List of dicts [{"content":..., "source":..., "original_id":...}]
    """
    print("\n--- Pipeline Started ---")
    
    all_graphs = []
    folder_paths = []
    all_revised_texts = []

    use_async_pipeline = bool(
        run_kg and run_rewrite and sampled_data
        and int(PIPELINE_CHUNK_SIZE) > 0
        and int(PIPELINE_MAX_IN_FLIGHT) > 0
    )
    if use_async_pipeline:
        return run_async_multistage_pipeline(
            sampled_data,
            run_genqa=run_genqa,
            run_genkg=run_genkg,
            skip_reasoning=skip_reasoning,
        )

    # 1. KG Extraction
    if run_kg:
        all_graphs, folder_paths = step_1_generate_kg(sampled_data)
        if ONLY_CHECK_CACHE:
            valid_mask = [is_valid_cached_result(graph) for graph in all_graphs]
            sampled_data, all_graphs, folder_paths = filter_aligned_records_by_mask(
                sampled_data, all_graphs, folder_paths,
                valid_mask=valid_mask,
                stage_name="Step 1 Cache Finalize"
            )
    else:
        print("⏭️ (Step 1) 跳过 KG Extraction")
        pass

    # 2. Text Revision
    if run_rewrite:
        if not all_graphs or not folder_paths:
             if not run_kg:
                 print("⚠️ 警告: 启用了 -rewrite 但未启用 -kg。正在尝试从磁盘加载 Step 1 的结果...")
                 if os.path.exists(KG_BASE_DIR):
                     folder_paths = sorted([os.path.join(KG_BASE_DIR, d) for d in os.listdir(KG_BASE_DIR) if d.startswith("文章顺序_")])
                     for fp in folder_paths:
                         json_path = os.path.join(fp, "2原始文章对应知识图谱.json")
                         if not os.path.exists(json_path): json_path = os.path.join(fp, "原始文章对应知识图谱.json")
                         if os.path.exists(json_path):
                             with open(json_path, 'r', encoding='utf-8') as f: all_graphs.append(json.load(f))
                         else: all_graphs.append({})
                     
                     # Reconstruct sampled_data (Partial)
                     sampled_data = []
                     for fp in folder_paths:
                         txt_path = os.path.join(fp, "1原始文章.txt")
                         if not os.path.exists(txt_path): txt_path = os.path.join(fp, "原始文章.txt")
                         if os.path.exists(txt_path):
                             with open(txt_path, 'r', encoding='utf-8') as f: 
                                 content = f.read()
                                 stable_article_id = make_stable_article_id(fp, f"Recovered_{os.path.basename(fp)}", 0, content)
                                 sampled_data.append({
                                     "content": content,
                                     "source": "Recovered_From_Disk",
                                     "original_id": "Unknown",
                                     "stable_article_id": stable_article_id,
                                     "sample_order": len(sampled_data),
                                 })
                         else:
                             stable_article_id = make_stable_article_id(fp, f"Recovered_{os.path.basename(fp)}", 0, "")
                             sampled_data.append({
                                 "content": "",
                                 "source": "Recovered_From_Disk",
                                 "original_id": "Unknown",
                                 "stable_article_id": stable_article_id,
                                 "sample_order": len(sampled_data),
                             })

        all_revised_texts = step_2_revise_text(sampled_data, all_graphs, folder_paths)
        if ONLY_CHECK_CACHE:
            valid_mask = [is_valid_cached_result(text) for text in all_revised_texts]
            sampled_data, all_graphs, folder_paths, all_revised_texts = filter_aligned_records_by_mask(
                sampled_data, all_graphs, folder_paths, all_revised_texts,
                valid_mask=valid_mask,
                stage_name="Step 2 Cache Finalize"
            )
        step_3_generate_final_json(all_revised_texts, all_graphs, folder_paths)
        step_4_generate_plots(folder_paths)
        rename_and_generate_statistics(folder_paths)
    else:
        print("⏭️ (Step 2-4) 跳过 Text Revision & Plotting")

    # 5. QA / KG / SFT Generation
    final_df = pd.DataFrame()
    kg_data_df = pd.DataFrame()
    sft_data_df = pd.DataFrame()
    
    kg_data_df = pd.DataFrame()
    
    if run_genqa or run_genkg:
        if not all_revised_texts:
             if not run_rewrite:
                 print("⚠️ 警告: 启用了生成任务 但未启用 -rewrite。尝试加载 Step 2 结果...")
                 if not folder_paths and os.path.exists(KG_BASE_DIR):
                      folder_paths = sorted([os.path.join(KG_BASE_DIR, d) for d in os.listdir(KG_BASE_DIR) if d.startswith("文章顺序_")])
                 
                 if not all_graphs:
                      for fp in folder_paths:
                         json_path = os.path.join(fp, "2原始文章对应知识图谱.json")
                         if not os.path.exists(json_path): json_path = os.path.join(fp, "原始文章对应知识图谱.json")
                         if os.path.exists(json_path):
                             with open(json_path, 'r', encoding='utf-8') as f: all_graphs.append(json.load(f))
                         else: all_graphs.append({})

                 all_revised_texts = []
                 for fp in folder_paths:
                     rev_path = os.path.join(fp, "3根据知识图谱修改后的文章.txt")
                     if not os.path.exists(rev_path): rev_path = os.path.join(fp, "根据知识图谱修改后的文章.txt")
                     if os.path.exists(rev_path):
                         with open(rev_path, 'r', encoding='utf-8') as f: all_revised_texts.append(f.read())
                     else: all_revised_texts.append("")
                 
                 # Reconstruct sampled_data if still missing
                 if not sampled_data or len(sampled_data) != len(all_revised_texts):
                     sampled_data = []
                     for fp in folder_paths:
                         txt_path = os.path.join(fp, "1原始文章.txt")
                         if not os.path.exists(txt_path): txt_path = os.path.join(fp, "原始文章.txt")
                         content = ""
                         if os.path.exists(txt_path):
                             with open(txt_path, 'r', encoding='utf-8') as f: content = f.read()
                         stable_article_id = make_stable_article_id(fp, f"Recovered_{os.path.basename(fp)}", 0, content)
                         sampled_data.append({
                             "content": content,
                             "source": "Recovered_From_Disk",
                             "original_id": "Unknown",
                             "stable_article_id": stable_article_id,
                             "sample_order": len(sampled_data),
                         })
                 
                 if not all_revised_texts:
                     print("❌ 无法加载 Step 2 数据，无法执行后续步骤。")
                     return pd.DataFrame(), pd.DataFrame()

                 valid_mask = [
                     (not is_invalid_text_payload(text)) and is_valid_graph_payload(graph)
                     for text, graph in zip(all_revised_texts, all_graphs)
                 ]
                 if valid_mask and not all(valid_mask):
                     sampled_data, all_graphs, folder_paths, all_revised_texts = filter_aligned_records_by_mask(
                         sampled_data, all_graphs, folder_paths, all_revised_texts,
                         valid_mask=valid_mask,
                         stage_name="Disk Reload Sanitize"
                     )

    if run_genqa:
        final_df = step_5_generate_qa(all_revised_texts, all_graphs, sampled_data)
    else:
        print("⏭️ (Step 5-A) 跳过 QA Generation")
    
    if run_genkg:
        
        kg_data_df = step_5bb_generate_sft_reasoning_data(all_revised_texts, all_graphs, sampled_data, skip_reasoning=skip_reasoning)
    else:
        print("⏭️ (Step 5-B) 跳过 KG Extraction Data Generation")

    return final_df, kg_data_df


# =============================================================================
# --- Part 3: Process and Save Locally ---
# =============================================================================

def collect_identity_hashes_from_generated_df(df, label=""):
    if df is None or df.empty:
        return set()

    hashes = set()
    missing = 0
    for row in df.itertuples(index=False):
        text = getattr(row, "text_raw_from_file", "")
        if is_invalid_text_payload(text):
            missing += 1
            continue
        hashes.add(make_content_identity(text))

    if label:
        print(f"🧩 [{label}] 已完成池 identity 统计: usable={len(hashes)}, missing_text={missing}")
    return hashes


def align_generated_dataframe_with_source_lookup(df, article_lookup, label=""):
    if df is None or df.empty or not article_lookup:
        return df

    aligned_df = df.copy(deep=True)
    matched = 0
    for idx in range(len(aligned_df)):
        text = aligned_df.at[idx, "text_raw_from_file"] if "text_raw_from_file" in aligned_df.columns else ""
        meta = article_lookup.get(make_content_identity(text))
        if not meta:
            continue
        matched += 1
        aligned_df.at[idx, "stable_article_id"] = meta["stable_article_id"]
        aligned_df.at[idx, "original_id"] = meta["original_id"]
        aligned_df.at[idx, "source_file"] = meta["source_file"]

    if label:
        print(f"🧩 [{label}] 旧完成池对齐 canonical stable_article_id: matched={matched}, total={len(aligned_df)}")
    return aligned_df


def merge_generated_dataframes(existing_df, new_df, label=""):
    frames = []
    if existing_df is not None and not existing_df.empty:
        frames.append(existing_df.copy(deep=True))
    if new_df is not None and not new_df.empty:
        frames.append(new_df.copy(deep=True))

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        merged_df = frames[0].copy(deep=True)
    else:
        merged_df = pd.concat(frames, ignore_index=True)

    dedup_seen = set()
    keep_indices = []
    dropped = 0
    for idx, row in enumerate(merged_df.itertuples(index=False)):
        stable_article_id = getattr(row, "stable_article_id", "") or ""
        if stable_article_id:
            dedup_key = ("stable", stable_article_id)
        else:
            dedup_key = ("content", make_content_identity(getattr(row, "text_raw_from_file", "")))
        if dedup_key in dedup_seen:
            dropped += 1
            continue
        dedup_seen.add(dedup_key)
        keep_indices.append(idx)

    merged_df = merged_df.iloc[keep_indices].reset_index(drop=True)
    if "sample_order" in merged_df.columns:
        merged_df["sample_order"] = list(range(len(merged_df)))
    if "index" in merged_df.columns:
        merged_df["index"] = list(range(len(merged_df)))
    if "totalnum" in merged_df.columns:
        merged_df["totalnum"] = len(merged_df)

    if label:
        print(
            f"🧩 [{label}] 合并完成: existing={0 if existing_df is None else len(existing_df)}, "
            f"new={0 if new_df is None else len(new_df)}, merged={len(merged_df)}, dropped_dup={dropped}"
        )
    return merged_df


def process_and_save_dataset_locally(df, name, output_dir, data_source, ability, extra_info=None, split_ratio=0.8, train_type="RL", real_test_df=None):
    # 2026-03-21:
    
    
    
    export_variant_suffix = {
        "full_train": "(full_nosplit_useastrain)",
        "train_split": "(0.7_split_from_full)",
        "split_test": "(0.3_split_from_full)",
        # 2026-03-23:
        
        
        "real_test": "(real_useastest)",
    }

    if df.empty and (real_test_df is None or real_test_df.empty):
        print("⚠️ DataFrame is empty, skipping save.")
        return

    background_text = getattr(
        tools,
        "简短的背景让QA数据集的读者理解这个任务_260303",
        QA_BACKGROUND_TEXT_260303,
    ) or QA_BACKGROUND_TEXT_260303

    def _build_chat_prompt(content, question, options, extra_text=""):
        part1 = f"""You are an expert AI assistant specializing in Cyber Threat Intelligence (CTI) and knowledge graph (KG) extraction.

Your **primary task** is to answer a multiple-choice question about a piece of CTI text.
This question is designed to evaluate whether a KG extraction model would make a mistake or succeed, based on a specific set of rules.

First, carefully review the **reference rules**:
--- REFERENCE: KG EXTRACTION RULES START ---
{background_text}
--- REFERENCE: KG EXTRACTION RULES END ---

Now, using these rules as your guide, analyze the following context and answer the question.

**Context:**
{content}

**Question:**
{question}

**Options:**
{options}

**Your Task (Instruction):**
Think the task through step-by-step, write down your reasoning process, and then determine which options are correct according to the reference rules.
"""
        part2 = """
Select any applicable options from [\"A\",\"B\",\"C\",\"D\"] (Must be \", not \')
Final output the final answer, strictly in python list format, and prefix it with four hash symbols.
Example:
#### ["A","C"]
If none of the options are correct, output:
#### []
"""
        q = part1 + extra_text + part2
        return [{"role": "user", "content": q}]

    def _answers_to_json_array(ans_raw):
        letters = re.findall(r"[A-D]", ans_raw or "", flags=re.I)
        uniq = sorted(list(set(l.upper() for l in letters)))
        return json.dumps(uniq, ensure_ascii=False)

    mode = train_type.strip().upper()
    print(f"\n--- Part 3: Processing and Saving Dataset (mode={mode}) ---")
    
    text_to_append = """Before you think how to solve the question, write down the words <mythink> to mark your reasoning process step by step, and when you decide to give the final answer, write down </mythink> to end your reasoning. Then output the final answer.""" if mode == "RL" else """Before you think how to solve the question, write down <mythink> to mark your reasoning process step by step, and when you decide to give the final answer, write down </mythink> to end your reasoning. Then output the final answer."""

    def process_row(row):
        # Handle pre-computed prompt/ground_truth (e.g. from step_6, step_7)
        if hasattr(row, 'prompt') and hasattr(row, 'ground_truth'):
            # For SFT step_7 data: prompt is already list of dict, ground_truth is string (Reasoning...)
            # For RL step_6 data: prompt is list of dict, ground_truth is string (Text###KG)
            return row.prompt, row.ground_truth

        # For QA step_5 data: we need to construct prompt
        prompt = _build_chat_prompt(row.Content, row.Question, row.Options, extra_text=text_to_append)
        if mode == "SFT":
            gt = str(row.sft_answer or "")
        else:
            gt = _answers_to_json_array(row.Answers)
        return prompt, gt

    if df.empty:
        results = []
        prompts, gts = [], []
    else:
        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(executor.map(process_row, df.itertuples(index=False)))
        prompts, gts = zip(*results)
    
    # --- UNIFIED EXTRA INFO STRUCT ---
    final_extra_info_list = []
    
    # Helper to safely get value from row or extra_info
    def get_val(row, key, default=""):
        if hasattr(row, key):
             return getattr(row, key)
        if extra_info and key in extra_info:
             return extra_info[key]
        return default

    # Current date
    current_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    def _infer_reward_provider(model_name):
        model_name = str(model_name or "").strip()
        lower_name = model_name.lower()
        if lower_name in {"super", "normal", "ultra"}:
            return lower_name
        if lower_name.startswith("gpt") or "openai" in lower_name:
            return "api"
        return "api"
    
    for i, row in enumerate(df.itertuples(index=False)):
        # Determine specific ground truth value for extra info
        # logic: we just store the "top level" values as requested
        
        # Prepare Info Dict
        info = {
            "text_raw_from_file": get_val(row, "text_raw_from_file", ""),
            "graph_from_text_raw_from_file": get_val(row, "graph_from_text_raw_from_file", ""),
            "text_fixed_by_revision": get_val(row, "text_fixed_by_revision", ""),
            "source_file": get_val(row, "source_file", ""),
            "original_id": get_val(row, "original_id", ""),
            "stable_article_id": get_val(row, "stable_article_id", ""),
            "sample_order": get_val(row, "sample_order", i),
            
            "stage1_llm_labeler": STAGE1_CONFIG["model"],
            "stage1_prompt_name": STAGE1_CONFIG["prompt_func"],
            "stage2_llm_labeler": STAGE2_CONFIG["model"],
            "stage2_prompt_name": STAGE2_CONFIG["prompt_func"],
            
            "date": current_date,
            "index": getattr(row, 'index', i),
            "totalnum": len(df), 
            "split_strategy": "stable_hash_by_article_id_only",
            
            "top_level_prompt": prompts[i],
            "top_level_data_source": data_source, 
            "top_level_ground_truth": gts[i],
            "top_level_sft_ground_truth": gts[i],
        }
        final_extra_info_list.append(info)

    # Create Base DataFrame (Unified Schema)
    data_dict = {
        "prompt": prompts,
        "data_source": [data_source] * len(prompts),
        "ability": [ability] * len(prompts),
        "ground_truth": gts,
        "extra_info": final_extra_info_list,
        "reward_model": [{"style":"rule","ground_truth": gt} for gt in gts],
        "sft_ground_truth": gts,
        "_stable_article_id": [
            (info.get("stable_article_id") or f"row_{idx}")
            for idx, info in enumerate(final_extra_info_list)
        ],
        "_sample_order": [
            info.get("sample_order", idx)
            for idx, info in enumerate(final_extra_info_list)
        ],
    }

    base = pd.DataFrame(data_dict)

    def filter_invalid_dataset_rows(dframe, label):
        if dframe.empty:
            return dframe

        invalid_text = dframe["extra_info"].apply(
            lambda info: is_invalid_text_payload((info or {}).get("text_fixed_by_revision", ""))
        )
        invalid_prompt = dframe["prompt"].apply(prompt_contains_invalid_input_text)
        empty_graph = dframe["extra_info"].apply(
            lambda info: not is_valid_graph_payload((info or {}).get("graph_from_text_raw_from_file", ""))
        )
        valid_mask = ~(invalid_text | invalid_prompt | empty_graph)
        dropped = int((~valid_mask).sum())
        if dropped:
            print(
                f"🧹 [{label}] 过滤坏样本: kept={int(valid_mask.sum())}, dropped={dropped}, "
                f"bad_text={int(invalid_text.sum())}, bad_prompt={int(invalid_prompt.sum())}, empty_graph={int(empty_graph.sum())}"
            )
        return dframe.loc[valid_mask].copy()

    base = filter_invalid_dataset_rows(base, "Generated Dataset Finalize")

    
    
    
    split_mask = pd.Series([
        is_train_split(stable_article_id=stable_article_id, split_ratio=split_ratio)
        for stable_article_id in base["_stable_article_id"].tolist()
    ], index=base.index)
    full_train_df = base.sort_values("_sample_order").copy()
    train_df = base[split_mask].sort_values("_sample_order").copy()
    test_df = base[~split_mask].sort_values("_sample_order").copy()
    print(f"📦 稳定切分完成: full_train={len(full_train_df)}, train={len(train_df)}, test={len(test_df)}, split_ratio={split_ratio}")
    
    # Update totalnum in extra_info based on split?
    
    # So we should update it for train and test DFs separately.
    
    def update_extra_info(dframe, dataset_type, test_variant=""):
        t_num = len(dframe)
        new_extras = []
        for idx, val in enumerate(dframe['extra_info']):
            new_val = val.copy()
            new_val['totalnum'] = t_num
            new_val['index'] = idx
            new_val['dataset_type'] = dataset_type
            if test_variant:
                new_val['test_variant'] = test_variant
            new_extras.append(new_val)
        dframe['extra_info'] = new_extras
        return dframe

    full_train_df = update_extra_info(full_train_df.copy(), "train", test_variant="full_nosplit")
    train_df = update_extra_info(train_df.copy(), "train", test_variant="split_some_to_build_trainbased_test")
    test_df = update_extra_info(test_df.copy(), "test", test_variant="split_from_same_source_as_train")

    split_test_df = test_df.copy()
    real_test_output_df = None

    # Handle Real Test Set
    if real_test_df is not None and not real_test_df.empty:
        print("🧪 Using REAL TEST SET in addition to generated split test set.")
        
        # We need to map real_test_df to Unified Schema
        # real_test_df columns: prompt, ground_truth, [metadata columns if available]
        
        real_prompts = real_test_df['prompt'].tolist()
        real_gts = real_test_df['ground_truth'].tolist()
        
        real_extras = []
        for i, row in enumerate(real_test_df.itertuples(index=False)):
            info = {
                "text_raw_from_file": getattr(row, "text_raw_from_file", ""),
                "graph_from_text_raw_from_file": getattr(row, "graph_from_text_raw_from_file", ""),
                "text_fixed_by_revision": getattr(row, "text_fixed_by_revision", ""),
                "source_file": getattr(row, "source_file", "Real_Test_Set_JSONs"),
                "original_id": getattr(row, "original_id", ""),
                "stable_article_id": getattr(row, "stable_article_id", f"real_test_{i}"),
                "sample_order": getattr(row, "sample_order", i),
                "stage1_llm_labeler": "",
                "stage1_prompt_name": "",
                "stage2_llm_labeler": "",
                "stage2_prompt_name": "",
                "date": current_date,
                "index": i,
                "totalnum": len(real_test_df),
                "top_level_prompt": real_prompts[i],
                "top_level_data_source": data_source,
                "top_level_ground_truth": real_gts[i],
                "top_level_sft_ground_truth": real_gts[i],
            }
            real_extras.append(info)

        real_data_dict = {
            "prompt": real_prompts,
            "data_source": [data_source] * len(real_prompts),
            "ability": [ability] * len(real_prompts),
            "ground_truth": real_gts,
            "extra_info": real_extras,
            "reward_model": [{"style":"rule","ground_truth": gt} for gt in real_gts],
            "sft_ground_truth": real_gts,
            "_stable_article_id": [getattr(row, "stable_article_id", f"real_test_{i}") for i, row in enumerate(real_test_df.itertuples(index=False))],
            "_sample_order": [getattr(row, "sample_order", i) for i, row in enumerate(real_test_df.itertuples(index=False))],
        }
        real_test_output_df = pd.DataFrame(real_data_dict)
        real_test_output_df = filter_invalid_dataset_rows(real_test_output_df, "Real Test Finalize")
        real_test_output_df = update_extra_info(real_test_output_df, "test", test_variant="real")

    # Internal helper columns should not leak into final parquet/json.
    drop_internal_cols = ["_stable_article_id", "_sample_order"]
    full_train_df = full_train_df.drop(columns=[c for c in drop_internal_cols if c in full_train_df.columns])
    train_df = train_df.drop(columns=[c for c in drop_internal_cols if c in train_df.columns])
    split_test_df = split_test_df.drop(columns=[c for c in drop_internal_cols if c in split_test_df.columns])
    if real_test_output_df is not None:
        real_test_output_df = real_test_output_df.drop(columns=[c for c in drop_internal_cols if c in real_test_output_df.columns])

    os.makedirs(output_dir, exist_ok=True)
    
    def save_dataset_variant(dframe, variant_suffix):
        parquet_path = os.path.join(output_dir, f"{name}_{variant_suffix}.parquet")
        json_path = os.path.join(output_dir, f"JSON版_{name}_{variant_suffix}.json")
        dframe.to_parquet(parquet_path, index=False)
        dframe.to_json(json_path, orient='records', force_ascii=False, indent=2)
        print(f"✅ [Done] Saved {len(dframe)} rows -> {os.path.basename(parquet_path)}")
        return parquet_path

    save_jobs = [
        (full_train_df, export_variant_suffix["full_train"]),
        (train_df, export_variant_suffix["train_split"]),
    ]
    split_test_suffix = export_variant_suffix["split_test"]
    save_jobs.append((split_test_df, split_test_suffix))
    if real_test_output_df is not None:
        save_jobs.append((real_test_output_df, export_variant_suffix["real_test"]))

    with ThreadPoolExecutor(max_workers=min(4, len(save_jobs), os.cpu_count() or 4)) as executor:
        list(executor.map(lambda args: save_dataset_variant(*args), save_jobs))
    
    
    check_dir = os.path.join(output_dir, "检查用")
    os.makedirs(check_dir, exist_ok=True)
    sample_count = 5
    
    if len(full_train_df) > 0:
        full_train_sample = full_train_df.sample(min(len(full_train_df), sample_count), random_state=SEED)
        full_train_sample.to_json(
            os.path.join(check_dir, f"{name}_{export_variant_suffix['full_train']}-{sample_count}个随机样本.json"),
            orient='records', force_ascii=False, indent=2
        )
    if len(train_df) > 0:
        train_sample = train_df.sample(min(len(train_df), sample_count), random_state=SEED)
        train_sample.to_json(
            os.path.join(check_dir, f"{name}_{export_variant_suffix['train_split']}-{sample_count}个随机样本.json"),
            orient='records', force_ascii=False, indent=2
        )
    if len(split_test_df) > 0:
        test_sample = split_test_df.sample(min(len(split_test_df), sample_count), random_state=SEED)
        test_sample.to_json(
            os.path.join(check_dir, f"{name}_{split_test_suffix}-{sample_count}个随机样本.json"),
            orient='records', force_ascii=False, indent=2
        )
    if real_test_output_df is not None and len(real_test_output_df) > 0:
        real_test_sample = real_test_output_df.sample(min(len(real_test_output_df), sample_count), random_state=SEED)
        real_test_sample.to_json(
            os.path.join(check_dir, f"{name}_{export_variant_suffix['real_test']}-{sample_count}个随机样本.json"),
            orient='records', force_ascii=False, indent=2
        )
    print(f"🔍 [Done] Saved {sample_count} random samples to {check_dir}")

    # --- Print Token Statistics ---
    def print_stats(df, label):
        if df.empty:
            print(f"📊 {label}: Empty DataFrame")
            return
        
        print(f"\n📊 Token Statistics for {label} ({len(df)} rows):")
        
        cols_to_check = ['prompt', 'reward_model', 'sft_ground_truth']
        for col in cols_to_check:
            if col not in df.columns: continue
            
            target_list = []
            vals = df[col].tolist()
            if col == 'prompt':
                for v in vals: target_list.append(str(v))
            elif col == 'reward_model':
                 for v in vals:
                     if isinstance(v, dict): target_list.append(str(v.get('ground_truth', '')))
                     else: target_list.append(str(v))
            elif col == 'sft_ground_truth':
                 for v in vals: target_list.append(str(v))

            if target_list:
                with ThreadPoolExecutor(max_workers=min(64, os.cpu_count() or 32, len(target_list))) as executor:
                    lens = list(executor.map(lambda x: tools.tokenlen(str(x)), target_list))
                print(f"  - {col}: Avg={np.mean(lens):.1f}, Max={np.max(lens)}, Min={np.min(lens)}")

    print_stats(full_train_df, export_variant_suffix["full_train"])
    print_stats(train_df, export_variant_suffix["train_split"])
    print_stats(split_test_df, export_variant_suffix["split_test"])
    if real_test_output_df is not None:
        print_stats(real_test_output_df, export_variant_suffix["real_test"])


# =============================================================================
# --- Main Execution ---
# =============================================================================

def load_yaml_config(yaml_path):
    global INPUT_SOURCES, OUTPUT_DIR, OUTPUT_NAME, KG_BASE_DIR
    global TARGET_COUNT, MAX_TOKENS, SEED, SPLIT_RATIO, PIPELINE_CHUNK_SIZE, PIPELINE_MAX_IN_FLIGHT, PIPELINE_FILL_RATE_PER_SEC
    global PIPELINE_STAGE1_SOFT_LIMIT, PIPELINE_STAGE1_HARD_LIMIT, PIPELINE_PRIORITY_DOWNSTREAM_STAGES, ONLY_CHECK_CACHE
    global GENERATED_DATASET_ONLY_CHECK_CACHE, GENERATED_DATASET_FAST_FINALIZE_FROM_DISK, REAL_TESTSET_ONLY_CHECK_CACHE, REAL_TESTSET_PRIORITY_FIRST
    global COMPAT_OLD_YAML_PATH, COMPAT_RUN_OLD_YAML_FIRST, COMPAT_USE_OLD_REAL_TEST_DF_FOR_CURRENT_OUTPUT
    global BASE_VERY_SIMPLE_PROMPT_FUNC, PRIORITY_REFERENCE_PARQUET_PATHS, PRIORITY_UNLOCK_LOW_PRIORITY_THRESHOLD
    global STAGE1_CONFIG, STAGE2_CONFIG, STAGE5A_CONFIG, STAGE5B_CONFIG, REAL_TESTSET_CONFIG, REAL_TEST_STAGE5B_CONFIG, VERL_ROUTING_CONFIG
    
    if not os.path.exists(yaml_path):
        print(f"❌ YAML 配置文件不存在: {yaml_path}")
        sys.exit(1)
    
    print(f"📄 正在加载 YAML 配置: {yaml_path}")

    
    
    PIPELINE_STAGE1_SOFT_LIMIT = 0
    PIPELINE_STAGE1_HARD_LIMIT = 0
    PIPELINE_PRIORITY_DOWNSTREAM_STAGES = True
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if not config:
        print("⚠️ YAML 配置文件为空")
        return {}
    
    
    if 'input_sources' in config:
        INPUT_SOURCES = config['input_sources']
        print(f"  ✓ INPUT_SOURCES: 加载了 {len(INPUT_SOURCES)} 个数据源")
        for i, src in enumerate(INPUT_SOURCES):
            src_name = src.get('name', f'source_{i}')
            print(f"      [{i+1}] {src_name}: {os.path.basename(src.get('path', 'N/A'))}")
    if 'output_dir' in config:
        OUTPUT_DIR = config['output_dir']
        print(f"  ✓ OUTPUT_DIR = {OUTPUT_DIR}")
    if 'output_name' in config:
        OUTPUT_NAME = config['output_name']
        print(f"  ✓ OUTPUT_NAME = {OUTPUT_NAME}")
    if 'kg_base_dir' in config:
        KG_BASE_DIR = config['kg_base_dir']
        print(f"  ✓ KG_BASE_DIR = {KG_BASE_DIR}")
    if 'target_count' in config:
        TARGET_COUNT = int(config['target_count'])
        print(f"  ✓ TARGET_COUNT = {TARGET_COUNT}")
    if 'max_tokens' in config:
        MAX_TOKENS = int(config['max_tokens'])
        print(f"  ✓ MAX_TOKENS = {MAX_TOKENS}")
    if 'seed' in config:
        SEED = int(config['seed'])
        print(f"  ✓ SEED = {SEED}")
    if 'split_ratio' in config:
        SPLIT_RATIO = float(config['split_ratio'])
        print(f"  ✓ SPLIT_RATIO = {SPLIT_RATIO}")
    if 'pipeline_chunk_size' in config:
        PIPELINE_CHUNK_SIZE = int(config['pipeline_chunk_size'])
        print(f"  ✓ PIPELINE_CHUNK_SIZE = {PIPELINE_CHUNK_SIZE}")
    if 'pipeline_max_in_flight' in config:
        PIPELINE_MAX_IN_FLIGHT = int(config['pipeline_max_in_flight'])
        print(f"  ✓ PIPELINE_MAX_IN_FLIGHT = {PIPELINE_MAX_IN_FLIGHT}")
    if 'pipeline_fill_rate_per_sec' in config:
        PIPELINE_FILL_RATE_PER_SEC = float(config['pipeline_fill_rate_per_sec'])
        print(f"  ✓ PIPELINE_FILL_RATE_PER_SEC = {PIPELINE_FILL_RATE_PER_SEC} (仅限制未命中缓存的云端发送速率)")
    if 'pipeline_stage1_soft_limit' in config:
        PIPELINE_STAGE1_SOFT_LIMIT = int(config['pipeline_stage1_soft_limit'])
        print(f"  ✓ PIPELINE_STAGE1_SOFT_LIMIT = {PIPELINE_STAGE1_SOFT_LIMIT} (0 表示自动推导)")
    if 'pipeline_stage1_hard_limit' in config:
        PIPELINE_STAGE1_HARD_LIMIT = int(config['pipeline_stage1_hard_limit'])
        print(f"  ✓ PIPELINE_STAGE1_HARD_LIMIT = {PIPELINE_STAGE1_HARD_LIMIT} (0 表示自动推导)")
    if 'pipeline_priority_downstream_stages' in config:
        PIPELINE_PRIORITY_DOWNSTREAM_STAGES = bool(config['pipeline_priority_downstream_stages'])
        print(f"  ✓ PIPELINE_PRIORITY_DOWNSTREAM_STAGES = {PIPELINE_PRIORITY_DOWNSTREAM_STAGES}")
    if 'only_check_cache' in config:
        ONLY_CHECK_CACHE = bool(config['only_check_cache'])
        print(f"  ✓ ONLY_CHECK_CACHE = {ONLY_CHECK_CACHE}")
    if 'generated_dataset_only_check_cache' in config:
        GENERATED_DATASET_ONLY_CHECK_CACHE = bool(config['generated_dataset_only_check_cache'])
        print(f"  ✓ GENERATED_DATASET_ONLY_CHECK_CACHE = {GENERATED_DATASET_ONLY_CHECK_CACHE}")
    if 'generated_dataset_fast_finalize_from_disk' in config:
        GENERATED_DATASET_FAST_FINALIZE_FROM_DISK = bool(config['generated_dataset_fast_finalize_from_disk'])
        print(f"  ✓ GENERATED_DATASET_FAST_FINALIZE_FROM_DISK = {GENERATED_DATASET_FAST_FINALIZE_FROM_DISK}")
    if 'real_testset_only_check_cache' in config:
        REAL_TESTSET_ONLY_CHECK_CACHE = bool(config['real_testset_only_check_cache'])
        print(f"  ✓ REAL_TESTSET_ONLY_CHECK_CACHE = {REAL_TESTSET_ONLY_CHECK_CACHE}")
    if 'real_testset_priority_first' in config:
        REAL_TESTSET_PRIORITY_FIRST = bool(config['real_testset_priority_first'])
        print(f"  ✓ REAL_TESTSET_PRIORITY_FIRST = {REAL_TESTSET_PRIORITY_FIRST}")
    if 'compat_old_yaml_path' in config:
        COMPAT_OLD_YAML_PATH = normalize_compat_yaml_paths(config['compat_old_yaml_path'])
        print(f"  ✓ COMPAT_OLD_YAML_PATH = {COMPAT_OLD_YAML_PATH}")
    if 'compat_run_old_yaml_first' in config:
        COMPAT_RUN_OLD_YAML_FIRST = bool(config['compat_run_old_yaml_first'])
        print(f"  ✓ COMPAT_RUN_OLD_YAML_FIRST = {COMPAT_RUN_OLD_YAML_FIRST}")
    if 'compat_use_old_real_test_df_for_current_output' in config:
        COMPAT_USE_OLD_REAL_TEST_DF_FOR_CURRENT_OUTPUT = bool(config['compat_use_old_real_test_df_for_current_output'])
        print(
            "  ✓ COMPAT_USE_OLD_REAL_TEST_DF_FOR_CURRENT_OUTPUT = "
            f"{COMPAT_USE_OLD_REAL_TEST_DF_FOR_CURRENT_OUTPUT}"
        )
    if 'base_very_simple_prompt_func' in config:
        BASE_VERY_SIMPLE_PROMPT_FUNC = str(config['base_very_simple_prompt_func'] or "").strip()
        print(f"  ✓ BASE_VERY_SIMPLE_PROMPT_FUNC = {BASE_VERY_SIMPLE_PROMPT_FUNC}")
    if 'priority_reference_parquet_paths' in config:
        PRIORITY_REFERENCE_PARQUET_PATHS = normalize_compat_yaml_paths(config['priority_reference_parquet_paths'])
        print(f"  ✓ PRIORITY_REFERENCE_PARQUET_PATHS = {PRIORITY_REFERENCE_PARQUET_PATHS}")
    if 'priority_unlock_low_priority_threshold' in config:
        PRIORITY_UNLOCK_LOW_PRIORITY_THRESHOLD = int(config['priority_unlock_low_priority_threshold'])
        print(f"  ✓ PRIORITY_UNLOCK_LOW_PRIORITY_THRESHOLD = {PRIORITY_UNLOCK_LOW_PRIORITY_THRESHOLD}")
    
    
    if 'stage1_config' in config:
        stage1_yaml = config['stage1_config']
        for key in ['model', 'token', 'temp', 'flex', 'prompt_func', 'retry', 'max_retry_attempts', 'max_total_seconds', 'stream_stall_seconds']:
            if key in stage1_yaml:
                STAGE1_CONFIG[key] = stage1_yaml[key]
        print(f"  ✓ STAGE1_CONFIG: model={STAGE1_CONFIG['model']}, token={STAGE1_CONFIG['token']}, temp={STAGE1_CONFIG['temp']}")
        print(f"                   prompt_func={STAGE1_CONFIG['prompt_func']}")
        print(f"                   retry={STAGE1_CONFIG['retry']}, max_retry_attempts={STAGE1_CONFIG['max_retry_attempts']}, max_total_seconds={STAGE1_CONFIG['max_total_seconds']}, stream_stall_seconds={STAGE1_CONFIG['stream_stall_seconds']}")
    
    if 'stage2_config' in config:
        stage2_yaml = config['stage2_config']
        for key in ['model', 'token', 'temp', 'flex', 'prompt_func', 'retry', 'max_retry_attempts', 'max_total_seconds', 'stream_stall_seconds']:
            if key in stage2_yaml:
                STAGE2_CONFIG[key] = stage2_yaml[key]
        print(f"  ✓ STAGE2_CONFIG: model={STAGE2_CONFIG['model']}, token={STAGE2_CONFIG['token']}, temp={STAGE2_CONFIG['temp']}")
        print(f"                   prompt_func={STAGE2_CONFIG['prompt_func']}")
        print(f"                   retry={STAGE2_CONFIG['retry']}, max_retry_attempts={STAGE2_CONFIG['max_retry_attempts']}, max_total_seconds={STAGE2_CONFIG['max_total_seconds']}, stream_stall_seconds={STAGE2_CONFIG['stream_stall_seconds']}")
    
    if 'stage5a_config' in config:
        stage5a_yaml = config['stage5a_config']
        for key in ['model', 'token', 'temp', 'flex', 'prompt_func', 'retry', 'max_retry_attempts', 'max_total_seconds', 'stream_stall_seconds']:
            if key in stage5a_yaml:
                STAGE5A_CONFIG[key] = stage5a_yaml[key]
        print(f"  ✓ STAGE5A_CONFIG: model={STAGE5A_CONFIG['model']}, token={STAGE5A_CONFIG['token']}, temp={STAGE5A_CONFIG['temp']}")
        print(f"                    prompt_func={STAGE5A_CONFIG['prompt_func']}")
        print(f"                    retry={STAGE5A_CONFIG['retry']}, max_retry_attempts={STAGE5A_CONFIG['max_retry_attempts']}, max_total_seconds={STAGE5A_CONFIG['max_total_seconds']}, stream_stall_seconds={STAGE5A_CONFIG['stream_stall_seconds']}")
    
    if 'stage5b_config' in config:
        stage5b_yaml = config['stage5b_config']
        for key in ['model', 'token', 'temp', 'flex', 'prompt_func', 'request_batch_size', 'retry', 'max_retry_attempts', 'max_total_seconds', 'stream_stall_seconds']:
            if key in stage5b_yaml:
                STAGE5B_CONFIG[key] = stage5b_yaml[key]
        print(f"  ✓ STAGE5B_CONFIG: model={STAGE5B_CONFIG['model']}, token={STAGE5B_CONFIG['token']}, temp={STAGE5B_CONFIG['temp']}")
        print(f"                    prompt_func={STAGE5B_CONFIG['prompt_func']}")
        print(f"                    request_batch_size={STAGE5B_CONFIG['request_batch_size']}, retry={STAGE5B_CONFIG['retry']}, max_retry_attempts={STAGE5B_CONFIG['max_retry_attempts']}, max_total_seconds={STAGE5B_CONFIG['max_total_seconds']}, stream_stall_seconds={STAGE5B_CONFIG['stream_stall_seconds']}")

    if 'real_testset_config' in config:
        real_test_yaml = config['real_testset_config']
        for key in ['dataset_path', 'sources', 'sample_size_per_source', 'seed', 'mode']:
            if key in real_test_yaml:
                REAL_TESTSET_CONFIG[key] = real_test_yaml[key]
        print(
            "  ✓ REAL_TESTSET_CONFIG: "
            f"dataset_path={REAL_TESTSET_CONFIG['dataset_path']}, "
            f"sources={REAL_TESTSET_CONFIG['sources']}, "
            f"sample_size_per_source={REAL_TESTSET_CONFIG['sample_size_per_source']}, "
            f"seed={REAL_TESTSET_CONFIG['seed']}, mode={REAL_TESTSET_CONFIG['mode']}"
        )

    if 'real_test_stage5b_config' in config:
        real_test_stage5b_yaml = config['real_test_stage5b_config']
        for key in ['model', 'token', 'temp', 'flex', 'prompt_func', 'request_batch_size', 'retry', 'max_retry_attempts', 'max_total_seconds', 'stream_stall_seconds']:
            if key in real_test_stage5b_yaml:
                REAL_TEST_STAGE5B_CONFIG[key] = real_test_stage5b_yaml[key]
        print(
            "  ✓ REAL_TEST_STAGE5B_CONFIG: "
            f"model={REAL_TEST_STAGE5B_CONFIG['model']}, token={REAL_TEST_STAGE5B_CONFIG['token']}, "
            f"temp={REAL_TEST_STAGE5B_CONFIG['temp']}, flex={REAL_TEST_STAGE5B_CONFIG['flex']}, "
            f"prompt_func={REAL_TEST_STAGE5B_CONFIG['prompt_func']}, request_batch_size={REAL_TEST_STAGE5B_CONFIG['request_batch_size']}"
        )

    if 'verl_routing_config' in config:
        verl_routing_yaml = config['verl_routing_config']
        for key in ['train_reward_model', 'train_reward_flex', 'test_reward_model', 'test_reward_flex']:
            if key in verl_routing_yaml:
                VERL_ROUTING_CONFIG[key] = verl_routing_yaml[key]
        print(
            "  ✓ VERL_ROUTING_CONFIG: "
            f"train_reward_model={VERL_ROUTING_CONFIG['train_reward_model']} (flex={VERL_ROUTING_CONFIG['train_reward_flex']}), "
            f"test_reward_model={VERL_ROUTING_CONFIG['test_reward_model']} (flex={VERL_ROUTING_CONFIG['test_reward_flex']})"
        )
    
    print("✅ YAML 配置加载完成")
    return config


def detect_fast_finalize_candidates(base_dir):
    if not base_dir or not os.path.exists(base_dir):
        return 0

    recovered = 0
    with os.scandir(base_dir) as it:
        for entry in it:
            if not entry.is_dir() or not entry.name.startswith("文章顺序_"):
                continue
            graph_path = os.path.join(entry.path, "原始文章对应知识图谱.json")
            if not os.path.exists(graph_path):
                graph_path = os.path.join(entry.path, "2原始文章对应知识图谱.json")
            revised_path = os.path.join(entry.path, "根据知识图谱修改后的文章.txt")
            if not os.path.exists(revised_path):
                revised_path = os.path.join(entry.path, "3根据知识图谱修改后的文章.txt")
            if os.path.exists(graph_path) and os.path.exists(revised_path):
                recovered += 1
    return recovered


def build_argument_parser():
    """Build the shared CLI parser for the main pipeline and compatibility flows."""
    parser = argparse.ArgumentParser(
        description="Generate GRID training data artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CLI-only mode
  python generate_train_data_pipeline.py -kg -rewrite -genkg

  # YAML-driven mode (the YAML steps section overrides CLI stage flags)
  python generate_train_data_pipeline.py -yaml config.yaml

  # Mixed mode (YAML for paths and configs, CLI for stage selection)
  python generate_train_data_pipeline.py -yaml config.yaml -kg -rewrite

  # Standalone Step6 export from an existing Step5 parquet
  python generate_train_data_pipeline.py -yaml config.yaml --individual-step6

  # Derive random-order and complex-order variants from an existing Step6 parquet
  python generate_train_data_pipeline.py -yaml config.yaml --split-step6-to-random-and-complex-2vers
"""
    )
    parser.add_argument('-yaml', '--yaml', dest='yaml_config', type=str, default=None,
                        help='Path to a YAML config file used to override default settings.')
    parser.add_argument('--input-file', type=str, nargs='+', default=None,
                        help='One or more input files. When provided, these files override INPUT_SOURCES and share the same column settings.')
    parser.add_argument('--input-type', type=str, default='auto',
                        help='Input file type: auto/xlsx/csv/parquet.')
    parser.add_argument('--content-col', type=str, default='content',
                        help='Column containing the article text. Default: content.')
    parser.add_argument('--id-col', type=str, default='id',
                        help='Column containing the stable article identifier. Default: id.')
    parser.add_argument('--id-prefix', type=str, default='',
                        help='Optional identifier prefix. Default: empty.')
    parser.add_argument('--input-name', type=str, default=None,
                        help='Display name of the input source. Defaults to the file name.')
    parser.add_argument('--filter-col', type=str, default=None,
                        help='Optional filter column.')
    parser.add_argument('--filter-value', type=str, default=None,
                        help='Optional filter value.')
    parser.add_argument('--target-count', type=int, default=None,
                        help='Number of articles to sample. Use <=0 to keep all length-qualified articles.')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Maximum input-token threshold used for filtering.')
    parser.add_argument('--split-ratio', type=float, default=None,
                        help='Train split ratio. Default: 0.8. The split is stable because it only depends on stable_article_id hashing.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed used for stable sampling order.')
    parser.add_argument('--pipeline-chunk-size', type=int, default=None,
                        help='Legacy compatibility flag. In the article-level async pipeline this should normally stay at 1.')
    parser.add_argument('--pipeline-max-in-flight', type=int, default=None,
                        help='Maximum number of articles allowed to remain in flight in the async pipeline. Default: 2048.')
    parser.add_argument('--pipeline-fill-rate-per-sec', type=float, default=None,
                        help='Rate limit for uncached prompts sent to the backend per second. Cache hits are not throttled by this option.')
    parser.add_argument('--only-check-cache', action='store_true',
                        help='Reuse existing asks cache only and send no new requests. Useful for finalizing parquet outputs after an interrupted run.')
    parser.add_argument('--individual-step6', action='store_true',
                        help='Run standalone Step6 export: read an existing Step5 parquet and export VERL parquet from the asks cache without rerunning Steps 1-5.')
    parser.add_argument('--split-step6-to-random-and-complex-2vers', dest='split_step6_to_random_and_complex_2vers', action='store_true',
                        help='Derive two additional Step6 datasets from an existing Step6 parquet: random_order and complex_order.')
    parser.add_argument('--gen-random-and-complex', dest='split_step6_to_random_and_complex_2vers', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--filter-by-rollout-model', action='store_true',
                        help='Run local rollout-based filtering on an existing Step6 parquet and keep medium-difficulty samples by average accuracy.')
    parser.add_argument('--rollout-model-location', type=str, default=None,
                        help='Physical path to the rollout model. The script will try to start the same local model on super/normal/ultra.')
    parser.add_argument('--filter-by-rollout-model-sample-number', type=int, default=1000,
                        help='Number of samples used during rollout filtering. Default: 1000.')
    parser.add_argument('--rollout-filter-output-parquet', type=str, default=None,
                        help='Output parquet for rollout filtering. By default it is created next to the input parquet using a filter-min33max66-sampleXXX suffix.')
    parser.add_argument('--rollout-repeat-number', type=int, default=STEP6_ROLLOUT_DEFAULT_REPEAT,
                        help='Number of rollout responses per sample. Default: 8.')
    parser.add_argument('--rollout-keep-min-accuracy', type=float, default=STEP6_ROLLOUT_DEFAULT_KEEP_MIN,
                        help='Minimum average accuracy required to keep a sample. Default: 0.33.')
    parser.add_argument('--rollout-keep-max-accuracy', type=float, default=STEP6_ROLLOUT_DEFAULT_KEEP_MAX,
                        help='Maximum average accuracy allowed to keep a sample. Default: 0.66.')
    parser.add_argument('--rollout-max-tokens', type=int, default=STEP6_ROLLOUT_DEFAULT_TOKEN,
                        help='max_tokens used by the local rollout model. Default: 16384.')
    parser.add_argument('--rollout-temperature', type=float, default=STEP6_ROLLOUT_DEFAULT_TEMP,
                        help='Temperature used by the local rollout model. Default: 0.7.')
    parser.add_argument('--rollout-vllm-workers', type=int, default=STEP6_ROLLOUT_DEFAULT_VLLM_WORKERS,
                        help='Worker count for local rollout asks. Default: 1024.')
    parser.add_argument('--step6-input-parquet', type=str, default=None,
                        help='Input parquet for standalone Step6 mode. By default the script infers the full_nosplit Step5 parquet from output_dir/output_name.')
    parser.add_argument('--step6-output-parquet', type=str, default=None,
                        help='Output parquet for standalone Step6 mode. By default it is inferred from output_dir/output_name or the input parquet.')
    parser.add_argument('--step6-article-limit', type=int, default=10000,
                        help='Number of source articles read in standalone Step6 mode. Default: 10000.')
    parser.add_argument('--step6-items-per-pair', type=int, default=20,
                        help='Number of atomic items generated or recovered per task-version pair in standalone Step6 mode. Default: 20.')
    parser.add_argument('--step6-model', type=str, default='gpt-5.4-mini',
                        help='Model name used for standalone Step6 cache-key reconstruction. Default: gpt-5.4-mini.')
    parser.add_argument('--step6-max-tokens', type=int, default=131072,
                        help='max_tokens used for standalone Step6 cache-key reconstruction. Default: 131072.')
    parser.add_argument('--step6-temperature', type=float, default=0.3,
                        help='Temperature used for standalone Step6 cache-key reconstruction. Default: 0.3.')
    parser.add_argument('--step6-think', type=int, default=1,
                        help='Thinking level used for standalone Step6 cache-key reconstruction. Default: 1.')
    parser.add_argument('--step6-dataset-type', type=str, default='test', choices=['train', 'test'],
                        help='dataset_type written into standalone Step6 outputs. Default: test.')
    parser.add_argument('--step6-reason', dest='step6_reason', action='store_true',
                        help='In standalone Step6 mode, rebuild cache keys with Reason=true and prefer recovering the cached reasoning field into sft_ground_truth.')
    parser.add_argument('--step6-disable-reason', dest='step6_reason', action='store_false',
                        help='Disable Reason=true cache-key reconstruction in standalone Step6 mode and fall back to the legacy fixed mythink stub behavior.')
    parser.add_argument('--step6-exclude-existing-step6-parquet', type=str, default=None,
                        help='Exclude articles already present in an existing Step6 parquet by extra_info.stable_article_id and export only the incremental remainder.')
    parser.add_argument('--step6-record-workers', type=int, default=min(48, os.cpu_count() or 8),
                        help='Number of worker processes used to rebuild article-to-job mappings in standalone Step6 mode. Default: min(48, CPU).')
    parser.add_argument('--step6-record-chunk-articles', type=int, default=64,
                        help='Number of articles per record-rebuild chunk in standalone Step6 mode. Default: 64.')
    parser.add_argument('--step6-export-workers', type=int, default=min(32, os.cpu_count() or 8),
                        help='Number of worker processes used to expand cache-hit jobs into atomic items in standalone Step6 mode. Default: min(32, CPU).')
    parser.add_argument('--step6-export-job-chunk-size', type=int, default=128,
                        help='Number of cache-hit jobs per export chunk in standalone Step6 mode. Default: 128.')
    parser.add_argument('--step6-disable-complexity', action='store_true',
                        help='Disable article, edge, and task complexity computation in standalone Step6 mode.')
    parser.add_argument('--step6-complexity-embedding-enabled', action='store_true',
                        help='Enable embeddings during standalone Step6 complexity computation. Disabled by default because it is slower.')
    parser.add_argument('--step6-cache-prompt-git-ref', type=str, default=None,
                        help='Git ref of the Step6 prompt module used for standalone Step6 cache-key reconstruction. Default: HEAD.')
    parser.add_argument('--step6-fallback-cache-prompt-git-ref', type=str, default='8925968d',
                        help='Fallback git ref used when HEAD yields zero cache hits in standalone Step6 mode. Default: 8925968d.')
    parser.add_argument('--step6-complex-block-size', type=int, default=256,
                        help='Block size used for complex_order generation. Default: 256.')
    parser.add_argument('--step6-random-seed', type=int, default=9999,
                        help='Stable random seed used for random_order generation. Default: 9999.')
    parser.set_defaults(step6_reason=False)
    parser.add_argument('-kg', action='store_true', help='Run Step 1: knowledge-graph extraction.')
    parser.add_argument('-rewrite', action='store_true', help='Run Steps 2-4: article rewriting, final JSON regeneration, and visualization.')
    parser.add_argument('-genqa', action='store_true', help='Generate QA-style training outputs for RL.')
    parser.add_argument('-genkg', action='store_true', help='Generate KG extraction training data. By default this includes reasoning traces in SFT format.')
    parser.add_argument('-gen_plussft', action='store_true', help='Deprecated: generate extra SFT outputs alongside -genqa.')
    parser.add_argument('-genkg_plussft', action='store_true', help='Deprecated: equivalent to -genkg.')
    parser.add_argument('-skip-reason-trace-re-generate', dest='skip_reasoning', action='store_true', 
                        help='Skip LLM-generated reasoning traces and use placeholders instead.')
    parser.add_argument('-real_testset', action='store_true', help='Also generate real-testset artifacts and keep the split_from_same_source_as_train test split.')
    return parser


def build_default_args():
    return argparse.Namespace(
        yaml_config=None,
        input_file=None,
        input_type='auto',
        content_col='content',
        id_col='id',
        id_prefix='',
        input_name=None,
        filter_col=None,
        filter_value=None,
        target_count=None,
        max_tokens=None,
        split_ratio=None,
        seed=None,
        pipeline_chunk_size=None,
        pipeline_max_in_flight=None,
        pipeline_fill_rate_per_sec=None,
        only_check_cache=False,
        individual_step6=False,
        split_step6_to_random_and_complex_2vers=False,
        filter_by_rollout_model=False,
        rollout_model_location=None,
        filter_by_rollout_model_sample_number=1000,
        rollout_filter_output_parquet=None,
        rollout_repeat_number=STEP6_ROLLOUT_DEFAULT_REPEAT,
        rollout_keep_min_accuracy=STEP6_ROLLOUT_DEFAULT_KEEP_MIN,
        rollout_keep_max_accuracy=STEP6_ROLLOUT_DEFAULT_KEEP_MAX,
        rollout_max_tokens=STEP6_ROLLOUT_DEFAULT_TOKEN,
        rollout_temperature=STEP6_ROLLOUT_DEFAULT_TEMP,
        rollout_vllm_workers=STEP6_ROLLOUT_DEFAULT_VLLM_WORKERS,
        step6_input_parquet=None,
        step6_output_parquet=None,
        step6_article_limit=10000,
        step6_items_per_pair=20,
        step6_model='gpt-5.4-mini',
        step6_max_tokens=131072,
        step6_temperature=0.3,
        step6_think=1,
        step6_dataset_type='test',
        step6_record_workers=min(48, os.cpu_count() or 8),
        step6_record_chunk_articles=64,
        step6_export_workers=min(32, os.cpu_count() or 8),
        step6_export_job_chunk_size=128,
        step6_disable_complexity=False,
        step6_complexity_embedding_enabled=False,
        step6_cache_prompt_git_ref=None,
        step6_fallback_cache_prompt_git_ref='8925968d',
        step6_complex_block_size=256,
        step6_random_seed=9999,
        kg=False,
        rewrite=False,
        genqa=False,
        genkg=False,
        gen_plussft=False,
        genkg_plussft=False,
        skip_reasoning=False,
        real_testset=False,
    )


def apply_steps_from_yaml_to_args(args, yaml_config):
    steps = yaml_config.get('steps', {})
    if not steps:
        return

    print("📋 从 YAML 加载执行步骤:")
    if 'kg' in steps:
        args.kg = bool(steps['kg'])
        print(f"  ✓ kg = {args.kg}")
    if 'rewrite' in steps:
        args.rewrite = bool(steps['rewrite'])
        print(f"  ✓ rewrite = {args.rewrite}")
    if 'genqa' in steps:
        args.genqa = bool(steps['genqa'])
        print(f"  ✓ genqa = {args.genqa}")
    if 'genkg' in steps:
        args.genkg = bool(steps['genkg'])
        print(f"  ✓ genkg = {args.genkg}")
    if 'gen_plussft' in steps:
        args.gen_plussft = bool(steps['gen_plussft'])
        print(f"  ✓ gen_plussft = {args.gen_plussft}")
    if 'genkg_plussft' in steps:
        args.genkg_plussft = bool(steps['genkg_plussft'])
        print(f"  ✓ genkg_plussft = {args.genkg_plussft} (Deprecated, merged into genkg)")
    if 'skip_reasoning' in steps:
        yaml_skip_reasoning = bool(steps['skip_reasoning'])
        # 2026-03-19:
        
        
        args.skip_reasoning = bool(args.skip_reasoning or yaml_skip_reasoning)
        print(f"  ✓ skip_reasoning = {args.skip_reasoning}")
    if 'real_testset' in steps:
        args.real_testset = bool(steps['real_testset'])
        print(f"  ✓ real_testset = {args.real_testset}")
    if 'individual_step6' in steps:
        args.individual_step6 = bool(steps['individual_step6'])
        print(f"  ✓ individual_step6 = {args.individual_step6}")
    if 'split_step6_to_random_and_complex_2vers' in steps:
        args.split_step6_to_random_and_complex_2vers = bool(steps['split_step6_to_random_and_complex_2vers'])
        print(f"  ✓ split_step6_to_random_and_complex_2vers = {args.split_step6_to_random_and_complex_2vers}")
    if 'gen_random_and_complex' in steps:
        args.split_step6_to_random_and_complex_2vers = bool(steps['gen_random_and_complex'])
        print(f"  ✓ split_step6_to_random_and_complex_2vers = {args.split_step6_to_random_and_complex_2vers} (legacy key gen_random_and_complex)")
    if 'filter_by_rollout_model' in steps:
        args.filter_by_rollout_model = bool(steps['filter_by_rollout_model'])
        print(f"  ✓ filter_by_rollout_model = {args.filter_by_rollout_model}")


def _load_module_from_path_runtime(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"❌ 无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_step6_exporter_module():
    return _load_module_from_path_runtime(
        STEP6_EXPORTER_PATH,
        f"grid_step6_export_runtime_{os.getpid()}",
    )


def _candidate_step5_generated_parquets():
    return [
        os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_KG_extraction_(full_nosplit_useastrain).parquet"),
        os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_KG_extraction_train(full_nosplit).parquet"),
    ]


def resolve_individual_step6_input_parquet(explicit_path=None):
    if explicit_path:
        resolved = os.path.abspath(str(explicit_path))
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"❌ --step6-input-parquet 不存在: {resolved}")
        return resolved

    candidates = _candidate_step5_generated_parquets()
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "❌ 无法自动推断 Step5 生成的 parquet。"
        f" 已尝试: {candidates}。"
        " 请显式传入 --step6-input-parquet。"
    )


def resolve_individual_step6_output_parquet(explicit_path, input_parquet):
    if explicit_path:
        return os.path.abspath(str(explicit_path))
    if str(OUTPUT_DIR or "").strip() and str(OUTPUT_NAME or "").strip():
        return os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_step6_easyreward_compute_realtest.parquet")
    stem, _ = os.path.splitext(str(input_parquet))
    return stem + "__step6_easyreward_compute_realtest.parquet"


def _load_extra_info_dict(extra_info_value):
    if isinstance(extra_info_value, dict):
        return copy.deepcopy(extra_info_value)
    if isinstance(extra_info_value, str):
        try:
            parsed = json_repair.loads(extra_info_value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            parsed = json.loads(extra_info_value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _extract_step6_question_complexity_from_loaded(extra_info):
    try:
        raw_question_complexity = float(extra_info.get("step6_题目复杂度", 0.0) or 0.0)
    except Exception:
        raw_question_complexity = 0.0

    if raw_question_complexity > 0.0:
        return raw_question_complexity

    
    
    gold_edge_payload = extra_info.get("step6_gold_edge复杂度") or {}
    positive_edge_scores = []
    for score in gold_edge_payload.get("gold_edge_complexity_scores") or []:
        try:
            score_value = float(score)
        except Exception:
            continue
        if score_value > 0.0:
            positive_edge_scores.append(score_value)
    if positive_edge_scores:
        return float(sum(positive_edge_scores) / len(positive_edge_scores))

    # 2026-03-29:
    
    
    
    article_complexity_payload = extra_info.get("文章复杂度") or {}
    if isinstance(article_complexity_payload, dict):
        for key_name in ("final_complexity_score", "base_complexity_score"):
            try:
                score_value = float(article_complexity_payload.get(key_name, 0.0) or 0.0)
            except Exception:
                score_value = 0.0
            if score_value > 0.0:
                return score_value

    try:
        article_complexity = float(extra_info.get("文章复杂度分数", 0.0) or 0.0)
        if article_complexity > 0.0:
            return article_complexity
    except Exception:
        pass
    return 0.0


def _extract_step6_question_complexity(extra_info_value):
    extra_info = _load_extra_info_dict(extra_info_value)
    return _extract_step6_question_complexity_from_loaded(extra_info)


def _extract_step6_reward_mode(ground_truth_value):
    if isinstance(ground_truth_value, dict):
        return str(ground_truth_value.get("reward_mode", "") or "")
    if isinstance(ground_truth_value, str):
        try:
            parsed = json_repair.loads(ground_truth_value)
            if isinstance(parsed, dict):
                return str(parsed.get("reward_mode", "") or "")
        except Exception:
            pass
        try:
            parsed = json.loads(ground_truth_value)
            if isinstance(parsed, dict):
                return str(parsed.get("reward_mode", "") or "")
        except Exception:
            pass
    return ""


def _iter_chunk_ranges(total_size, chunk_size):
    chunk_size = max(1, int(chunk_size))
    for start in range(0, int(total_size), chunk_size):
        end = min(int(total_size), start + chunk_size)
        yield start, end


def _extract_step6_sort_metadata_chunk(extra_info_values_chunk, ground_truth_values_chunk=None):
    rows = []
    if ground_truth_values_chunk is None:
        ground_truth_values_chunk = [None] * len(extra_info_values_chunk)
    for extra_info_value, ground_truth_value in zip(extra_info_values_chunk, ground_truth_values_chunk):
        extra_info = _load_extra_info_dict(extra_info_value)
        reward_mode = _extract_step6_reward_mode(ground_truth_value)
        keep_row = reward_mode != "unsupported_step6"
        rows.append(
            (
                float(_extract_step6_question_complexity_from_loaded(extra_info)),
                str(extra_info.get("stable_article_id", "") or ""),
                str(extra_info.get("step6_task_type", "") or ""),
                str(reward_mode or ""),
                bool(keep_row),
            )
        )
    return rows


def _parallel_extract_step6_sort_metadata(extra_info_values, ground_truth_values, worker_count, chunk_size):
    total_size = len(extra_info_values)
    if total_size == 0:
        return np.array([], dtype=np.float64), [], [], [], np.array([], dtype=bool)

    worker_count = max(1, int(worker_count))
    chunk_size = max(1, int(chunk_size))
    chunk_ranges = list(_iter_chunk_ranges(total_size, chunk_size))
    chunk_outputs = [None] * len(chunk_ranges)

    print(
        f"🧮 [Step6 Reorder] 并行解析排序元数据: rows={total_size}, "
        f"workers={worker_count}, chunk_size={chunk_size}, chunks={len(chunk_ranges)}"
    )

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_idx = {
            executor.submit(
                _extract_step6_sort_metadata_chunk,
                extra_info_values[start:end],
                ground_truth_values[start:end],
            ): chunk_idx
            for chunk_idx, (start, end) in enumerate(chunk_ranges)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            chunk_idx = future_to_idx[future]
            chunk_outputs[chunk_idx] = future.result()
            completed += 1
            if completed == len(chunk_ranges) or completed % max(1, len(chunk_ranges) // 10) == 0:
                print(f"🧮 [Step6 Reorder] 排序元数据进度: {completed}/{len(chunk_ranges)} chunks")

    complexities = np.empty(total_size, dtype=np.float64)
    stable_article_ids = [""] * total_size
    task_types = [""] * total_size
    reward_modes = [""] * total_size
    keep_mask = np.empty(total_size, dtype=bool)
    cursor = 0
    for chunk_rows in chunk_outputs:
        for complexity, stable_article_id, task_type, reward_mode, keep_row in chunk_rows:
            complexities[cursor] = float(complexity)
            stable_article_ids[cursor] = stable_article_id
            task_types[cursor] = task_type
            reward_modes[cursor] = reward_mode
            keep_mask[cursor] = bool(keep_row)
            cursor += 1

    return complexities, stable_article_ids, task_types, reward_modes, keep_mask


def _build_step6_reordered_output_path(base_parquet_path, order_suffix):
    stem, ext = os.path.splitext(str(base_parquet_path))
    return f"{stem}_{order_suffix}{ext or '.parquet'}"


def _decorate_step6_extra_info_for_order(
    extra_info_value,
    *,
    order_mode,
    order_position,
    dataset_type_override,
    block_size,
    random_seed,
    block_index,
    within_block_index,
    block_mean_question_complexity,
    effective_question_complexity=None,
):
    extra_info = _load_extra_info_dict(extra_info_value)
    if effective_question_complexity is not None:
        original_question_complexity = extra_info.get("step6_题目复杂度", 0.0)
        extra_info["step6_题目复杂度_原始值"] = original_question_complexity
        extra_info["step6_题目复杂度"] = float(effective_question_complexity)
        extra_info["step6_题目复杂度_排序有效值"] = float(effective_question_complexity)
    extra_info["step6_order_mode"] = str(order_mode)
    # 2026-03-23:
    
    
    
    
    if dataset_type_override:
        extra_info["dataset_type"] = str(dataset_type_override)
    extra_info["step6_order_position"] = int(order_position)
    extra_info["step6_complex_block_size"] = int(block_size)
    extra_info["step6_random_seed"] = int(random_seed)
    extra_info["step6_block_index"] = int(block_index)
    extra_info["step6_within_block_index"] = int(within_block_index)
    extra_info["step6_block_mean_question_complexity"] = float(block_mean_question_complexity)
    return json.dumps(extra_info, ensure_ascii=False)


def _write_step6_reordered_dataset(df, output_parquet_path):
    output_dir = os.path.dirname(str(output_parquet_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_parquet_path, index=False)
    print(f"✅ [Step6 Reorder] 已保存: {output_parquet_path} | rows={len(df)}")


def _decorate_step6_extra_info_chunk(
    extra_info_values_chunk,
    *,
    order_mode,
    start_position,
    dataset_type_override,
    block_size,
    random_seed,
    effective_question_complexities_chunk,
    block_indices_chunk,
    block_mean_values_chunk,
):
    decorated = []
    for local_index, extra_info_value in enumerate(extra_info_values_chunk):
        decorated.append(
            _decorate_step6_extra_info_for_order(
                extra_info_value=extra_info_value,
                order_mode=order_mode,
                order_position=start_position + local_index,
                dataset_type_override=dataset_type_override,
                block_size=block_size,
                random_seed=random_seed,
                block_index=int(block_indices_chunk[local_index]),
                within_block_index=int(local_index % block_size),
                block_mean_question_complexity=float(block_mean_values_chunk[local_index]),
                effective_question_complexity=float(effective_question_complexities_chunk[local_index]),
            )
        )
    return decorated


def _compute_block_mean_values(ordered_complexities, block_size):
    block_means = []
    block_mean_values = np.empty(len(ordered_complexities), dtype=np.float64)
    for start, end in _iter_chunk_ranges(len(ordered_complexities), block_size):
        chunk = ordered_complexities[start:end]
        block_mean = float(np.mean(chunk)) if len(chunk) else 0.0
        block_means.append(block_mean)
        block_mean_values[start:end] = block_mean
    return block_means, block_mean_values


def _parallel_build_decorated_extra_infos(
    ordered_extra_info_values,
    *,
    ordered_complexities,
    order_mode,
    dataset_type_override,
    block_size,
    random_seed,
    worker_count,
    chunk_size,
):
    total_size = len(ordered_extra_info_values)
    if total_size == 0:
        return [], []

    worker_count = max(1, int(worker_count))
    chunk_size = max(1, int(chunk_size))
    row_positions = np.arange(total_size, dtype=np.int64)
    block_indices = row_positions // block_size
    block_means, block_mean_values = _compute_block_mean_values(np.asarray(ordered_complexities, dtype=np.float64), block_size=block_size)

    chunk_ranges = list(_iter_chunk_ranges(total_size, chunk_size))
    chunk_outputs = [None] * len(chunk_ranges)
    print(
        f"🧩 [Step6 Reorder] 并行回写 extra_info: mode={order_mode}, rows={total_size}, "
        f"workers={worker_count}, chunk_size={chunk_size}, chunks={len(chunk_ranges)}"
    )

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_idx = {
            executor.submit(
                _decorate_step6_extra_info_chunk,
                ordered_extra_info_values[start:end],
                order_mode=order_mode,
                start_position=start,
                dataset_type_override=dataset_type_override,
                block_size=block_size,
                random_seed=random_seed,
                effective_question_complexities_chunk=np.asarray(ordered_complexities[start:end], dtype=np.float64),
                block_indices_chunk=block_indices[start:end],
                block_mean_values_chunk=block_mean_values[start:end],
            ): chunk_idx
            for chunk_idx, (start, end) in enumerate(chunk_ranges)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            chunk_idx = future_to_idx[future]
            chunk_outputs[chunk_idx] = future.result()
            completed += 1
            if completed == len(chunk_ranges) or completed % max(1, len(chunk_ranges) // 10) == 0:
                print(f"🧩 [Step6 Reorder] extra_info 回写进度: {completed}/{len(chunk_ranges)} chunks")

    decorated_extra_infos = []
    for chunk_output in chunk_outputs:
        decorated_extra_infos.extend(chunk_output)
    return decorated_extra_infos, block_means


def _write_reordered_parquet_via_tmp_arrow(
    *,
    base_table,
    ordered_indices,
    decorated_extra_infos,
    tmp_output_path,
    final_output_path,
):
    ordered_index_array = pa.array([int(x) for x in ordered_indices], type=pa.int64())
    reordered_table = base_table.take(ordered_index_array)
    extra_info_col_idx = reordered_table.schema.get_field_index("extra_info")
    reordered_table = reordered_table.set_column(
        extra_info_col_idx,
        "extra_info",
        pa.array(decorated_extra_infos),
    )
    pq.write_table(
        reordered_table,
        tmp_output_path,
        compression="snappy",
        use_dictionary=True,
        row_group_size=65536,
    )
    os.makedirs(os.path.dirname(str(final_output_path)), exist_ok=True)
    shutil.move(tmp_output_path, final_output_path)
    print(f"✅ [Step6 Reorder] 已保存: {final_output_path} | rows={reordered_table.num_rows}")


def _cast_arrow_string_columns_to_large_string(table):
    target_fields = []
    need_cast = False
    for field in table.schema:
        if pa.types.is_string(field.type):
            need_cast = True
            target_fields.append(pa.field(field.name, pa.large_string(), nullable=field.nullable, metadata=field.metadata))
        else:
            target_fields.append(field)
    if not need_cast:
        return table
    print("🔧 [Step6 Reorder] 检测到大字符串列，统一转为 large_string 以避免 Arrow offset overflow")
    return table.cast(pa.schema(target_fields))


def _save_step6_reorder_plot(random_block_means, complex_block_means, png_path, block_size):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(random_block_means)), random_block_means, label="random_order", alpha=0.8)
    ax.plot(range(len(complex_block_means)), complex_block_means, label="complex_order", alpha=0.9)
    ax.set_title(f"Step6 Block Mean Question Complexity (block_size={block_size})")
    ax.set_xlabel("Block Index")
    ax.set_ylabel("Mean Question Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"🖼️ [Step6 Reorder] 已保存可视化: {png_path}")


def run_step6_random_and_complex_generation(args, base_step6_parquet_path=None):
    if args.yaml_config:
        load_yaml_config(args.yaml_config)

    if base_step6_parquet_path:
        base_parquet = os.path.abspath(str(base_step6_parquet_path))
    else:
        if args.step6_output_parquet and os.path.exists(str(args.step6_output_parquet)):
            base_parquet = os.path.abspath(str(args.step6_output_parquet))
        else:
            inferred_input = resolve_individual_step6_input_parquet(args.step6_input_parquet)
            base_parquet = resolve_individual_step6_output_parquet(args.step6_output_parquet, inferred_input)
        if not os.path.exists(base_parquet):
            raise FileNotFoundError(
                f"❌ 基础 Step6 parquet 不存在，无法生成 random/complex 版本: {base_parquet}"
            )

    block_size = max(1, int(args.step6_complex_block_size))
    random_seed = int(args.step6_random_seed)
    reorder_workers = min(64, max(4, os.cpu_count() or 8))
    metadata_chunk_size = 8192
    decorate_chunk_size = 8192

    print(f"🎲 [Step6 Reorder] 读取基础 parquet: {base_parquet}")
    base_table = pq.read_table(base_parquet, memory_map=True)
    if base_table.num_rows == 0:
        raise ValueError(f"❌ 基础 Step6 parquet 为空: {base_parquet}")
    base_table = _cast_arrow_string_columns_to_large_string(base_table)

    extra_info_values = base_table.column("extra_info").to_pylist()
    ground_truth_values = base_table.column("ground_truth").to_pylist()
    question_complexities, stable_article_ids, step6_task_types, reward_modes, keep_mask = _parallel_extract_step6_sort_metadata(
        extra_info_values,
        ground_truth_values,
        worker_count=reorder_workers,
        chunk_size=metadata_chunk_size,
    )

    unsupported_rows = int(np.sum(~keep_mask))
    if unsupported_rows > 0:
        print(
            f"🚮 [Step6 Reorder] 检测到 unsupported_step6 样本 {unsupported_rows} 条，"
            "在 random/complex split 阶段直接过滤，不重生 base parquet。"
        )
        keep_indices = np.nonzero(keep_mask)[0].astype(np.int64)
        ordered_keep_array = pa.array([int(x) for x in keep_indices], type=pa.int64())
        base_table = base_table.take(ordered_keep_array)
        extra_info_values = [extra_info_values[int(idx)] for idx in keep_indices]
        question_complexities = question_complexities[keep_indices]
        stable_article_ids = [stable_article_ids[int(idx)] for idx in keep_indices]
        step6_task_types = [step6_task_types[int(idx)] for idx in keep_indices]
        reward_modes = [reward_modes[int(idx)] for idx in keep_indices]
        print(f"✅ [Step6 Reorder] 过滤后保留样本数: {base_table.num_rows}")

    sort_df = pd.DataFrame(
        {
            "__orig_idx": np.arange(len(extra_info_values), dtype=np.int64),
            "__question_complexity": question_complexities,
            "__stable_article_id": stable_article_ids,
            "__step6_task_type": step6_task_types,
        }
    )

    rng = np.random.default_rng(random_seed)
    random_order_indices = rng.permutation(len(extra_info_values)).astype(np.int64)
    complex_order_indices = sort_df.sort_values(
        by=["__question_complexity", "__stable_article_id", "__step6_task_type"],
        ascending=[True, True, True],
        kind="mergesort",
    )["__orig_idx"].to_numpy(dtype=np.int64)

    random_output = _build_step6_reordered_output_path(base_parquet, "random_order")
    complex_output = _build_step6_reordered_output_path(base_parquet, "complex_order")
    summary_path = _build_step6_reordered_output_path(base_parquet, "random_and_complex_summary").replace(".parquet", ".json")
    plot_path = _build_step6_reordered_output_path(base_parquet, "random_and_complex_block_means").replace(".parquet", ".png")
    split_dataset_type = "train"

    with tempfile.TemporaryDirectory(prefix="step6_reorder_", dir="/tmp") as tmp_dir:
        print(f"🧊 [Step6 Reorder] 使用 /tmp staging 目录: {tmp_dir}")
        ordered_random_extra_infos = [extra_info_values[int(idx)] for idx in random_order_indices]
        ordered_complex_extra_infos = [extra_info_values[int(idx)] for idx in complex_order_indices]
        ordered_random_complexities = question_complexities[random_order_indices]
        ordered_complex_complexities = question_complexities[complex_order_indices]

        random_decorated_extra_infos, random_block_means = _parallel_build_decorated_extra_infos(
            ordered_random_extra_infos,
            ordered_complexities=ordered_random_complexities,
            order_mode="random_order",
            dataset_type_override=split_dataset_type,
            block_size=block_size,
            random_seed=random_seed,
            worker_count=reorder_workers,
            chunk_size=decorate_chunk_size,
        )
        complex_decorated_extra_infos, complex_block_means = _parallel_build_decorated_extra_infos(
            ordered_complex_extra_infos,
            ordered_complexities=ordered_complex_complexities,
            order_mode="complex_order",
            dataset_type_override=split_dataset_type,
            block_size=block_size,
            random_seed=random_seed,
            worker_count=reorder_workers,
            chunk_size=decorate_chunk_size,
        )

        _write_reordered_parquet_via_tmp_arrow(
            base_table=base_table,
            ordered_indices=random_order_indices,
            decorated_extra_infos=random_decorated_extra_infos,
            tmp_output_path=os.path.join(tmp_dir, os.path.basename(random_output)),
            final_output_path=random_output,
        )
        _write_reordered_parquet_via_tmp_arrow(
            base_table=base_table,
            ordered_indices=complex_order_indices,
            decorated_extra_infos=complex_decorated_extra_infos,
            tmp_output_path=os.path.join(tmp_dir, os.path.basename(complex_output)),
            final_output_path=complex_output,
        )

    summary = {
        "base_parquet": base_parquet,
        "random_output_parquet": random_output,
        "complex_output_parquet": complex_output,
        "rows": int(base_table.num_rows),
        "dataset_type_override": split_dataset_type,
        "filtered_unsupported_step6_rows": int(unsupported_rows),
        "block_size": int(block_size),
        "random_seed": int(random_seed),
        "random_block_count": int(len(random_block_means)),
        "complex_block_count": int(len(complex_block_means)),
        "complex_block_mean_is_non_decreasing": bool(
            all(complex_block_means[i] <= complex_block_means[i + 1] + 1e-12 for i in range(len(complex_block_means) - 1))
        ),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📝 [Step6 Reorder] 已保存 summary: {summary_path}")

    _save_step6_reorder_plot(random_block_means, complex_block_means, plot_path, block_size)

    _safe_notify_step6_completion(
        f"✅ Step6 random/complex 数据集已完成: "
        f"{os.path.basename(random_output)} | {os.path.basename(complex_output)}"
    )
    return summary


def _load_step6_reward_module():
    return _load_module_from_path_runtime(
        STEP6_REWARD_MODULE_PATH,
        f"grid_step6_reward_runtime_{os.getpid()}",
    )


def resolve_rollout_filter_input_parquet(args):
    if args.step6_input_parquet:
        resolved = os.path.abspath(str(args.step6_input_parquet))
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"❌ rollout 过滤输入 parquet 不存在: {resolved}")
        return resolved

    candidates = []
    try:
        inferred_input = resolve_individual_step6_input_parquet(None)
        base_step6_output = resolve_individual_step6_output_parquet(args.step6_output_parquet, inferred_input)
        candidates.extend([
            _build_step6_reordered_output_path(base_step6_output, "random_order"),
            base_step6_output,
            _build_step6_reordered_output_path(base_step6_output, "complex_order"),
        ])
    except Exception:
        pass

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        "❌ 无法自动推断 rollout 过滤输入 parquet。"
        " 请显式提供 --step6-input-parquet（通常应指向 *_random_order.parquet）。"
    )


def resolve_rollout_filter_output_parquet(args, input_parquet, sample_number):
    if args.rollout_filter_output_parquet:
        return os.path.abspath(str(args.rollout_filter_output_parquet))

    stem, ext = os.path.splitext(str(input_parquet))
    min_tag = int(round(float(args.rollout_keep_min_accuracy) * 100.0))
    max_tag = int(round(float(args.rollout_keep_max_accuracy) * 100.0))
    return f"{stem}_filter-min{min_tag}max{max_tag}-sample{int(sample_number)}{ext or '.parquet'}"


def _normalize_model_hint_text(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _query_rollout_server_model_name(server_name):
    try:
        return tools.llmname(vllm_server_name=server_name, shortname=False)
    except TypeError:
        try:
            return tools.llmname(server_name)
        except Exception:
            return None
    except Exception:
        return None


def _check_rollout_servers_ready(model_location, target_servers):
    expected_hint = os.path.basename(str(model_location or "").rstrip("/"))
    expected_hint_normalized = _normalize_model_hint_text(expected_hint)
    status = {}
    ready = True

    for server_name in target_servers:
        current_name = _query_rollout_server_model_name(server_name)
        status[server_name] = current_name
        if not current_name:
            ready = False
            continue
        if expected_hint_normalized:
            current_normalized = _normalize_model_hint_text(current_name)
            if expected_hint_normalized not in current_normalized:
                ready = False
    return ready, status


def _ensure_rollout_model_ready(model_location, target_servers):
    if not model_location:
        raise ValueError("❌ --filter-by-rollout-model 模式需要提供 --rollout-model-location")

    model_location = os.path.abspath(str(model_location))
    if not os.path.exists(model_location):
        raise FileNotFoundError(f"❌ rollout 模型路径不存在: {model_location}")

    ready, status = _check_rollout_servers_ready(model_location, target_servers)
    print(f"🛰️ [Rollout Filter] 当前本地模型状态: {status}")
    if ready:
        print("✅ [Rollout Filter] 三台服务器已经加载了目标 rollout 模型，直接复用。")
        return status

    llm_cmd = ["llm", "--model-path", model_location, "-all", "-y", "--detach"]
    print("🚀 [Rollout Filter] 检测到本地模型未就绪，开始在 super/normal/ultra 上启动 rollout 模型…")
    print(f"🚀 [Rollout Filter] 启动命令: {' '.join(llm_cmd)}")
    start_result = subprocess.run(llm_cmd, check=False, text=True, timeout=1800)
    if start_result.returncode != 0:
        raise RuntimeError(f"❌ 启动 rollout 模型失败，返回码: {start_result.returncode}")

    deadline = time.time() + 1800
    poll_round = 0
    while time.time() < deadline:
        poll_round += 1
        ready, status = _check_rollout_servers_ready(model_location, target_servers)
        print(f"⏳ [Rollout Filter] 等待本地模型就绪: poll={poll_round}, status={status}")
        if ready:
            print("✅ [Rollout Filter] 三台服务器 rollout 模型均已就绪。")
            return status
        time.sleep(15)

    raise TimeoutError(f"❌ rollout 模型在 1800 秒内未就绪: {status}")


def _load_rollout_filter_head_dataframe(input_parquet, sample_number):
    parquet_file = pq.ParquetFile(input_parquet)
    remaining = max(0, int(sample_number))
    tables = []
    for row_group_idx in range(parquet_file.num_row_groups):
        if remaining <= 0:
            break
        table = parquet_file.read_row_group(row_group_idx)
        if table.num_rows > remaining:
            table = table.slice(0, remaining)
        tables.append(table)
        remaining -= table.num_rows

    if not tables:
        return pd.DataFrame()

    combined_table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    combined_table = combined_table.slice(0, min(int(sample_number), combined_table.num_rows))
    return combined_table.to_pandas()


def _decorate_rollout_filter_extra_info(
    extra_info_value,
    *,
    rollout_scores,
    rollout_accuracy,
    rollout_model_location,
    rollout_model_names,
    source_row_index,
    repeat_number,
    keep_min,
    keep_max,
    input_parquet,
):
    extra_info = _load_extra_info_dict(extra_info_value)
    step6_task_type = str(extra_info.get("step6_task_type", "") or "")
    step6_prompt_version = str(extra_info.get("step6_prompt_version", "") or "")
    extra_info["step6_rollout_filter_enabled"] = True
    extra_info["step6_rollout_filter_input_parquet"] = str(input_parquet)
    extra_info["step6_rollout_filter_source_row_index"] = int(source_row_index)
    extra_info["step6_rollout_filter_repeat_number"] = int(repeat_number)
    extra_info["step6_rollout_filter_scores"] = [float(score) for score in rollout_scores]
    extra_info["step6_rollout_filter_accuracy"] = float(rollout_accuracy)
    extra_info["step6_rollout_filter_keep_min_accuracy"] = float(keep_min)
    extra_info["step6_rollout_filter_keep_max_accuracy"] = float(keep_max)
    extra_info["step6_rollout_filter_kept"] = bool(float(keep_min) <= float(rollout_accuracy) <= float(keep_max))
    extra_info["step6_rollout_filter_model_location"] = str(rollout_model_location)
    extra_info["step6_rollout_filter_model_names"] = dict(rollout_model_names or {})
    extra_info["step6_rollout_filter_prompt_only"] = True
    extra_info["step6_rollout_filter_ground_truth_only"] = True
    extra_info["step6_rollout_filter_reward_module"] = os.path.basename(STEP6_REWARD_MODULE_PATH)
    extra_info["step6_rollout_filter_task_type"] = step6_task_type
    extra_info["step6_rollout_filter_prompt_version"] = step6_prompt_version
    return json.dumps(extra_info, ensure_ascii=False)


def _save_rollout_filter_plot(accuracies, keep_min, keep_max, png_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes
    ax1.hist(accuracies, bins=30, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax1.axvline(keep_min, color="#DD8452", linestyle="--", label=f"min={keep_min:.2f}")
    ax1.axvline(keep_max, color="#55A868", linestyle="--", label=f"max={keep_max:.2f}")
    ax1.set_title("Rollout Accuracy Histogram")
    ax1.set_xlabel("Mean reward over rollout samples")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    sorted_acc = np.sort(np.asarray(accuracies, dtype=np.float64))
    ax2.plot(sorted_acc, linewidth=1.0, color="#8172B3")
    ax2.axhline(keep_min, color="#DD8452", linestyle="--")
    ax2.axhline(keep_max, color="#55A868", linestyle="--")
    ax2.set_title("Sorted Rollout Accuracy")
    ax2.set_xlabel("Sample rank")
    ax2.set_ylabel("Mean reward")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"🖼️ [Rollout Filter] 已保存可视化: {png_path}")


def run_step6_rollout_filter(args):
    if args.yaml_config:
        load_yaml_config(args.yaml_config)

    input_parquet = resolve_rollout_filter_input_parquet(args)
    sample_number = max(1, int(args.filter_by_rollout_model_sample_number))
    repeat_number = max(1, int(args.rollout_repeat_number))
    keep_min = float(args.rollout_keep_min_accuracy)
    keep_max = float(args.rollout_keep_max_accuracy)
    output_parquet = resolve_rollout_filter_output_parquet(args, input_parquet, sample_number)
    summary_path = output_parquet.replace(".parquet", ".summary.json")
    plot_path = output_parquet.replace(".parquet", ".accuracy.png")

    if keep_min > keep_max:
        raise ValueError("❌ rollout 保留区间非法：min 不能大于 max")

    rollout_model_names = _ensure_rollout_model_ready(
        model_location=args.rollout_model_location,
        target_servers=STEP6_ROLLOUT_DEFAULT_TARGET_SERVERS,
    )

    print(f"📥 [Rollout Filter] 输入 parquet: {input_parquet}")
    print(f"📤 [Rollout Filter] 输出 parquet: {output_parquet}")
    print(
        f"🎛️ [Rollout Filter] sample_number={sample_number}, repeat_number={repeat_number}, "
        f"keep_range=[{keep_min:.2f}, {keep_max:.2f}], rollout_model={args.rollout_model_location}"
    )

    start_time = time.time()
    sample_df = _load_rollout_filter_head_dataframe(input_parquet, sample_number)
    if sample_df.empty:
        raise ValueError(f"❌ rollout 过滤输入为空: {input_parquet}")

    sample_df = sample_df.reset_index(drop=True)
    sample_df["__rollout_source_row_index"] = np.arange(len(sample_df), dtype=np.int64)
    sample_df["step6_task_type"] = sample_df["extra_info"].map(
        lambda value: str(_load_extra_info_dict(value).get("step6_task_type", "") or "")
    )
    sample_df["step6_prompt_version"] = sample_df["extra_info"].map(
        lambda value: str(_load_extra_info_dict(value).get("step6_prompt_version", "") or "")
    )
    actual_sample_count = len(sample_df)
    print(f"🧮 [Rollout Filter] 实际载入样本数: {actual_sample_count}")

    def _augment_step6_rollout_prompt_for_repeat(prompt_text, repeat_idx, ground_truth_text):
        prompt_text = str(prompt_text or "")
        try:
            reward_payload = json.loads(str(ground_truth_text or ""))
        except Exception:
            reward_payload = {}
        reward_mode = str(reward_payload.get("reward_mode") or "").strip().lower()
        repeat_id = int(repeat_idx) + 1
        extra_lines = [
            "",
            "Temporary rollout sampling note:",
            f"- Sampling variant id for this attempt: {repeat_id}",
            "- This sampling variant id exists only to diversify repeated attempts.",
            "- Do not mention the sampling variant id in your answer.",
        ]
        if reward_mode == "mcq_xml":
            extra_lines.extend(
                [
                    "- Strict output reminder for multiple-choice items:",
                    "- Inside <myresult>, you must output exactly one tag in the form <mychoice>X</mychoice>.",
                    "- X must be exactly one uppercase option letter from A to J.",
                    "- Do not output the option content, entity name, relation name, explanation, or any text other than the single option letter.",
                ]
            )
        return prompt_text.rstrip() + "\n" + "\n".join(extra_lines) + "\n"

    prompt_list = []
    sample_mapping = []
    prompt_series = sample_df["prompt"].tolist()
    ground_truth_series = sample_df["ground_truth"].tolist()
    print(
        "🔧 [Rollout Filter] 临时 prompt patch 已启用：为每次重复回答追加 Sampling variant id=1..K，"
        "并对 MCQ 追加“<mychoice> 里只能输出 A-J 单个大写字母”的硬约束。"
    )
    for row_idx, prompt_text in enumerate(prompt_series):
        prompt_text = str(prompt_text or "")
        ground_truth_text = str(ground_truth_series[row_idx] or "")
        for repeat_idx in range(repeat_number):
            diversified_prompt = _augment_step6_rollout_prompt_for_repeat(
                prompt_text=prompt_text,
                repeat_idx=repeat_idx,
                ground_truth_text=ground_truth_text,
            )
            prompt_list.append([{"role": "user", "content": diversified_prompt}])
            sample_mapping.append((row_idx, repeat_idx))

    asks_fn = getattr(tools, "asks", None) or getattr(tools, "ask_group_link")
    rollout_inference_model = next(
        (str(model_name) for model_name in rollout_model_names.values() if str(model_name or "").strip()),
        "local",
    )
    print(f"🧠 [Rollout Filter] 钉死 llmname 为真实模型名: {rollout_inference_model}")
    rollout_elapsed_start = time.time()
    original_llmname_fn = tools.llmname

    def _pinned_rollout_llmname(vllm_server_name='super', shortname=True):
        model_name = str(rollout_inference_model)
        if shortname:
            return os.path.basename(model_name.rstrip('/'))
        return model_name

    try:
        
        
        
        
        tools.llmname = _pinned_rollout_llmname
        rollout_outputs = asks_fn(
            prompt_list=prompt_list,
            model="local",
            token=int(args.rollout_max_tokens),
            temp=float(args.rollout_temperature),
            retry=3,
            check_history_cache=False,
            think=0,
            count=True,
            note=f"step6-rollout-filter-{actual_sample_count}x{repeat_number}",
            max_workers_Vllm=int(args.rollout_vllm_workers),
            prompt_send_weight_VllmNotSmartMode={server_name: 1 for server_name in STEP6_ROLLOUT_DEFAULT_TARGET_SERVERS},
        )
    finally:
        tools.llmname = original_llmname_fn
    rollout_elapsed_seconds = time.time() - rollout_elapsed_start
    print(
        f"🤖 [Rollout Filter] rollout 已完成: prompts={len(prompt_list)}, "
        f"elapsed={rollout_elapsed_seconds:.2f}s"
    )

    reward_module = _load_step6_reward_module()
    normalized_outputs = [str(output or "") for output in rollout_outputs]
    repeated_ground_truths = [str(sample_df.iloc[row_idx]["ground_truth"]) for row_idx, _ in sample_mapping]
    repeated_data_sources = [str(sample_df.iloc[row_idx].get("data_source", "grid_dataset")) for row_idx, _ in sample_mapping]
    repeated_extra_infos = []
    for row_idx, _ in sample_mapping:
        row_extra_info = _load_extra_info_dict(sample_df.iloc[row_idx].get("extra_info"))
        
        
        
        
        row_extra_info["step6_prompt_text"] = str(sample_df.iloc[row_idx].get("prompt") or "")
        repeated_extra_infos.append(row_extra_info)
    reward_scores = reward_module.compute_score_kg_batch(
        solution_str=normalized_outputs,
        ground_truth=repeated_ground_truths,
        data_source=repeated_data_sources,
        extra_info=repeated_extra_infos,
        max_workers=min(max(8, os.cpu_count() or 8), len(normalized_outputs)),
    )
    reward_scores = [float(score) for score in reward_scores]

    per_row_scores = [[] for _ in range(actual_sample_count)]
    for (row_idx, _repeat_idx), score in zip(sample_mapping, reward_scores):
        per_row_scores[row_idx].append(float(score))

    rollout_accuracies = []
    kept_row_indices = []
    for row_idx, score_list in enumerate(per_row_scores):
        mean_score = float(np.mean(score_list)) if score_list else 0.0
        rollout_accuracies.append(mean_score)
        if keep_min <= mean_score <= keep_max:
            kept_row_indices.append(row_idx)

    filtered_df = sample_df.iloc[kept_row_indices].copy(deep=True)
    decorated_extra_infos = []
    for row_idx in kept_row_indices:
        decorated_extra_infos.append(
            _decorate_rollout_filter_extra_info(
                sample_df.iloc[row_idx].get("extra_info"),
                rollout_scores=per_row_scores[row_idx],
                rollout_accuracy=rollout_accuracies[row_idx],
                rollout_model_location=args.rollout_model_location,
                rollout_model_names=rollout_model_names,
                source_row_index=int(sample_df.iloc[row_idx]["__rollout_source_row_index"]),
                repeat_number=repeat_number,
                keep_min=keep_min,
                keep_max=keep_max,
                input_parquet=input_parquet,
            )
        )
    if not filtered_df.empty:
        filtered_df.loc[:, "extra_info"] = decorated_extra_infos
    filtered_df = filtered_df.drop(columns=["__rollout_source_row_index"], errors="ignore")

    output_dir = os.path.dirname(output_parquet)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="step6_rollout_filter_", dir="/tmp") as tmp_dir:
        tmp_output_parquet = os.path.join(tmp_dir, os.path.basename(output_parquet))
        filtered_df.to_parquet(tmp_output_parquet, index=False)
        shutil.move(tmp_output_parquet, output_parquet)

    total_elapsed_seconds = time.time() - start_time
    accuracy_array = np.asarray(rollout_accuracies, dtype=np.float64)
    per_type_stats = {}
    grouped = {}
    for row_idx, score in enumerate(rollout_accuracies):
        task_type = str(sample_df.iloc[row_idx].get("step6_task_type", "") or "")
        grouped.setdefault(task_type, []).append(float(score))
    for task_type, score_list in sorted(grouped.items(), key=lambda item: item[0]):
        score_arr = np.asarray(score_list, dtype=np.float64)
        kept_count = int(np.sum((score_arr >= keep_min) & (score_arr <= keep_max)))
        per_type_stats[task_type] = {
            "rows": int(score_arr.size),
            "kept_rows": kept_count,
            "keep_ratio": float(kept_count / score_arr.size) if score_arr.size else 0.0,
            "mean_accuracy": float(np.mean(score_arr)) if score_arr.size else 0.0,
            "min_accuracy": float(np.min(score_arr)) if score_arr.size else 0.0,
            "max_accuracy": float(np.max(score_arr)) if score_arr.size else 0.0,
            "std_accuracy": float(np.std(score_arr)) if score_arr.size else 0.0,
        }
    summary = {
        "input_parquet": input_parquet,
        "output_parquet": output_parquet,
        "rollout_model_location": os.path.abspath(str(args.rollout_model_location)),
        "rollout_model_names": rollout_model_names,
        "sample_number_requested": int(sample_number),
        "sample_number_actual": int(actual_sample_count),
        "repeat_number": int(repeat_number),
        "keep_min_accuracy": float(keep_min),
        "keep_max_accuracy": float(keep_max),
        "total_rollout_prompts": int(len(prompt_list)),
        "kept_rows": int(len(filtered_df)),
        "dropped_rows": int(actual_sample_count - len(filtered_df)),
        "keep_ratio": float(len(filtered_df) / actual_sample_count) if actual_sample_count else 0.0,
        "rollout_elapsed_seconds": float(rollout_elapsed_seconds),
        "total_elapsed_seconds": float(total_elapsed_seconds),
        "mean_accuracy": float(np.mean(accuracy_array)) if accuracy_array.size else 0.0,
        "min_accuracy": float(np.min(accuracy_array)) if accuracy_array.size else 0.0,
        "max_accuracy": float(np.max(accuracy_array)) if accuracy_array.size else 0.0,
        "std_accuracy": float(np.std(accuracy_array)) if accuracy_array.size else 0.0,
        "per_type_stats": per_type_stats,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📝 [Rollout Filter] 已保存 summary: {summary_path}")

    _save_rollout_filter_plot(rollout_accuracies, keep_min, keep_max, plot_path)
    _safe_notify_step6_completion(
        f"✅ Step6 rollout 过滤已完成: {os.path.basename(output_parquet)} | kept={len(filtered_df)}/{actual_sample_count}"
    )
    return summary


def _safe_notify_step6_completion(message):
    message = "" if message is None else str(message)
    
    if not message.strip():
        print("⚠️ [Individual Step6] 检测到空通知正文，已跳过，避免发送 triggered。")
        return
    notify_cmd = [
        "curl",
        "--max-time",
        "10",
        "-H",
        "Content-Type: text/plain; charset=utf-8",
        "-d",
        message,
        "https://ntfy.sh/IpwyTq4gKsCbWHpAgIvHorBJDldJQq1t5RTlNHUtrhW6Kqvnkpv8ZpAc88WRF1Ex",
    ]
    try:
        result = subprocess.run(notify_cmd, check=False, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            print(f"⚠️ [Individual Step6] ntfy 通知失败但主流程已成功: code={result.returncode}")
    except Exception as exc:
        print(f"⚠️ [Individual Step6] ntfy 通知异常但主流程已成功: {exc}")


def run_individual_step6_group(args):
    if args.yaml_config:
        load_yaml_config(args.yaml_config)

    yaml_payload = {}
    if args.yaml_config and yaml is not None:
        with open(args.yaml_config, 'r', encoding='utf-8') as handle:
            yaml_payload = yaml.safe_load(handle) or {}
        if not isinstance(yaml_payload, dict):
            yaml_payload = {}

    yaml_input_parquet = yaml_payload.get('input_parquet')
    yaml_output_json = yaml_payload.get('output_json')
    yaml_reason = yaml_payload.get('reason')
    yaml_exclude_existing = yaml_payload.get('exclude_existing_step6_parquet')
    yaml_article_limit = yaml_payload.get('article_limit')
    yaml_items_per_pair = yaml_payload.get('items_per_pair')
    yaml_model = yaml_payload.get('model')
    yaml_max_tokens = yaml_payload.get('max_tokens')
    yaml_temperature = yaml_payload.get('temperature')
    yaml_think = yaml_payload.get('think')

    resolved_input_arg = args.step6_input_parquet or yaml_input_parquet
    input_parquet = resolve_individual_step6_input_parquet(resolved_input_arg)
    resolved_output_arg = args.step6_output_parquet
    if not resolved_output_arg and yaml_output_json:
        resolved_output_arg = str(Path(str(yaml_output_json)).with_suffix('.parquet'))
    output_parquet = resolve_individual_step6_output_parquet(resolved_output_arg, input_parquet)
    resolved_reason_enabled = bool(args.step6_reason or bool(yaml_reason))
    resolved_exclude_existing = args.step6_exclude_existing_step6_parquet or yaml_exclude_existing
    resolved_article_limit = int(yaml_article_limit) if (args.step6_article_limit == 10000 and yaml_article_limit is not None) else int(args.step6_article_limit)
    resolved_items_per_pair = int(yaml_items_per_pair) if (args.step6_items_per_pair == 20 and yaml_items_per_pair is not None) else int(args.step6_items_per_pair)
    resolved_model = str(yaml_model) if (args.step6_model == 'gpt-5.4-mini' and yaml_model is not None) else str(args.step6_model)
    resolved_max_tokens = int(yaml_max_tokens) if (args.step6_max_tokens == 131072 and yaml_max_tokens is not None) else int(args.step6_max_tokens)
    resolved_temperature = float(yaml_temperature) if (abs(args.step6_temperature - 0.3) < 1e-12 and yaml_temperature is not None) else float(args.step6_temperature)
    resolved_think = int(yaml_think) if (args.step6_think == 1 and yaml_think is not None) else int(args.step6_think)

    print("🧩 [Individual Step6] 已进入独立 Step6 大功能群。")
    print("🧩 [Individual Step6] Step1-Step5 是一个大功能群；Step6 是另一个大功能群。")
    print("🧩 [Individual Step6] 当前不会重跑 Step1-Step5，只会读取 Step5 已生成好的 parquet。")
    print("🧩 [Individual Step6] 当前由主脚本统一调度，但 Step6 导出核心实现源仍是 exporter 模块。")
    print(f"📥 [Individual Step6] Step5 输入 parquet: {input_parquet}")
    print(f"📤 [Individual Step6] Step6 输出 parquet: {output_parquet}")
    print(
        f"🧩 [Individual Step6] reason_enabled={resolved_reason_enabled}, "
        f"exclude_existing_step6_parquet={resolved_exclude_existing or ''}, "
        f"article_limit={resolved_article_limit}, items_per_pair={resolved_items_per_pair}, "
        f"model={resolved_model}, max_tokens={resolved_max_tokens}, temperature={resolved_temperature}, think={resolved_think}"
    )

    exporter_module = _load_step6_exporter_module()
    summary = exporter_module.run_step6_export(
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        article_limit=int(resolved_article_limit),
        items_per_pair=int(resolved_items_per_pair),
        model=str(resolved_model),
        max_tokens=int(resolved_max_tokens),
        temperature=float(resolved_temperature),
        think=int(resolved_think),
        dataset_type=str(args.step6_dataset_type),
        reason_enabled=bool(resolved_reason_enabled),
        exclude_existing_step6_parquet=resolved_exclude_existing,
        record_workers=int(args.step6_record_workers),
        record_chunk_articles=int(args.step6_record_chunk_articles),
        export_workers=int(args.step6_export_workers),
        export_job_chunk_size=int(args.step6_export_job_chunk_size),
        attach_complexity=(not bool(args.step6_disable_complexity)),
        complexity_embedding_enabled=bool(args.step6_complexity_embedding_enabled),
        cache_prompt_git_ref=args.step6_cache_prompt_git_ref,
        fallback_cache_prompt_git_ref=args.step6_fallback_cache_prompt_git_ref,
    )
    _safe_notify_step6_completion(
        f"✅ generate_train_data_pipeline.py Step6 export finished: "
        f"{os.path.basename(output_parquet)} rows={summary.get('exported_rows')}"
    )
    return summary


def apply_cli_overrides_to_runtime(args):
    global INPUT_SOURCES, TARGET_COUNT, MAX_TOKENS, SPLIT_RATIO, SEED
    global PIPELINE_CHUNK_SIZE, PIPELINE_MAX_IN_FLIGHT, PIPELINE_FILL_RATE_PER_SEC, ONLY_CHECK_CACHE

    if args.input_file:
        input_type = "" if str(args.input_type).lower() == "auto" else args.input_type
        cli_paths = []
        for item in args.input_file:
            if isinstance(item, (list, tuple)):
                cli_paths.extend(item)
            else:
                cli_paths.append(item)
        INPUT_SOURCES[:] = [
            build_single_input_source(
                path=path,
                input_type=input_type,
                content_col=args.content_col,
                id_col=args.id_col,
                id_prefix=args.id_prefix,
                name=(args.input_name if len(cli_paths) == 1 else None),
                filter_col=args.filter_col,
                filter_value=args.filter_value,
            )
            for path in cli_paths
        ]
        print(f"📥 命令行覆盖输入源: {len(INPUT_SOURCES)} 个文件")
        for idx, src in enumerate(INPUT_SOURCES):
            print(f"   [{idx+1}] {src['path']} (type={src['type']}, content_col={src['content_col']}, id_col={src['id_col']})")

    if args.target_count is not None:
        TARGET_COUNT = int(args.target_count)
        print(f"🎯 命令行覆盖 TARGET_COUNT = {TARGET_COUNT}")
    if args.max_tokens is not None:
        MAX_TOKENS = int(args.max_tokens)
        print(f"📏 命令行覆盖 MAX_TOKENS = {MAX_TOKENS}")
    if args.split_ratio is not None:
        SPLIT_RATIO = float(args.split_ratio)
        print(f"🧪 命令行覆盖 SPLIT_RATIO = {SPLIT_RATIO}")
    if args.seed is not None:
        SEED = int(args.seed)
        print(f"🎲 命令行覆盖 SEED = {SEED}")
    if args.pipeline_chunk_size is not None:
        PIPELINE_CHUNK_SIZE = int(args.pipeline_chunk_size)
        print(f"🧩 命令行覆盖 PIPELINE_CHUNK_SIZE = {PIPELINE_CHUNK_SIZE}")
    if args.pipeline_max_in_flight is not None:
        PIPELINE_MAX_IN_FLIGHT = int(args.pipeline_max_in_flight)
        print(f"🚦 命令行覆盖 PIPELINE_MAX_IN_FLIGHT = {PIPELINE_MAX_IN_FLIGHT}")
    if args.pipeline_fill_rate_per_sec is not None:
        PIPELINE_FILL_RATE_PER_SEC = float(args.pipeline_fill_rate_per_sec)
        print(f"💧 命令行覆盖 PIPELINE_FILL_RATE_PER_SEC = {PIPELINE_FILL_RATE_PER_SEC}")
    if args.only_check_cache:
        ONLY_CHECK_CACHE = True
        print("🛑 命令行开启 ONLY_CHECK_CACHE = True")


def prepare_runtime_from_args(args):
    yaml_config = {}
    if args.yaml_config:
        yaml_config = load_yaml_config(args.yaml_config)
        apply_steps_from_yaml_to_args(args, yaml_config)

    apply_cli_overrides_to_runtime(args)
    return yaml_config


def normalize_compat_yaml_paths(raw_value):
    normalized = []
    seen = set()

    def _append_one(item):
        text = str(item or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        normalized.append(text)

    if raw_value is None:
        return []
    if isinstance(raw_value, (str, os.PathLike)):
        _append_one(raw_value)
        return normalized
    if isinstance(raw_value, (list, tuple)):
        for item in raw_value:
            if isinstance(item, (list, tuple)):
                for sub_item in item:
                    _append_one(sub_item)
            else:
                _append_one(item)
        return normalized

    _append_one(raw_value)
    return normalized


def run_pipeline_for_current_runtime(args, external_real_test_df=None, existing_generated_df=None, save_outputs=True):
    global ONLY_CHECK_CACHE

    generated_dataset_cache_mode = bool(ONLY_CHECK_CACHE or GENERATED_DATASET_ONLY_CHECK_CACHE)
    real_test_cache_mode = bool(ONLY_CHECK_CACHE or REAL_TESTSET_ONLY_CHECK_CACHE)
    print(f"🧊 生成数据集缓存模式: {generated_dataset_cache_mode}")
    print(f"🧪 Real Test 缓存模式: {real_test_cache_mode}")

    # Determine pipeline flags based on modes
    run_genqa = args.genqa or args.gen_plussft
    
    run_genkg_pipeline = args.genkg or args.genkg_plussft

    fast_finalize_count = 0
    use_fast_finalize_from_disk = False
    if GENERATED_DATASET_FAST_FINALIZE_FROM_DISK and run_genkg_pipeline and not run_genqa:
        fast_finalize_count = detect_fast_finalize_candidates(KG_BASE_DIR)
        if fast_finalize_count > 0:
            use_fast_finalize_from_disk = True
            print(
                f"⚡ [Fast Finalize] 发现 {fast_finalize_count} 个已落盘完成 Step1+Step2 的文章目录；"
                "本轮跳过全量输入扫描，直接从 KG_BASE_DIR 恢复并生成 parquet。"
            )

    print_stage_policy_warnings()

    
    
    if args.kg and os.path.exists(KG_BASE_DIR):
        if generated_dataset_cache_mode or use_fast_finalize_from_disk:
            print(f"🧊 保留已有 KG_BASE_DIR，不清空 {KG_BASE_DIR}")
        else:
            shutil.rmtree(KG_BASE_DIR)
            print(f"🧹 清空了 {KG_BASE_DIR}")
    
    
    
    if save_outputs and os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"🧹 清空了 {OUTPUT_DIR}")

    prioritized_real_test_df_sft = None
    should_prioritize_real_testset = bool(
        REAL_TESTSET_PRIORITY_FIRST
        and args.real_testset
        and external_real_test_df is None
    )
    if should_prioritize_real_testset:
        print("🚨 [Real Test Priority First] 已启用：先处理外部正式测试集，再进入训练侧大流水线。")
        prioritized_real_test_df_sft = load_real_testset(
            REAL_TESTSET_CONFIG["dataset_path"],
            mode=str(REAL_TESTSET_CONFIG.get("mode", "SFT") or "SFT"),
            skip_reasoning=args.skip_reasoning,
            sources=REAL_TESTSET_CONFIG.get("sources", ["all"]),
            sample_size_per_source=REAL_TESTSET_CONFIG.get("sample_size_per_source", 50),
            seed=REAL_TESTSET_CONFIG.get("seed", 42),
            stage5b_config=REAL_TEST_STAGE5B_CONFIG,
            only_check_cache=real_test_cache_mode,
        )
        print(
            f"✅ [Real Test Priority First] 外部正式测试集预处理完成: "
            f"{len(prioritized_real_test_df_sft)} rows"
        )

    
    exclude_content_identities = collect_identity_hashes_from_generated_df(
        existing_generated_df,
        label="旧训练完成池" if existing_generated_df is not None else "",
    )
    article_lookup = None
    if use_fast_finalize_from_disk:
        sampled_texts = []
        print("⚡ [Fast Finalize] sampled_texts 留空，后续 pipeline 直接走磁盘恢复路径。")
    else:
        sampled_payload = load_and_sample_articles(
            exclude_content_identities=exclude_content_identities,
            return_article_lookup=bool(existing_generated_df is not None and not existing_generated_df.empty),
        )
        if isinstance(sampled_payload, tuple):
            sampled_texts, article_lookup = sampled_payload
        else:
            sampled_texts = sampled_payload
    
    original_only_check_cache = ONLY_CHECK_CACHE
    ONLY_CHECK_CACHE = generated_dataset_cache_mode
    print(f"🚦 训练侧/同源切分测试集阶段 ONLY_CHECK_CACHE = {ONLY_CHECK_CACHE}")

    
    final_df, kg_data_df = generate_qa_dataframe_pipeline(
        sampled_texts, 
        run_kg=(False if use_fast_finalize_from_disk else args.kg), 
        run_rewrite=(False if use_fast_finalize_from_disk else args.rewrite), 
        run_genqa=run_genqa,
        run_genkg=run_genkg_pipeline,
        skip_reasoning=args.skip_reasoning
    )

    if run_genkg_pipeline:
        print(f"🧊 [训练侧仅缓存统计] KG SFT 最终保留文章数: {len(kg_data_df)}")
        if existing_generated_df is not None and not existing_generated_df.empty:
            aligned_existing_generated_df = align_generated_dataframe_with_source_lookup(
                existing_generated_df,
                article_lookup or {},
                label="旧训练完成池",
            )
            kg_data_df = merge_generated_dataframes(
                aligned_existing_generated_df,
                kg_data_df,
                label="训练完成池合并",
            )
    
    # Load Real Test Set if requested
    real_test_df_sft = prioritized_real_test_df_sft

    if external_real_test_df is not None:
        real_test_df_sft = external_real_test_df.copy(deep=True)
        print(
            "🧩 [Compat] 当前阶段直接复用旧 YAML 已构建好的 real_test_df，"
            "不再重新读取旧 parquet，也不再重新触发 real test 生成。"
        )
        print(f"🧪 [Compat] 复用的 Real Test 样本数: {len(real_test_df_sft)}")
    elif prioritized_real_test_df_sft is not None:
        print(f"🧪 [Real Test Priority First] 复用预处理结果: {len(real_test_df_sft)}")
    elif args.real_testset:
        if run_genkg_pipeline:
            real_test_df_sft = load_real_testset(
                REAL_TESTSET_CONFIG["dataset_path"],
                mode=str(REAL_TESTSET_CONFIG.get("mode", "SFT") or "SFT"),
                skip_reasoning=args.skip_reasoning,
                sources=REAL_TESTSET_CONFIG.get("sources", ["all"]),
                sample_size_per_source=REAL_TESTSET_CONFIG.get("sample_size_per_source", 50),
                seed=REAL_TESTSET_CONFIG.get("seed", 42),
                stage5b_config=REAL_TEST_STAGE5B_CONFIG,
                only_check_cache=real_test_cache_mode,
            )
            print(f"🧪 [Real Test] 最终样本数: {len(real_test_df_sft)}")

    ONLY_CHECK_CACHE = original_only_check_cache
    
    if save_outputs:
        
        if run_genqa and not final_df.empty:
            process_and_save_dataset_locally(
                df=final_df,
                name=OUTPUT_NAME + "_QA_selection",
                output_dir=OUTPUT_DIR,
                train_type="RL",
                data_source="grid_dataset",
                ability="knowledge_graph_extraction",
                split_ratio=SPLIT_RATIO,
            )

        
        if args.gen_plussft and not final_df.empty:
            process_and_save_dataset_locally(
                df=final_df,
                name=OUTPUT_NAME + "_QA_selection",
                output_dir=OUTPUT_DIR,
                train_type="SFT",
                data_source="grid_dataset",
                ability="knowledge_graph_extraction",
                split_ratio=SPLIT_RATIO,
            )
        
        
        if run_genkg_pipeline and (not kg_data_df.empty or (real_test_df_sft is not None and not real_test_df_sft.empty)):
            process_and_save_dataset_locally(
                df=kg_data_df,
                name=OUTPUT_NAME + "_KG_extraction",
                output_dir=OUTPUT_DIR,
                train_type="SFT", 
                data_source="grid_dataset_kg_sft",
                ability="knowledge_graph_extraction",
                split_ratio=SPLIT_RATIO,
                real_test_df=real_test_df_sft
            )
    else:
        print("🧩 当前阶段仅构造内存数据池，不落地 parquet/json。")
    return {
        "final_df": final_df,
        "kg_data_df": kg_data_df,
        "real_test_df_sft": real_test_df_sft,
    }


def peek_compat_config_from_yaml(yaml_path):
    if not yaml_path or not os.path.exists(yaml_path):
        return {
            "compat_old_yaml_path": [],
            "compat_run_old_yaml_first": False,
            "compat_use_old_real_test_df_for_current_output": False,
            "current_skip_reasoning": False,
        }

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    steps = config.get("steps", {}) or {}
    return {
        "compat_old_yaml_path": normalize_compat_yaml_paths(config.get("compat_old_yaml_path")),
        "compat_run_old_yaml_first": bool(config.get("compat_run_old_yaml_first", False)),
        "compat_use_old_real_test_df_for_current_output": bool(
            config.get("compat_use_old_real_test_df_for_current_output", False)
        ),
        "current_skip_reasoning": bool(steps.get("skip_reasoning", False)),
    }


def inflate_saved_generated_dataframe(saved_df):
    if saved_df is None or saved_df.empty or "extra_info" not in saved_df.columns:
        return saved_df

    inflated_df = saved_df.copy(deep=True)

    def _extra_get(extra, key, default=None):
        if isinstance(extra, dict):
            return extra.get(key, default)
        return default

    fields_to_restore = [
        "text_raw_from_file",
        "graph_from_text_raw_from_file",
        "text_fixed_by_revision",
        "source_file",
        "original_id",
        "stable_article_id",
        "sample_order",
        "index",
        "totalnum",
    ]

    for field_name in fields_to_restore:
        needs_restore = field_name not in inflated_df.columns
        if not needs_restore:
            current_series = inflated_df[field_name]
            needs_restore = bool(current_series.isna().all())
        if not needs_restore:
            continue
        inflated_df[field_name] = inflated_df["extra_info"].apply(
            lambda extra: _extra_get(extra, field_name)
        )

    if "ground_truth" not in inflated_df.columns and "sft_ground_truth" in inflated_df.columns:
        inflated_df["ground_truth"] = inflated_df["sft_ground_truth"]
    if "sft_ground_truth" not in inflated_df.columns and "ground_truth" in inflated_df.columns:
        inflated_df["sft_ground_truth"] = inflated_df["ground_truth"]

    return inflated_df


def try_load_completed_compat_outputs_from_yaml(yaml_path):
    if not yaml_path or not os.path.exists(yaml_path):
        return None

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    output_dir = str(config.get("output_dir", "") or "").strip()
    output_name = str(config.get("output_name", "") or "").strip()
    if not output_dir or not output_name:
        return None

    train_full_candidates = [
        os.path.join(
            output_dir,
            f"{output_name}_KG_extraction_(full_nosplit_useastrain).parquet",
        ),
        os.path.join(
            output_dir,
            f"{output_name}_KG_extraction_train(full_nosplit).parquet",
        ),
    ]
    real_test_candidates = [
        os.path.join(
            output_dir,
            f"{output_name}_KG_extraction_(real_useastest).parquet",
        ),
        os.path.join(
            output_dir,
            f"{output_name}_KG_extraction_(real_usetrain).parquet",
        ),
        os.path.join(
            output_dir,
            f"{output_name}_KG_extraction_(real_useastrain).parquet",
        ),
        os.path.join(
            output_dir,
            f"{output_name}_KG_extraction_test(real).parquet",
        ),
    ]

    train_full_path = next((path for path in train_full_candidates if os.path.exists(path)), train_full_candidates[0])
    real_test_path = next((path for path in real_test_candidates if os.path.exists(path)), real_test_candidates[0])

    loaded_any = False
    loaded_train_df = None
    loaded_real_test_df = None

    if os.path.exists(train_full_path):
        loaded_train_df = inflate_saved_generated_dataframe(pd.read_parquet(train_full_path))
        loaded_any = True
        print(f"📦 [Compat Reuse] 复用历史训练 parquet: {train_full_path} ({len(loaded_train_df)} rows)")
    else:
        print(f"📦 [Compat Reuse] 历史训练 parquet 不存在，回退执行旧 YAML: {train_full_path}")

    if os.path.exists(real_test_path):
        loaded_real_test_df = inflate_saved_generated_dataframe(pd.read_parquet(real_test_path))
        loaded_any = True
        print(f"📦 [Compat Reuse] 复用历史 real_test parquet: {real_test_path} ({len(loaded_real_test_df)} rows)")
    else:
        print(f"📦 [Compat Reuse] 历史 real_test parquet 不存在，回退执行旧 YAML: {real_test_path}")

    if not loaded_any:
        return None

    return {
        "kg_data_df": loaded_train_df,
        "real_test_df_sft": loaded_real_test_df,
        "yaml_config": yaml_path,
        "reused_saved_outputs": True,
    }


def run_configuration_with_args(args, phase_label="", external_real_test_df=None, existing_generated_df=None, save_outputs=True):
    print(f"\n{'=' * 20} 🚀 开始阶段: {phase_label or 'Main'} {'=' * 20}")
    restore_runtime_state(DEFAULT_RUNTIME_STATE)
    yaml_config = prepare_runtime_from_args(args)
    results = run_pipeline_for_current_runtime(
        args,
        external_real_test_df=external_real_test_df,
        existing_generated_df=existing_generated_df,
        save_outputs=save_outputs,
    )
    print(f"{'=' * 20} ✅ 阶段完成: {phase_label or 'Main'} {'=' * 20}\n")
    results["yaml_config"] = yaml_config
    return results


if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.individual_step6:
        step6_summary = run_individual_step6_group(args)
        if args.split_step6_to_random_and_complex_2vers:
            run_step6_random_and_complex_generation(
                args,
                base_step6_parquet_path=step6_summary.get("output_parquet"),
            )
        raise SystemExit(0)

    if args.split_step6_to_random_and_complex_2vers:
        run_step6_random_and_complex_generation(args)
        raise SystemExit(0)

    if args.filter_by_rollout_model:
        run_step6_rollout_filter(args)
        raise SystemExit(0)

    compat_meta = peek_compat_config_from_yaml(args.yaml_config) if args.yaml_config else {
        "compat_old_yaml_path": [],
        "compat_run_old_yaml_first": False,
        "compat_use_old_real_test_df_for_current_output": False,
        "current_skip_reasoning": False,
    }

    compat_real_test_df = None
    compat_generated_df = None
    compat_old_yaml_paths = normalize_compat_yaml_paths(compat_meta.get("compat_old_yaml_path"))
    compat_run_old_yaml_first = compat_meta.get("compat_run_old_yaml_first")
    compat_use_old_real_test_df = compat_meta.get("compat_use_old_real_test_df_for_current_output")
    compat_force_skip_reasoning = bool(args.skip_reasoning or compat_meta.get("current_skip_reasoning", False))

    if compat_run_old_yaml_first:
        if not compat_old_yaml_paths:
            raise ValueError("❌ compat_run_old_yaml_first=true 但未提供 compat_old_yaml_path")
        for compat_old_yaml_path in compat_old_yaml_paths:
            if not os.path.exists(compat_old_yaml_path):
                raise FileNotFoundError(f"❌ 兼容旧 YAML 不存在: {compat_old_yaml_path}")

        print(
            "🔁 [Compat] 先执行旧 YAML 阶段，保持旧 YAML 自己的缓存/模型/real-test 配置。"
        )
        print(f"🧩 [Compat] 历史 YAML 列表: {compat_old_yaml_paths}")

        for idx, compat_old_yaml_path in enumerate(compat_old_yaml_paths, start=1):
            compat_args = build_default_args()
            compat_args.yaml_config = compat_old_yaml_path
            compat_args.skip_reasoning = compat_force_skip_reasoning
            compat_results = run_configuration_with_args(
                compat_args,
                phase_label=f"Compat-Old-YAML[{idx}/{len(compat_old_yaml_paths)}]::{os.path.basename(compat_old_yaml_path)}",
                save_outputs=False,
            )

            compat_stage_real_test_df = compat_results.get("real_test_df_sft")
            compat_stage_generated_df = compat_results.get("kg_data_df")

            if compat_stage_generated_df is not None and not compat_stage_generated_df.empty:
                compat_generated_df = merge_generated_dataframes(
                    compat_generated_df,
                    compat_stage_generated_df,
                    label=f"Compat 历史训练池合并[{idx}]",
                )

            if compat_stage_real_test_df is not None and not compat_stage_real_test_df.empty:
                compat_real_test_df = merge_generated_dataframes(
                    compat_real_test_df,
                    compat_stage_real_test_df,
                    label=f"Compat 历史 real_test 合并[{idx}]",
                )

        if compat_use_old_real_test_df:
            if compat_real_test_df is None or compat_real_test_df.empty:
                raise RuntimeError(
                    "❌ 兼容旧 YAML 阶段未成功产出 real_test_df，无法继续用旧 YAML 的 real test 构建当前输出。"
                )
            print(
                f"🧩 [Compat] 已捕获历史 YAML real_test_df: {len(compat_real_test_df)} 条，"
                "当前阶段将直接复用这份内存数据。"
            )
        if compat_generated_df is not None and not compat_generated_df.empty:
            print(
                f"🧩 [Compat] 已捕获历史 YAML 训练完成池: {len(compat_generated_df)} 条，"
                "当前阶段会先从候选池中扣除这些旧完成样本。"
            )

    current_results = run_configuration_with_args(
        args,
        phase_label=f"Current-YAML::{os.path.basename(args.yaml_config) if args.yaml_config else 'CLI'}",
        external_real_test_df=(compat_real_test_df if compat_use_old_real_test_df else None),
        existing_generated_df=compat_generated_df,
    )

    tools.msg("generate_train_data_pipeline.py finished")
