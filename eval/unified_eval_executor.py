# -*- coding: utf-8 -*-
"""
Filename: unified_eval_executor.py
Description: Unified evaluation entrypoint for the packaged GRID artifact.
Keywords: evaluation, YAML, benchmark, cache, judge, reporting

Workflow:
1. Load YAML or CLI overrides and normalize method specifications.
2. Load the canonical benchmark parquet from the packaged runtime input directory.
3. Reuse or refresh per-method generation caches under the packaged evaluation workspace.
4. Run generation and evaluation phases with the configured judge backend.
5. Export structured reports into the packaged result directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
DROPBOX_PATH = Path(os.path.expanduser("~/Dropbox"))
SRC_GRID_DIR = REPO_ROOT / "src" / "grid"
SRC_BASELINE_DIR = REPO_ROOT / "src" / "baseline"
EVAL_ROOT = CURRENT_DIR
APPROACH_DIR = SRC_BASELINE_DIR
EXPERIMENT_YAML_DIR = EVAL_ROOT / "experiment_yaml"
INPUT_ARTICLE_KG_DIR = REPO_ROOT / "benchmark" / "runtime_input"
GENERATED_KG_CONTENT_DIR = EVAL_ROOT / "generated_runs"
EFFECTIVENESS_RESULT_DIR = REPO_ROOT / "result" / "eval_runs"
TOOLS_PROMPT_PATH = DROPBOX_PATH / "tools_prompt.py"
LEGACY_CORE_PATH = EVAL_ROOT / "ultimate_eval_core.py"
KG_REWARD_BACKEND_PATH = DROPBOX_PATH / "VerlDockerFiles/GRID_dataset/reward/kg_reward.py"
DEFAULT_GRID_RL_MODEL_PATH = str(REPO_ROOT / "models" / "task_bank_reward")

for p in [str(DROPBOX_PATH), str(EVAL_ROOT), str(SRC_GRID_DIR), str(SRC_BASELINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from shared_eval_backend import (  # noqa: E402
    DEFAULT_SHARED_QWEN_MODEL_PATH,
    ResourceMonitor,
    SharedLLMBackendManager,
    build_default_shared_backend,
    build_method_init_kwargs,
    create_runtime_context,
    is_native_baseline,
    summarize_latency_logs,
    uses_shared_qwen,
)


def _load_legacy_core():
    spec = importlib.util.spec_from_file_location("ultimate_eval_legacy_core", str(LEGACY_CORE_PATH))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load evaluation core: {LEGACY_CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ultimate_eval_legacy_core"] = module
    spec.loader.exec_module(module)
    return module


LEGACY_CORE = _load_legacy_core()


def _load_shared_kg_reward():
    module_name = "ultimate_eval_shared_kg_reward"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not KG_REWARD_BACKEND_PATH.exists():
        raise ImportError(f"Missing shared reward backend: {KG_REWARD_BACKEND_PATH}")
    spec = importlib.util.spec_from_file_location(module_name, str(KG_REWARD_BACKEND_PATH))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load shared reward backend: {KG_REWARD_BACKEND_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_kg_reward_route(judge_model: str) -> Tuple[List[str], Optional[str]]:
    text = str(judge_model or "").strip()
    lower = text.lower()
    if not lower:
        raise ValueError("judge_model must not be empty")

    if lower in {"super", "normal", "ultra", "api"}:
        return [lower], None
    if lower.startswith("gpt") or lower.startswith("o1") or lower.startswith("o3"):
        return ["api"], text
    if "gemini" in lower:
        return ["api"], text
    if "/" in text:
        return ["api"], text
    return ["api"], text


class KgRewardEvaluator:

    def __init__(
        self,
        judge_model: str = "gpt-5.4-mini",
        prompt_mode: str = "grid_judge_fav",
        token: int = 64 * 1024,
        temp: float = 0.1,
        think: int = 2,
        flex: bool = False,
        runtime_context: Optional[Dict[str, Any]] = None,
    ):
        self.judge_model = judge_model
        self.prompt_mode_requested = prompt_mode
        self.token = token
        self.temp = temp
        self.think = think
        self.flex = flex
        self.runtime_context = runtime_context or {}
        self.judge_backend_name = "kg_reward"
        self.kg_reward = _load_shared_kg_reward()

        prompt_bundle = LEGACY_CORE.resolve_judge_prompt_bundle(prompt_mode)
        self.prompt_mode = str(prompt_bundle["canonical_mode"])
        self.precision_prompt_name = str(prompt_bundle["precision_prompt_name"])
        self.recall_prompt_name = str(prompt_bundle["recall_prompt_name"])
        self.recall_prompt = prompt_bundle["recall_prompt"] or ""
        self.precision_prompt = prompt_bundle["precision_prompt"] or ""
        self.route_mode_list, self.route_model_name = _resolve_kg_reward_route(judge_model)
        self.batch_workers = max(1, int(self.runtime_context.get("kg_reward_batch_workers", 8) or 8))

    def _format_kg(self, relations: List[Dict[str, Any]], prefix: str = "item") -> str:
        items = []
        for idx, rel in enumerate(LEGACY_CORE.normalize_relation_records(relations)):
            items.append(
                {
                    "index": f"{prefix}_{idx}",
                    "sub": rel.get("sub", ""),
                    "rel": rel.get("rel", ""),
                    "obj": rel.get("obj", ""),
                }
            )
        return json.dumps(items, indent=2, ensure_ascii=False)

    def _parse_eval_result(self, response: str, eval_type: str) -> Dict[str, Any]:
        from json_repair import loads as json_repair_loads

        response_text = str(response or "").strip()
        if not response_text:
            return {
                "score": 0.0,
                "TP": 0,
                "error_type": 0,
                "parse_success": False,
                "error": "Empty response",
            }

        parse_errors: List[str] = []
        for strategy, candidate_text in LEGACY_CORE.build_json_parse_candidates(response_text):
            try:
                parsed = json_repair_loads(candidate_text)
            except Exception as exc:
                parse_errors.append(f"{strategy}: {exc}")
                continue

            results = LEGACY_CORE.find_eval_result_list(parsed)
            if results is None:
                parse_errors.append(f"{strategy}: No result list found in parsed object")
                continue

            metrics = self.kg_reward.calculate_metrics_details(results, eval_type=eval_type)
            metrics["parser_strategy"] = strategy
            return metrics

        return {
            "score": 0.0,
            "TP": 0,
            "error_type": 0,
            "parse_success": False,
            "error": "; ".join(parse_errors[:6]) if parse_errors else "No JSON found",
        }

    def _normalize_prediction(self, prediction: Any) -> Dict[str, Any]:
        if isinstance(prediction, list):
            return {"relations": prediction, "entities": []}
        if prediction is None:
            return {"relations": [], "entities": []}
        if isinstance(prediction, dict):
            return {
                "relations": prediction.get("relations", []),
                "entities": prediction.get("entities", []),
            }
        return {"relations": [], "entities": [], "error": f"Unsupported prediction type: {type(prediction).__name__}"}

    def _build_ground_truth_entities(
        self,
        ground_truth_relations: List[Dict[str, Any]],
        entities_from_extra: Optional[List[Dict[str, Any]]],
        source: Optional[str],
    ) -> List[Dict[str, Any]]:
        entities_set = set()
        for rel in ground_truth_relations:
            entities_set.add(rel.get("sub", ""))
            entities_set.add(rel.get("obj", ""))

        final_entities = [{"name": e} for e in sorted(entities_set) if e]
        if source == "grid" and entities_from_extra:
            existing_names = {item.get("name", "") for item in final_entities}
            for extra_e in entities_from_extra:
                if isinstance(extra_e, dict):
                    name = str(extra_e.get("name", "")).strip()
                    payload = dict(extra_e)
                else:
                    name = str(extra_e or "").strip()
                    payload = {"name": name}
                if name and name not in existing_names:
                    final_entities.append(payload)
                    existing_names.add(name)
        return final_entities

    def _build_failed_result(self, ground_truth: List[Dict[str, Any]], prediction: Any, error_text: str) -> Dict[str, Any]:
        pred_dict = self._normalize_prediction(prediction)
        pred_relations = LEGACY_CORE.normalize_relation_records(pred_dict.get("relations", []))
        gt_relations = LEGACY_CORE.normalize_relation_records(ground_truth or [])
        detail = {
            "score": 0.0,
            "TP": 0,
            "total": 0,
            "parse_success": False,
            "error": str(error_text or "kg_reward exception"),
        }
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "pred_count": len(pred_relations),
            "gt_count": len(gt_relations),
            "precision_details": dict(detail),
            "recall_details": dict(detail),
            "precision_prompt": "",
            "recall_prompt": "",
            "precision_response": "",
            "recall_response": "",
            "evaluation_time": 0.0,
            "error": str(error_text or "kg_reward exception"),
        }

    def evaluate(
        self,
        ground_truth: List[Dict],
        prediction: Dict,
        article_text: str = "",
        entities_from_extra: List[Dict] = None,
        source: str = None,
    ) -> Dict:
        pred_dict = self._normalize_prediction(prediction)
        gt_relations = LEGACY_CORE.normalize_relation_records(ground_truth or [])
        pred_relations = LEGACY_CORE.normalize_relation_records(pred_dict.get("relations", []))
        gt_entities = self._build_ground_truth_entities(gt_relations, entities_from_extra, source)
        pred_entities = pred_dict.get("entities", []) or []

        try:
            ground_truth_text = self.kg_reward.build_kg_control_block_text(gt_relations, entities=gt_entities)
            prediction_text = self.kg_reward.build_kg_control_block_text(pred_relations, entities=pred_entities)
            extra_info = {
                "dataset_type": "test",
                "text_fixed_by_revision": (article_text or ""),
                "source_approach_provided_dataset": source or "",
                "judge_backend": self.judge_backend_name,
                "judge_backend_requested_model": self.judge_model,
            }
            result = self.kg_reward.compute_score_kg(
                data_source="formal_eval_host",
                solution_str=prediction_text,
                ground_truth=ground_truth_text,
                extra_info=extra_info,
                return_details=True,
                save_debug_record=False,
                force_mode=self.route_mode_list,
                force_model_name=self.route_model_name,
                force_max_tokens=self.token,
                force_temperature=self.temp,
                force_service_tier="flex" if self.flex else None,
                precision_prompt_text=self.precision_prompt,
                recall_prompt_text=self.recall_prompt,
            )
            result["pred_count"] = len(pred_relations)
            result["gt_count"] = len(gt_relations)
            result["evaluation_time"] = float(result.get("reward_eval_wall_seconds") or 0.0)
            return result
        except Exception as exc:
            return self._build_failed_result(ground_truth, prediction, f"kg_reward evaluate exception: {exc}")

    def batch_evaluate(self, data_list: List[Dict], predictions: List[Dict]) -> List[Dict]:
        total = len(data_list)
        print(f"   🧑‍⚖️ [kg_reward] 准备评估 {total} 个样本...")
        if total <= 0:
            return []
        if total == 1 or self.batch_workers <= 1:
            item = data_list[0]
            pred = predictions[0]
            return [
                self.evaluate(
                    ground_truth=item.get("ground_truth", []),
                    prediction=pred,
                    article_text=item.get("content", ""),
                    entities_from_extra=(item.get("extra_info", {}) or {}).get("实体列表(更新别名后)", []),
                    source=item.get("source_approach_provided_dataset", item.get("source")),
                )
            ]

        results: List[Optional[Dict[str, Any]]] = [None] * total
        max_workers = min(self.batch_workers, total)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ultimate_eval_kg_reward") as executor:
            future_map = {}
            for idx, (item, pred) in enumerate(zip(data_list, predictions)):
                future = executor.submit(
                    self.evaluate,
                    ground_truth=item.get("ground_truth", []),
                    prediction=pred,
                    article_text=item.get("content", ""),
                    entities_from_extra=(item.get("extra_info", {}) or {}).get("实体列表(更新别名后)", []),
                    source=item.get("source_approach_provided_dataset", item.get("source")),
                )
                future_map[future] = (idx, item, pred)

            for future in as_completed(future_map):
                idx, item, pred = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = self._build_failed_result(
                        item.get("ground_truth", []),
                        pred,
                        f"kg_reward batch future exception: {exc}",
                    )

        return [item or self._build_failed_result([], {}, "kg_reward returned empty result") for item in results]


DEFAULT_TOP_LEVEL_CFG: Dict[str, Any] = {
    "是否跳过生成文章KG对而是直接使用现有的parquet文件作为真实值": False,
    "作为真实值的parquet文件名": "",
    "是否跳过生成快进到评估并统计": False,
    "是否跳过生成和评估快进到统计": False,
    "如果生成是否重试生成失败的文章": True,
    "如果生成是否重试已经成功的文章": False,
    "如果评估是否重试评估失败的文章": True,
    "如果评估是否重试已经成功的结果": False,
    "是否从默认的方法级切换到最大化并行流水线的文章级": False,
    "最大本地VLLM生成并行任务数": 64,
    "单任务最大存活分钟数": 120,
    "裁判用LLM模型": "gpt-5.4-mini",
    
    "judge_backend": "kg_reward",
    "裁判用LLM的提示词模式": "grid_judge_fav",
    "裁判用LLM输出token上限": 65536,
    "裁判用LLM温度": 0.1,
    "裁判用LLM思考强度": 2,
    "裁判用LLM是否开启Flex": True,
    "数据集路径": str(
        DROPBOX_PATH
        / "项目GRID-GIT投稿用/train-data/data/real_testset/benchmark_full249.json"
    ),
    "数据来源筛选": "all",
    "每个数据源采样数量": 50,
    "随机种子": 42,
    "输出目录": None,
    "共享LLM后端模型路径": DEFAULT_SHARED_QWEN_MODEL_PATH,
    
    "共享LLM服务器列表": "super",
    "共享LLM是否检查历史缓存": True,
    "本地VLLM流式静默超时秒": 1800,
    "本地VLLM请求总时长上限秒": 14400,
    "资源采样间隔秒": 5.0,
    "多进程数量": 64,
    "方法级评估并行车道数": 2,
}

FLOW_CONTROL_OVERRIDE_KEYS = [
    "是否跳过生成快进到评估并统计",
    "是否跳过生成和评估快进到统计",
    "如果生成是否重试生成失败的文章",
    "如果生成是否重试已经成功的文章",
    "如果评估是否重试评估失败的文章",
    "如果评估是否重试已经成功的结果",
    "是否从默认的方法级切换到最大化并行流水线的文章级",
    "最大本地VLLM生成并行任务数",
    "单任务最大存活分钟数",
]

TIMEOUT_KEYWORDS = (
    "timeout",
    "timed out",
    "read timed out",
    "deadline",
    "stall",
    "超过",
    "超时",
)


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _hash_obj(value: Any, length: int = 16) -> str:
    return hashlib.sha256(_stable_json_dumps(value).encode("utf-8")).hexdigest()[:length]


def _hash_text(text: str, length: int = 16) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:length]


def _safe_name(name: str) -> str:
    text = str(name or "")
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def _compact_path_component(name: str, max_len: int = 96) -> str:
    safe = _safe_name(name)
    if len(safe) <= max_len:
        return safe
    digest = _hash_text(safe, length=12)
    keep = max(8, max_len - len(digest) - 2)
    return f"{safe[:keep]}__{digest}"


def _now_iso() -> str:
    return datetime.now().isoformat()


def _method_stem(method_file: str) -> str:
    return os.path.splitext(os.path.basename(method_file))[0]


def _base_method_name(method_file: str) -> str:
    return _method_stem(method_file).replace("Approach_", "")


def _short_model_name(model_value: Any) -> str:
    text = str(model_value or "").strip()
    if not text:
        return "UnknownModel"
    normalized = text.rstrip("/").rstrip("\\")
    if "/" in normalized or "\\" in normalized:
        return os.path.basename(normalized)
    return normalized


def _resolve_display_model_value(method_params: Dict[str, Any]) -> Any:
    direct_model = method_params.get("model")
    vllm_model_path = method_params.get("vllm_model_path")
    shared_backend = dict(method_params.get("shared_llm_backend") or {})
    shared_backend_model_path = shared_backend.get("model_path")

    direct_model_text = str(direct_model or "").strip().lower()
    if direct_model_text and direct_model_text not in {"local", "none", "null"}:
        return direct_model
    if vllm_model_path:
        return vllm_model_path
    if shared_backend_model_path:
        return shared_backend_model_path
    return direct_model


def _display_number(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return "Unknown"
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
        return format(numeric, "g")
    except Exception:
        return text


def _build_method_display_name(method_file: str, method_params: Dict[str, Any]) -> str:
    base_name = _base_method_name(method_file)
    if is_native_baseline(method_file):
        return f"{base_name}(not LLM-based)"

    model_value = _resolve_display_model_value(method_params)
    token_value = method_params.get("token", 32768)
    temp_value = method_params.get("temp", 0.7)
    return (
        f"{base_name}(LLM-based,"
        f"{_short_model_name(model_value)},"
        f"{_display_number(token_value)},"
        f"{_display_number(temp_value)})"
    )


def _build_method_output_slug(method_file: str, method_params: Dict[str, Any]) -> str:
    base_name = _base_method_name(method_file)
    if is_native_baseline(method_file):
        return f"{base_name}__not-LLM-based"
    model_value = _short_model_name(_resolve_display_model_value(method_params) or "UnknownModel")
    token_value = _display_number(method_params.get("token", 32768))
    temp_value = _display_number(method_params.get("temp", 0.7))
    return _safe_name(f"{base_name}__{model_value}__{token_value}__{temp_value}")


def _notify(message: str) -> None:
    try:
        subprocess.run(
            [
                "curl",
                "--max-time",
                "10",
                "-H",
                "Content-Type: text/plain; charset=utf-8",
                "-d",
                message,
                "https://ntfy.sh/IpwyTq4gKsCbWHpAgIvHorBJDldJQq1t5RTlNHUtrhW6Kqvnkpv8ZpAc88WRF1Ex",
            ],
            timeout=10,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        pass


def _stop_vllm_before_non_llm_phase() -> None:
    stop_fn = getattr(LEGACY_CORE, "stop_shared_vllm_cluster_for_non_llm_phase", None)
    if callable(stop_fn):
        stop_fn()
        _notify("🛑 非LLM阶段开始，已停止三机vLLM以释放GPU")
        return
    print("⚠️ 未找到旧核心中的 stop_shared_vllm_cluster_for_non_llm_phase，跳过 vLLM 清理")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _git_state() -> Tuple[str, bool]:
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(EVAL_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
            or "Unknown"
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(EVAL_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
        return commit, bool(dirty)
    except Exception:
        return "Unknown", True


def _is_timeout_text(text: str) -> bool:
    lower = str(text or "").lower()
    return any(keyword in lower for keyword in TIMEOUT_KEYWORDS)


def _build_sample_id(item: Dict[str, Any], index: int) -> str:
    source = item.get("source_approach_provided_dataset", "unknown")
    extra_info = item.get("extra_info", {}) or {}
    file_name = extra_info.get("file_name", f"item_{index:04d}")
    return f"{source}__{file_name}"


def _normalize_item(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    norm = dict(item)
    norm["ground_truth"] = list(norm.get("ground_truth", []) or [])
    extra_info = dict(norm.get("extra_info", {}) or {})
    source = norm.get("source_approach_provided_dataset", "unknown")
    file_name = extra_info.get("file_name", f"item_{index:04d}")
    extra_info["file_name"] = file_name
    norm["extra_info"] = extra_info
    norm["sample_index"] = index
    norm["sample_id"] = _build_sample_id(norm, index)
    norm["source"] = source
    norm["file_name"] = file_name
    return norm


def _build_input_parquet_name(dataset_path: str, sources: Optional[List[str]], sample_size: Optional[int], seed: int, total: int) -> str:
    source_tag = "all" if not sources else "-".join(sorted(sources))
    sample_tag = "all" if sample_size is None else str(sample_size)
    dataset_tag = _safe_name(Path(dataset_path).stem)[:32]
    return f"articles_kg__dataset-{dataset_tag}__sources-{source_tag}__sample-{sample_tag}__seed-{seed}__total-{total}.parquet"


def _materialize_input_articleandkg(
    data: List[Dict[str, Any]],
    *,
    dataset_path: str,
    sources: Optional[List[str]],
    sample_size: Optional[int],
    seed: int,
) -> str:
    INPUT_ARTICLE_KG_DIR.mkdir(parents=True, exist_ok=True)
    file_name = _build_input_parquet_name(dataset_path, sources, sample_size, seed, len(data))
    parquet_path = INPUT_ARTICLE_KG_DIR / file_name
    rows = []
    for item in data:
        rows.append(
            {
                "sample_index": item["sample_index"],
                "sample_id": item["sample_id"],
                "source": item["source"],
                "file_name": item["file_name"],
                "content": item.get("content", ""),
                "ground_truth_json": _stable_json_dumps(item.get("ground_truth", [])),
                "extra_info_json": _stable_json_dumps(item.get("extra_info", {})),
                "dataset_path": dataset_path,
                "sample_size_per_source": sample_size if sample_size is not None else -1,
                "seed": seed,
            }
        )
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return str(parquet_path)


def _load_input_articleandkg_from_parquet(parquet_file: str) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_file)
    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        extra_info = json.loads(row.get("extra_info_json") or "{}")
        ground_truth = json.loads(row.get("ground_truth_json") or "[]")
        item = {
            "content": row.get("content", ""),
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "source_approach_provided_dataset": row.get("source", "unknown"),
        }
        item = _normalize_item(item, int(row.get("sample_index", len(records))))
        records.append(item)
    return records


def _prepare_input_data(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    dataset_path = cfg["数据集路径"]
    sources = None if str(cfg["数据来源筛选"]).lower() == "all" else [s.strip() for s in str(cfg["数据来源筛选"]).split(",") if s.strip()]
    sample_size = None if str(cfg["每个数据源采样数量"]).lower() == "all" else int(cfg["每个数据源采样数量"])
    seed = int(cfg["随机种子"])

    if cfg["是否跳过生成文章KG对而是直接使用现有的parquet文件作为真实值"]:
        parquet_name = str(cfg["作为真实值的parquet文件名"] or "").strip()
        if not parquet_name:
            raise ValueError("❌ 已要求跳过生成 InputArticleandKG，但未提供 作为真实值的parquet文件名")
        parquet_path = Path(parquet_name)
        if not parquet_path.is_absolute():
            repo_candidate = REPO_ROOT / parquet_name
            input_candidate = INPUT_ARTICLE_KG_DIR / parquet_name
            parquet_path = repo_candidate if repo_candidate.exists() else input_candidate
        if not parquet_path.exists():
            raise FileNotFoundError(f"❌ parquet 文件不存在: {parquet_path}")
        data = _load_input_articleandkg_from_parquet(str(parquet_path))
        if sources and sources != ["all"]:
            source_set = {str(source).strip().lower() for source in sources}
            data = [item for item in data if str(item.get("source", "")).strip().lower() in source_set]
        if sample_size is not None:
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for item in data:
                grouped.setdefault(str(item.get("source", "unknown")), []).append(item)
            import random
            random.seed(seed)
            filtered: List[Dict[str, Any]] = []
            for _, items in grouped.items():
                if sample_size < len(items):
                    filtered.extend(random.sample(items, sample_size))
                else:
                    filtered.extend(items)
            data = filtered
        return data, str(parquet_path)

    raw_data = LEGACY_CORE.load_ultimate_dataset(
        dataset_path=dataset_path,
        sources=sources,
        sample_size=sample_size,
        seed=seed,
    )
    data = [_normalize_item(item, idx) for idx, item in enumerate(raw_data)]
    parquet_path = _materialize_input_articleandkg(
        data,
        dataset_path=dataset_path,
        sources=sources,
        sample_size=sample_size,
        seed=seed,
    )
    return data, parquet_path


def _extract_method_flow_override(raw_spec: Dict[str, Any]) -> Dict[str, Any]:
    legacy_ctrl = dict(raw_spec.get("运行控制") or {})
    new_ctrl = dict(raw_spec.get("override流程控制") or {})
    merged_ctrl = dict(legacy_ctrl)
    merged_ctrl.update(new_ctrl)
    return {
        key: merged_ctrl[key]
        for key in FLOW_CONTROL_OVERRIDE_KEYS
        if key in merged_ctrl
    }


def _map_old_method_spec(raw_spec: Any) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    if isinstance(raw_spec, str):
        file_name = raw_spec
        params: Dict[str, Any] = {}
        run_ctrl: Dict[str, Any] = {}
    elif isinstance(raw_spec, dict):
        file_name = raw_spec.get("文件名") or raw_spec.get("method") or raw_spec.get("name")
        params = dict(raw_spec.get("参数") or {})
        run_ctrl = _extract_method_flow_override(raw_spec)
        if not file_name:
            raise ValueError(f"❌ 方法配置缺少 文件名: {raw_spec}")
    else:
        raise TypeError(f"❌ 不支持的方法配置类型: {type(raw_spec)}")

    old_map: Dict[str, Tuple[str, Dict[str, Any]]] = {
        "方法_qwen-3-4B.py": ("Approach_GRIDSingleInAndOut.py", {"llm_backend": "shared_vllm", "model": DEFAULT_SHARED_QWEN_MODEL_PATH}),
        "方法_gpt-5-nano.py": ("Approach_GRIDSingleInAndOut.py", {"llm_backend": "cloud_api", "model": "gpt-5-nano"}),
        "方法_1000个文章3flash生成的提取题-SFT和RL-3flash打分.py": ("Approach_GRIDSingleInAndOut.py", {"llm_backend": "dedicated_vllm", "model": DEFAULT_GRID_RL_MODEL_PATH}),
        "方法_CTINexus.py": ("Approach_CTINexus.py", {"llm_backend": "shared_vllm", "model": DEFAULT_SHARED_QWEN_MODEL_PATH}),
        "方法_CTINexus_本地vllm.py": ("Approach_CTINexus.py", {"llm_backend": "shared_vllm", "model": DEFAULT_SHARED_QWEN_MODEL_PATH}),
        "方法_CTINexus_2steps.py": ("Approach_CTINexus_2steps.py", {"llm_backend": "shared_vllm", "model": DEFAULT_SHARED_QWEN_MODEL_PATH}),
        "方法_GTIKGResearch.py": ("Approach_GTIKGResearch.py", {"llm_backend": "shared_vllm", "model": DEFAULT_SHARED_QWEN_MODEL_PATH}),
        "方法_GTIKGResearch_本地vllm.py": ("Approach_GTIKGResearch.py", {"llm_backend": "shared_vllm", "model": DEFAULT_SHARED_QWEN_MODEL_PATH}),
        "方法_graphrag.py": ("Approach_graphrag.py", {"llm_backend": "shared_vllm", "model": DEFAULT_SHARED_QWEN_MODEL_PATH}),
        "方法_graphrag_本地vllm.py": ("Approach_graphrag.py", {"llm_backend": "shared_vllm", "model": DEFAULT_SHARED_QWEN_MODEL_PATH}),
        "方法_cognee.py": ("Approach_cognee.py", {"llm_backend": "shared_vllm", "model": DEFAULT_SHARED_QWEN_MODEL_PATH}),
        "方法_graphiti.py": ("Approach_graphiti.py", {"llm_backend": "shared_vllm", "model": DEFAULT_SHARED_QWEN_MODEL_PATH}),
        "方法_llm_akg.py": ("Approach_llm_akg.py", {"llm_backend": "shared_vllm", "model": DEFAULT_SHARED_QWEN_MODEL_PATH}),
        "方法_AttacKG_plus.py": ("Approach_AttacKG_plus.py", {"llm_backend": "shared_vllm", "model": DEFAULT_SHARED_QWEN_MODEL_PATH}),
        "方法_rebel.py": ("Approach_rebel.py", {}),
        "方法_knowgl.py": ("Approach_knowgl.py", {}),
        "方法_UIE.py": ("Approach_UIE.py", {}),
        "方法_EXTRACTOR.py": ("Approach_EXTRACTOR.py", {}),
    }
    normalized_file, default_params = old_map.get(file_name, (file_name, {}))
    merged_params = dict(default_params)
    merged_params.update(params)

    if uses_shared_qwen(normalized_file) and "llm_backend" not in merged_params:
        merged_params["llm_backend"] = "shared_vllm"
        merged_params["model"] = DEFAULT_SHARED_QWEN_MODEL_PATH
    if uses_shared_qwen(normalized_file):
        merged_params.setdefault("token", 32768)
        merged_params.setdefault("temp", 0.7)
    if normalized_file == "Approach_GRIDSingleInAndOut.py":
        merged_params.setdefault("token", 32768)
        merged_params.setdefault("temp", 0.7)

    return normalized_file, merged_params, run_ctrl


def _normalize_method_specs(args, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_methods = cfg.get("生成方法")
    if raw_methods is None:
        if not args.method:
            raise ValueError("❌ 必须在 YAML 的 生成方法 中指定方法列表，或通过 --method 提供单方法")
        raw_methods = [args.method]
    if not isinstance(raw_methods, list):
        raw_methods = [raw_methods]

    specs: List[Dict[str, Any]] = []
    for raw_spec in raw_methods:
        file_name, params, run_ctrl = _map_old_method_spec(raw_spec)
        specs.append(
            {
                "文件名": file_name,
                "参数": params,
                "运行控制": run_ctrl,
                "override流程控制": run_ctrl,
            }
        )
    return specs


def _build_effective_method_cfg(global_cfg: Dict[str, Any], method_spec: Dict[str, Any]) -> Dict[str, Any]:
    effective_cfg = dict(global_cfg)
    override_ctrl = dict(method_spec.get("override流程控制") or method_spec.get("运行控制") or {})
    for key in FLOW_CONTROL_OVERRIDE_KEYS:
        value = override_ctrl.get(key, None)
        if value is not None:
            effective_cfg[key] = value
    return effective_cfg


def _build_paths_for_method(
    method_file: str,
    method_output_slug: str,
    params_hash: str,
    run_id: str,
    yaml_stem: str,
    explicit_output_dir: Optional[str],
    is_multi_method: bool,
    cache_namespace: Optional[str] = None,
) -> Dict[str, Path]:
    method_stem = _method_stem(method_file)
    method_result_root = EFFECTIVENESS_RESULT_DIR / method_stem
    if explicit_output_dir:
        output_dir = Path(explicit_output_dir) / method_stem / method_output_slug if is_multi_method else Path(explicit_output_dir)
    else:
        output_dir = method_result_root / method_output_slug
    generated_root = GENERATED_KG_CONTENT_DIR / method_stem
    canonical_cache_root = generated_root / "cache" / params_hash
    if cache_namespace:
        namespace_safe = _safe_name(str(cache_namespace))
        variant_root = generated_root / "judge_variants" / namespace_safe
        cache_root = variant_root / "cache" / params_hash
        run_root = variant_root / "runs"
    else:
        cache_root = canonical_cache_root
        run_root = generated_root / "runs"
    run_yaml_slug = _compact_path_component(yaml_stem, max_len=48)
    run_method_slug = _compact_path_component(method_output_slug, max_len=120)
    run_dir = run_root / f"{run_id}__{run_yaml_slug}__{run_method_slug}"
    paths = {
        "output_dir": output_dir,
        "generated_root": generated_root,
        "cache_root": cache_root,
        "canonical_cache_root": canonical_cache_root,
        "cache_generated_dir": canonical_cache_root / "generated",
        "cache_evaluated_dir": cache_root / "evaluated",
        "cache_status_dir": cache_root / "status",
        "run_dir": run_dir,
        "run_generated_dir": run_dir / "generated",
        "run_evaluated_dir": run_dir / "evaluated",
        "run_debug_dir": run_dir / "debug",
    }
    for path in paths.values():
        if path.suffix:
            continue
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _build_status_record(
    *,
    sample: Dict[str, Any],
    approach_instance_key: str,
    generation_status: str,
    judge_status: str,
    overall_status: str,
    generate_attempt_count: int,
    judge_attempt_count: int,
    generation_model_resolved: str,
    judge_model_resolved: str,
    end_to_end_seconds: float,
    failure_type: str = "",
    failure_reason: str = "",
    last_execution_mode: str = "方法级",
    last_run_id: str = "",
) -> Dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "sample_index": sample["sample_index"],
        "source": sample["source"],
        "file_name": sample["file_name"],
        "approach_instance_key": approach_instance_key,
        "generation_status": generation_status,
        "judge_status": judge_status,
        "overall_status": overall_status,
        "generate_attempt_count": generate_attempt_count,
        "judge_attempt_count": judge_attempt_count,
        "generation_model_resolved": generation_model_resolved,
        "judge_model_resolved": judge_model_resolved,
        "end_to_end_seconds": end_to_end_seconds,
        "failure_type": failure_type,
        "failure_reason": failure_reason,
        "last_execution_mode": last_execution_mode,
        "last_run_id": last_run_id,
        "updated_at": _now_iso(),
    }


def _maybe_prompt_text(method: Any, content: str) -> Tuple[str, str]:
    try:
        if hasattr(method, "_create_prompt"):
            prompt_data = method._create_prompt(content)
            return _stable_json_dumps(prompt_data), _hash_obj(prompt_data)
    except Exception:
        pass
    return "", ""


def _normalize_generation_entities(value: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    def _visit(item: Any) -> None:
        if item in (None, ""):
            return
        if isinstance(item, dict):
            normalized.append(dict(item))
            return
        if isinstance(item, (list, tuple, set)):
            for sub_item in item:
                _visit(sub_item)
            return
        text = str(item).strip()
        if text:
            normalized.append({"name": text})

    _visit(value)
    return normalized


def _normalize_generation_relations(value: Any) -> List[Dict[str, Any]]:
    try:
        flattened = LEGACY_CORE.normalize_relation_records(value)
        return [dict(item) for item in flattened if isinstance(item, dict)]
    except Exception:
        return []


def _coerce_generation_payload(payload: Any, raw_output: str = "") -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "entities": [],
        "relations": [],
        "raw_output": raw_output,
    }

    if isinstance(payload, dict):
        normalized.update(dict(payload))
    elif isinstance(payload, list):
        relation_items = _normalize_generation_relations(payload)
        entity_items = _normalize_generation_entities(payload)
        if relation_items:
            normalized["relations"] = relation_items
        elif entity_items:
            normalized["entities"] = entity_items

    normalized["entities"] = _normalize_generation_entities(normalized.get("entities"))
    normalized["relations"] = _normalize_generation_relations(normalized.get("relations"))
    normalized["raw_output"] = str(normalized.get("raw_output", "") or raw_output or "")
    return normalized


def _normalize_generation_prediction(pred: Any) -> Dict[str, Any]:
    if isinstance(pred, (dict, list)):
        raw_output = pred.get("raw_output", "") if isinstance(pred, dict) else _stable_json_dumps(pred)
        normalized = _coerce_generation_payload(pred, raw_output=raw_output)
    else:
        raw_output = "" if pred is None else str(pred)
        normalized = _coerce_generation_payload({}, raw_output=raw_output)
        raw_text = raw_output.strip()
        if raw_text:
            parsed_from_text = None
            try:
                parsed_from_text = json.loads(raw_text)
            except Exception:
                try:
                    from json_repair import loads as json_repair_loads

                    parsed_from_text = json_repair_loads(raw_text)
                except Exception:
                    parsed_from_text = None
            if isinstance(parsed_from_text, (dict, list)):
                parsed_payload = _coerce_generation_payload(parsed_from_text, raw_output=raw_output)
                normalized.update(parsed_payload)

    normalized["entities"] = _normalize_generation_entities(normalized.get("entities"))
    normalized["relations"] = _normalize_generation_relations(normalized.get("relations"))
    normalized["raw_output"] = str(normalized.get("raw_output", "") or "")
    return normalized


def _is_generation_parse_success(pred: Any) -> Tuple[bool, str]:
    normalized = _normalize_generation_prediction(pred)
    error_text = str(normalized.get("error", "") or "").strip()
    if error_text:
        return False, error_text

    has_structure = ("entities" in normalized) or ("relations" in normalized) or bool(normalized.get("raw_output"))
    if not has_structure:
        return False, "生成结果缺少可解析结构"

    return True, ""


def _generation_record(
    sample: Dict[str, Any],
    *,
    method_file: str,
    approach_instance_key: str,
    resolved_params: Dict[str, Any],
    execution_mode: str,
    generation_status: str,
    generation_failure_type: str,
    generation_failure_reason: str,
    generation_started_at: str,
    generation_finished_at: str,
    generation_wall_seconds: float,
    generation_attempt_count: int,
    generation_model_requested: str,
    generation_model_resolved: str,
    generation_backend_type: str,
    generation_token: Any,
    generation_temp: Any,
    generation_flex: Any,
    prompt_text: str,
    prompt_hash: str,
    raw_response_text: str,
    parsed_kg_json: Dict[str, Any],
    parser_success: bool,
) -> Dict[str, Any]:
    ground_truth_hash = _hash_obj(sample.get("ground_truth", []))
    return {
        "sample_id": sample["sample_id"],
        "sample_index": sample["sample_index"],
        "source": sample["source"],
        "file_name": sample["file_name"],
        "approach_file": method_file,
        "approach_instance_key": approach_instance_key,
        "resolved_params": resolved_params,
        "execution_mode": execution_mode,
        "generation_status": generation_status,
        "generation_failure_type": generation_failure_type,
        "generation_failure_reason": generation_failure_reason,
        "generation_started_at": generation_started_at,
        "generation_finished_at": generation_finished_at,
        "generation_wall_seconds": generation_wall_seconds,
        "generation_attempt_count": generation_attempt_count,
        "generation_model_requested": generation_model_requested,
        "generation_model_resolved": generation_model_resolved,
        "generation_backend_type": generation_backend_type,
        "generation_backend_server": "",
        "generation_backend_server_group": resolved_params.get("llm_backend", ""),
        "generation_token": generation_token,
        "generation_temp": generation_temp,
        "generation_think": resolved_params.get("think", 2),
        "generation_flex": generation_flex,
        "generation_prompt_name": "grid_kg_single_prompt_maker_very_simple_20260303" if prompt_text else "",
        "generation_prompt_text": prompt_text,
        "generation_prompt_hash": prompt_hash,
        "generation_raw_response_text": raw_response_text,
        "generation_raw_response_hash": _hash_text(raw_response_text),
        "parsed_kg_json": parsed_kg_json,
        "parsed_kg_hash": _hash_obj(parsed_kg_json),
        "parser_name": "method_internal_parser",
        "parser_success": parser_success,
        "input_article_excerpt": sample.get("content", "")[:1000],
        "ground_truth_hash": ground_truth_hash,
    }


def _evaluation_record(
    sample: Dict[str, Any],
    *,
    method_file: str,
    approach_instance_key: str,
    judge_status: str,
    judge_failure_type: str,
    judge_failure_reason: str,
    judge_started_at: str,
    judge_finished_at: str,
    judge_wall_seconds_total: float,
    judge_attempt_count: int,
    judge_model_requested: str,
    judge_model_resolved: str,
    judge_backend_type: str,
    judge_token: Any,
    judge_temp: Any,
    judge_think: Any,
    judge_flex: bool,
    judge_prompt_name_precision: str,
    judge_prompt_name_recall: str,
    scores: Dict[str, Any],
    prediction_hash: str,
    ground_truth_hash: str,
    fallback_triggered: bool,
    fallback_reason: str,
) -> Dict[str, Any]:
    precision_details = scores.get("precision_details", {})
    recall_details = scores.get("recall_details", {})
    return {
        "sample_id": sample["sample_id"],
        "sample_index": sample["sample_index"],
        "source": sample["source"],
        "file_name": sample["file_name"],
        "approach_file": method_file,
        "approach_instance_key": approach_instance_key,
        "judge_status": judge_status,
        "judge_failure_type": judge_failure_type,
        "judge_failure_reason": judge_failure_reason,
        "judge_started_at": judge_started_at,
        "judge_finished_at": judge_finished_at,
        "judge_wall_seconds_total": judge_wall_seconds_total,
        "judge_attempt_count": judge_attempt_count,
        "judge_model_requested": judge_model_requested,
        "judge_model_resolved": judge_model_resolved,
        "judge_backend_type": judge_backend_type,
        "judge_backend_provider_group": judge_model_resolved,
        "judge_token": judge_token,
        "judge_temp": judge_temp,
        "judge_think": judge_think,
        "judge_flex": judge_flex,
        "judge_prompt_name_precision": judge_prompt_name_precision,
        "judge_prompt_name_recall": judge_prompt_name_recall,
        "judge_prompt_text_precision": scores.get("precision_prompt", ""),
        "judge_prompt_text_recall": scores.get("recall_prompt", ""),
        "judge_prompt_hash_precision": _hash_text(scores.get("precision_prompt", "")),
        "judge_prompt_hash_recall": _hash_text(scores.get("recall_prompt", "")),
        "judge_raw_response_precision": scores.get("precision_response", ""),
        "judge_raw_response_recall": scores.get("recall_response", ""),
        "judge_raw_response_hash_precision": _hash_text(scores.get("precision_response", "")),
        "judge_raw_response_hash_recall": _hash_text(scores.get("recall_response", "")),
        "precision_parse_success": precision_details.get("parse_success", False),
        "recall_parse_success": recall_details.get("parse_success", False),
        "precision_parse_error": precision_details.get("error", ""),
        "recall_parse_error": recall_details.get("error", ""),
        "precision": scores.get("precision", 0.0),
        "recall": scores.get("recall", 0.0),
        "f1": scores.get("f1", 0.0),
        "prediction_hash": prediction_hash,
        "ground_truth_hash": ground_truth_hash,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
    }


def _judge_failure_reason_from_scores(scores: Dict[str, Any]) -> str:
    reasons: List[str] = []
    precision_details = scores.get("precision_details", {}) or {}
    recall_details = scores.get("recall_details", {}) or {}
    if not precision_details.get("parse_success", False):
        err = str(precision_details.get("error", "") or "").strip() or "precision parse failed"
        reasons.append(f"precision: {err}")
    if not recall_details.get("parse_success", False):
        err = str(recall_details.get("error", "") or "").strip() or "recall parse failed"
        reasons.append(f"recall: {err}")
    return "; ".join(reasons)


def _build_failed_judge_scores(item: Dict[str, Any], pred: Any, error_text: str) -> Dict[str, Any]:
    pred_dict = pred if isinstance(pred, dict) else _normalize_generation_prediction(pred)
    pred_count = len(LEGACY_CORE.normalize_relation_records(pred_dict.get("relations", [])))
    gt_count = len(LEGACY_CORE.normalize_relation_records(item.get("ground_truth", [])))
    detail = {
        "score": 0.0,
        "TP": 0,
        "total": 0,
        "parse_success": False,
        "error": str(error_text or "judge item exception"),
    }
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "pred_count": pred_count,
        "gt_count": gt_count,
        "precision_details": dict(detail),
        "recall_details": dict(detail),
        "precision_prompt": "",
        "recall_prompt": "",
        "precision_response": "",
        "recall_response": "",
        "evaluation_time": 0.0,
    }


def _batch_evaluate_with_item_isolation(
    evaluator: Any,
    data_list: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool, str]:
    try:
        return evaluator.batch_evaluate(data_list, predictions), False, ""
    except Exception as batch_exc:
        print(f"   ⚠️ Judge 批量评估异常，切换到单样本隔离: {batch_exc}")
        isolated_scores: List[Dict[str, Any]] = []
        failed_indices: List[int] = []
        for idx, (item, pred) in enumerate(zip(data_list, predictions)):
            try:
                score = evaluator.evaluate(
                    ground_truth=item.get("ground_truth", []),
                    prediction=pred if isinstance(pred, dict) else _normalize_generation_prediction(pred),
                    article_text=item.get("content", ""),
                    entities_from_extra=(item.get("extra_info", {}) or {}).get("实体列表(更新别名后)", []),
                    source=item.get("source_approach_provided_dataset", item.get("source")),
                )
            except Exception as item_exc:
                failed_indices.append(idx)
                score = _build_failed_judge_scores(item, pred, f"judge item exception: {item_exc}")
            isolated_scores.append(score)
        reason = (
            f"batch_evaluate exception: {batch_exc}; auto isolated {len(failed_indices)}/{len(data_list)} items"
        )
        return isolated_scores, True, reason


def _judge_batch_with_fallback(evaluator: Any, data_list: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str, bool, str]:
    backend_name = str(getattr(evaluator, "judge_backend_name", "kg_reward")).strip().lower()
    if backend_name != "kg_reward":
        raise RuntimeError(f"formal 目录已停用内置 judge backend，当前不支持: {backend_name}")

    scores, isolated, isolation_reason = _batch_evaluate_with_item_isolation(
        evaluator,
        data_list,
        predictions,
    )
    return scores, str(evaluator.judge_model), False, isolation_reason if isolated else ""


def _load_existing_status(paths: Dict[str, Path], sample_id: str) -> Dict[str, Any]:
    return _read_json(paths["cache_status_dir"] / f"{sample_id}.json") or {
        "generation_status": "missing",
        "judge_status": "missing",
        "overall_status": "failed",
        "generate_attempt_count": 0,
        "judge_attempt_count": 0,
        "failure_type": "",
        "failure_reason": "",
        "generation_model_resolved": "",
        "judge_model_resolved": "",
        "end_to_end_seconds": 0.0,
    }


def _select_generation_items(data: List[Dict[str, Any]], paths: Dict[str, Path], cfg: Dict[str, Any], run_ctrl: Dict[str, Any]) -> List[Dict[str, Any]]:
    force_success = run_ctrl.get("如果生成是否重试已经成功的文章")
    retry_failed = run_ctrl.get("如果生成是否重试生成失败的文章")
    skip_generate_only = run_ctrl.get("是否跳过生成快进到评估并统计")
    skip_generate_and_eval = run_ctrl.get("是否跳过生成和评估快进到统计")
    if force_success is None:
        force_success = cfg["如果生成是否重试已经成功的文章"]
    if retry_failed is None:
        retry_failed = cfg["如果生成是否重试生成失败的文章"]
    if skip_generate_only is None:
        skip_generate_only = cfg["是否跳过生成快进到评估并统计"]
    if skip_generate_and_eval is None:
        skip_generate_and_eval = cfg["是否跳过生成和评估快进到统计"]

    fast_skip_generate = bool(skip_generate_only or skip_generate_and_eval)
    selected: List[Dict[str, Any]] = []
    for item in data:
        sample_id = item["sample_id"]
        status = _load_existing_status(paths, sample_id)
        if fast_skip_generate:
            continue
        if force_success:
            selected.append(item)
            continue
        if retry_failed and status.get("generation_status") != "success":
            selected.append(item)
            continue
        if status.get("generation_status") != "success" or not (paths["cache_generated_dir"] / f"{sample_id}.json").exists():
            selected.append(item)
    return selected


def _select_evaluation_items(data: List[Dict[str, Any]], paths: Dict[str, Path], cfg: Dict[str, Any], run_ctrl: Dict[str, Any]) -> List[Dict[str, Any]]:
    force_success = run_ctrl.get("如果评估是否重试已经成功的结果")
    retry_failed = run_ctrl.get("如果评估是否重试评估失败的文章")
    skip_generate_and_eval = run_ctrl.get("是否跳过生成和评估快进到统计")
    if force_success is None:
        force_success = cfg["如果评估是否重试已经成功的结果"]
    if retry_failed is None:
        retry_failed = cfg["如果评估是否重试评估失败的文章"]
    if skip_generate_and_eval is None:
        skip_generate_and_eval = cfg["是否跳过生成和评估快进到统计"]

    full_skip = bool(skip_generate_and_eval)
    selected: List[Dict[str, Any]] = []
    for item in data:
        sample_id = item["sample_id"]
        status = _load_existing_status(paths, sample_id)
        if full_skip:
            continue
        if force_success:
            selected.append(item)
            continue
        if retry_failed and status.get("judge_status") != "success":
            selected.append(item)
            continue
        if status.get("judge_status") != "success" or not (paths["cache_evaluated_dir"] / f"{sample_id}.json").exists():
            selected.append(item)
    return selected


def _compute_pending_work_counts(
    *,
    data: List[Dict[str, Any]],
    paths: Dict[str, Path],
    cfg: Dict[str, Any],
    run_ctrl: Dict[str, Any],
) -> Tuple[int, int]:
    generate_count = len(_select_generation_items(data, paths, cfg, run_ctrl))
    eval_count = len(_select_evaluation_items(data, paths, cfg, run_ctrl))
    return generate_count, eval_count


def _collect_prediction_from_cache(paths: Dict[str, Path], sample_id: str) -> Optional[Dict[str, Any]]:
    record = _read_json(paths["cache_generated_dir"] / f"{sample_id}.json")
    if not record:
        return None
    parsed_original = record.get("parsed_kg_json", {})
    parsed = _normalize_generation_prediction(parsed_original)
    if not parsed:
        return None

    if _stable_json_dumps(parsed_original) != _stable_json_dumps(parsed):
        repaired_record = dict(record)
        repaired_record["parsed_kg_json"] = parsed
        repaired_record["parsed_kg_hash"] = _hash_obj(parsed)
        parser_success, parser_error = _is_generation_parse_success(parsed)
        repaired_record["parser_success"] = parser_success
        if parser_success and repaired_record.get("generation_failure_type") == "generate_parse_failed":
            repaired_record["generation_status"] = "success"
            repaired_record["generation_failure_type"] = ""
            repaired_record["generation_failure_reason"] = ""
        elif not parser_success:
            repaired_record["generation_status"] = "failed"
            repaired_record["generation_failure_type"] = "generate_parse_failed"
            repaired_record["generation_failure_reason"] = parser_error
        _write_json(paths["cache_generated_dir"] / f"{sample_id}.json", repaired_record)
        _write_json(paths["run_generated_dir"] / f"{sample_id}.json", repaired_record)
    return parsed


def _persist_generation(paths: Dict[str, Path], sample_id: str, generation_record: Dict[str, Any], status_record: Dict[str, Any]) -> None:
    _write_json(paths["cache_generated_dir"] / f"{sample_id}.json", generation_record)
    _write_json(paths["run_generated_dir"] / f"{sample_id}.json", generation_record)
    _write_json(paths["cache_status_dir"] / f"{sample_id}.json", status_record)


def _persist_evaluation(paths: Dict[str, Path], sample_id: str, evaluation_record: Dict[str, Any], status_record: Dict[str, Any]) -> None:
    _write_json(paths["cache_evaluated_dir"] / f"{sample_id}.json", evaluation_record)
    _write_json(paths["run_evaluated_dir"] / f"{sample_id}.json", evaluation_record)
    _write_json(paths["cache_status_dir"] / f"{sample_id}.json", status_record)


def _apply_metric_inclusion_policy(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    for col in ["generation_status", "judge_status", "overall_status", "failure_type", "failure_reason"]:
        if col not in work_df.columns:
            work_df[col] = ""
    for col in ["precision", "recall", "f1", "gt_count", "pred_count", "generate_seconds", "judge_seconds", "end_to_end_seconds"]:
        if col not in work_df.columns:
            work_df[col] = pd.NA
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    metric_included = (
        work_df["generation_status"].fillna("").astype(str).eq("success")
        & work_df["judge_status"].fillna("").astype(str).eq("success")
    )
    work_df["metric_included"] = metric_included
    work_df["metric_skipped_reason"] = ""
    work_df.loc[~metric_included, "metric_skipped_reason"] = (
        work_df.loc[~metric_included, "failure_reason"].fillna("").astype(str).str.strip()
    )
    work_df.loc[
        ~metric_included & work_df["metric_skipped_reason"].eq(""),
        "metric_skipped_reason",
    ] = "generation/judge 未成功，不纳入均值"

    for metric_col in ["precision", "recall", "f1"]:
        work_df.loc[~metric_included, metric_col] = pd.NA
    return work_df


def _build_by_source_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "sample_count",
                "included_metric_count",
                "skipped_metric_count",
                "avg_precision",
                "avg_recall",
                "avg_f1",
                "avg_generate_seconds",
                "avg_judge_seconds",
                "avg_end_to_end_seconds",
            ]
        )

    by_source = (
        df.groupby("source", dropna=False)
        .agg(
            sample_count=("sample_id", "count"),
            included_metric_count=("metric_included", "sum"),
            avg_precision=("precision", "mean"),
            avg_recall=("recall", "mean"),
            avg_f1=("f1", "mean"),
            avg_generate_seconds=("generate_seconds", "mean"),
            avg_judge_seconds=("judge_seconds", "mean"),
            avg_end_to_end_seconds=("end_to_end_seconds", "mean"),
        )
        .reset_index()
    )
    by_source["included_metric_count"] = by_source["included_metric_count"].fillna(0).astype(int)
    by_source["skipped_metric_count"] = by_source["sample_count"] - by_source["included_metric_count"]
    return by_source


def _build_method_summary_from_rows(
    df: pd.DataFrame,
    *,
    method_display_name: str,
    method_file: str,
    approach_instance_key: str,
) -> Dict[str, Any]:
    total_count = int(len(df))
    included_df = df[df["metric_included"]].copy()
    included_count = int(len(included_df))
    failed_count = int(total_count - included_count)
    timeout_count = int(df["failure_type"].fillna("").astype(str).str.contains("timeout", case=False).sum())
    parse_fail_count = int(df["failure_type"].fillna("").astype(str).str.contains("parse", case=False).sum())
    return {
        "method": method_display_name,
        "method_file": method_file,
        "approach_instance_key": approach_instance_key,
        "sample_count": total_count,
        "included_sample_count": included_count,
        "skipped_sample_count": failed_count,
        "avg_precision": float(included_df["precision"].mean()) if included_count else 0.0,
        "avg_recall": float(included_df["recall"].mean()) if included_count else 0.0,
        "avg_f1": float(included_df["f1"].mean()) if included_count else 0.0,
        "avg_generate_seconds": float(df["generate_seconds"].mean()) if total_count else 0.0,
        "avg_judge_seconds": float(df["judge_seconds"].mean()) if total_count else 0.0,
        "avg_end_to_end_seconds": float(df["end_to_end_seconds"].mean()) if total_count else 0.0,
        "failure_rate_total": (failed_count / total_count) if total_count else 0.0,
        "timeout_rate": (timeout_count / total_count) if total_count else 0.0,
        "parse_failure_rate": (parse_fail_count / total_count) if total_count else 0.0,
        "success_count": included_count,
        "failed_count": failed_count,
    }


def _build_method_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "method_file",
                "approach_instance_key",
                "sample_count",
                "included_sample_count",
                "skipped_sample_count",
                "avg_precision",
                "avg_recall",
                "avg_f1",
                "avg_generate_seconds",
                "avg_judge_seconds",
                "avg_end_to_end_seconds",
                "failure_rate_total",
                "timeout_rate",
                "parse_failure_rate",
                "success_count",
                "failed_count",
            ]
        )

    summary_rows: List[Dict[str, Any]] = []
    for (method_name, method_file, approach_instance_key), group_df in df.groupby(
        ["method", "method_file", "approach_instance_key"],
        dropna=False,
        sort=False,
    ):
        summary_rows.append(
            _build_method_summary_from_rows(
                group_df,
                method_display_name=str(method_name),
                method_file="" if pd.isna(method_file) else str(method_file),
                approach_instance_key="" if pd.isna(approach_instance_key) else str(approach_instance_key),
            )
        )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and "avg_f1" in summary_df.columns:
        summary_df = summary_df.sort_values("avg_f1", ascending=False).reset_index(drop=True)
    return summary_df


def _render_method_metric_bar_png(summary_df: pd.DataFrame, out_png: Path, title: str) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        import matplotlib as mpl
    except Exception as exc:
        print(f"⚠️ 无法导入 matplotlib，跳过方法级柱状图: {exc}")
        return None

    if summary_df.empty:
        print("⚠️ 方法级柱状图输入为空，跳过生成")
        return None

    plot_df = summary_df.copy()
    plot_df["display_method"] = plot_df["method"].astype(str)
    x = list(range(len(plot_df)))
    width = 0.24

    cjk_font = None
    for candidate in ("Source Han Sans SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Noto Sans CJK JP"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            cjk_font = candidate
            break
        except Exception:
            continue
    if cjk_font:
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [cjk_font, "DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False

    fig_w = max(12, 0.85 * len(plot_df) + 6)
    fig, ax = plt.subplots(figsize=(fig_w, 7), dpi=220)
    ax.bar([i - width for i in x], plot_df["avg_precision"], width=width, label="Precision")
    ax.bar(x, plot_df["avg_recall"], width=width, label="Recall")
    ax.bar([i + width for i in x], plot_df["avg_f1"], width=width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["display_method"], rotation=28, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️ 已生成方法级柱状图: {out_png}")
    return str(out_png)


def _write_excluded_source_comparison_views(
    detailed_df: pd.DataFrame,
    comparison_dir: Path,
    *,
    excluded_source: str = "casie",
    excluded_sources: Optional[List[str]] = None,
    current_prompt_label: str = "",
    input_parquet_path: str = "",
) -> None:
    if detailed_df.empty or "source" not in detailed_df.columns:
        return

    normalized_sources = [
        str(src).strip()
        for src in (excluded_sources if excluded_sources else [excluded_source])
        if str(src).strip()
    ]
    if not normalized_sources:
        return

    excluded_lower_set = {src.lower() for src in normalized_sources}
    filtered_df = detailed_df[
        ~detailed_df["source"].fillna("").astype(str).str.lower().isin(excluded_lower_set)
    ].copy()
    if filtered_df.empty:
        print(f"⚠️ 去掉来源 {', '.join(normalized_sources)} 后没有剩余样本，跳过额外可视化")
        return

    summary_df = _build_method_summary_frame(filtered_df)
    excluded_upper_names = [src.upper() for src in normalized_sources]
    excluded_label = " + ".join(excluded_upper_names)
    suffix = f"去掉{'和'.join(excluded_upper_names)}后"
    summary_csv = comparison_dir / f"methods_overview__{suffix}.csv"
    filtered_detailed_csv = comparison_dir / f"all_methods_detailed__{suffix}.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    filtered_df.to_csv(filtered_detailed_csv, index=False, encoding="utf-8-sig")
    prompt_prefix = f"当前裁判prompt={current_prompt_label}\n" if str(current_prompt_label).strip() else ""
    _render_method_metric_bar_png(
        summary_df,
        comparison_dir / f"{suffix}_方法表现.png",
        title=f"{prompt_prefix}Method Performance Excluding {excluded_label}",
    )

    try:
        LEGACY_CORE.generate_source_pr_split_heatmap(
            input_csv=str(filtered_detailed_csv),
            output_png=str(comparison_dir / f"按来源_PR双子格热力图__{suffix}.png"),
            methods_overview_csv=str(summary_csv),
            title=f"{prompt_prefix}P/R by Source (+AVG, excluding {excluded_label})",
            input_parquet_path=input_parquet_path or None,
        )
    except Exception as exc:
        print(f"⚠️ 生成去掉 {excluded_label} 后的来源热力图失败: {exc}")


def _write_global_comparison_bundle(
    *,
    comparison_dir: Path,
    all_rows: List[Dict[str, Any]],
    run_id: str,
    input_parquet_path: str,
    cfg: Dict[str, Any],
) -> None:
    if not all_rows:
        return

    comparison_dir.mkdir(parents=True, exist_ok=True)
    detailed_df = _apply_metric_inclusion_policy(pd.DataFrame(all_rows))
    if detailed_df.empty:
        return

    summary_df = _build_method_summary_frame(detailed_df)
    methods_overview_csv = comparison_dir / "methods_overview.csv"
    detailed_csv = comparison_dir / "all_methods_detailed.csv"
    summary_df.to_csv(methods_overview_csv, index=False, encoding="utf-8-sig")
    detailed_df.to_csv(detailed_csv, index=False, encoding="utf-8-sig")

    _render_method_metric_bar_png(
        summary_df,
        comparison_dir / "方法级总览.png",
        title=(
            "Formal Ultimate Eval | "
            f"{cfg.get('裁判用LLM模型', '')} | {cfg.get('裁判用LLM的提示词模式', '')}"
        ),
    )

    best_method = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    _write_json(
        comparison_dir / "comparison_report.json",
        {
            "run_id": run_id,
            "input_articleandkg_file": input_parquet_path,
            "methods": summary_df.to_dict(orient="records"),
            "best_method": best_method,
        },
    )

    try:
        current_prompt_label = str(
            LEGACY_CORE.resolve_judge_prompt_bundle(str(cfg.get("裁判用LLM的提示词模式", ""))).get(
                "canonical_mode",
                str(cfg.get("裁判用LLM的提示词模式", "")),
            )
        )
        LEGACY_CORE.generate_source_pr_split_heatmap(
            input_csv=str(detailed_csv),
            output_png=str(comparison_dir / "按来源_PR双子格热力图.png"),
            methods_overview_csv=str(methods_overview_csv),
            title=f"当前裁判prompt={current_prompt_label}\nP/R by Source (+AVG)",
            input_parquet_path=input_parquet_path,
        )
        _write_excluded_source_comparison_views(
            detailed_df,
            comparison_dir,
            excluded_source="casie",
            current_prompt_label=current_prompt_label,
            input_parquet_path=input_parquet_path,
        )
        _write_excluded_source_comparison_views(
            detailed_df,
            comparison_dir,
            excluded_sources=["casie", "ctinexus"],
            current_prompt_label=current_prompt_label,
            input_parquet_path=input_parquet_path,
        )
    except Exception as exc:
        print(f"⚠️ 增量刷新多方法对比 PNG 失败: {exc}")


def _aggregate_method_outputs(
    data: List[Dict[str, Any]],
    paths: Dict[str, Path],
    *,
    method_file: str,
    method_display_name: str,
    approach_instance_key: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []

    for item in data:
        sample_id = item["sample_id"]
        status = _load_existing_status(paths, sample_id)
        generation_record = _read_json(paths["cache_generated_dir"] / f"{sample_id}.json") or {}
        evaluation_record = _read_json(paths["cache_evaluated_dir"] / f"{sample_id}.json") or {}
        row = {
            "method": method_display_name,
            "method_file": method_file,
            "approach_instance_key": approach_instance_key,
            "sample_id": sample_id,
            "item_id": item["file_name"],
            "source": item["source"],
            "precision": evaluation_record.get("precision", 0.0),
            "recall": evaluation_record.get("recall", 0.0),
            "f1": evaluation_record.get("f1", 0.0),
            "gt_count": len(item.get("ground_truth", [])),
            "pred_count": len((generation_record.get("parsed_kg_json", {}) or {}).get("relations", [])),
            "generation_status": status.get("generation_status", "missing"),
            "judge_status": status.get("judge_status", "missing"),
            "overall_status": status.get("overall_status", "failed"),
            "failure_type": status.get("failure_type", ""),
            "failure_reason": status.get("failure_reason", ""),
            "generation_model_resolved": status.get("generation_model_resolved", ""),
            "judge_model_resolved": status.get("judge_model_resolved", ""),
            "generate_seconds": generation_record.get("generation_wall_seconds", 0.0),
            "judge_seconds": evaluation_record.get("judge_wall_seconds_total", 0.0),
            "end_to_end_seconds": status.get("end_to_end_seconds", 0.0),
        }
        rows.append(row)
        if row["overall_status"] != "success":
            failed_rows.append(
                {
                    "method": method_display_name,
                    "approach_file": method_file,
                    "approach_instance_key": approach_instance_key,
                    "sample_index": item["sample_index"],
                    "sample_id": sample_id,
                    "source": item["source"],
                    "stage": "generate" if row["generation_status"] != "success" else "judge",
                    "failure_type": row["failure_type"],
                    "failure_reason": row["failure_reason"],
                    "generate_attempt_count": status.get("generate_attempt_count", 0),
                    "judge_attempt_count": status.get("judge_attempt_count", 0),
                    "end_to_end_seconds": row["end_to_end_seconds"],
                }
            )

    df = _apply_metric_inclusion_policy(pd.DataFrame(rows))
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "逐条评估结果.csv", index=False, encoding="utf-8-sig")
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(output_dir / "failed_tasks.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["approach_file", "sample_index", "sample_id", "source", "stage", "failure_type", "failure_reason", "generate_attempt_count", "judge_attempt_count", "end_to_end_seconds"]).to_csv(
            output_dir / "failed_tasks.csv", index=False, encoding="utf-8-sig"
        )

    by_source = _build_by_source_stats(df)
    by_source.to_csv(output_dir / "按来源统计.csv", index=False, encoding="utf-8-sig")
    by_source.to_csv(output_dir / "按数据来源统计.csv", index=False, encoding="utf-8-sig")

    summary = _build_method_summary_from_rows(
        df,
        method_display_name=method_display_name,
        method_file=method_file,
        approach_instance_key=approach_instance_key,
    )
    pd.DataFrame([summary]).to_csv(output_dir / "方法级总览.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "总样本数": summary["sample_count"],
                "纳入均值样本数": summary["included_sample_count"],
                "跳过样本数": summary["skipped_sample_count"],
                "平均Precision": summary["avg_precision"],
                "平均Recall": summary["avg_recall"],
                "平均F1": summary["avg_f1"],
                "平均生成耗时秒": summary["avg_generate_seconds"],
                "平均评估耗时秒": summary["avg_judge_seconds"],
                "平均端到端耗时秒": summary["avg_end_to_end_seconds"],
                "失败率": summary["failure_rate_total"],
                "超时率": summary["timeout_rate"],
                "解析失败率": summary["parse_failure_rate"],
            }
        ]
    ).to_csv(output_dir / "总体统计.csv", index=False, encoding="utf-8-sig")

    run_overview = {
        "method": method_display_name,
        "method_file": method_file,
        "approach_instance_key": approach_instance_key,
        "output_dir": str(output_dir),
        "summary": summary,
    }
    _write_json(output_dir / "run_overview.json", run_overview)
    return rows, summary


def _prepare_method_context(
    *,
    method_spec: Dict[str, Any],
    cfg: Dict[str, Any],
    args,
    input_parquet_path: str,
    shared_backend: Dict[str, Any],
    shared_vllm_servers: List[str],
    run_id: str,
    yaml_stem: str,
    is_multi_method: bool,
) -> Dict[str, Any]:
    method_file = method_spec["文件名"]
    method_params = dict(method_spec.get("参数") or {})
    run_ctrl = dict(method_spec.get("运行控制") or {})
    override_flow_ctrl = dict(method_spec.get("override流程控制") or run_ctrl)
    method_display_name = _build_method_display_name(method_file, method_params)
    method_output_slug = _build_method_output_slug(method_file, method_params)
    params_hash = _hash_obj({"method_file": method_file, "method_params": method_params})
    approach_instance_key = f"{_method_stem(method_file)}__{params_hash}"
    paths = _build_paths_for_method(
        method_file,
        method_output_slug,
        params_hash,
        run_id,
        yaml_stem,
        cfg.get("输出目录"),
        is_multi_method,
    )

    runtime_context = create_runtime_context(
        run_id=run_id,
        method_file=method_file,
        output_dir=str(paths["output_dir"]),
        resource_interval_seconds=float(cfg.get("资源采样间隔秒", 5.0)),
    )
    runtime_context.update(
        {
            "approach_instance_key": approach_instance_key,
            "method_display_name": method_display_name,
            "params_hash": params_hash,
            "yaml_stem": yaml_stem,
            "input_articleandkg_file": input_parquet_path,
            "generated_root": str(paths["generated_root"]),
            "cache_root": str(paths["cache_root"]),
        }
    )

    init_kwargs = build_method_init_kwargs(
        method_file,
        method_params=method_params,
        shared_backend=shared_backend if uses_shared_qwen(method_file) else None,
        runtime_context=runtime_context,
    )
    method = LEGACY_CORE.load_method(method_file, init_kwargs=init_kwargs)
    if hasattr(method, "name"):
        runtime_context["method_name"] = method.name

    judge_backend_name = str(cfg.get("judge_backend", "kg_reward") or "kg_reward").strip().lower()
    if judge_backend_name not in {"kg_reward", "kgreward"}:
        raise RuntimeError(
            f"formal 目录已停用内置 judge backend；请改用 judge_backend=kg_reward，当前收到: {judge_backend_name}"
        )
    judge_backend_name = "kg_reward"
    evaluator = KgRewardEvaluator(
        judge_model=str(cfg["裁判用LLM模型"]),
        prompt_mode=str(cfg["裁判用LLM的提示词模式"]),
        token=int(cfg.get("裁判用LLM输出token上限", 32768) or 32768),
        temp=float(cfg.get("裁判用LLM温度", 0.1) or 0.1),
        think=int(cfg.get("裁判用LLM思考强度", 2) or 2),
        flex=bool(cfg.get("裁判用LLM是否开启Flex", True)),
        runtime_context=runtime_context,
    )
    judge_backend_type = "kg_reward"
    judge_prompt_bundle = LEGACY_CORE.resolve_judge_prompt_bundle(str(cfg["裁判用LLM的提示词模式"]))

    commit_id, git_dirty = _git_state()
    manifest = {
        "run_id": run_id,
        "start_time": _now_iso(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "git_commit_id": commit_id,
        "git_dirty": git_dirty,
        "yaml_file_name": yaml_stem,
        "yaml_abs_path": str(args.yaml) if args.yaml else "",
        "yaml_content": cfg,
        "execution_mode": "文章级" if cfg["是否从默认的方法级切换到最大化并行流水线的文章级"] else "方法级",
        "input_articleandkg_file": input_parquet_path,
        "generation_prompt_name": "grid_kg_single_prompt_maker_very_simple_20260303",
        "judge_prompt_name": str(judge_prompt_bundle.get("canonical_mode", str(cfg["裁判用LLM的提示词模式"]))),
        "judge_prompt_name_precision": str(judge_prompt_bundle.get("precision_prompt_name", "")),
        "judge_prompt_name_recall": str(judge_prompt_bundle.get("recall_prompt_name", "")),
        "judge_model_requested": str(cfg["裁判用LLM模型"]),
        "judge_model_resolved": str(cfg["裁判用LLM模型"]),
        "judge_backend_name": judge_backend_name,
        "judge_backend_type": judge_backend_type,
        "judge_fallback_enabled": False,
        "judge_fallback_triggered": False,
        "judge_fallback_reason": "",
        "judge_fallback_model_requested": "gpt-5.4-mini",
        "judge_fallback_model_resolved": "gpt-5.4-mini",
        "judge_token": int(cfg.get("裁判用LLM输出token上限", 32768) or 32768),
        "judge_temp": float(cfg.get("裁判用LLM温度", 0.1) or 0.1),
        "judge_think": int(cfg.get("裁判用LLM思考强度", 2) or 2),
        "judge_flex": bool(cfg.get("裁判用LLM是否开启Flex", True)),
        "generation_check_cache_enabled": bool(method_params.get("check_cache", True)),
        "judge_check_cache_enabled": True,
        "local_vllm_model_path": shared_backend.get("model_path"),
        "local_vllm_target_servers": shared_vllm_servers,
        "max_local_vllm_parallelism": int(cfg["最大本地VLLM生成并行任务数"]),
        "approach_file": method_file,
        "approach_instance_key": approach_instance_key,
        "method_display_name": method_display_name,
        "method_output_slug": method_output_slug,
        "resolved_params": init_kwargs,
        "override_flow_control": override_flow_ctrl,
        "effective_flow_control": {
            key: cfg.get(key)
            for key in FLOW_CONTROL_OVERRIDE_KEYS
        },
    }
    _write_json(paths["run_dir"] / "run_manifest.json", manifest)
    (paths["run_dir"] / "yaml_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    resource_monitor = ResourceMonitor(
        runtime_context=runtime_context,
        servers=shared_vllm_servers,
        interval_seconds=float(cfg.get("资源采样间隔秒", 5.0)),
    )
    return {
        "method_file": method_file,
        "method_params": method_params,
        "run_ctrl": run_ctrl,
        "effective_cfg": cfg,
        "method_display_name": method_display_name,
        "method_output_slug": method_output_slug,
        "params_hash": params_hash,
        "approach_instance_key": approach_instance_key,
        "paths": paths,
        "runtime_context": runtime_context,
        "init_kwargs": init_kwargs,
        "method": method,
        "evaluator": evaluator,
        "judge_backend_name": judge_backend_name,
        "judge_backend_type": judge_backend_type,
        "judge_prompt_bundle": judge_prompt_bundle,
        "manifest": manifest,
        "resource_monitor": resource_monitor,
        "run_id": run_id,
    }


def _start_method_resources(context: Dict[str, Any]) -> None:
    context["resource_monitor"].start()


def _stop_method_resources(context: Dict[str, Any]) -> None:
    try:
        context["resource_monitor"].stop()
    finally:
        summarize_latency_logs(context["runtime_context"])
        method = context.get("method")
        if method is not None and hasattr(method, "cleanup"):
            method.cleanup()


def _ensure_shared_backend_for_context(
    context: Dict[str, Any],
    shared_backend_manager: Optional[SharedLLMBackendManager],
) -> None:
    if shared_backend_manager is None:
        return
    method = context.get("method")
    desired_backend = dict(getattr(method, "shared_llm_backend", {}) or {})
    if not desired_backend.get("enabled"):
        return
    desired_model = str(desired_backend.get("model_path", "") or "").strip()
    if desired_model:
        print(f"🔁 共享 vLLM 切换校验: {context['method_file']} -> {desired_model}")
    shared_backend_manager.ensure_backend(desired_backend)


def _method_uses_local_vllm(context: Dict[str, Any]) -> bool:
    method_params = context.get("method_params") or {}
    backend = str(method_params.get("llm_backend", "") or "").strip().lower()
    if backend in {"shared_vllm", "dedicated_vllm"}:
        return True

    method = context.get("method")
    shared_backend = dict(getattr(method, "shared_llm_backend", {}) or {})
    if shared_backend.get("enabled"):
        return True

    use_cloud_or_vllm = str(getattr(method, "use_cloud_or_vllm", "") or "").strip().lower()
    return use_cloud_or_vllm == "vllm"


def _resolve_generation_workers(context: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    if _method_uses_local_vllm(context):
        return max(1, int(cfg.get("最大本地VLLM生成并行任务数", 64) or 64))
    return max(1, int(cfg.get("多进程数量", 1) or 1))


def _run_generation_phase(
    *,
    context: Dict[str, Any],
    data: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> None:
    method_file = context["method_file"]
    method_params = context["method_params"]
    approach_instance_key = context["approach_instance_key"]
    paths = context["paths"]
    init_kwargs = context["init_kwargs"]
    method = context["method"]
    run_ctrl = context["run_ctrl"]
    run_id = context["run_id"]

    selected_generate_items = _select_generation_items(data, paths, cfg, run_ctrl)
    context["selected_generate_count"] = len(selected_generate_items)
    print(f"   📦 需要生成: {len(selected_generate_items)}/{len(data)}")

    if not selected_generate_items:
        return

    generation_started_at = _now_iso()
    generation_batch_start = time.time()
    generation_workers = _resolve_generation_workers(context, cfg)
    generation_worker_source = (
        "最大本地VLLM生成并行任务数"
        if _method_uses_local_vllm(context)
        else "多进程数量"
    )
    print(f"   ⚙️ 生成并发: {generation_workers} (来源: {generation_worker_source})")
    try:
        contents = [item.get("content", "") for item in selected_generate_items]
        predictions = method.batch_generate(contents, num_workers=generation_workers)
        if len(predictions) != len(selected_generate_items):
            raise RuntimeError(f"生成返回条数异常: 期望 {len(selected_generate_items)}，实际 {len(predictions)}")
        batch_duration = time.time() - generation_batch_start
        per_item_duration = batch_duration / max(len(selected_generate_items), 1)
        for item, pred in zip(selected_generate_items, predictions):
            try:
                prompt_text, prompt_hash = _maybe_prompt_text(method, item.get("content", ""))
                normalized_pred = _normalize_generation_prediction(pred)
                raw_output = normalized_pred.get("raw_output", "")
                parser_success, parser_error = _is_generation_parse_success(normalized_pred)
                failure_type = "" if parser_success else "generate_parse_failed"
                failure_reason = "" if parser_success else (parser_error or "无法解析出KG结果")
                generation_record = _generation_record(
                    item,
                    method_file=method_file,
                    approach_instance_key=approach_instance_key,
                    resolved_params=init_kwargs,
                    execution_mode="方法级",
                    generation_status="success" if parser_success else "failed",
                    generation_failure_type=failure_type,
                    generation_failure_reason=failure_reason,
                    generation_started_at=generation_started_at,
                    generation_finished_at=_now_iso(),
                    generation_wall_seconds=per_item_duration,
                    generation_attempt_count=1,
                    generation_model_requested=str(method_params.get("model", "")),
                    generation_model_resolved=str(LEGACY_CORE._resolve_method_model_name(method)),
                    generation_backend_type=str(method_params.get("llm_backend", "native" if "Approach_" in method_file and not uses_shared_qwen(method_file) else "shared_vllm")),
                    generation_token=method_params.get("token", getattr(method, "token", None)),
                    generation_temp=method_params.get("temp", getattr(method, "temp", None)),
                    generation_flex=method_params.get("flex", False),
                    prompt_text=prompt_text,
                    prompt_hash=prompt_hash,
                    raw_response_text=raw_output,
                    parsed_kg_json=normalized_pred,
                    parser_success=parser_success,
                )
                status_record = _build_status_record(
                    sample=item,
                    approach_instance_key=approach_instance_key,
                    generation_status="success" if parser_success else "failed",
                    judge_status="missing",
                    overall_status="success" if parser_success else "failed",
                    generate_attempt_count=1,
                    judge_attempt_count=0,
                    generation_model_resolved=generation_record["generation_model_resolved"],
                    judge_model_resolved="",
                    end_to_end_seconds=generation_record["generation_wall_seconds"],
                    failure_type=failure_type,
                    failure_reason=failure_reason,
                    last_execution_mode="方法级",
                    last_run_id=run_id,
                )
                _persist_generation(paths, item["sample_id"], generation_record, status_record)
            except Exception as item_exc:
                prompt_text, prompt_hash = _maybe_prompt_text(method, item.get("content", ""))
                generation_record = _generation_record(
                    item,
                    method_file=method_file,
                    approach_instance_key=approach_instance_key,
                    resolved_params=init_kwargs,
                    execution_mode="方法级",
                    generation_status="failed",
                    generation_failure_type="generate_parse_failed",
                    generation_failure_reason=str(item_exc),
                    generation_started_at=generation_started_at,
                    generation_finished_at=_now_iso(),
                    generation_wall_seconds=per_item_duration,
                    generation_attempt_count=1,
                    generation_model_requested=str(method_params.get("model", "")),
                    generation_model_resolved=str(LEGACY_CORE._resolve_method_model_name(method)),
                    generation_backend_type=str(method_params.get("llm_backend", "")),
                    generation_token=method_params.get("token", getattr(method, "token", None)),
                    generation_temp=method_params.get("temp", getattr(method, "temp", None)),
                    generation_flex=method_params.get("flex", False),
                    prompt_text=prompt_text,
                    prompt_hash=prompt_hash,
                    raw_response_text="",
                    parsed_kg_json={"relations": [], "entities": []},
                    parser_success=False,
                )
                status_record = _build_status_record(
                    sample=item,
                    approach_instance_key=approach_instance_key,
                    generation_status="failed",
                    judge_status="missing",
                    overall_status="failed",
                    generate_attempt_count=1,
                    judge_attempt_count=0,
                    generation_model_resolved=generation_record["generation_model_resolved"],
                    judge_model_resolved="",
                    end_to_end_seconds=per_item_duration,
                    failure_type="generate_parse_failed",
                    failure_reason=str(item_exc),
                    last_execution_mode="方法级",
                    last_run_id=run_id,
                )
                _persist_generation(paths, item["sample_id"], generation_record, status_record)
    except Exception as exc:
        batch_duration = time.time() - generation_batch_start
        per_item_duration = batch_duration / max(len(selected_generate_items), 1)
        failure_type = "generate_timeout" if _is_timeout_text(str(exc)) else "generate_parse_failed"
        for item in selected_generate_items:
            prompt_text, prompt_hash = _maybe_prompt_text(method, item.get("content", ""))
            generation_record = _generation_record(
                item,
                method_file=method_file,
                approach_instance_key=approach_instance_key,
                resolved_params=init_kwargs,
                execution_mode="方法级",
                generation_status="failed",
                generation_failure_type=failure_type,
                generation_failure_reason=str(exc),
                generation_started_at=generation_started_at,
                generation_finished_at=_now_iso(),
                generation_wall_seconds=per_item_duration,
                generation_attempt_count=1,
                generation_model_requested=str(method_params.get("model", "")),
                generation_model_resolved=str(LEGACY_CORE._resolve_method_model_name(method)),
                generation_backend_type=str(method_params.get("llm_backend", "")),
                generation_token=method_params.get("token", getattr(method, "token", None)),
                generation_temp=method_params.get("temp", getattr(method, "temp", None)),
                generation_flex=method_params.get("flex", False),
                prompt_text=prompt_text,
                prompt_hash=prompt_hash,
                raw_response_text="",
                parsed_kg_json={"relations": [], "entities": []},
                parser_success=False,
            )
            status_record = _build_status_record(
                sample=item,
                approach_instance_key=approach_instance_key,
                generation_status="failed",
                judge_status="missing",
                overall_status="failed",
                generate_attempt_count=1,
                judge_attempt_count=0,
                generation_model_resolved=generation_record["generation_model_resolved"],
                judge_model_resolved="",
                end_to_end_seconds=per_item_duration,
                failure_type=failure_type,
                failure_reason=str(exc),
                last_execution_mode="方法级",
                last_run_id=run_id,
            )
            _persist_generation(paths, item["sample_id"], generation_record, status_record)


def _run_evaluation_phase(
    *,
    context: Dict[str, Any],
    data: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> None:
    method_file = context["method_file"]
    run_ctrl = context["run_ctrl"]
    approach_instance_key = context["approach_instance_key"]
    paths = context["paths"]
    evaluator = context["evaluator"]
    judge_prompt_bundle = context.get("judge_prompt_bundle") or {}
    manifest = context["manifest"]
    run_id = context["run_id"]
    judge_backend_type = context.get("judge_backend_type", str(cfg["裁判用LLM模型"]))

    selected_eval_items = _select_evaluation_items(data, paths, cfg, run_ctrl)
    context["selected_eval_count"] = len(selected_eval_items)
    print(f"   🧑‍⚖️ 需要评估: {len(selected_eval_items)}/{len(data)}")

    if not selected_eval_items:
        return

    eval_items: List[Dict[str, Any]] = []
    eval_predictions: List[Dict[str, Any]] = []
    for item in selected_eval_items:
        pred = _collect_prediction_from_cache(paths, item["sample_id"])
        if pred and pred.get("relations") is not None:
            eval_items.append(item)
            eval_predictions.append(pred)

    if not eval_items:
        return

    judge_started_at = _now_iso()
    judge_batch_start = time.time()
    try:
        scores_list, judge_model_resolved, fallback_triggered, fallback_reason = _judge_batch_with_fallback(
            evaluator,
            eval_items,
            eval_predictions,
        )
        if fallback_triggered:
            manifest["judge_fallback_triggered"] = True
            manifest["judge_fallback_reason"] = fallback_reason
            manifest["judge_model_resolved"] = judge_model_resolved
            _write_json(paths["run_dir"] / "run_manifest.json", manifest)

        batch_duration = time.time() - judge_batch_start
        per_item_duration = batch_duration / max(len(eval_items), 1)
        for item, pred, scores in zip(eval_items, eval_predictions, scores_list):
            precision_ok = scores.get("precision_details", {}).get("parse_success", False)
            recall_ok = scores.get("recall_details", {}).get("parse_success", False)
            judge_ok = precision_ok and recall_ok
            judge_failure_type = "" if judge_ok else "judge_parse_failed"
            judge_failure_reason = "" if judge_ok else (_judge_failure_reason_from_scores(scores) or "无法解析出评估结果")
            evaluation_record = _evaluation_record(
                item,
                method_file=method_file,
                approach_instance_key=approach_instance_key,
                judge_status="success" if judge_ok else "failed",
                judge_failure_type=judge_failure_type,
                judge_failure_reason=judge_failure_reason,
                judge_started_at=judge_started_at,
                judge_finished_at=_now_iso(),
                judge_wall_seconds_total=per_item_duration,
                judge_attempt_count=1,
                judge_model_requested=str(cfg["裁判用LLM模型"]),
                judge_model_resolved=judge_model_resolved,
                judge_backend_type=judge_backend_type,
                judge_token=int(cfg.get("裁判用LLM输出token上限", 32768) or 32768),
                judge_temp=float(cfg.get("裁判用LLM温度", 0.1) or 0.1),
                judge_think=int(cfg.get("裁判用LLM思考强度", 2) or 2),
                judge_flex=bool(cfg.get("裁判用LLM是否开启Flex", True)) or fallback_triggered,
                judge_prompt_name_precision=str(judge_prompt_bundle.get("precision_prompt_name", "")),
                judge_prompt_name_recall=str(judge_prompt_bundle.get("recall_prompt_name", "")),
                scores=scores,
                prediction_hash=_hash_obj(pred),
                ground_truth_hash=_hash_obj(item.get("ground_truth", [])),
                fallback_triggered=fallback_triggered,
                fallback_reason=fallback_reason,
            )
            prev_status = _load_existing_status(paths, item["sample_id"])
            end_to_end_seconds = prev_status.get("end_to_end_seconds", 0.0) + per_item_duration
            if end_to_end_seconds > int(cfg["单任务最大存活分钟数"]) * 60:
                judge_ok = False
                judge_failure_type = "end_to_end_timeout"
                judge_failure_reason = f"单任务超过 {cfg['单任务最大存活分钟数']} 分钟"
                evaluation_record["judge_status"] = "failed"
                evaluation_record["judge_failure_type"] = judge_failure_type
                evaluation_record["judge_failure_reason"] = judge_failure_reason
            status_record = _build_status_record(
                sample=item,
                approach_instance_key=approach_instance_key,
                generation_status=prev_status.get("generation_status", "success"),
                judge_status="success" if judge_ok else "failed",
                overall_status="success" if judge_ok and prev_status.get("generation_status") == "success" else "failed",
                generate_attempt_count=int(prev_status.get("generate_attempt_count", 1)),
                judge_attempt_count=int(prev_status.get("judge_attempt_count", 0)) + 1,
                generation_model_resolved=prev_status.get("generation_model_resolved", ""),
                judge_model_resolved=judge_model_resolved,
                end_to_end_seconds=end_to_end_seconds,
                failure_type=judge_failure_type,
                failure_reason=judge_failure_reason,
                last_execution_mode="方法级",
                last_run_id=run_id,
            )
            _persist_evaluation(paths, item["sample_id"], evaluation_record, status_record)
    except Exception as exc:
        batch_duration = time.time() - judge_batch_start
        per_item_duration = batch_duration / max(len(eval_items), 1)
        failure_type = "judge_timeout" if _is_timeout_text(str(exc)) else "judge_parse_failed"
        for item in eval_items:
            prev_status = _load_existing_status(paths, item["sample_id"])
            evaluation_record = _evaluation_record(
                item,
                method_file=method_file,
                approach_instance_key=approach_instance_key,
                judge_status="failed",
                judge_failure_type=failure_type,
                judge_failure_reason=str(exc),
                judge_started_at=judge_started_at,
                judge_finished_at=_now_iso(),
                judge_wall_seconds_total=per_item_duration,
                judge_attempt_count=1,
                judge_model_requested=str(cfg["裁判用LLM模型"]),
                judge_model_resolved=str(cfg["裁判用LLM模型"]),
                judge_backend_type=judge_backend_type,
                judge_token=int(cfg.get("裁判用LLM输出token上限", 32768) or 32768),
                judge_temp=float(cfg.get("裁判用LLM温度", 0.1) or 0.1),
                judge_think=int(cfg.get("裁判用LLM思考强度", 2) or 2),
                judge_flex=bool(cfg.get("裁判用LLM是否开启Flex", True)),
                judge_prompt_name_precision=str(judge_prompt_bundle.get("precision_prompt_name", "")),
                judge_prompt_name_recall=str(judge_prompt_bundle.get("recall_prompt_name", "")),
                scores={},
                prediction_hash="",
                ground_truth_hash=_hash_obj(item.get("ground_truth", [])),
                fallback_triggered=False,
                fallback_reason="",
            )
            status_record = _build_status_record(
                sample=item,
                approach_instance_key=approach_instance_key,
                generation_status=prev_status.get("generation_status", "success"),
                judge_status="failed",
                overall_status="failed",
                generate_attempt_count=int(prev_status.get("generate_attempt_count", 1)),
                judge_attempt_count=int(prev_status.get("judge_attempt_count", 0)) + 1,
                generation_model_resolved=prev_status.get("generation_model_resolved", ""),
                judge_model_resolved=str(cfg["裁判用LLM模型"]),
                end_to_end_seconds=prev_status.get("end_to_end_seconds", 0.0) + per_item_duration,
                failure_type=failure_type,
                failure_reason=str(exc),
                last_execution_mode="方法级",
                last_run_id=run_id,
            )
            _persist_evaluation(paths, item["sample_id"], evaluation_record, status_record)


def _finalize_method_context(
    *,
    context: Dict[str, Any],
    data: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    method_file = context["method_file"]
    method_display_name = context["method_display_name"]
    approach_instance_key = context["approach_instance_key"]
    paths = context["paths"]
    manifest = context["manifest"]

    max_lifetime_seconds = int(cfg["单任务最大存活分钟数"]) * 60
    for item in data:
        sample_id = item["sample_id"]
        status = _load_existing_status(paths, sample_id)
        if float(status.get("end_to_end_seconds", 0.0)) > max_lifetime_seconds:
            status["judge_status"] = "failed"
            status["overall_status"] = "failed"
            status["failure_type"] = "end_to_end_timeout"
            status["failure_reason"] = f"单任务超过 {cfg['单任务最大存活分钟数']} 分钟"
            status["updated_at"] = _now_iso()
            _write_json(paths["cache_status_dir"] / f"{sample_id}.json", status)

    rows, summary = _aggregate_method_outputs(
        data,
        paths,
        method_file=method_file,
        method_display_name=method_display_name,
        approach_instance_key=approach_instance_key,
    )
    manifest["end_time"] = _now_iso()
    manifest["summary"] = summary
    _write_json(paths["run_dir"] / "run_manifest.json", manifest)
    return rows, summary


def _run_method_eval_and_finalize(
    *,
    context: Dict[str, Any],
    data: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        _run_evaluation_phase(context=context, data=data, cfg=cfg)
    finally:
        _stop_method_resources(context)
    return _finalize_method_context(context=context, data=data, cfg=cfg)


def _run_single_method(
    *,
    data: List[Dict[str, Any]],
    method_spec: Dict[str, Any],
    cfg: Dict[str, Any],
    args,
    input_parquet_path: str,
    shared_backend: Dict[str, Any],
    shared_backend_manager: Optional[SharedLLMBackendManager],
    shared_vllm_servers: List[str],
    run_id: str,
    yaml_stem: str,
    is_multi_method: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    context = _prepare_method_context(
        method_spec=method_spec,
        cfg=cfg,
        args=args,
        input_parquet_path=input_parquet_path,
        shared_backend=shared_backend,
        shared_vllm_servers=shared_vllm_servers,
        run_id=run_id,
        yaml_stem=yaml_stem,
        is_multi_method=is_multi_method,
    )
    print(f"\n🚀 方法: {context['method_file']}")
    pending_generate_count, pending_eval_count = _compute_pending_work_counts(
        data=data,
        paths=context["paths"],
        cfg=cfg,
        run_ctrl=context["run_ctrl"],
    )
    context["pending_generate_count"] = pending_generate_count
    context["pending_eval_count"] = pending_eval_count
    context["manifest"]["selected_generate_count"] = pending_generate_count
    context["manifest"]["selected_eval_count"] = pending_eval_count
    _write_json(context["paths"]["run_dir"] / "run_manifest.json", context["manifest"])
    print(f"   📦 待生成: {pending_generate_count}/{len(data)} | 🧑‍⚖️ 待评估: {pending_eval_count}/{len(data)}")
    if pending_generate_count == 0 and pending_eval_count == 0:
        print("   ⏭️ 本方法实例本轮无待执行工作，直接复用现有结果进行聚合。")
        return _finalize_method_context(context=context, data=data, cfg=cfg)
    if pending_generate_count > 0:
        _ensure_shared_backend_for_context(context, shared_backend_manager)
    if cfg["是否从默认的方法级切换到最大化并行流水线的文章级"]:
        print("   ⚠️ 当前先走稳定的方法级执行器；文章级开关已记录到结果与 manifest 中。")
    _start_method_resources(context)
    try:
        _run_generation_phase(context=context, data=data, cfg=cfg)
        _run_evaluation_phase(context=context, data=data, cfg=cfg)
    finally:
        _stop_method_resources(context)
    return _finalize_method_context(context=context, data=data, cfg=cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified evaluation executor for the packaged GRID benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-yaml", "--yaml", type=str, help="Path to an experiment YAML file.")
    parser.add_argument("--method", type=str, help="Single method file name for legacy compatibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.yaml and not args.method:
        raise SystemExit("Please provide --yaml or --method.")

    cfg = dict(DEFAULT_TOP_LEVEL_CFG)
    yaml_stem = "cli"
    if args.yaml:
        yaml_path = Path(args.yaml)
        yaml_stem = yaml_path.stem
        loaded_cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        cfg.update(loaded_cfg)

    method_specs = _normalize_method_specs(args, cfg)
    data, input_parquet_path = _prepare_input_data(cfg)
    shared_vllm_servers = [s.strip() for s in str(cfg.get("共享LLM服务器列表", "super")).split(",") if s.strip()]
    shared_llm_check_history_cache_raw = cfg.get("共享LLM是否检查历史缓存", True)
    if isinstance(shared_llm_check_history_cache_raw, str):
        shared_llm_check_history_cache = shared_llm_check_history_cache_raw.strip().lower() in {
            "1", "true", "yes", "y", "on"
        }
    else:
        shared_llm_check_history_cache = bool(shared_llm_check_history_cache_raw)
    shared_backend = build_default_shared_backend(
        model_path=str(cfg.get("共享LLM后端模型路径") or DEFAULT_SHARED_QWEN_MODEL_PATH),
        servers=shared_vllm_servers,
        check_history_cache=shared_llm_check_history_cache,
        stream_stall_seconds=cfg.get("本地VLLM流式静默超时秒", 1800),
        request_max_total_seconds=cfg.get("本地VLLM请求总时长上限秒", 14400),
    )

    need_shared_backend = any(
        uses_shared_qwen(spec["文件名"]) and str((spec.get("参数") or {}).get("llm_backend", "shared_vllm")).lower() != "dedicated_vllm"
        for spec in method_specs
    )
    shared_backend_manager = None
    if need_shared_backend:
        print("🚀 已启用共享三机 vLLM 管理器（按需懒切模型，不再全局提前预热）...")
        shared_backend_manager = SharedLLMBackendManager(shared_backend)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(
        "🧠 asks() 检查缓存状态: "
        f"生成默认开启={bool(shared_backend.get('check_history_cache', True))} | 评估默认开启=True"
    )
    _notify(f"🚀 究极测试集统一评估开始\nrun_id={run_id}\n方法数={len(method_specs)}\n样本数={len(data)}")

    method_cfgs: List[Dict[str, Any]] = [
        _build_effective_method_cfg(cfg, method_spec)
        for method_spec in method_specs
    ]

    all_summaries: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    try:
        use_method_pipeline = (
            all(
                not bool(method_cfg["是否从默认的方法级切换到最大化并行流水线的文章级"])
                for method_cfg in method_cfgs
            )
            and len(method_specs) > 1
        )
        if use_method_pipeline:
            eval_lane_workers = max(1, int(cfg.get("方法级评估并行车道数", 2) or 2))
            non_llm_phase_started = False
            print(
                "🚦 启用方法级多车道流水线："
                "生成保持顺序推进，已完成生成的方法立即并行进入评估。"
            )
            print(f"   🧑‍⚖️ 评估并行车道数: {eval_lane_workers}")
            eval_executor = ThreadPoolExecutor(max_workers=eval_lane_workers, thread_name_prefix="ultimate_eval_judge_lane")
            queued_evals: Dict[Any, str] = {}
            try:
                for idx, (method_spec, method_cfg) in enumerate(zip(method_specs, method_cfgs), 1):
                    if is_native_baseline(method_spec["文件名"]) and not non_llm_phase_started:
                        _stop_vllm_before_non_llm_phase()
                        non_llm_phase_started = True
                    print("\n" + "=" * 80)
                    print(f"📊 [{idx}/{len(method_specs)}] 开始方法生成: {method_spec['文件名']}")
                    print("=" * 80)
                    context = _prepare_method_context(
                        method_spec=method_spec,
                        cfg=method_cfg,
                        args=args,
                        input_parquet_path=input_parquet_path,
                        shared_backend=shared_backend,
                        shared_vllm_servers=shared_vllm_servers,
                        run_id=run_id,
                        yaml_stem=yaml_stem,
                        is_multi_method=len(method_specs) > 1,
                    )
                    print(f"\n🚀 方法: {context['method_file']}")
                    pending_generate_count, pending_eval_count = _compute_pending_work_counts(
                        data=data,
                        paths=context["paths"],
                        cfg=method_cfg,
                        run_ctrl=context["run_ctrl"],
                    )
                    context["pending_generate_count"] = pending_generate_count
                    context["pending_eval_count"] = pending_eval_count
                    context["manifest"]["selected_generate_count"] = pending_generate_count
                    context["manifest"]["selected_eval_count"] = pending_eval_count
                    _write_json(context["paths"]["run_dir"] / "run_manifest.json", context["manifest"])
                    print(f"   📦 待生成: {pending_generate_count}/{len(data)} | 🧑‍⚖️ 待评估: {pending_eval_count}/{len(data)}")
                    if pending_generate_count == 0 and pending_eval_count == 0:
                        print("   ⏭️ 本方法实例本轮无待执行工作，直接复用现有结果进行聚合。")
                        rows, summary = _finalize_method_context(context=context, data=data, cfg=method_cfg)
                        all_rows.extend([{**row, "method": summary["method"]} for row in rows])
                        all_summaries.append(summary)
                        _write_global_comparison_bundle(
                            comparison_dir=Path(cfg.get("输出目录") or (EFFECTIVENESS_RESULT_DIR / "多方法对比")),
                            all_rows=all_rows,
                            run_id=run_id,
                            input_parquet_path=input_parquet_path,
                            cfg=cfg,
                        )
                        _notify(
                            f"✅ 方法完成\nrun_id={run_id}\n方法={summary['method']}\n"
                            f"F1={summary['avg_f1']:.4f}\n失败率={summary['failure_rate_total']:.4f}"
                        )
                        continue
                    if pending_generate_count > 0:
                        _ensure_shared_backend_for_context(context, shared_backend_manager)
                    _start_method_resources(context)
                    try:
                        _run_generation_phase(context=context, data=data, cfg=method_cfg)
                    except Exception:
                        _stop_method_resources(context)
                        raise
                    if is_native_baseline(method_spec["文件名"]):
                        method = context.get("method")
                        if method is not None and hasattr(method, "cleanup"):
                            print("   🧹 非LLM方法生成完成，提前释放 GPU 模型显存...")
                            method.cleanup()
                    future = eval_executor.submit(
                        _run_method_eval_and_finalize,
                        context=context,
                        data=data,
                        cfg=method_cfg,
                    )
                    queued_evals[future] = context["method_file"]
                    if idx < len(method_specs):
                        print(
                            f"   🔁 {context['method_file']} 已进入并行评估车道；"
                            f"接下来开始 {method_specs[idx]['文件名']} 的生成。"
                        )

                for future in as_completed(list(queued_evals.keys())):
                    method_file = queued_evals[future]
                    rows, summary = future.result()
                    all_rows.extend([{**row, "method": summary["method"]} for row in rows])
                    all_summaries.append(summary)
                    _write_global_comparison_bundle(
                        comparison_dir=Path(cfg.get("输出目录") or (EFFECTIVENESS_RESULT_DIR / "多方法对比")),
                        all_rows=all_rows,
                        run_id=run_id,
                        input_parquet_path=input_parquet_path,
                        cfg=cfg,
                    )
                    _notify(
                        f"✅ 方法完成\nrun_id={run_id}\n方法={summary['method']}\n"
                        f"F1={summary['avg_f1']:.4f}\n失败率={summary['failure_rate_total']:.4f}"
                    )
            finally:
                eval_executor.shutdown(wait=True)
        else:
            if len(method_specs) > 1 and not use_method_pipeline:
                print("🚦 检测到至少一个方法启用了方法级 override流程控制中的文章级开关；本轮回退为逐方法执行。")
            for idx, (method_spec, method_cfg) in enumerate(zip(method_specs, method_cfgs), 1):
                print("\n" + "=" * 80)
                print(f"📊 [{idx}/{len(method_specs)}] 开始方法: {method_spec['文件名']}")
                print("=" * 80)
                rows, summary = _run_single_method(
                    data=data,
                    method_spec=method_spec,
                    cfg=method_cfg,
                    args=args,
                    input_parquet_path=input_parquet_path,
                    shared_backend=shared_backend,
                    shared_vllm_servers=shared_vllm_servers,
                    run_id=run_id,
                    yaml_stem=yaml_stem,
                    is_multi_method=len(method_specs) > 1,
                    shared_backend_manager=shared_backend_manager,
                )
                all_rows.extend([{**row, "method": summary["method"]} for row in rows])
                all_summaries.append(summary)
                _write_global_comparison_bundle(
                    comparison_dir=Path(cfg.get("输出目录") or (EFFECTIVENESS_RESULT_DIR / "多方法对比")),
                    all_rows=all_rows,
                    run_id=run_id,
                    input_parquet_path=input_parquet_path,
                    cfg=cfg,
                )
                _notify(
                    f"✅ 方法完成\nrun_id={run_id}\n方法={summary['method']}\n"
                    f"F1={summary['avg_f1']:.4f}\n失败率={summary['failure_rate_total']:.4f}"
                )
    finally:
        if shared_backend_manager is not None:
            shared_backend_manager.cleanup()

    comparison_dir = Path(cfg.get("输出目录") or (EFFECTIVENESS_RESULT_DIR / "多方法对比"))
    comparison_dir.mkdir(parents=True, exist_ok=True)
    if all_summaries:
        _write_global_comparison_bundle(
            comparison_dir=comparison_dir,
            all_rows=all_rows,
            run_id=run_id,
            input_parquet_path=input_parquet_path,
            cfg=cfg,
        )
        summary_df = _build_method_summary_frame(_apply_metric_inclusion_policy(pd.DataFrame(all_rows)))
        best = summary_df.iloc[0]
        _notify(
            f"🎯 究极测试集统一评估完成\nrun_id={run_id}\n最佳方法={best['method']}\n"
            f"F1={best['avg_f1']:.4f}\n失败率={best['failure_rate_total']:.4f}"
        )


if __name__ == "__main__":
    main()
