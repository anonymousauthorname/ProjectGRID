# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import socket
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
DROPBOX_PATH = Path(os.path.expanduser("~/Dropbox"))
if str(DROPBOX_PATH) not in os.sys.path:
    os.sys.path.insert(0, str(DROPBOX_PATH))

TOOLS_FILE = str(DROPBOX_PATH / "tools.py")
loaded_tools = os.sys.modules.get("tools")
if loaded_tools is None or os.path.abspath(getattr(loaded_tools, "__file__", "")) != TOOLS_FILE:
    spec = importlib.util.spec_from_file_location("tools", TOOLS_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"❌ 无法加载 Dropbox tools.py: {TOOLS_FILE}")
    loaded_tools = importlib.util.module_from_spec(spec)
    os.sys.modules["tools"] = loaded_tools
    spec.loader.exec_module(loaded_tools)
tools = loaded_tools
from vllm_environment_setup import SERVERS, VLLMEnvironmentManager

HOSTNAME_SHORT = socket.gethostname().split(".", 1)[0]
HOSTNAME_TO_SERVER = {
    "en4230616l": "super",
    "en4217394l": "normal",
    "en4234871l": "ultra",
}
CURRENT_SERVER_NAME = HOSTNAME_TO_SERVER.get(HOSTNAME_SHORT)


DEFAULT_SHARED_QWEN_MODEL_PATH = str(REPO_ROOT / "models" / "base_model")
DEFAULT_SHARED_VLLM_PROMPT_SEND_WEIGHT = {"super": 2, "ultra": 4, "normal": 1}

DEFAULT_FULL_METHODS = [
    "Approach_GRIDSingleInAndOut.py",
    "GRID_Ours.py",
    "Approach_CTINexus.py",
    "Approach_graphrag.py",
    "Approach_GTIKGResearch.py",
    "Approach_cognee.py",
    "Approach_graphiti.py",
    "Approach_llm_akg.py",
    "Approach_AttacKG_plus.py",
    "Approach_rebel.py",
    "Approach_knowgl.py",
    "Approach_UIE.py",
    "Approach_EXTRACTOR.py",
]

SHARED_QWEN_METHODS = {
    "Approach_GRIDSingleInAndOut.py",
    "GRID_Ours.py",
    "Approach_CTINexus.py",
    "Approach_CTINexus_2steps.py",
    "Approach_graphrag.py",
    "Approach_GTIKGResearch.py",
    "Approach_cognee.py",
    "Approach_graphiti.py",
    "Approach_llm_akg.py",
    "Approach_AttacKG_plus.py",
}

NATIVE_BASELINE_METHODS = {
    "Approach_rebel.py",
    "Approach_knowgl.py",
    "Approach_UIE.py",
    "Approach_EXTRACTOR.py",
}

METHOD_INIT_PRESETS: Dict[str, Dict[str, Any]] = {
    "Approach_GRIDSingleInAndOut.py": {},

    "GRID_Ours.py": {
        "llm_backend": "shared_vllm",
        "model": DEFAULT_SHARED_QWEN_MODEL_PATH,
    },
    "Approach_CTINexus.py": {
        "model": "local",
        "use_cloud_or_vllm": "vllm",
        "vllm_model_path": DEFAULT_SHARED_QWEN_MODEL_PATH,
    },
    
    
    "Approach_CTINexus_2steps.py": {
        "model": "local",
        "use_cloud_or_vllm": "vllm",
        "vllm_model_path": DEFAULT_SHARED_QWEN_MODEL_PATH,
    },
    "Approach_graphrag.py": {
        "model": "local",
        "use_cloud_or_vllm": "vllm",
        "vllm_model_path": DEFAULT_SHARED_QWEN_MODEL_PATH,
    },
    "Approach_GTIKGResearch.py": {
        "model": "local",
        "use_cloud_or_vllm": "vllm",
        "vllm_model_path": DEFAULT_SHARED_QWEN_MODEL_PATH,
    },
    "Approach_cognee.py": {"model": "local"},
    "Approach_graphiti.py": {"model": "local"},
    "Approach_llm_akg.py": {"model": "local"},
    "Approach_AttacKG_plus.py": {"model": "local"},
    
    "Approach_rebel.py": {"inference_batch_size": 64, "num_beams": 2, "max_devices": 2},
    
    
    "Approach_knowgl.py": {"inference_batch_size": 64, "num_beams": 1, "max_devices": 2},
    "Approach_UIE.py": {"inference_batch_size": 64, "num_beams": 1, "max_devices": 2},
    
    
    "Approach_EXTRACTOR.py": {"max_batch_workers": 64, "timeout": 600},
}


def build_default_shared_backend(
    model_path: str = DEFAULT_SHARED_QWEN_MODEL_PATH,
    
    
    servers: Sequence[str] = ("super", "ultra", "normal"),
    check_history_cache: bool = True,
    stream_stall_seconds: Optional[float] = 1800,
    request_max_total_seconds: Optional[float] = 14400,
) -> Dict[str, Any]:
    server_list = list(servers)
    
    
    filtered_weights = {
        server: DEFAULT_SHARED_VLLM_PROMPT_SEND_WEIGHT[server]
        for server in server_list
        if server in DEFAULT_SHARED_VLLM_PROMPT_SEND_WEIGHT
    }
    return {
        "enabled": True,
        "model": "local",
        "model_path": model_path,
        "servers": server_list,
        
        
        "smart_mode": False,
        "max_workers_vllm": 64,
        "prompt_send_weight_vllm": filtered_weights,
        
        
        
        "check_history_cache": bool(check_history_cache),
        "note_prefix": "ultimate_eval",
        "gpu_cache_threshold": 90.0,
        "ready_wait_seconds": 120,
        "auto_cleanup": False,
        
        
        
        "stream_stall_seconds": stream_stall_seconds,
        "request_max_total_seconds": request_max_total_seconds,
    }


def uses_shared_qwen(method_file: str) -> bool:
    return method_file in SHARED_QWEN_METHODS


def is_native_baseline(method_file: str) -> bool:
    return method_file in NATIVE_BASELINE_METHODS


def sort_methods_for_async_pipeline(methods: Sequence[str]) -> List[str]:
    llm_methods = [method for method in methods if not is_native_baseline(method)]
    native_methods = [method for method in methods if is_native_baseline(method)]
    return llm_methods + native_methods


def build_method_init_kwargs(
    method_file: str,
    method_params: Optional[Dict[str, Any]] = None,
    shared_backend: Optional[Dict[str, Any]] = None,
    runtime_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(METHOD_INIT_PRESETS.get(method_file, {}))
    method_params = dict(method_params or {})
    llm_backend = str(method_params.pop("llm_backend", "") or "").strip().lower()
    model_value = method_params.pop("model", None)

    
    if llm_backend:
        if method_file in {"Approach_GRIDSingleInAndOut.py", "GRID_Ours.py"}:
            kwargs["llm_backend"] = llm_backend
            if model_value is not None:
                kwargs["model"] = model_value
        else:
            if llm_backend == "cloud_api":
                kwargs["use_cloud_or_vllm"] = "cloud"
                if model_value is not None:
                    kwargs["model"] = model_value
            elif llm_backend in {"shared_vllm", "dedicated_vllm"}:
                kwargs["use_cloud_or_vllm"] = "vllm"
                if model_value is not None:
                    kwargs["vllm_model_path"] = model_value

    for key, value in method_params.items():
        kwargs[key] = value

    if runtime_context is not None:
        kwargs["runtime_context"] = runtime_context
    if uses_shared_qwen(method_file):
        backend_cfg = dict(shared_backend or build_default_shared_backend())
        if llm_backend == "dedicated_vllm":
            backend_cfg["enabled"] = False
        elif llm_backend == "shared_vllm" or not llm_backend:
            backend_cfg["enabled"] = True
        if model_value is not None and llm_backend == "shared_vllm":
            backend_cfg["model_path"] = model_value
        
        if "vllm_model_path" in kwargs:
            kwargs["vllm_model_path"] = backend_cfg.get("model_path", kwargs["vllm_model_path"])
        if backend_cfg.get("enabled"):
            kwargs["shared_llm_backend"] = backend_cfg
    return kwargs


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prompt_hash(prompt: Any) -> str:
    try:
        return tools.get_prompt_hash(prompt)
    except Exception:
        return hashlib.md5(_stable_json_dumps(prompt).encode("utf-8")).hexdigest()


def content_hash(content: str) -> str:
    return hashlib.sha1((content or "").encode("utf-8")).hexdigest()


def build_content_ref_map(data: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    ref_map: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(data):
        content = item.get("content", "")
        extra_info = item.get("extra_info", {}) or {}
        ref_map[content_hash(content)] = {
            "sample_index": idx,
            "item_id": extra_info.get("file_name", f"item_{idx}"),
            "source": item.get("source_approach_provided_dataset", "unknown"),
        }
    return ref_map


def lookup_sample_ref(runtime_context: Optional[Dict[str, Any]], content: str) -> Dict[str, Any]:
    if not runtime_context:
        return {}
    return dict(runtime_context.get("content_ref_map", {}).get(content_hash(content), {}))


class JsonlWriter:

    def __init__(self, file_path: str):
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def create_runtime_context(
    *,
    run_id: str,
    method_file: str,
    output_dir: str,
    resource_interval_seconds: float = 5.0,
) -> Dict[str, Any]:
    log_dir = Path(output_dir) / "logs" / run_id
    return {
        "run_id": run_id,
        "method_file": method_file,
        "method_name": method_file.replace(".py", ""),
        "output_dir": output_dir,
        "log_dir": str(log_dir),
        "request_writer": JsonlWriter(str(log_dir / "request_events.jsonl")),
        "judge_writer": JsonlWriter(str(log_dir / "judge_events.jsonl")),
        "resource_writer": JsonlWriter(str(log_dir / "resource_snapshots.jsonl")),
        "resource_interval_seconds": float(resource_interval_seconds),
    }


def _build_stats_meta_map(
    prompt_list: Sequence[Any],
    prompt_metadata_list: Optional[Sequence[Optional[Dict[str, Any]]]],
    *,
    phase: str,
    runtime_context: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not prompt_metadata_list:
        return {}

    stats_meta_map: Dict[str, Dict[str, Any]] = {}
    for prompt, metadata in zip(prompt_list, prompt_metadata_list):
        if metadata is None:
            continue
        merged_meta = dict(metadata)
        if runtime_context:
            merged_meta.setdefault("run_id", runtime_context.get("run_id"))
            merged_meta.setdefault("method_file", runtime_context.get("method_file"))
            merged_meta.setdefault("method_name", runtime_context.get("method_name"))
        merged_meta.setdefault("phase", phase)
        stats_meta_map[prompt_hash(prompt)] = merged_meta
    return stats_meta_map


def _write_stats_records(
    *,
    runtime_context: Optional[Dict[str, Any]],
    phase: str,
    stats_collector: Sequence[Dict[str, Any]],
) -> None:
    if not runtime_context or not stats_collector:
        return

    writer_key = "judge_writer" if phase == "judge" else "request_writer"
    writer: Optional[JsonlWriter] = runtime_context.get(writer_key)
    if writer is None:
        return

    default_fields = {
        "run_id": runtime_context.get("run_id"),
        "method_file": runtime_context.get("method_file"),
        "method_name": runtime_context.get("method_name"),
        "phase": phase,
        "logged_at_iso": datetime.now().isoformat(),
    }

    for item in stats_collector:
        record = dict(default_fields)
        record.update(item)
        writer.write(record)


def run_logged_asks(
    prompt_list: List[List[Dict[str, Any]]],
    *,
    model: str,
    token: int,
    temp: float,
    think: int = 0,
    runtime_context: Optional[Dict[str, Any]] = None,
    phase: str = "generate",
    prompt_metadata_list: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    note: Optional[str] = None,
    check_history_cache: bool = True,
    VllmSmartMode: bool = False,
    max_workers_Vllm: Any = 64,
    prompt_send_weight_VllmNotSmartMode: Optional[Dict[str, int]] = None,
    vllm_server_name: Optional[str] = None,
    retry: bool = False,
    force_api_do_huge_input_Cloud: bool = True,
    flex: bool = False,
    top_p: Any = "NotSet",
    top_k: Any = "NotSet",
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    stats_collector: List[Dict[str, Any]] = []
    stats_meta_map = _build_stats_meta_map(
        prompt_list,
        prompt_metadata_list,
        phase=phase,
        runtime_context=runtime_context,
    )

    if note is None:
        prefix = "ultimate_eval"
        if runtime_context:
            prefix = runtime_context.get("run_id", prefix)
        note = f"{prefix}|{phase}|{model}"

    
    
    
    if model == "local":
        target_servers: List[str] = []
        if vllm_server_name:
            target_servers = [vllm_server_name]
        elif isinstance(prompt_send_weight_VllmNotSmartMode, dict) and prompt_send_weight_VllmNotSmartMode:
            
            
            target_servers = [str(s) for s in prompt_send_weight_VllmNotSmartMode.keys()]
        elif isinstance(max_workers_Vllm, (list, tuple)) and max_workers_Vllm:
            target_servers = [str(s) for s in max_workers_Vllm]
        else:
            target_servers = ["super", "ultra", "normal"]

        healthy_servers = []
        for server in target_servers:
            try:
                if tools.llmname(server, shortname=True):
                    healthy_servers.append(server)
            except Exception:
                continue
        if not healthy_servers:
            raise RuntimeError(
                f"❌ 共享 vLLM 后端未就绪: {target_servers} 均未返回模型名。"
                " 请先检查 llm-docker.sh 启动结果、8000/8001/8002 端口与 /v1/models 健康状态。"
            )

    ask_kwargs: Dict[str, Any] = {
        "prompt_list": prompt_list,
        "model": model,
        "token": token,
        "temp": temp,
        "think": think,
        "streamprint": False,
        "check_history_cache": check_history_cache,
        "retry": retry,
        "force_api_do_huge_input_Cloud": force_api_do_huge_input_Cloud,
        "VllmSmartMode": VllmSmartMode,
        "max_workers_Vllm": max_workers_Vllm,
        "prompt_send_weight_VllmNotSmartMode": (
            dict(prompt_send_weight_VllmNotSmartMode)
            if isinstance(prompt_send_weight_VllmNotSmartMode, dict)
            else dict(DEFAULT_SHARED_VLLM_PROMPT_SEND_WEIGHT)
        ),
        "flex": flex,
        "count": True,
        "stats_collector": stats_collector,
        "stats_meta_map": stats_meta_map,
        "note": note,
    }
    if vllm_server_name:
        ask_kwargs["vllm_server_name"] = vllm_server_name
    if top_p != "NotSet":
        ask_kwargs["top_p"] = top_p
    if top_k != "NotSet":
        ask_kwargs["top_k"] = top_k
    if extra_kwargs:
        ask_kwargs.update(extra_kwargs)

    responses = tools.ask_group_link(**ask_kwargs)
    _write_stats_records(runtime_context=runtime_context, phase=phase, stats_collector=stats_collector)
    return [resp if resp else "" for resp in responses]


def summarize_latency_logs(runtime_context: Optional[Dict[str, Any]]) -> None:
    if not runtime_context:
        return

    log_dir = Path(runtime_context["log_dir"])
    rows: List[Dict[str, Any]] = []

    for name, phase in (("request_events.jsonl", "generate"), ("judge_events.jsonl", "judge")):
        path = log_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    payload.setdefault("phase", phase)
                    rows.append(payload)
                except Exception:
                    continue

    if not rows:
        return

    def _safe_mean(values: Iterable[float]) -> float:
        clean = [float(v) for v in values if isinstance(v, (int, float))]
        return statistics.fmean(clean) if clean else 0.0

    def _safe_quantile(values: List[float], q: float) -> float:
        clean = sorted(float(v) for v in values if isinstance(v, (int, float)))
        if not clean:
            return 0.0
        if len(clean) == 1:
            return clean[0]
        idx = (len(clean) - 1) * q
        lo = int(idx)
        hi = min(lo + 1, len(clean) - 1)
        frac = idx - lo
        return clean[lo] * (1 - frac) + clean[hi] * frac

    summary_rows: List[Dict[str, Any]] = []
    for phase in ("generate", "judge"):
        phase_rows = [r for r in rows if r.get("phase") == phase]
        if not phase_rows:
            continue
        durations = [float(r.get("duration_s", 0.0) or 0.0) for r in phase_rows]
        cache_hits = [bool(r.get("cache_hit", False)) for r in phase_rows]
        server_counts: Dict[str, int] = {}
        for row in phase_rows:
            server_name = row.get("server_name") or "unknown"
            server_counts[server_name] = server_counts.get(server_name, 0) + 1

        summary_rows.append(
            {
                "phase": phase,
                "request_count": len(phase_rows),
                "cache_hit_rate": round(sum(cache_hits) / len(cache_hits), 6) if cache_hits else 0.0,
                "avg_latency_s": round(_safe_mean(durations), 6),
                "p50_latency_s": round(_safe_quantile(durations, 0.50), 6),
                "p95_latency_s": round(_safe_quantile(durations, 0.95), 6),
                "server_distribution": json.dumps(server_counts, ensure_ascii=False, sort_keys=True),
            }
        )

    if not summary_rows:
        return

    import csv

    csv_path = log_dir / "latency_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


class SharedLLMBackendManager:

    def __init__(self, shared_backend: Optional[Dict[str, Any]] = None):
        self.config = dict(shared_backend or build_default_shared_backend())
        self.manager = self._build_manager(self.config)
        self._ready = False
        self._started_by_this_run = False

    def _build_manager(self, config: Dict[str, Any]) -> VLLMEnvironmentManager:
        return VLLMEnvironmentManager(
            model_path=config["model_path"],
            skip_verification=False,
            auto_cleanup=bool(config.get("auto_cleanup", False)),
            
            vllm_servers={server: 1 for server in config.get("servers", ["super", "ultra", "normal"])},
        )

    def ensure_backend(self, shared_backend: Optional[Dict[str, Any]] = None) -> None:
        desired_config = dict(shared_backend or self.config or build_default_shared_backend())
        if desired_config != self.config:
            self.config = desired_config
            self.manager = self._build_manager(self.config)
            self._ready = False
            self._started_by_this_run = False
        self.ensure_ready()

    def ensure_ready(self) -> None:
        if self._ready:
            return

        pre_running = False
        try:
            pre_running = self.manager.is_model_running_on_servers()
        except Exception:
            pre_running = False

        if pre_running:
            wait_ok = self.manager.wait_for_ready(max_wait=int(self.config.get("ready_wait_seconds", 120)))
            if wait_ok:
                self._ready = True
                self._started_by_this_run = False
                return

        self.manager.deploy(force=not pre_running)
        self._ready = True
        self._started_by_this_run = True

    def cleanup(self) -> None:
        
        if self._started_by_this_run and self.config.get("auto_cleanup", False):
            self.manager.cleanup()
            self._ready = False


@dataclass
class _ResourceSample:
    server: str
    ts: float
    gpu_records: List[Dict[str, Any]]
    waiting: Any
    running: Any
    kv_usage: Any
    iteration_tokens: Any
    metrics_backend: Any
    health: bool
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        gpu_utils = [record.get("gpu_util") for record in self.gpu_records if isinstance(record.get("gpu_util"), (int, float))]
        mem_used = [record.get("memory_used_mb") for record in self.gpu_records if isinstance(record.get("memory_used_mb"), (int, float))]
        mem_total = [record.get("memory_total_mb") for record in self.gpu_records if isinstance(record.get("memory_total_mb"), (int, float))]
        return {
            "ts": self.ts,
            "ts_iso": datetime.fromtimestamp(self.ts).isoformat(),
            "server": self.server,
            "gpu_records": self.gpu_records,
            "gpu_util_avg": round(statistics.fmean(gpu_utils), 4) if gpu_utils else None,
            "mem_used_total_mb": round(sum(mem_used), 4) if mem_used else None,
            "mem_total_total_mb": round(sum(mem_total), 4) if mem_total else None,
            "waiting": self.waiting,
            "running": self.running,
            "kv_usage": self.kv_usage,
            "iteration_tokens": self.iteration_tokens,
            "metrics_backend": self.metrics_backend,
            "health": self.health,
            "error": self.error,
        }


class ResourceMonitor:

    def __init__(
        self,
        *,
        runtime_context: Optional[Dict[str, Any]],
        servers: Sequence[str],
        interval_seconds: float = 5.0,
    ):
        self.runtime_context = runtime_context
        self.servers = list(servers)
        self.interval_seconds = max(1.0, float(interval_seconds))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run_shell(self, command: str, *, server: Optional[str] = None, timeout_seconds: int = 8) -> subprocess.CompletedProcess:
        if server and server != CURRENT_SERVER_NAME:
            ssh_host = SERVERS[server]["ssh"]
            cmd = ["ssh", ssh_host, command]
        else:
            cmd = ["bash", "-lc", command]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

    def _sample_server(self, server: str) -> _ResourceSample:
        now = time.time()
        gpu_records: List[Dict[str, Any]] = []
        error_messages: List[str] = []

        nvidia_cmd = (
            "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader,nounits"
        )
        try:
            result = self._run_shell(nvidia_cmd, server=server)
            if result.returncode == 0:
                for raw_line in result.stdout.splitlines():
                    parts = [part.strip() for part in raw_line.split(",")]
                    if len(parts) < 5:
                        continue
                    gpu_records.append(
                        {
                            "gpu_index": int(parts[0]),
                            "gpu_name": parts[1],
                            "gpu_util": float(parts[2]),
                            "memory_used_mb": float(parts[3]),
                            "memory_total_mb": float(parts[4]),
                        }
                    )
            else:
                error_messages.append(f"nvidia-smi rc={result.returncode}: {result.stderr.strip()[:160]}")
        except Exception as exc:
            error_messages.append(f"nvidia-smi exception: {type(exc).__name__}: {exc}")

        waiting = running = kv_usage = iteration_tokens = metrics_backend = None
        health = False
        try:
            stats = tools.get_vllm_realtime_stats(server)
            waiting = stats.get("waiting")
            running = stats.get("running")
            kv_usage = stats.get("kv_usage")
            iteration_tokens = stats.get("iteration_tokens")
            metrics_backend = stats.get("metrics_backend")
            health = bool(stats.get("success"))
        except Exception as exc:
            error_messages.append(f"vllm_stats exception: {type(exc).__name__}: {exc}")

        return _ResourceSample(
            server=server,
            ts=now,
            gpu_records=gpu_records,
            waiting=waiting,
            running=running,
            kv_usage=kv_usage,
            iteration_tokens=iteration_tokens,
            metrics_backend=metrics_backend,
            health=health,
            error=" | ".join(error_messages),
        )

    def _loop(self) -> None:
        writer: Optional[JsonlWriter] = None
        if self.runtime_context:
            writer = self.runtime_context.get("resource_writer")

        while not self._stop_event.is_set():
            cycle_start = time.time()
            for server in self.servers:
                sample = self._sample_server(server).as_dict()
                if self.runtime_context:
                    sample.setdefault("run_id", self.runtime_context.get("run_id"))
                    sample.setdefault("method_file", self.runtime_context.get("method_file"))
                    sample.setdefault("method_name", self.runtime_context.get("method_name"))
                if writer is not None:
                    writer.write(sample)

            elapsed = time.time() - cycle_start
            wait_seconds = max(0.0, self.interval_seconds - elapsed)
            self._stop_event.wait(wait_seconds)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="grid_resource_monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval_seconds + 2.0)
