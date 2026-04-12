# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DROPBOX_PATH = os.path.join(os.path.expanduser("~"), "Dropbox")
if DROPBOX_PATH not in sys.path:
    sys.path.insert(0, DROPBOX_PATH)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

TOOLS_FILE = os.path.join(DROPBOX_PATH, "tools.py")
_loaded_tools = sys.modules.get("tools")
if _loaded_tools is None or os.path.abspath(getattr(_loaded_tools, "__file__", "")) != TOOLS_FILE:
    spec = importlib.util.spec_from_file_location("tools", TOOLS_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"❌ 无法加载 Dropbox tools.py: {TOOLS_FILE}")
    _loaded_tools = importlib.util.module_from_spec(spec)
    sys.modules["tools"] = _loaded_tools
    spec.loader.exec_module(_loaded_tools)
tools = _loaded_tools

from vllm_environment_setup import VLLMEnvironmentManager
from shared_eval_backend import lookup_sample_ref, run_logged_asks

try:
    from json_repair import loads as json_repair_loads
except Exception:  
    json_repair_loads = None


def _safe_json_load(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except Exception:
        if json_repair_loads is None:
            raise
        return json_repair_loads(candidate)


def parse_json_payload(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None

    candidates: List[str] = [raw]

    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    candidates.extend(block.strip() for block in code_blocks if block.strip())

    for pattern in (r"(\{[\s\S]*\})", r"(\[[\s\S]*\])"):
        match = re.search(pattern, raw)
        if match:
            candidates.append(match.group(1).strip())

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return _safe_json_load(candidate)
        except Exception:
            continue
    return None


def normalize_relations(raw_relations: Iterable[Any]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    seen = set()

    for item in raw_relations or []:
        if isinstance(item, dict):
            sub = item.get("sub") or item.get("subject") or item.get("source_node_name") or item.get("source")
            rel = item.get("rel") or item.get("relation") or item.get("predicate") or item.get("fact") or item.get("relationship_name")
            obj = item.get("obj") or item.get("object") or item.get("target_node_name") or item.get("target")
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            sub, rel, obj = item[0], item[1], item[2]
        else:
            continue

        sub = str(sub or "").strip()
        rel = str(rel or "").strip()
        obj = str(obj or "").strip()
        if not (sub and rel and obj):
            continue

        key = (sub.lower(), rel.lower(), obj.lower())
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"sub": sub, "rel": rel, "obj": obj})

    return normalized


def sentence_chunk_text(
    content: str,
    target_chars: int = 9000,
    overlap_chars: int = 500,
    min_chunk_chars: int = 500,
) -> List[str]:
    text = str(content or "").strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    normalized_units: List[str] = []
    sentence_splitter = re.compile(r"(?<=[。！？!?\.])\s+")

    for paragraph in paragraphs:
        if len(paragraph) <= target_chars:
            normalized_units.append(paragraph)
            continue
        sentences = [s.strip() for s in sentence_splitter.split(paragraph) if s.strip()]
        if not sentences:
            sentences = [paragraph]
        current = ""
        for sentence in sentences:
            if not current:
                current = sentence
                continue
            tentative = f"{current} {sentence}"
            if len(tentative) <= target_chars:
                current = tentative
            else:
                normalized_units.append(current.strip())
                current = sentence
        if current.strip():
            normalized_units.append(current.strip())

    chunks: List[str] = []
    current = ""
    for unit in normalized_units:
        if not current:
            current = unit
            continue
        tentative = f"{current}\n\n{unit}"
        if len(tentative) <= target_chars:
            current = tentative
            continue
        if len(current.strip()) >= min_chunk_chars:
            chunks.append(current.strip())
        else:
            chunks.append(tentative[:target_chars].strip())
            current = ""
            continue
        overlap = current[-overlap_chars:].strip() if overlap_chars > 0 else ""
        current = f"{overlap}\n\n{unit}".strip() if overlap else unit

    if current.strip():
        chunks.append(current.strip())

    cleaned: List[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk and (not cleaned or cleaned[-1] != chunk):
            cleaned.append(chunk)
    return cleaned


def render_jinja_like_template(template_text: str, context: Dict[str, Any]) -> str:
    rendered = str(template_text)
    for key, value in context.items():
        rendered = rendered.replace(f"{{{{ {key} }}}}", str(value))
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


def load_text(path: os.PathLike[str] | str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class SharedPromptBaseline:

    def __init__(
        self,
        *,
        name: str,
        model: str = "gpt-5-nano",
        token: int = 16 * 1024,
        temp: float = 0.1,
        use_cloud_or_vllm: str = "cloud",
        vllm_model_path: Optional[str] = None,
        shared_llm_backend: Optional[Dict[str, Any]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        **_: Any,
    ):
        self.name = name
        self.shared_llm_backend = dict(shared_llm_backend or {})
        self.runtime_context = runtime_context or {}
        self.use_cloud_or_vllm = (
            "vllm" if self.shared_llm_backend.get("enabled") else str(use_cloud_or_vllm or "cloud").lower()
        )
        self.vllm_model_path = self.shared_llm_backend.get("model_path", vllm_model_path)
        self.model = self.shared_llm_backend.get(
            "model",
            model if self.use_cloud_or_vllm == "cloud" else "local",
        )
        self.token = int(token or 0)
        self.temp = float(temp)
        self.ask_check_history_cache = self.shared_llm_backend.get("check_history_cache", True)
        self.ask_vllm_smart_mode = self.shared_llm_backend.get("smart_mode", True)
        self.ask_max_workers_vllm = self.shared_llm_backend.get("max_workers_vllm", "auto")
        self.ask_prompt_send_weight_vllm = self.shared_llm_backend.get(
            "prompt_send_weight_vllm",
            {"super": 2, "ultra": 4, "normal": 1},
        )
        self.ask_vllm_server_name = self.shared_llm_backend.get("vllm_server_name")
        self.vllm_manager: Optional[VLLMEnvironmentManager] = None
        if self.use_cloud_or_vllm == "vllm" and not self.shared_llm_backend.get("enabled"):
            if not self.vllm_model_path:
                raise ValueError(f"{self.name} 使用 dedicated_vllm 时必须提供 vllm_model_path")
            self.vllm_manager = VLLMEnvironmentManager(self.vllm_model_path)
        print(f"✅ {self.name} 初始化 (backend={self.use_cloud_or_vllm}, model={self.model})")

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        *,
        sample_content: str,
        stage: str,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.vllm_manager:
            self.vllm_manager.ensure_ready()
        prompt_metadata = [dict(lookup_sample_ref(self.runtime_context, sample_content), stage=stage)]
        result = run_logged_asks(
            [messages],
            model=self.model,
            token=self.token,
            temp=self.temp,
            runtime_context=self.runtime_context,
            phase="generate",
            prompt_metadata_list=prompt_metadata,
            check_history_cache=self.ask_check_history_cache,
            VllmSmartMode=self.ask_vllm_smart_mode,
            max_workers_Vllm=self.ask_max_workers_vllm,
            prompt_send_weight_VllmNotSmartMode=self.ask_prompt_send_weight_vllm,
            vllm_server_name=self.ask_vllm_server_name,
            **(extra_kwargs or {}),
        )
        return result[0] if result else ""

    def batch_generate(self, contents: List[str], num_workers: int = 1, **kwargs) -> List[Dict[str, Any]]:
        if not contents:
            return []
        worker_num = max(1, min(int(num_workers or 1), len(contents)))
        if worker_num == 1:
            return [self.generate(content, **kwargs) for content in contents]

        results: List[Optional[Dict[str, Any]]] = [None] * len(contents)
        print(f"🚀 [{self.name}] 文章级并发生成: workers={worker_num}, articles={len(contents)}")
        with ThreadPoolExecutor(max_workers=worker_num, thread_name_prefix=f"{self.name}_batch") as executor:
            future_to_idx = {
                executor.submit(self.generate, content, **kwargs): idx
                for idx, content in enumerate(contents)
            }
            done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                done += 1
                results[idx] = future.result()
                print(f"  ✅ [{self.name}] 批量进度 {done}/{len(contents)}")
        return [item or {"relations": [], "raw_output": ""} for item in results]
