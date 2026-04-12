# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import re
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

from shared_eval_backend import lookup_sample_ref, run_logged_asks

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXTERNAL_REPRO_ROOT = PROJECT_ROOT / "dataset" / "26年2月24日大型复现计划" / "复现成功"

_EXTERNAL_MODULE_CACHE: Dict[str, Any] = {}


def load_external_method_module(relative_path: Sequence[str], module_tag: str):
    file_path = EXTERNAL_REPRO_ROOT.joinpath(*relative_path).resolve()
    cache_key = f"{module_tag}:{file_path}"
    if cache_key in _EXTERNAL_MODULE_CACHE:
        return _EXTERNAL_MODULE_CACHE[cache_key]

    if not file_path.exists():
        raise FileNotFoundError(f"❌ 外部方法文件不存在: {file_path}")

    spec = importlib.util.spec_from_file_location(f"grid_related_{module_tag}", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"❌ 无法为外部方法创建 import spec: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EXTERNAL_MODULE_CACHE[cache_key] = module
    return module


def _try_json_load(text: str):
    if not isinstance(text, str):
        return None

    candidates = [text.strip()]

    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        candidates.append(code_block.group(1).strip())

    brace_match = re.search(r"(\{[\s\S]*\})", text)
    if brace_match:
        candidates.append(brace_match.group(1).strip())

    bracket_match = re.search(r"(\[[\s\S]*\])", text)
    if bracket_match:
        candidates.append(bracket_match.group(1).strip())

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                import json_repair
                return json_repair.loads(candidate)
            except Exception:
                continue
    return None


def _normalize_entities(raw_entities: Any) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    if not isinstance(raw_entities, list):
        return entities

    for item in raw_entities:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                entity = dict(item)
                entity["name"] = name
                entities.append(entity)
        elif isinstance(item, str):
            name = item.strip()
            if name:
                entities.append({"name": name})
    return entities


def _normalize_single_relation(item: Any) -> Dict[str, str] | None:
    if isinstance(item, dict):
        sub = (
            item.get("sub")
            or item.get("subject")
            or item.get("source_node_name")
            or item.get("source")
            or item.get("head")
        )
        rel = (
            item.get("rel")
            or item.get("relation")
            or item.get("predicate")
            or item.get("fact")
            or item.get("edge")
        )
        obj = (
            item.get("obj")
            or item.get("object")
            or item.get("target_node_name")
            or item.get("target")
            or item.get("tail")
        )
    elif isinstance(item, (list, tuple)) and len(item) >= 3:
        sub, rel, obj = item[0], item[1], item[2]
    else:
        return None

    sub = str(sub or "").strip()
    rel = str(rel or "").strip()
    obj = str(obj or "").strip()
    if not (sub and rel and obj):
        return None

    return {"sub": sub, "rel": rel, "obj": obj}


def normalize_relations(raw_relations: Any) -> List[Dict[str, str]]:
    relations: List[Dict[str, str]] = []
    if not isinstance(raw_relations, list):
        return relations

    for item in raw_relations:
        relation = _normalize_single_relation(item)
        if relation:
            relations.append(relation)
    return relations


def dedupe_relations(relations: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    unique: List[Dict[str, str]] = []
    seen = set()
    for item in relations:
        relation = _normalize_single_relation(item)
        if not relation:
            continue
        key = (
            relation["sub"].strip().lower(),
            relation["rel"].strip().lower(),
            relation["obj"].strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(relation)
    return unique


def normalize_prediction(prediction: Any) -> Dict[str, Any]:
    raw_output = prediction if isinstance(prediction, str) else None
    data = prediction

    if isinstance(prediction, str):
        data = _try_json_load(prediction)
    elif prediction is None:
        data = {}

    entities: List[Dict[str, Any]] = []
    relations: List[Dict[str, str]] = []
    extras: Dict[str, Any] = {}

    if isinstance(data, dict):
        entities = _normalize_entities(data.get("entities", []))
        relation_source = (
            data.get("relations")
            if "relations" in data
            else data.get("triplets", data.get("kg", []))
        )
        relations = normalize_relations(relation_source)
        extras = {
            key: value
            for key, value in data.items()
            if key not in {"entities", "relations", "triplets", "kg", "raw_output"}
        }
        if raw_output is None:
            raw_output = data.get("raw_output")
    elif isinstance(data, list):
        normalized_relations = normalize_relations(data)
        if normalized_relations:
            relations = normalized_relations
        else:
            entities = _normalize_entities(data)

    if raw_output is None:
        try:
            raw_output = json.dumps(data, ensure_ascii=False)
        except Exception:
            raw_output = str(prediction)

    result = {
        "entities": entities,
        "relations": dedupe_relations(relations),
        "raw_output": raw_output,
    }
    result.update(extras)
    return result


def parallel_batch_generate(
    generate_fn: Callable[[str], Dict[str, Any]],
    contents: List[str],
    num_workers: int,
    method_name: str,
) -> List[Dict[str, Any]]:
    if not contents:
        return []

    max_workers = max(1, min(int(num_workers or 1), len(contents)))
    results: List[Dict[str, Any]] = [None] * len(contents)  # type: ignore[assignment]

    if max_workers == 1:
        for idx, content in enumerate(contents, 1):
            print(f"  📝 [{method_name}] 串行生成 {idx}/{len(contents)}")
            results[idx - 1] = generate_fn(content)
        return results

    print(f"  🚀 [{method_name}] 启动文章级并发: workers={max_workers}, samples={len(contents)}")
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"{method_name}_kg") as executor:
        futures = {
            executor.submit(generate_fn, content): idx
            for idx, content in enumerate(contents)
        }
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            completed += 1
            try:
                results[idx] = future.result()
            except Exception as exc:
                print(f"  ⚠️ [{method_name}] 样本 {idx} 失败: {exc}")
                results[idx] = {
                    "entities": [],
                    "relations": [],
                    "raw_output": str(exc),
                    "error": str(exc),
                }
            print(f"  🔄 [{method_name}] 批量进度: {completed}/{len(contents)}")
    return results


class ExternalPromptMethodAdapter:

    def __init__(
        self,
        *,
        name: str,
        module_tag: str,
        external_relative_path: Sequence[str],
        shared_llm_backend: Dict[str, Any] | None = None,
        runtime_context: Dict[str, Any] | None = None,
        **kwargs,
    ):
        self.name = name
        self.shared_llm_backend = dict(shared_llm_backend or {})
        self.runtime_context = runtime_context or {}
        self._external_module = load_external_method_module(external_relative_path, module_tag)
        method_cls = getattr(self._external_module, "Method", None)
        if method_cls is None:
            raise AttributeError(f"❌ 外部模块缺少 Method 类: {external_relative_path}")

        wrapped_kwargs = dict(kwargs)
        self._wrapped = method_cls(**wrapped_kwargs)
        self._wrapped._grid_runtime_context = self.runtime_context
        self._wrapped._grid_shared_llm_backend = self.shared_llm_backend
        if self.shared_llm_backend.get("enabled"):
            if hasattr(self._wrapped, "model"):
                self._wrapped.model = self.shared_llm_backend.get("model", getattr(self._wrapped, "model", "local"))
            self._install_logged_llm_bridge()

    def _install_logged_llm_bridge(self) -> None:
        if not hasattr(self._wrapped, "_call_llm"):
            return

        backend_cfg = self.shared_llm_backend
        runtime_context = self.runtime_context

        def _patched_call_llm(inner_self, prompt_or_messages):
            is_batch = False
            if isinstance(prompt_or_messages, list) and prompt_or_messages and isinstance(prompt_or_messages[0], list):
                prompt_list = prompt_or_messages
                is_batch = True
            elif isinstance(prompt_or_messages, list) and prompt_or_messages and isinstance(prompt_or_messages[0], dict):
                prompt_list = [prompt_or_messages]
            else:
                prompt_list = [[{"role": "user", "content": prompt_or_messages}]]

            sample_ref = dict(getattr(inner_self, "_grid_current_sample_ref", {}) or {})
            stage_name = getattr(inner_self, "_grid_current_stage", "generate")
            prompt_metadata_list = []
            for _ in prompt_list:
                metadata = dict(sample_ref)
                metadata["stage"] = stage_name
                prompt_metadata_list.append(metadata)

            responses = run_logged_asks(
                prompt_list,
                model=getattr(inner_self, "model", backend_cfg.get("model", "local")),
                token=int(getattr(inner_self, "token", 8192)),
                temp=float(getattr(inner_self, "temp", 0.1)),
                runtime_context=runtime_context,
                phase="generate",
                prompt_metadata_list=prompt_metadata_list,
                check_history_cache=backend_cfg.get("check_history_cache", True),
                VllmSmartMode=backend_cfg.get("smart_mode", True),
                max_workers_Vllm=backend_cfg.get("max_workers_vllm", "auto"),
                prompt_send_weight_VllmNotSmartMode=backend_cfg.get(
                    "prompt_send_weight_vllm",
                    {"super": 2, "ultra": 4, "normal": 1},
                ),
                vllm_server_name=backend_cfg.get("vllm_server_name"),
            )
            return responses if is_batch else (responses[0] if responses else "")

        self._wrapped._call_llm = types.MethodType(_patched_call_llm, self._wrapped)

    def generate(self, content: str) -> Dict[str, Any]:
        sample_ref = lookup_sample_ref(self.runtime_context, content)
        self._wrapped._grid_current_sample_ref = sample_ref
        self._wrapped._grid_current_stage = "generate"
        try:
            return normalize_prediction(self._wrapped.generate(content))
        finally:
            self._wrapped._grid_current_sample_ref = {}
            self._wrapped._grid_current_stage = None

    def batch_generate(self, contents: List[str], num_workers: int = 1, **kwargs) -> List[Dict[str, Any]]:
        return parallel_batch_generate(self.generate, contents, num_workers=num_workers, method_name=self.name)
