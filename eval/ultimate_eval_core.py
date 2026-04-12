# -*- coding: utf-8 -*-
"""
Filename: ultimate_eval_core.py
Description: Core evaluation module for the packaged GRID benchmark artifact.
Keywords: evaluation, knowledge graph extraction, LLM judge, reporting

Quick Start:
1. YAML-driven execution:
   python %(prog)s --yaml eval/experiment_yaml/smoke.yaml

2. Single-method execution:
   python %(prog)s --method GRID_Ours.py --sources casie --sample_size 3

3. Verification mode:
   python %(prog)s --verify --method Approach_CTINexus.py
"""

import os
import sys
import json
import argparse
import random
import time
import hashlib
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Iterator
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DROPBOX_PATH = os.path.join(os.path.expanduser("~"), 'Dropbox')
if DROPBOX_PATH not in sys.path:
    sys.path.insert(0, DROPBOX_PATH)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

REPO_ROOT = str(Path(SCRIPT_DIR).resolve().parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
METHOD_DIRS = [
    os.path.join(REPO_ROOT, "src", "grid"),
    os.path.join(REPO_ROOT, "src", "baseline"),
]
for method_dir in METHOD_DIRS:
    if method_dir not in sys.path:
        sys.path.insert(0, method_dir)
METHOD_DIR = METHOD_DIRS[0]

from shared_eval_backend import (
    DEFAULT_FULL_METHODS,
    build_content_ref_map,
    build_default_shared_backend,
    build_method_init_kwargs,
    create_runtime_context,
    is_native_baseline,
    run_logged_asks,
    sort_methods_for_async_pipeline,
    summarize_latency_logs,
    uses_shared_qwen,
    SharedLLMBackendManager,
    ResourceMonitor,
)
from src import tools_prompt_nano


DEFAULT_DATASET_PATH = os.path.join(
    DROPBOX_PATH,
    "项目GRID-GIT投稿用/train-data/data/real_testset/benchmark_full249.json",
)

AVAILABLE_SOURCES = ['casie', 'ctinexus', 'malkg', 'securenlp', 'grid']
INPUT_ARTICLE_KG_DIR = os.path.join(SCRIPT_DIR, "InputArticleandKG")
GENERATED_KG_CONTENT_DIR = os.path.join(SCRIPT_DIR, "GeneratedKGContent")
EFFECTIVENESS_RESULT_DIR = os.path.join(SCRIPT_DIR, "EffectivenessResult")


def get_file_md5(file_path: str) -> str:
    """Return the MD5 hash of a file."""
    import hashlib
    if not os.path.exists(file_path):
        return "Unknown"
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def save_detailed_log(dir_name: str, method_file: str, log_data: Dict):
    """Write detailed logs into the packaged legacy log directory."""
    base_dir = os.path.join(GENERATED_KG_CONTENT_DIR, "_LegacyDetailedLogs", dir_name, method_file)
    os.makedirs(base_dir, exist_ok=True)
    
    dataset_name = log_data.get('数据集名称', 'No_Dataset_name')
    dataset_index = log_data.get('数据索引', 'No_Dataset_Index')
    file_name = f"{dataset_name}_{dataset_index}.json"
    
    path = os.path.join(base_dir, file_name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)


def stop_shared_vllm_cluster_for_non_llm_phase() -> None:
    """Stop shared vLLM services before non-LLM baselines reclaim GPU memory."""
    stop_bin = shutil.which("stop") or os.path.expanduser("~/bin/stop")
    if not stop_bin or not os.path.exists(stop_bin):
        print("⚠️ 未找到 stop 命令，跳过三机 vLLM 清理")
        return

    print("🛑 进入非LLM阶段，停止三机 vLLM/llm-docker 容器以释放 GPU ...")
    try:
        result = subprocess.run(
            ["bash", stop_bin, "--only-llm-docker", "-all"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        if result.returncode == 0:
            print("✅ 三机 vLLM 已停止，开始非LLM阶段")
        else:
            print(f"⚠️ 停止 vLLM 返回非零退出码: {result.returncode}")
    except subprocess.TimeoutExpired:
        print("⚠️ 停止三机 vLLM 超时，继续进入非LLM阶段")
    except Exception as exc:
        print(f"⚠️ 停止三机 vLLM 失败: {exc}")

# ================================================================================
# ================================================================================

def is_failed_sample(prediction: Dict) -> bool:
    if isinstance(prediction, str):
        try:
            prediction = json.loads(prediction)
        except json.JSONDecodeError:
            return True  
    
    if not isinstance(prediction, dict):
        return True  
    
    return bool(prediction.get('error'))


def iter_complete_json_arrays(text: str) -> Iterator[str]:
    in_string = False
    escape = False
    depth = 0
    start = None

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start: idx + 1]
                start = None


def iter_array_start_positions(text: str) -> Iterator[int]:
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            yield idx


def is_eval_result_dict_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, dict) for item in value)
        and all(("result" in item) or ("label" in item) or ("decision" in item) for item in value)
    )


def is_eval_result_token_list(value: Any) -> bool:
    allowed = {"TP", "FP", "FN"}
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and item.strip().upper() in allowed for item in value)
    )


def find_eval_result_list(value: Any) -> Optional[List[Dict[str, Any]]]:
    if is_eval_result_dict_list(value):
        normalized: List[Dict[str, Any]] = []
        for item in value:
            result = str(item.get("result") or item.get("label") or item.get("decision") or "").strip().upper()
            if result not in {"TP", "FP", "FN"}:
                return None
            normalized_item = dict(item)
            normalized_item["result"] = result
            normalized.append(normalized_item)
        return normalized
    if is_eval_result_token_list(value):
        return [{"result": str(item).strip().upper()} for item in value]
    if isinstance(value, list):
        for item in value:
            found = find_eval_result_list(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = find_eval_result_list(item)
            if found is not None:
                return found
    return None


def build_json_parse_candidates(response: str) -> List[Tuple[str, str]]:
    import re

    candidates: List[Tuple[str, str]] = []
    seen: set[str] = set()

    def add_candidate(strategy: str, text: str) -> None:
        normalized = str(text or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append((strategy, normalized))

    for match in re.finditer(r"```json\s*([\s\S]*?)(?:```|$)", response, flags=re.IGNORECASE):
        add_candidate("json_fence", match.group(1))

    think_split = response.rsplit("</think>", 1)
    if len(think_split) == 2:
        add_candidate("after_think_tail", think_split[1])

    if "<Fin>" in response:
        add_candidate("before_fin", response.split("<Fin>", 1)[0])

    complete_arrays = list(iter_complete_json_arrays(response))
    for array_text in reversed(complete_arrays):
        add_candidate("complete_array", array_text)

    array_starts = list(iter_array_start_positions(response))
    for start in reversed(array_starts[-12:]):
        add_candidate("partial_array_tail", response[start:])

    add_candidate("full_response", response)
    return candidates


def normalize_relation_records(relations: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    def _visit(value: Any) -> None:
        if isinstance(value, dict):
            if any(key in value for key in ("sub", "rel", "obj")):
                normalized.append(value)
            return
        if isinstance(value, list):
            for item in value:
                _visit(item)

    _visit(relations)
    return normalized


# ================================================================================
# ================================================================================


def load_ultimate_dataset(
    dataset_path: str = None,
    sources: List[str] = None,
    sample_size: int = None,
    seed: int = 42
) -> List[Dict]:
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_PATH
    
    print(f"📂 加载数据集: {os.path.basename(dataset_path)}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"   总样本数: {len(all_data)}")
    
    by_source = defaultdict(list)
    for item in all_data:
        src = item.get('source_approach_provided_dataset', 'unknown')
        by_source[src].append(item)
    
    print(f"   数据来源分布: {dict((k, len(v)) for k, v in by_source.items())}")
    
    if sources and sources != ['all']:
        by_source = {k: v for k, v in by_source.items() if k in sources}
        print(f"   筛选后来源: {list(by_source.keys())}")
    
    random.seed(seed)
    result = []
    
    for src, items in by_source.items():
        if sample_size is not None and sample_size < len(items):
            sampled = random.sample(items, sample_size)
            print(f"   {src}: 采样 {sample_size}/{len(items)}")
        else:
            sampled = items
            print(f"   {src}: 全部 {len(items)}")
        result.extend(sampled)
    
    print(f"✅ 最终样本数: {len(result)}")
    return result


# ================================================================================
# ================================================================================

def load_method(method_file: str, init_kwargs: Optional[Dict[str, Any]] = None):
    import importlib.util
    
    print(f"🔌 加载方法: {method_file}")
    
    if not method_file.endswith('.py'):
        raise ValueError(f"❌ 方法名必须是 .py 文件: {method_file}")
    
    file_path = _resolve_method_path(method_file)

    module_name = method_file.replace('.py', '').replace('-', '_').replace(' ', '_')
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'Method'):
        init_kwargs = init_kwargs or {}
        return normalize_method_interface(module.Method(**init_kwargs))
    else:
        raise AttributeError(f"❌ 文件 {method_file} 中未找到 'Method' 类")


def _resolve_method_path(method_file: str) -> str:
    for method_dir in METHOD_DIRS:
        candidate = os.path.join(method_dir, method_file)
        if os.path.exists(candidate):
            return candidate
    searched = ", ".join(METHOD_DIRS)
    raise FileNotFoundError(f"❌ 方法文件不存在: {method_file} (searched: {searched})")


def normalize_method_interface(method):
    if not hasattr(method, 'generate'):
        raise AttributeError(f"❌ 方法 {type(method).__name__} 缺少 generate() 接口")

    if not hasattr(method, 'single_generate'):
        method.single_generate = method.generate

    if not hasattr(method, 'batch_generate'):
        def _parallel_batch_generate(contents: List[str], num_workers: int = 1, **kwargs) -> List[Dict[str, Any]]:
            method_name = getattr(method, 'name', type(method).__name__)
            if not contents:
                return []

            max_workers = max(1, min(int(num_workers or 1), len(contents)))
            if max_workers == 1:
                results = []
                for idx, content in enumerate(contents, 1):
                    print(f"  📝 [{method_name}] batch单线程 {idx}/{len(contents)}")
                    try:
                        results.append(method.generate(content))
                    except Exception as exc:
                        print(f"  ⚠️ [{method_name}] 单线程样本 {idx} 失败: {exc}")
                        results.append({"entities": [], "relations": [], "raw_output": str(exc), "error": str(exc)})
                return results

            print(f"  🚀 [{method_name}] batch并发启动: workers={max_workers}, samples={len(contents)}")
            results: List[Optional[Dict[str, Any]]] = [None] * len(contents)

            def _generate_single(idx: int, content: str):
                try:
                    return idx, method.generate(content)
                except Exception as exc:
                    print(f"  ⚠️ [{method_name}] 并发样本 {idx + 1} 失败: {exc}")
                    return idx, {"entities": [], "relations": [], "raw_output": str(exc), "error": str(exc)}

            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"{method_name}_batch") as executor:
                futures = {
                    executor.submit(_generate_single, idx, content): idx
                    for idx, content in enumerate(contents)
                }
                completed = 0
                from concurrent.futures import as_completed
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
                    completed += 1
                    if completed % 5 == 0 or completed == len(contents):
                        print(f"  🔄 [{method_name}] batch并发进度: {completed}/{len(contents)}")

            return [item if item is not None else {"entities": [], "relations": [], "raw_output": "", "error": "Unknown batch gap"} for item in results]

        method.batch_generate = _parallel_batch_generate

    method.supported_modes = ['single', 'batch']
    return method



# ================================================================================
# ================================================================================

def resolve_judge_prompt_bundle(prompt_mode: str) -> Dict[str, Any]:
    """
    Resolve the packaged judge prompt bundle from the local prompt module.
    """
    return tools_prompt_nano.get_judge_prompt_bundle(prompt_mode or "grid_judge_fav")


class SimpleKGEvaluator:
    
    def __init__(
        self,
        judge_model: str = 'gpt-5.4-mini',
        prompt_mode: str = 'grid_judge_fav',
        token: int = 64 * 1024,
        temp: float = 0.1,
        think: int = 2,
        flex: bool = False,  
        runtime_context: Optional[Dict[str, Any]] = None,
    ):
        raise RuntimeError(
            "formal 目录已停用内置 SimpleKGEvaluator；请改用 "
            "the KgRewardEvaluator in unified_eval_executor.py, "
            "或在 YAML 中设置 judge_backend=kg_reward。"
        )
    
    def _default_recall_prompt(self):
        return """You are evaluating the RECALL of a knowledge graph extraction.

For each item in the Ground Truth, determine if it is correctly captured in the Prediction.
- TP (True Positive): Item in Ground Truth is correctly found in Prediction
- FN (False Negative): Item in Ground Truth is NOT found in Prediction

Output a JSON array:
```json
[
  {"index": "truth_0", "result": "TP/FN", "reason": "brief reason"},
  ...
]
```

"""
    
    def _default_precision_prompt(self):
        return """You are evaluating the PRECISION of a knowledge graph extraction.

For each item in the Prediction, determine if it is correct based on Ground Truth and article.
- TP (True Positive): Predicted item is correct
- FP (False Positive): Predicted item is incorrect/hallucinated

Output a JSON array:
```json
[
  {"index": "pred_0", "result": "TP/FP", "reason": "brief reason"},
  ...
]
```

"""
    
    def _format_kg(self, relations: List[Dict], prefix: str = "item") -> str:
        items = []
        for i, rel in enumerate(normalize_relation_records(relations)):
            items.append({
                'index': f'{prefix}_{i}',
                'sub': rel.get('sub', ''),
                'rel': rel.get('rel', ''),
                'obj': rel.get('obj', '')
            })
        return json.dumps(items, indent=2, ensure_ascii=False)
    
    def _parse_eval_result(self, response: str, eval_type: str) -> Dict:
        from json_repair import loads as json_repair_loads
        
        if not str(response or "").strip():
            return {'score': 0.0, 'TP': 0, 'error_type': 0, 'parse_success': False, 'error': 'Empty response'}

        parse_errors: List[str] = []
        for strategy, candidate_text in build_json_parse_candidates(str(response)):
            try:
                parsed = json_repair_loads(candidate_text)
            except Exception as exc:
                parse_errors.append(f"{strategy}: {exc}")
                continue

            results = find_eval_result_list(parsed)
            if results is None:
                parse_errors.append(f"{strategy}: No result list found in parsed object")
                continue

            tp = sum(1 for item in results if str(item.get('result', '')).upper() == 'TP')
            total = len(results)
            score = tp / total if total > 0 else 0.0
            out = {
                'score': score,
                'TP': tp,
                'total': total,
                'raw_results': results,
                'parse_success': True,
                'parser_strategy': strategy,
            }
            out['FP' if eval_type == 'precision' else 'FN'] = total - tp
            return out

        return {
            'score': 0.0,
            'TP': 0,
            'error_type': 0,
            'parse_success': False,
            'error': '; '.join(parse_errors[:6]) if parse_errors else 'No JSON found',
        }
    
    def evaluate(
        self,
        ground_truth: List[Dict],
        prediction: Dict,
        article_text: str = "",
        entities_from_extra: List[Dict] = None,
        source: str = None
    ) -> Dict:
        if isinstance(prediction, list):
            prediction = {'relations': prediction, 'entities': []}
        elif prediction is None:
            prediction = {'relations': [], 'entities': []}
        elif not isinstance(prediction, dict):
            prediction = {'relations': [], 'entities': [], 'error': f'Unsupported prediction type: {type(prediction).__name__}'}

        gt_relations = normalize_relation_records(ground_truth)
        pred_relations = normalize_relation_records(prediction.get('relations', []))
        
        if not pred_relations:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'pred_count': 0,
                'gt_count': len(gt_relations),
                'error': 'Empty prediction'
            }
        
        gt_formatted = self._format_kg(gt_relations, "truth")
        pred_formatted = self._format_kg(pred_relations, "pred")
        
        entities_set = set()
        for rel in gt_relations:
            entities_set.add(rel.get('sub', ''))
            entities_set.add(rel.get('obj', ''))
        
        final_entities = [{'name': e} for e in sorted(entities_set) if e]  
        if source == 'grid' and entities_from_extra:
            existing_names = set(e.get('name', '') for e in final_entities)
            for extra_e in entities_from_extra:
                name = extra_e.get('name', '')
                if name and name not in existing_names:
                    final_entities.append(extra_e)
        
        entities_str = json.dumps(final_entities, indent=2, ensure_ascii=False)
        
        article_context_text = article_text if article_text else 'N/A'

        recall_content = f"""{self.recall_prompt}
--- Ground Truth Entities ---
{entities_str}

--- Ground Truth Relations ---
{gt_formatted}

--- Prediction Pool Relations ---
{pred_formatted}

--- Article (Context) ---
{article_context_text}
"""
        
        precision_content = f"""{self.precision_prompt}
--- Prediction Relations to Evaluate ---
{pred_formatted}

--- Ground Truth Reference Entities ---
{entities_str}

--- Ground Truth Reference Relations ---
{gt_formatted}

--- Article (Context) ---
{article_context_text}
"""
        
        prompts = [
            [{"role": "user", "content": recall_content}],
            [{"role": "user", "content": precision_content}]
        ]
        
        debug_file_path = "DEBUG_JUDGE_PROMPT.log"
        if not os.path.exists(debug_file_path):
            with open(debug_file_path, "w", encoding="utf-8") as f:
                f.write("=== Recall Prompt ===\n")
                f.write(recall_content)
                f.write("\n\n=== Precision Prompt ===\n")
                f.write(precision_content)
                f.write("\n\n")
        
        prompt_metadata_list = [
            {"sample_index": 0, "stage": "judge_recall", "eval_type": "recall"},
            {"sample_index": 0, "stage": "judge_precision", "eval_type": "precision"},
        ]
        start_eval = time.time()
        responses = run_logged_asks(
            prompts,
            model=self.judge_model,
            token=self.token,
            temp=self.temp,
            think=self.think,
            runtime_context=self.runtime_context,
            phase="judge",
            prompt_metadata_list=prompt_metadata_list,
            retry=True,
            check_history_cache=True,
            force_api_do_huge_input_Cloud=True,
            flex=self.flex,
        )
        eval_duration = time.time() - start_eval
        
        recall_result = self._parse_eval_result(responses[0] if responses else '', 'recall')
        precision_result = self._parse_eval_result(responses[1] if len(responses) > 1 else '', 'precision')
        
        recall = recall_result['score']
        precision = precision_result['score']
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pred_count': len(pred_relations),
            'gt_count': len(gt_relations),
            'recall_details': recall_result,
            'precision_details': precision_result,
            'recall_prompt': recall_content,       
            'precision_prompt': precision_content, 
            'recall_response': responses[0] if responses else '',           
            'precision_response': responses[1] if len(responses) > 1 else '', 
            'evaluation_time': eval_duration,
        }

    def batch_evaluate(
        self,
        data_list: List[Dict],
        predictions: List[Dict]
    ) -> List[Dict]:
        all_prompts = []
        indices = [] 
        
        print(f"   🧑‍⚖️ 准备批量评估 {len(data_list)} 个样本 (共 {len(data_list)*2} 个请求)...")
        
        processed_predictions = []
        for pred in predictions:
            if isinstance(pred, str):
                try:
                    parsed_pred = json.loads(pred)
                    if isinstance(parsed_pred, list):
                        processed_predictions.append({'relations': parsed_pred, 'entities': []})
                    elif isinstance(parsed_pred, dict):
                        processed_predictions.append(parsed_pred)
                    else:
                        processed_predictions.append({'relations': [], 'entities': [], 'error': f'Unsupported JSON root: {type(parsed_pred).__name__}'})
                except json.JSONDecodeError:
                    processed_predictions.append({'relations': [], 'entities': [], 'error': 'JSON parse failed'})
            elif isinstance(pred, list):
                processed_predictions.append({'relations': pred, 'entities': []})
            elif isinstance(pred, dict):
                processed_predictions.append(pred)
            elif pred is None:
                processed_predictions.append({'relations': [], 'entities': []})
            else:
                processed_predictions.append({'relations': [], 'entities': [], 'error': f'Unsupported prediction type: {type(pred).__name__}'})
        predictions = processed_predictions
        normalized_pred_counts: List[int] = []
        normalized_gt_counts: List[int] = []
        
        for i, (item, pred) in enumerate(zip(data_list, predictions)):
            content = item.get('content', '')
            ground_truth = item.get('ground_truth', [])
            source = item.get('source_approach_provided_dataset', 'unknown')
            extra_info = item.get('extra_info', {})
            
            ground_truth_relations = normalize_relation_records(ground_truth)
            pred_relations = normalize_relation_records(pred.get('relations', []))
            normalized_pred_counts.append(len(pred_relations))
            normalized_gt_counts.append(len(ground_truth_relations))
            
            gt_formatted = self._format_kg(ground_truth_relations, "truth")
            pred_formatted = self._format_kg(pred_relations, "pred")
            
            entities_set = set()
            for rel in ground_truth_relations:
                entities_set.add(rel.get('sub', ''))
                entities_set.add(rel.get('obj', ''))
            
            final_entities = [{'name': e} for e in sorted(entities_set) if e]  
            if source == 'grid':
                entities_from_extra = extra_info.get('实体列表(更新别名后)', [])
                existing_names = set(e.get('name', '') for e in final_entities)
                for extra_e in entities_from_extra:
                    name = extra_e.get('name', '')
                    if name and name not in existing_names:
                        final_entities.append(extra_e)
            
            entities_str = json.dumps(final_entities, indent=2, ensure_ascii=False)
            
            article_context_text = content if content else 'N/A'

            # Recall prompt
            recall_content = f"""{self.recall_prompt}
--- Ground Truth Entities ---
{entities_str}

--- Ground Truth Relations ---
{gt_formatted}

--- Prediction Pool Relations ---
{pred_formatted}

--- Article (Context) ---
{article_context_text}
"""
            # Precision prompt
            precision_content = f"""{self.precision_prompt}
--- Prediction Relations to Evaluate ---
{pred_formatted}

--- Ground Truth Reference Entities ---
{entities_str}

--- Ground Truth Reference Relations ---
{gt_formatted}

--- Article (Context) ---
{article_context_text}
"""
            all_prompts.append([{"role": "user", "content": recall_content}])
            all_prompts.append([{"role": "user", "content": precision_content}])
            indices.append((i, 'recall'))
            indices.append((i, 'precision'))
            
        prompt_metadata_list = []
        for sample_index, eval_type in indices:
            item = data_list[sample_index]
            extra_info = item.get('extra_info', {}) or {}
            prompt_metadata_list.append({
                "sample_index": sample_index,
                "item_id": extra_info.get('file_name', f'item_{sample_index}'),
                "source": item.get('source_approach_provided_dataset', 'unknown'),
                "stage": f"judge_{eval_type}",
                "eval_type": eval_type,
            })

        start_eval = time.time()
        responses = []

        responses = run_logged_asks(
            all_prompts,
            model=self.judge_model,
            token=self.token,
            temp=self.temp,
            think=self.think,
            runtime_context=self.runtime_context,
            phase="judge",
            prompt_metadata_list=prompt_metadata_list,
            retry=True,
            check_history_cache=True,
            force_api_do_huge_input_Cloud=True,
            flex=self.flex,
        )

        eval_duration = time.time() - start_eval
        
        final_scores = [{} for _ in range(len(data_list))]
        for resp, (idx, eval_type) in zip(responses, indices):
            res = self._parse_eval_result(resp if resp else '', eval_type)
            final_scores[idx][f"{eval_type}_result"] = res
            
        results = []
        for i, scores in enumerate(final_scores):
            recall_res = scores.get('recall_result', {'score': 0.0})
            precision_res = scores.get('precision_result', {'score': 0.0})
            
            recall = recall_res.get('score', 0.0)
            precision = precision_res.get('score', 0.0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results.append({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pred_count': normalized_pred_counts[i],
                'gt_count': normalized_gt_counts[i],
                'recall_details': recall_res,
                'precision_details': precision_res,
                'recall_prompt': all_prompts[2 * i][0]['content'],
                'precision_prompt': all_prompts[2 * i + 1][0]['content'],
                'recall_response': responses[2 * i] if len(responses) > 2 * i else '',
                'precision_response': responses[2 * i + 1] if len(responses) > 2 * i + 1 else '',
                'evaluation_time': eval_duration / max(len(data_list), 1),
            })
            
        return results



# ================================================================================
# ================================================================================

def run_verification(method, test_content: str = None):
    if test_content is None:
        target_file = os.path.join(DROPBOX_PATH, "项目GRID/dataset/LLM4CTI/格式化_根据关系删除和添加了实体_更新了别名/rq1_jun2025_index_1.json")
        try:
            if os.path.exists(target_file):
                print(f"📖 读取验证文件: {target_file}")
                with open(target_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    full_content = data.get("正文", "")
                    test_content = full_content[:1000]
                    print(f"📝 截取前1000字符作为输入 (总长 {len(full_content)})")
            else:
                 print(f"⚠️ 验证文件不存在: {target_file}")
        except Exception as e:
            print(f"⚠️ 读取验证文件失败: {e}")

        if test_content is None:
            test_content = "The attacker used the EternalBlue exploit to compromise the Windows 10 system."
    
    print(f"\n🧪 开始验证模式...")
    print(f"📝 测试请求内容(前100字): {test_content[:100]}...")
    
    try:
        start_t = time.time()
        result = method.generate(test_content)
        duration = time.time() - start_t
        
        print("\n" + "="*40)
        print("✅ 验证成功!")
        print(f"⏱️  生成耗时: {duration:.2f}s")
        print(f"📊 提取统计: {len(result.get('entities', []))} 实体 | {len(result.get('relations', []))} 关系")
        print("="*40)
        
        relations = result.get('relations', [])
        if relations:
            print("\n🔗 关系提取展示 (前5条):")
            for i, rel in enumerate(relations[:5]):
                print(f"   [{i+1}] {rel.get('sub')} -->> {rel.get('rel')} -->> {rel.get('obj')}")
        else:
            print("\n⚠️  未提取到任何关系。")

        print(f"\n📄 原始输出预览 (前300字):\n{'-'*20}\n{str(result.get('raw_output', ''))[:300]}...\n{'-'*20}")
        
    except Exception as e:
        print(f"\n❌ 验证过程中出现异常")
        print(f"⚠️  错误类型: {type(e).__name__}")
        print(f"💡 错误信息: {e}")
        import traceback
        print("\n🔍 详细堆栈:")
        traceback.print_exc()


def _initialize_batch_runtime(
    data: List[Dict],
    method,
    evaluator: Optional[SimpleKGEvaluator],
    runtime_context: Optional[Dict[str, Any]],
) -> None:
    if runtime_context is None:
        return
    runtime_context["content_ref_map"] = build_content_ref_map(data)
    runtime_context["sample_count"] = len(data)
    if hasattr(method, "runtime_context"):
        method.runtime_context = runtime_context
    if evaluator is not None and hasattr(evaluator, "runtime_context"):
        evaluator.runtime_context = runtime_context


def _build_cache_manager_for_method(method_file: Optional[str]):
    if not method_file:
        return None
    try:
        method_path = _resolve_method_path(method_file)
        method_dir = os.path.dirname(method_path)
        sys.path.insert(0, method_dir)
        from article_io_cache_parser import KGCacheManager
        return KGCacheManager(method_file, method_path)
    except ImportError as e:
        print(f"   ⚠️ 缓存模块加载失败: {e}")
        return None


def _resolve_method_model_name(method) -> str:
    if hasattr(method, 'model'):
        return getattr(method, 'model')
    if hasattr(method, 'config') and getattr(method, 'config', {}).get('model'):
        return method.config.get('model')
    return "Unknown"


def _run_generation_phase(
    data: List[Dict],
    method,
    *,
    method_file: Optional[str],
    use_cache: bool,
    num_workers: int,
    runtime_context: Optional[Dict[str, Any]],
    force_batch_single_thread: bool = False,
) -> Dict[str, Any]:
    total = len(data)
    start_time = time.time()
    contents = [item.get('content', '') for item in data]

    _initialize_batch_runtime(data, method, None, runtime_context)

    cache_manager = _build_cache_manager_for_method(method_file)

    predictions = [None] * total
    generation_times = [0.0] * total
    cache_hit_flags = [False] * total
    effective_batch_workers = 1 if force_batch_single_thread else num_workers

    if use_cache and cache_manager:
        print(f"📖 从缓存加载知识图谱...")
        cached_results, uncached_indices = cache_manager.batch_load(contents)

        for idx, result in cached_results.items():
            predictions[idx] = result
            generation_times[idx] = 0.0
            cache_hit_flags[idx] = True

        print(f"   ✅ 缓存命中: {len(cached_results)}/{total}, 未命中: {len(uncached_indices)}")

        if uncached_indices:
            print(f"   🔧 生成 {len(uncached_indices)} 个未缓存的知识图谱...")
            uncached_contents = [contents[i] for i in uncached_indices]

            if hasattr(method, 'batch_generate'):
                uncached_preds = method.batch_generate(uncached_contents, num_workers=effective_batch_workers)
                for i, idx in enumerate(uncached_indices):
                    predictions[idx] = uncached_preds[i]
            elif num_workers > 1 and not force_batch_single_thread:
                mp_results = _multiprocess_generate(method, uncached_contents, num_workers)
                for i, idx in enumerate(uncached_indices):
                    predictions[idx] = mp_results[i][0]
                    generation_times[idx] = mp_results[i][1]
                    cache_hit_flags[idx] = False
            else:
                for i, idx in enumerate(uncached_indices):
                    start_gen = time.time()
                    predictions[idx] = method.single_generate(uncached_contents[i])
                    generation_times[idx] = time.time() - start_gen
                    cache_hit_flags[idx] = False

            for idx in uncached_indices:
                if cache_manager and predictions[idx]:
                    cache_manager.save(contents[idx], predictions[idx])

    else:
        print(f"🔧 批量生成 {total} 个知识图谱...")
        if hasattr(method, 'batch_generate'):
            predictions = method.batch_generate(contents, num_workers=effective_batch_workers)
        elif num_workers > 1 and not force_batch_single_thread:
            mp_results = _multiprocess_generate(method, contents, num_workers)
            for i, (result, gen_time, hit) in enumerate(mp_results):
                predictions[i] = result
                generation_times[i] = gen_time
                cache_hit_flags[i] = hit
        else:
            print("   ⚠️ 方法不支持 batch_generate，将采用串行生成...")
            for i, content in enumerate(contents):
                print(f"   [{i+1}/{total}] 正在生成...")
                start_gen = time.time()
                predictions[i] = method.single_generate(content)
                generation_times[i] = time.time() - start_gen

        if cache_manager:
            saved = cache_manager.batch_save(contents, predictions)
            print(f"   💾 已保存 {saved} 个结果到缓存")

    try:
        method_path = _resolve_method_path(method_file) if method_file else "Unknown"
    except Exception:
        method_path = method_file or "Unknown"
    method_md5 = get_file_md5(method_path)
    current_model_name = _resolve_method_model_name(method)

    print(f"📝 保存生成日志 (Level 1)...")
    for i, (item, pred) in enumerate(zip(data, predictions)):
        source = item.get('source_approach_provided_dataset', 'unknown')
        extra_info = item.get('extra_info', {})
        item_id = extra_info.get('file_name', f'item_{i}')
        gen_log = {
            "py输入内容": item.get('content', ''),
            "py原始输出": pred.get('raw_output', '') if isinstance(pred, dict) else str(pred),
            "解析知识图谱": {"entities": pred.get('entities', []), "relations": pred.get('relations', [])} if isinstance(pred, dict) else {},
            "方法文件名": method_file,
            "方法文件MD5": method_md5,
            "数据集名称": source,
            "数据索引": item_id,
            "记录时间": datetime.now().isoformat(),
            "生成耗时": generation_times[i],
            "模型名称": current_model_name,
            "是否命中缓存": cache_hit_flags[i],
        }
        save_detailed_log("特定方法的生成内容", method_file or "UnknownMethod", gen_log)

    return {
        "data": data,
        "predictions": predictions,
        "generation_times": generation_times,
        "cache_hit_flags": cache_hit_flags,
        "method_file": method_file,
        "method_md5": method_md5,
        "current_model_name": current_model_name,
        "method_start_time": start_time,
    }


def _finalize_judge_and_results(
    generation_bundle: Dict[str, Any],
    evaluator: SimpleKGEvaluator,
    output_dir: str,
) -> List[Dict]:
    data = generation_bundle["data"]
    predictions = generation_bundle["predictions"]
    generation_times = generation_bundle["generation_times"]
    cache_hit_flags = generation_bundle["cache_hit_flags"]
    method_file = generation_bundle["method_file"]
    method_md5 = generation_bundle["method_md5"]
    current_model_name = generation_bundle["current_model_name"]
    start_time = generation_bundle["method_start_time"]

    total = len(data)
    print(f"🧑‍⚖️ 批量评估 {total} 个预测结果...")
    scores_list = evaluator.batch_evaluate(data, predictions)

    results = []
    for i, (item, pred, scores) in enumerate(zip(data, predictions, scores_list)):
        source = item.get('source_approach_provided_dataset', 'unknown')
        extra_info = item.get('extra_info', {})
        item_id = extra_info.get('file_name', f'item_{i}')

        gen_log = {
            "py输入内容": item.get('content', ''),
            "py原始输出": pred.get('raw_output', '') if isinstance(pred, dict) else str(pred),
            "解析知识图谱": {"entities": pred.get('entities', []), "relations": pred.get('relations', [])} if isinstance(pred, dict) else {},
            "方法文件名": method_file,
            "方法文件MD5": method_md5,
            "数据集名称": source,
            "数据索引": item_id,
            "记录时间": datetime.now().isoformat(),
            "生成耗时": generation_times[i],
            "模型名称": current_model_name,
            "是否命中缓存": cache_hit_flags[i]
        }

        eval_log = gen_log.copy()
        eval_log.update({
            "标准答案": item.get('ground_truth', []),
            "评估提示词": {
                "Recall_Prompt": scores.get('recall_prompt', ''),
                "Precision_Prompt": scores.get('precision_prompt', '')
            },
            "评估原始输出": {
                "Recall_Response": scores.get('recall_response', ''),
                "Precision_Response": scores.get('precision_response', '')
            },
            "精确率": scores['precision'],
            "召回率": scores['recall'],
            "F1值": scores['f1'],
            "提示词模式": evaluator.prompt_mode,
            "裁判模型": evaluator.judge_model,
            "是否提取失败": is_failed_sample(pred),
            "评估耗时": scores.get('evaluation_time', 0)
        })
        save_detailed_log("特定方法的生成内容-进一步评估", method_file or "UnknownMethod", eval_log)

        result = {
            'item_id': item_id,
            'source': source,
            'content_length': len(item.get('content', '')),
            'gt_count': len(item.get('ground_truth', [])),
            'pred_count': scores['pred_count'],
            'precision': scores['precision'],
            'recall': scores['recall'],
            'f1': scores['f1'],
            'prediction': pred,
            'ground_truth': item.get('ground_truth', []),
            'time_seconds': (time.time() - start_time) / max(total, 1),
            'is_failed': is_failed_sample(pred)
        }
        results.append(result)

        item_dir = os.path.join(output_dir, f"{source}_{item_id}")
        os.makedirs(item_dir, exist_ok=True)
        with open(os.path.join(item_dir, 'result.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    aggregate_results(results, output_dir)
    return results


# ================================================================================
# ================================================================================

def run_batch_evaluation(
    data: List[Dict],
    method,
    evaluator: SimpleKGEvaluator,
    output_dir: str,
    method_file: str = None,
    use_cache: bool = False,
    num_workers: int = 1,
    skip_eval: bool = False,
    runtime_context: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    total = len(data)

    print(f"\n📦 开始全量批处理评估 (共 {total} 条)...")
    print(f"   缓存模式: {'使用缓存' if use_cache else '重新生成'}")
    print(f"   多进程: {num_workers if num_workers > 1 else '单进程'}")
    print("=" * 60)

    _initialize_batch_runtime(data, method, evaluator, runtime_context)
    generation_bundle = _run_generation_phase(
        data,
        method,
        method_file=method_file,
        use_cache=use_cache,
        num_workers=num_workers,
        runtime_context=runtime_context,
        force_batch_single_thread=False,
    )

    if skip_eval:
        print(f"⏩ 跳过评估步骤 (仅生成验证模式)")
        return []

    results = _finalize_judge_and_results(generation_bundle, evaluator, output_dir)
    total_time = time.time() - start_time
    print(f"\n✅ 批量评估完成! 总用时: {total_time/60:.1f} 分钟")

    return results


def _build_output_dir(
    method_file: str,
    *,
    explicit_output_dir: Optional[str],
    is_multi_method: bool,
) -> str:
    method_stem = os.path.splitext(method_file)[0]
    if explicit_output_dir:
        return explicit_output_dir if not is_multi_method else os.path.join(explicit_output_dir, method_stem)
    return os.path.join(EFFECTIVENESS_RESULT_DIR, method_stem)


def _judge_finalize_worker(
    *,
    generation_bundle: Dict[str, Any],
    evaluator: SimpleKGEvaluator,
    output_dir: str,
    runtime_context: Optional[Dict[str, Any]],
) -> List[Dict]:
    try:
        results = _finalize_judge_and_results(generation_bundle, evaluator, output_dir)
        aggregate_results(results, output_dir)
        return results
    finally:
        summarize_latency_logs(runtime_context)


def _run_single_method_generation(
    *,
    data: List[Dict],
    method_file: str,
    args,
    shared_backend: Dict[str, Any],
    shared_vllm_servers: List[str],
    run_id: str,
    is_multi_method: bool,
) -> Tuple[object, SimpleKGEvaluator, Dict[str, Any], Dict[str, Any], str]:
    output_dir = _build_output_dir(
        method_file,
        explicit_output_dir=args.output_dir,
        is_multi_method=is_multi_method,
    )
    runtime_context = create_runtime_context(
        run_id=run_id,
        method_file=method_file,
        output_dir=output_dir,
        resource_interval_seconds=getattr(args, 'resource_monitor_interval', 5.0),
    )

    method = load_method(
        method_file,
        init_kwargs=build_method_init_kwargs(
            method_file,
            shared_backend=shared_backend if uses_shared_qwen(method_file) else None,
            runtime_context=runtime_context,
        ),
    )
    if hasattr(method, 'name'):
        runtime_context['method_name'] = method.name

    raise RuntimeError(
        "formal 目录已停用核心脚本内置 judge 流程；请改用 "
        "unified_eval_executor.py + judge_backend=kg_reward."
    )

    resource_monitor = ResourceMonitor(
        runtime_context=runtime_context,
        servers=shared_vllm_servers,
        interval_seconds=getattr(args, 'resource_monitor_interval', 5.0),
    )

    try:
        resource_monitor.start()
        generation_bundle = _run_generation_phase(
            data,
            method,
            method_file=method_file,
            use_cache=getattr(args, 'use_cache', False),
            num_workers=getattr(args, 'num_workers', 1),
            runtime_context=runtime_context,
            force_batch_single_thread=uses_shared_qwen(method_file),
        )
    finally:
        resource_monitor.stop()

    return method, evaluator, runtime_context, generation_bundle, output_dir


def run_async_method_pipeline(
    *,
    data: List[Dict],
    methods_to_run: List[str],
    args,
    shared_backend: Dict[str, Any],
    shared_backend_manager,
    shared_vllm_servers: List[str],
    run_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict]]]:
    sorted_methods = sort_methods_for_async_pipeline(methods_to_run)
    all_method_results: List[Dict[str, Any]] = []
    all_method_detailed_results: Dict[str, List[Dict]] = {}
    pending_judges: List[Tuple[Any, Dict[str, Any]]] = []
    non_llm_phase_started = False

    def _flush_one_judge(future, payload: Dict[str, Any]) -> None:
        results = future.result()
        method_file = payload['method_file']
        method = payload['method']
        output_dir = payload['output_dir']
        runtime_context = payload['runtime_context']

        avg_precision = sum(r.get('precision', 0) for r in results) / len(results) if results else 0
        avg_recall = sum(r.get('recall', 0) for r in results) / len(results) if results else 0
        avg_f1 = sum(r.get('f1', 0) for r in results) / len(results) if results else 0

        all_method_results.append({
            'method': method_file,
            'sample_count': len(results),
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'output_dir': output_dir
        })
        all_method_detailed_results[method_file] = results

        print(f"📁 结果保存至: {output_dir}")

        if hasattr(method, 'cleanup'):
            method.cleanup()

    judge_workers = max(1, len(sorted_methods))
    with ThreadPoolExecutor(max_workers=judge_workers, thread_name_prefix="judge_pipeline") as judge_executor:
        for method_idx, method_file in enumerate(sorted_methods, 1):
            is_llm_method = uses_shared_qwen(method_file)
            is_native_method = is_native_baseline(method_file)
            if len(sorted_methods) > 1:
                stage_label = "LLM优先阶段" if not is_native_method else "非LLM收尾阶段"
                print(f"\n{'='*60}")
                print(f"📊 [{method_idx}/{len(sorted_methods)}] {stage_label}: {method_file}")
                print("=" * 60)

            if is_native_method and not non_llm_phase_started:
                stop_shared_vllm_cluster_for_non_llm_phase()
                non_llm_phase_started = True

            if is_llm_method and shared_backend_manager is not None:
                print("🔌 检查共享 vLLM 后端就绪状态...")
                shared_backend_manager.ensure_ready()

            print(f"\n🚀 初始化并生成: {method_file} ...")
            method, evaluator, runtime_context, generation_bundle, output_dir = _run_single_method_generation(
                data=data,
                method_file=method_file,
                args=args,
                shared_backend=shared_backend,
                shared_vllm_servers=shared_vllm_servers,
                run_id=run_id,
                is_multi_method=True,
            )

            if getattr(args, 'skip_eval', False):
                results: List[Dict] = []
                all_method_results.append({
                    'method': method_file,
                    'sample_count': len(results),
                    'avg_precision': 0,
                    'avg_recall': 0,
                    'avg_f1': 0,
                    'output_dir': output_dir
                })
                all_method_detailed_results[method_file] = results
                print(f"📁 结果保存至: {output_dir}")
                if hasattr(method, 'cleanup'):
                    method.cleanup()
                summarize_latency_logs(runtime_context)
                continue

            judge_payload = {
                'method_file': method_file,
                'method': method,
                'runtime_context': runtime_context,
                'output_dir': output_dir,
            }
            judge_future = judge_executor.submit(
                _judge_finalize_worker,
                generation_bundle=generation_bundle,
                evaluator=evaluator,
                output_dir=output_dir,
                runtime_context=runtime_context,
            )
            pending_judges.append((judge_future, judge_payload))

        for judge_future, judge_payload in pending_judges:
            _flush_one_judge(judge_future, judge_payload)

    return all_method_results, all_method_detailed_results


def _multiprocess_generate(method, contents: List[str], num_workers: int, timeout_per_item: int = 300) -> List[tuple]:
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
    
    if hasattr(method, 'vllm_manager') and method.vllm_manager is not None:
        print(f"   🔌 预先部署 vLLM 服务...")
        method.vllm_manager.ensure_ready()
        print(f"   ✅ vLLM 服务已就绪，开始并发处理...")
    
    print(f"   🔄 使用 {num_workers} 线程并发生成 (超时: {timeout_per_item}s/任务)...")
    
    def _generate_single(idx_content):
        import random
        import time as time_module
        
        idx, content = idx_content
        
        delay = random.uniform(0, 5)
        time_module.sleep(delay)
        
        start_gen = time_module.time()
        cache_hit = False
        
        try:
            result = method.generate(content)
            if isinstance(result, dict) and result.get('_cache_hit'):
                cache_hit = True
                del result['_cache_hit']  
            gen_time = time_module.time() - start_gen
            return idx, result, gen_time, cache_hit
        except Exception as e:
            gen_time = time_module.time() - start_gen
            print(f"   ⚠️ 生成失败 [{idx}]: {e}")
            return idx, {"entities": [], "relations": [], "error": str(e)}, gen_time, False
    
    results = [None] * len(contents)
    generation_times = [0.0] * len(contents)
    cache_hits = [False] * len(contents)
    completed = 0
    timeout_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_generate_single, (i, c)): i for i, c in enumerate(contents)}
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result_tuple = future.result(timeout=timeout_per_item)
                _, result, gen_time, cache_hit = result_tuple
                results[idx] = result
                generation_times[idx] = gen_time
                cache_hits[idx] = cache_hit
            except FuturesTimeoutError:
                timeout_count += 1
                print(f"   ⏰ 任务 [{idx}] 超时 ({timeout_per_item}s)，跳过")
                results[idx] = {"entities": [], "relations": [], "error": f"Timeout after {timeout_per_item}s"}
                generation_times[idx] = timeout_per_item
                cache_hits[idx] = False
            except Exception as e:
                print(f"   ❌ 任务 [{idx}] 异常: {e}")
                results[idx] = {"entities": [], "relations": [], "error": str(e)}
                generation_times[idx] = 0.0
                cache_hits[idx] = False
            
            completed += 1
            if completed % 5 == 0 or completed == len(contents):
                print(f"   [{completed}/{len(contents)}] 线程池并发处理中... (超时: {timeout_count})")
    
    if timeout_count > 0:
        print(f"   ⚠️ 共 {timeout_count}/{len(contents)} 个任务超时")
    
    return list(zip(results, generation_times, cache_hits))

def run_streaming_evaluation(
    data: List[Dict],
    method,
    evaluator: SimpleKGEvaluator,
    output_dir: str,
    method_file: str = None,
    batch_size: int = 1
) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total = len(data)
    
    print(f"\n🔄 开始边生成边评估 (共 {total} 条)...")
    print("=" * 60)
    
    start_time = time.time()
    
    for i, item in enumerate(data):
        item_start = time.time()
        
        content = item.get('content', '')
        ground_truth = item.get('ground_truth', [])
        source = item.get('source_approach_provided_dataset', 'unknown')
        extra_info = item.get('extra_info', {})
        item_id = extra_info.get('file_name', f'item_{i}')
        
        print(f"\n📝 [{i+1}/{total}] {item_id} (来源: {source})")
        
        print("   🔧 生成知识图谱...")
        prediction = method.generate(content)
        pred_count = len(prediction.get('relations', []))
        print(f"   ✅ 提取到 {pred_count} 条关系")
        
        print("   🧑‍⚖️ LLM 裁判评估...")
        
        entities_from_extra = None
        if source == 'grid':
            entities_from_extra = extra_info.get('实体列表(更新别名后)', [])
        
        scores = evaluator.evaluate(
            ground_truth, prediction, content,
            entities_from_extra=entities_from_extra,
            source=source
        )
        print(f"   📊 P={scores['precision']:.3f} R={scores['recall']:.3f} F1={scores['f1']:.3f}")
        
        method_path = os.path.join(os.path.dirname(__file__), "生成知识图谱的方法", method_file) if method_file else "Unknown"
        method_md5 = get_file_md5(method_path)
        
        current_model_name = "Unknown"
        if hasattr(method, 'model'):
            current_model_name = method.model
        elif hasattr(method, 'config') and method.config.get('model'):
             current_model_name = method.config.get('model')

        gen_log = {
            "py输入内容": content,
            "py原始输出": prediction.get('raw_output', '') if isinstance(prediction, dict) else str(prediction),
            "解析知识图谱": {"entities": prediction.get('entities', []), "relations": prediction.get('relations', [])} if isinstance(prediction, dict) else {},
            "方法文件名": method_file,
            "方法文件MD5": method_md5,
            "数据集名称": source,
            "数据索引": item_id,
            "记录时间": datetime.now().isoformat(),
            "生成耗时": time.time() - item_start, 
            "模型名称": current_model_name,
            "是否命中缓存": False 
        }
        save_detailed_log("特定方法的生成内容", method_file or "UnknownMethod", gen_log)

        eval_log = gen_log.copy()
        eval_log.update({
            "标准答案": ground_truth,
            "评估提示词": {
                "Recall_Prompt": scores.get('recall_prompt', ''),
                "Precision_Prompt": scores.get('precision_prompt', '')
            },
            "评估原始输出": {
                "Recall_Response": scores.get('recall_response', ''),
                "Precision_Response": scores.get('precision_response', '')
            },
            "精确率": scores['precision'],
            "召回率": scores['recall'],
            "F1值": scores['f1'],
            "提示词模式": evaluator.prompt_mode,
            "裁判模型": evaluator.judge_model,
            "是否提取失败": is_failed_sample(prediction),
            "评估耗时": (time.time() - item_start) - gen_log['生成耗时']
        })
        save_detailed_log("特定方法的生成内容-进一步评估", method_file or "UnknownMethod", eval_log)

        result = {
            'item_id': item_id,
            'source': source,
            'content_length': len(content),
            'gt_count': len(ground_truth),
            'pred_count': pred_count,
            'precision': scores['precision'],
            'recall': scores['recall'],
            'f1': scores['f1'],
            'prediction': prediction,
            'ground_truth': ground_truth,
            'time_seconds': time.time() - item_start,
            'is_failed': is_failed_sample(prediction)  
        }
        results.append(result)
        
        item_dir = os.path.join(output_dir, f"{source}_{item_id}")
        os.makedirs(item_dir, exist_ok=True)
        
        with open(os.path.join(item_dir, 'result.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (total - i - 1)
        print(f"   ⏱️ 用时 {result['time_seconds']:.1f}s | 预计剩余 {remaining/60:.1f} 分钟")
    
    total_time = time.time() - start_time
    print(f"\n✅ 评估完成! 总用时: {total_time/60:.1f} 分钟")
    
    return results


# ================================================================================
# ================================================================================

def aggregate_results(results: List[Dict], output_dir: str) -> pd.DataFrame:
    if not results:
        print("⚠️ 无结果可汇总")
        return pd.DataFrame()
    
    df = pd.DataFrame([{
        'item_id': r['item_id'],
        'source': r['source'],
        'precision': r['precision'],
        'recall': r['recall'],
        'f1': r['f1'],
        'gt_count': r['gt_count'],
        'pred_count': r['pred_count'],
        'time_seconds': r['time_seconds']
    } for r in results])
    
    df.to_csv(os.path.join(output_dir, '逐条评估结果.csv'), index=False, encoding='utf-8-sig')
    
    by_source = df.groupby('source').agg({
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean',
        'item_id': 'count',
        'time_seconds': 'sum'
    }).rename(columns={'item_id': 'count'})
    
    by_source.to_csv(os.path.join(output_dir, '按数据来源统计.csv'), encoding='utf-8-sig')
    
    total_stats = pd.DataFrame([{
        '总样本数': len(results),
        '平均Precision': df['precision'].mean(),
        '平均Recall': df['recall'].mean(),
        '平均F1': df['f1'].mean(),
        '总耗时(分钟)': df['time_seconds'].sum() / 60
    }])
    
    total_stats.to_csv(os.path.join(output_dir, '总体统计.csv'), index=False, encoding='utf-8-sig')
    
    print("\n📊 评估结果汇总:")
    print("=" * 60)
    print(f"总样本数: {len(results)}")
    print(f"平均 Precision: {df['precision'].mean():.4f}")
    print(f"平均 Recall: {df['recall'].mean():.4f}")
    print(f"平均 F1: {df['f1'].mean():.4f}")
    print("\n按数据来源:")
    print(by_source.to_string())
    
    return df


def aggregate_results_with_filters(
    all_method_detailed_results: Dict[str, List[Dict]],
    output_dir: str = None
) -> None:
    if not all_method_detailed_results:
        print("⚠️ 无结果可汇总")
        return
    
    methods = list(all_method_detailed_results.keys())
    
    failed_items_by_method = {}  # {method: set(item_id)}
    all_item_ids = set()
    
    for method, results in all_method_detailed_results.items():
        failed_items_by_method[method] = set()
        for r in results:
            all_item_ids.add(r['item_id'])
            if r.get('is_failed', False):
                failed_items_by_method[method].add(r['item_id'])
    
    all_failed_items = set()
    for method, failed_set in failed_items_by_method.items():
        all_failed_items.update(failed_set)
    
    valid_items_intersection = all_item_ids - all_failed_items  
    
    print("\n" + "=" * 80)
    print("🧪 失败样本统计")
    print("=" * 80)
    for method in methods:
        failed_count = len(failed_items_by_method[method])
        total_count = len(all_method_detailed_results[method])
        print(f"   📌 {method}: {failed_count}/{total_count} 失败 ({100*failed_count/total_count:.1f}%)")
    print(f"   🔗 所有方法成功文章交集: {len(valid_items_intersection)}/{len(all_item_ids)} 篇")
    
    def _compute_stats(results: List[Dict], excluded_items: set = None) -> Dict:
        filtered_results = results
        if excluded_items:
            filtered_results = [r for r in results if r['item_id'] not in excluded_items]
        
        if not filtered_results:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'count': 0}
        
        avg_p = sum(r['precision'] for r in filtered_results) / len(filtered_results)
        avg_r = sum(r['recall'] for r in filtered_results) / len(filtered_results)
        avg_f1 = sum(r['f1'] for r in filtered_results) / len(filtered_results)
        
        return {'precision': avg_p, 'recall': avg_r, 'f1': avg_f1, 'count': len(filtered_results)}
    
    def _compute_by_source(results: List[Dict], excluded_items: set = None) -> Dict[str, Dict]:
        filtered_results = results
        if excluded_items:
            filtered_results = [r for r in results if r['item_id'] not in excluded_items]
        
        by_source = defaultdict(list)
        for r in filtered_results:
            by_source[r['source']].append(r)
        
        stats_by_source = {}
        for source, source_results in by_source.items():
            if source_results:
                stats_by_source[source] = {
                    'precision': sum(r['precision'] for r in source_results) / len(source_results),
                    'recall': sum(r['recall'] for r in source_results) / len(source_results),
                    'f1': sum(r['f1'] for r in source_results) / len(source_results),
                    'count': len(source_results)
                }
        return stats_by_source
    
    def _print_strategy_report(strategy_name: str, strategy_stats: Dict[str, Dict], 
                               strategy_by_source: Dict[str, Dict[str, Dict]]):
        print(f"\n{'='*80}")
        print(f"📊 策略: {strategy_name}")
        print("=" * 80)
        
        print("\n📈 综合性能 (按 F1 排序):")
        print(f"{'排名':<4} {'方法':<55} {'P':<9} {'R':<9} {'F1':<9} {'样本数':<8}")
        print("-" * 96)
        
        sorted_methods = sorted(strategy_stats.items(), key=lambda x: x[1]['f1'], reverse=True)
        for rank, (method, stats) in enumerate(sorted_methods, 1):
            print(f"{rank:<4} {method:<55} {stats['precision']:.4f}    {stats['recall']:.4f}    {stats['f1']:.4f}    {stats['count']:<8}")
        
        print("\n📊 各数据集性能:")
        
        all_sources = set()
        for method, sources_dict in strategy_by_source.items():
            all_sources.update(sources_dict.keys())
        all_sources = sorted(all_sources)
        
        for source in all_sources:
            print(f"\n   🗂️ {source}:")
            source_data = []
            for method in methods:
                if source in strategy_by_source.get(method, {}):
                    s = strategy_by_source[method][source]
                    source_data.append((method, s['precision'], s['recall'], s['f1'], s['count']))
            
            if source_data:
                source_data.sort(key=lambda x: x[3], reverse=True)  
                for method, p, r, f1, cnt in source_data:
                    method_short = method[:45] + "..." if len(method) > 48 else method
                    print(f"      {method_short:<48} P={p:.4f} R={r:.4f} F1={f1:.4f} (n={cnt})")
    
    # ================================================================================
    # ================================================================================
    strategy1_stats = {}
    strategy1_by_source = {}
    for method, results in all_method_detailed_results.items():
        strategy1_stats[method] = _compute_stats(results, excluded_items=None)
        strategy1_by_source[method] = _compute_by_source(results, excluded_items=None)
    
    _print_strategy_report("不过滤直接按照最终结果统计 (失败文章得0分)", 
                           strategy1_stats, strategy1_by_source)
    
    # ================================================================================
    # ================================================================================
    strategy2_stats = {}
    strategy2_by_source = {}
    for method, results in all_method_detailed_results.items():
        excluded = failed_items_by_method[method]
        strategy2_stats[method] = _compute_stats(results, excluded_items=excluded)
        strategy2_by_source[method] = _compute_by_source(results, excluded_items=excluded)
    
    _print_strategy_report("单独方法从统计中删除 (某方法失败时，该方法跳过该文章)", 
                           strategy2_stats, strategy2_by_source)
    
    # ================================================================================
    # ================================================================================
    strategy3_stats = {}
    strategy3_by_source = {}
    for method, results in all_method_detailed_results.items():
        excluded = all_failed_items
        strategy3_stats[method] = _compute_stats(results, excluded_items=excluded)
        strategy3_by_source[method] = _compute_by_source(results, excluded_items=excluded)
    
    _print_strategy_report("只保留所有方法成功文章交集 (所有方法都成功的文章)", 
                           strategy3_stats, strategy3_by_source)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            '失败样本统计': {
                method: {
                    'failed_count': len(failed_items_by_method[method]),
                    'total_count': len(all_method_detailed_results[method]),
                    'failed_items': list(failed_items_by_method[method])
                }
                for method in methods
            },
            '所有方法成功文章数': len(valid_items_intersection),
            '策略1_不过滤': strategy1_stats,
            '策略2_单独方法删除': strategy2_stats,
            '策略3_只保留交集': strategy3_stats
        }
        
        report_file = os.path.join(output_dir, 'multi_strategy_comparison.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 多策略对比报告保存至: {report_file}")


def generate_source_pr_split_heatmap(
    input_csv: str,
    output_png: str,
    methods_overview_csv: Optional[str] = None,
    title: str = "P/R by Source (+AVG)",
    input_parquet_path: Optional[str] = None,
) -> Optional[str]:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.patches import Rectangle
    except Exception as exc:
        print(f"⚠️ 无法导入 matplotlib，跳过双子格热力图生成: {exc}")
        return None

    input_path = Path(input_csv)
    output_path = Path(output_png)
    if not input_path.exists():
        print(f"⚠️ 热力图输入不存在，跳过: {input_path}")
        return None

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        print(f"⚠️ 读取热力图输入 CSV 失败: {exc}")
        return None

    if df.empty or "method" not in df.columns or "source" not in df.columns:
        print("⚠️ 热力图输入缺少 method/source 列，跳过生成")
        return None

    preferred_source_order = ["casie", "ctinexus", "grid", "malkg", "securenlp"]
    seen_sources = [s for s in df["source"].dropna().unique().tolist() if str(s).strip()]
    source_order = [s for s in preferred_source_order if s in seen_sources] + sorted(
        s for s in seen_sources if s not in preferred_source_order
    )

    methods = df["method"].dropna().unique().tolist()
    method_order = sorted(methods)

    def _dedupe_keep_order(items):
        seen = set()
        ordered_items = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered_items.append(item)
        return ordered_items

    if methods_overview_csv:
        overview_path = Path(methods_overview_csv)
        if overview_path.exists():
            try:
                overview_df = pd.read_csv(overview_path)
                ordered = _dedupe_keep_order([m for m in overview_df["method"].tolist() if m in methods])
                rest = [m for m in methods if m not in ordered]
                method_order = ordered + sorted(rest)
            except Exception as exc:
                print(f"⚠️ 读取 methods_overview.csv 失败，回退默认排序: {exc}")
    method_order = _dedupe_keep_order(method_order)

    stats = (
        df.groupby(["method", "source"], dropna=False)
        .agg(avg_precision=("precision", "mean"), avg_recall=("recall", "mean"))
        .reset_index()
    )
    avg_col = (
        df.groupby(["method"], dropna=False)
        .agg(avg_precision=("precision", "mean"), avg_recall=("recall", "mean"))
        .reset_index()
    )
    avg_col["source"] = "AVG"
    stats = pd.concat([stats, avg_col], ignore_index=True)
    stats["method"] = pd.Categorical(stats["method"], categories=method_order, ordered=True)
    stats["source"] = pd.Categorical(stats["source"], categories=source_order + ["AVG"], ordered=True)
    stats = stats.sort_values(["method", "source"]).reset_index(drop=True)

    def _short_method_name(name: str) -> str:
        method_name = str(name)
        if "(" in method_name and method_name.endswith(")"):
            return method_name
        method_file = method_name if method_name.endswith(".py") else f"{method_name}.py"
        base_name = method_name.replace("Approach_", "").replace(".py", "")
        suffix = "not LLM-based" if is_native_baseline(method_file) else "LLM-based"
        return f"{base_name} ({suffix})"

    def _short_source_name(name: str) -> str:
        return str(name).upper()

    def _compute_source_avg_token_map(parquet_path: Optional[str]) -> Dict[str, float]:
        if not parquet_path:
            return {}
        parquet_file = Path(parquet_path)
        if not parquet_file.exists():
            print(f"⚠️ token 统计 parquet 不存在，跳过列标题 token 注释: {parquet_file}")
            return {}
        try:
            import tiktoken
            article_df = pd.read_parquet(parquet_file, columns=["source", "content"])
        except Exception as exc:
            print(f"⚠️ 读取 token 统计 parquet 失败，跳过列标题 token 注释: {exc}")
            return {}

        if article_df.empty or "source" not in article_df.columns or "content" not in article_df.columns:
            return {}

        encoding = tiktoken.get_encoding("cl100k_base")
        article_df = article_df.copy()
        article_df["content"] = article_df["content"].fillna("").astype(str)
        article_df["token_len"] = article_df["content"].map(lambda text: len(encoding.encode(text)))
        return (
            article_df.groupby("source", dropna=False)["token_len"].mean().to_dict()
        )

    def _pct_text(value: float) -> str:
        value = 0.0 if pd.isna(value) else float(value)
        if 0.0 < value < 0.01:
            return "<1%"
        return f"{round(value * 100):.0f}%"

    source_avg_token_map = _compute_source_avg_token_map(input_parquet_path)

    cols = source_order + ["AVG"]
    rows = method_order
    if not cols or not rows:
        print("⚠️ 热力图缺少可视化维度，跳过生成")
        return None

    cmap = mpl.colormaps["RdYlGn"]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    
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

    cell_w = 1.65
    cell_h = 0.9
    left_margin = 2.7
    top_margin = 1.85
    bottom_margin = 1.1
    right_margin = 0.7

    width = left_margin + len(cols) * cell_w + right_margin
    height = top_margin + len(rows) * cell_h + bottom_margin

    fig = plt.figure(figsize=(width, height), dpi=220)
    ax = fig.add_axes([0, 0.11, 1, 0.84])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    ax.text(
        left_margin,
        0.55,
        title,
        fontsize=16,
        fontweight="bold",
        va="center",
        ha="left",
    )

    for j, source in enumerate(cols):
        x0 = left_margin + j * cell_w
        source_label = _short_source_name(source)
        token_line = ""
        if source != "AVG":
            token_val = source_avg_token_map.get(source)
            if token_val is not None and not pd.isna(token_val):
                token_line = f"(avg token={round(float(token_val))})"
        ax.text(
            x0 + cell_w / 2,
            top_margin - 0.52,
            source_label,
            fontsize=10.5,
            fontweight="bold",
            ha="center",
            va="center",
        )
        if token_line:
            ax.text(
                x0 + cell_w / 2,
                top_margin - 0.18,
                token_line,
                fontsize=8.7,
                ha="center",
                va="center",
                color="#444444",
            )

    line_color = "#333333"
    split_color = "#666666"
    text_color = "#111111"

    for i, method in enumerate(rows):
        y0 = top_margin + i * cell_h
        ax.text(
            left_margin - 0.18,
            y0 + cell_h / 2,
            _short_method_name(method),
            fontsize=10.5,
            ha="right",
            va="center",
        )
        for j, source in enumerate(cols):
            x0 = left_margin + j * cell_w
            row = stats[(stats["method"] == method) & (stats["source"] == source)]
            if row.empty:
                p_val = 0.0
                r_val = 0.0
            else:
                p_val = float(row.iloc[0]["avg_precision"])
                r_val = float(row.iloc[0]["avg_recall"])

            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    cell_w / 2,
                    cell_h,
                    facecolor=cmap(norm(p_val)),
                    edgecolor=split_color,
                    linewidth=0.8,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x0 + cell_w / 2, y0),
                    cell_w / 2,
                    cell_h,
                    facecolor=cmap(norm(r_val)),
                    edgecolor=split_color,
                    linewidth=0.8,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    cell_w,
                    cell_h,
                    facecolor="none",
                    edgecolor=line_color,
                    linewidth=1.0,
                )
            )
            ax.text(
                x0 + cell_w * 0.25,
                y0 + cell_h / 2,
                f"P:{_pct_text(p_val)}",
                fontsize=9.0,
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
            )
            ax.text(
                x0 + cell_w * 0.75,
                y0 + cell_h / 2,
                f"R:{_pct_text(r_val)}",
                fontsize=9.0,
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
            )

    avg_x = left_margin + (len(cols) - 1) * cell_w
    ax.plot([avg_x, avg_x], [top_margin, top_margin + len(rows) * cell_h], color="#000000", linewidth=1.8)

    cax = fig.add_axes([0.22, 0.04, 0.56, 0.03])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Color Scale (0 -> 1)", fontsize=10)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️ 已生成按来源 P/R 双子格热力图: {output_path}")
    return str(output_path)


# ================================================================================
# ================================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Packaged evaluation core for GRID knowledge graph extraction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s --yaml eval/experiment_yaml/smoke.yaml
  python %(prog)s --yaml eval/experiment_yaml/full.yaml
   python %(prog)s --method GRID_Ours.py --sources casie --sample_size 3
  python %(prog)s --verify --method Approach_CTINexus.py
        """
    )
    
    parser.add_argument('--dataset', type=str, default=None, 
                        help='Dataset path. Defaults to the packaged benchmark source.')
    parser.add_argument('--sources', type=str, default='all',
                        help='Comma-separated source filter or "all".')
    parser.add_argument('--sample_size', type=str, default='all',
                        help='Per-source sample count or "all".')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    
    parser.add_argument('--method', type=str,
                        help='Method Python file name, for example GRID_Ours.py.')
    
    parser.add_argument('--judge_model', type=str, default='gpt-5-nano', help='Judge model name.')
    parser.add_argument('--judge_flex', action='store_true', help='Enable Flex mode for the judge model.')
    parser.add_argument('--prompt_mode', type=str, default='grid_judge_fav',
                        choices=['grid_judge_fav'], help='Judge prompt mode.')
    
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory.')
    
    parser.add_argument('--use_cache', action='store_true', help='Reuse cached generation outputs when available.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes.')
    parser.add_argument('--shared_llm_model_path', type=str, default=None, help='Shared Qwen backend model path.')
    parser.add_argument('--shared_vllm_servers', type=str, default='super,ultra', help='Comma-separated shared vLLM servers.')
    parser.add_argument('--shared_llm_check_history_cache', type=str, default='true', help='Whether shared asks() checks history cache: true/false.')
    parser.add_argument('--resource_monitor_interval', type=float, default=5.0, help='Resource sampling interval in seconds.')
    
    parser.add_argument('-v', '--verify', action='store_true', help='Run verification mode instead of full evaluation.')
    parser.add_argument('--verify_content', type=str, default=None, help='Optional verification content.')
    
    parser.add_argument('-yaml', '--yaml', type=str, help='Path to an experiment YAML file.')
    
    parser.add_argument('--skip-eval', action='store_true', help='Skip the evaluation stage and only run generation.')
    
    return parser.parse_args()


def load_yaml_config(yaml_path: str) -> Dict:
    """Load a YAML configuration file."""
    import yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Run the standalone evaluation core."""
    venv_name = 'GRIDPYUV'
    project_root = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
    venv_path = os.path.join(project_root, venv_name)
    
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv and os.path.exists(venv_path):
        print(f"🖥️  Detected virtual environment: {venv_name}")
        print(f"💡 Activate it first: source {venv_path}/bin/activate")
        print(f"   Or run directly: {venv_path}/bin/python {' '.join(sys.argv)}")
        sys.exit(1)
    
    args = parse_args()
    
    if not args.method and not args.yaml:
        print("❌ Error: please provide --method or --yaml.\n")
        os.system(f"python {sys.argv[0]} --help")
        sys.exit(1)
    
    if args.yaml:
        print(f"📄 Loading YAML configuration: {args.yaml}")
        cfg = load_yaml_config(args.yaml)
        
        key_map = {
            '数据集路径': 'dataset',
            '数据来源筛选': 'sources',
            '每个数据源采样数量': 'sample_size',
            '随机种子': 'seed',
            '生成方法': 'method',
            '裁判用LLM模型': 'judge_model',
            '裁判用LLM是否开启Flex': 'judge_flex',
            '裁判用LLM的提示词模式': 'prompt_mode',
            '输出目录': 'output_dir',
            '使用临时结果缓存': 'use_cache',
            '多进程数量': 'num_workers',
            '共享LLM后端模型路径': 'shared_llm_model_path',
            '共享LLM服务器列表': 'shared_vllm_servers',
            '共享LLM是否检查历史缓存': 'shared_llm_check_history_cache',
            '资源采样间隔秒': 'resource_monitor_interval',
        }

        
        for cn_key, en_key in key_map.items():
            if cn_key in cfg and cfg[cn_key] is not None:
                setattr(args, en_key, cfg[cn_key])
    
    sources = args.sources.split(',') if args.sources != 'all' else None
    sample_size = None if str(args.sample_size).lower() == 'all' else int(args.sample_size)
    shared_vllm_servers = [
        server.strip()
        for server in str(getattr(args, 'shared_vllm_servers', 'super,ultra')).split(',')
        if server.strip()
    ] or ['super', 'ultra']
    shared_llm_check_history_cache_raw = getattr(args, 'shared_llm_check_history_cache', True)
    if isinstance(shared_llm_check_history_cache_raw, str):
        shared_llm_check_history_cache = shared_llm_check_history_cache_raw.strip().lower() in {
            '1', 'true', 'yes', 'y', 'on'
        }
    else:
        shared_llm_check_history_cache = bool(shared_llm_check_history_cache_raw)
    shared_backend = build_default_shared_backend(
        model_path=args.shared_llm_model_path or build_default_shared_backend()["model_path"],
        servers=shared_vllm_servers,
        check_history_cache=shared_llm_check_history_cache,
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    methods_to_run = args.method if isinstance(args.method, list) else [args.method]
    if methods_to_run == ['all']:
        methods_to_run = list(DEFAULT_FULL_METHODS)

    shared_backend_manager = None
    if any(uses_shared_qwen(method_file) for method_file in methods_to_run):
        print("\n🚀 预热共享三机 Qwen3-4B 后端...")
        shared_backend_manager = SharedLLMBackendManager(shared_backend)
        shared_backend_manager.ensure_ready()
    
    if args.verify:
        method_file = methods_to_run[0]
        print(f"\n🚀 初始化方法: {method_file} ...")
        method = load_method(
            method_file,
            init_kwargs=build_method_init_kwargs(
                method_file,
                shared_backend=shared_backend if uses_shared_qwen(method_file) else None,
                runtime_context=None,
            ),
        )
        run_verification(method, args.verify_content)
        if hasattr(method, 'cleanup'):
            method.cleanup()
        if shared_backend_manager:
            shared_backend_manager.cleanup()
        return
    
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    is_multi_method = len(methods_to_run) > 1
    
    if is_multi_method:
        print("\n" + "=" * 60)
        print("🚀 究极测试集评估系统 - 多方法对比模式")
        print("=" * 60)
        print(f"待评估方法 ({len(methods_to_run)} 个):")
        for i, m in enumerate(methods_to_run, 1):
            print(f"   [{i}] {m}")
        print(f"数据来源: {sources or 'all'}")
        print(f"采样数量: {sample_size or 'all'}")
        print(f"裁判模型: {args.judge_model}")
        print("=" * 60)
    
    data = load_ultimate_dataset(
        dataset_path=args.dataset,
        sources=sources,
        sample_size=sample_size,
        seed=args.seed
    )
    
    if not data:
        print("❌ 无数据可评估")
        return
    
    all_method_results = []
    all_method_detailed_results = {}  

    if is_multi_method:
        all_method_results, all_method_detailed_results = run_async_method_pipeline(
            data=data,
            methods_to_run=methods_to_run,
            args=args,
            shared_backend=shared_backend,
            shared_backend_manager=shared_backend_manager,
            shared_vllm_servers=shared_vllm_servers,
            run_id=run_id,
        )
    else:
        for method_idx, method_file in enumerate(methods_to_run, 1):
            if is_multi_method:
                print(f"\n{'='*60}")
                print(f"📊 [{method_idx}/{len(methods_to_run)}] 评估方法: {method_file}")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("🚀 究极测试集评估系统 - 完整评估模式")
                print("=" * 60)
                print(f"方法: {method_file}")
                print(f"数据来源: {sources or 'all'}")
                print(f"采样数量: {sample_size or 'all'}")
                print(f"裁判模型: {args.judge_model}")
                print("=" * 60)
            
            print(f"\n🚀 初始化方法: {method_file} ...")
            if args.output_dir:
                output_dir = args.output_dir if not is_multi_method else os.path.join(args.output_dir, method_file)
            else:
                output_dir = os.path.join(SCRIPT_DIR, 'Result', method_file)

            runtime_context = create_runtime_context(
                run_id=run_id,
                method_file=method_file,
                output_dir=output_dir,
                resource_interval_seconds=getattr(args, 'resource_monitor_interval', 5.0),
            )

            method = load_method(
                method_file,
                init_kwargs=build_method_init_kwargs(
                    method_file,
                    shared_backend=shared_backend if uses_shared_qwen(method_file) else None,
                    runtime_context=runtime_context,
                ),
            )
            if hasattr(method, 'name'):
                runtime_context['method_name'] = method.name

            raise RuntimeError(
                "formal 目录已停用核心脚本内置 judge 流程；请改用 "
                "unified_eval_executor.py + judge_backend=kg_reward."
            )

            resource_monitor = ResourceMonitor(
                runtime_context=runtime_context,
                servers=shared_vllm_servers,
                interval_seconds=getattr(args, 'resource_monitor_interval', 5.0),
            )

            try:
                resource_monitor.start()
                if len(data) <= 999999:
                    results = run_batch_evaluation(
                        data=data,
                        method=method,
                        evaluator=evaluator,
                        output_dir=output_dir,
                        method_file=method_file,
                        use_cache=getattr(args, 'use_cache', False),
                        num_workers=getattr(args, 'num_workers', 1),
                        skip_eval=getattr(args, 'skip_eval', False),
                        runtime_context=runtime_context,
                    )
                else:
                    results = run_streaming_evaluation(
                        data=data,
                        method=method,
                        evaluator=evaluator,
                        output_dir=output_dir,
                        method_file=method_file
                    )
            finally:
                resource_monitor.stop()
                summarize_latency_logs(runtime_context)

            aggregate_results(results, output_dir)
            
            if hasattr(method, 'cleanup'):
                method.cleanup()
            
            avg_precision = sum(r.get('precision', 0) for r in results) / len(results) if results else 0
            avg_recall = sum(r.get('recall', 0) for r in results) / len(results) if results else 0
            avg_f1 = sum(r.get('f1', 0) for r in results) / len(results) if results else 0
            
            all_method_results.append({
                'method': method_file,
                'sample_count': len(results),
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1': avg_f1,
                'output_dir': output_dir
            })
            
            all_method_detailed_results[method_file] = results
            
            print(f"📁 结果保存至: {output_dir}")

    if shared_backend_manager:
        shared_backend_manager.cleanup()
    
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    if is_multi_method:
        comparison_dir = args.output_dir or os.path.join(SCRIPT_DIR, 'Result', '多方法对比')
        aggregate_results_with_filters(all_method_detailed_results, comparison_dir)
    
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    if is_multi_method:
        print("\n" + "=" * 60)
        print("📊 多方法对比报告 (简版)")
        print("=" * 60)
        
        sorted_results = sorted(all_method_results, key=lambda x: x['avg_f1'], reverse=True)
        
        print(f"{'排名':<4} {'方法':<50} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 92)
        for rank, r in enumerate(sorted_results, 1):
            print(f"{rank:<4} {r['method']:<50} {r['avg_precision']:.4f}       {r['avg_recall']:.4f}       {r['avg_f1']:.4f}")
        
        comparison_dir = args.output_dir or os.path.join(SCRIPT_DIR, 'Result', '多方法对比')
        os.makedirs(comparison_dir, exist_ok=True)
        methods_overview_csv = os.path.join(comparison_dir, 'methods_overview.csv')
        detailed_csv = os.path.join(comparison_dir, 'all_methods_detailed.csv')
        comparison_file = os.path.join(comparison_dir, 'comparison_report.json')

        summary_df = pd.DataFrame(sorted_results)
        summary_df.to_csv(methods_overview_csv, index=False, encoding='utf-8-sig')

        detailed_rows = []
        for method_name, rows in all_method_detailed_results.items():
            for row in rows:
                if isinstance(row, dict):
                    detailed_rows.append({**row, 'method': method_name})
        if detailed_rows:
            pd.DataFrame(detailed_rows).to_csv(detailed_csv, index=False, encoding='utf-8-sig')
        
        import json
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump({
                'methods': all_method_results,
                'ranking': [r['method'] for r in sorted_results],
                'best_method': sorted_results[0]['method'],
                'best_f1': sorted_results[0]['avg_f1']
            }, f, ensure_ascii=False, indent=2)

        if detailed_rows:
            try:
                generate_source_pr_split_heatmap(
                    input_csv=detailed_csv,
                    output_png=os.path.join(comparison_dir, '按来源_PR双子格热力图.png'),
                    methods_overview_csv=methods_overview_csv,
                    title='P/R by Source (+AVG)',
                )
            except Exception as exc:
                print(f"⚠️ 生成按来源 P/R 双子格热力图失败: {exc}")
        
        print(f"\n📁 对比报告保存至: {comparison_file}")
        
        best = sorted_results[0]
        message = f"✅ 多方法对比评估完成\\n评估 {len(methods_to_run)} 个方法\\n🏆 最佳: {best['method']}\\nF1: {best['avg_f1']:.4f}"
    else:
        r = all_method_results[0]
        message = f"✅ 究极测试集评估完成\\n方法: {r['method']}\\n样本数: {r['sample_count']}\\n平均F1: {r['avg_f1']:.4f}"
    
    try:
        import subprocess
        subprocess.run(
            ['curl', '--max-time', '10', '-H', 'Content-Type: text/plain; charset=utf-8', '-d', message,
             'https://ntfy.sh/fwq3939'],
            timeout=10,
            capture_output=True
        )
        print("📤 已发送通知")
    except Exception as e:
        print(f"⚠️ 通知发送失败: {e}")


if __name__ == "__main__":
    bridge_path = os.path.join(SCRIPT_DIR, "unified_eval_executor.py")
    if os.path.exists(bridge_path):
        import importlib.util

        spec = importlib.util.spec_from_file_location("ultimate_eval_unified_runner", bridge_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"❌ 无法加载统一执行器: {bridge_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    else:
        main()
