#!/usr/bin/env python3

# Description: Official GRID triple-regex generator for KG matching. It reads article-level KG parquet, generates sub/rel/obj regex triples for every KG edge, and packages them into reward-ready training parquet. Keyword: GRID, triple regex, KG, regex_as_groundtruth, reward, parquet, gemini-2.5-flash.

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DATA_ROOT = REPO_ROOT / "train-data"
BASE_HELPER_SCRIPT = TRAIN_DATA_ROOT / "code" / "regex" / "generate_regex_for_recall_training_data.py"
DEFAULT_INPUT_PARQUET = TRAIN_DATA_ROOT / "data" / "qa_selection" / "representative20_qa_selection_train.parquet"
DEFAULT_OUTPUT_PARQUET = TRAIN_DATA_ROOT / "data" / "triple_regex" / "representative20_tripregex_train.parquet"
DEFAULT_OUTPUT_ROOT = TRAIN_DATA_ROOT / "data" / "triple_regex" / "debug_outputs"


def _import_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _import_module_from_path(
    f"grid_relregex_base_{__name__.replace('.', '_')}",
    BASE_HELPER_SCRIPT,
)

ensure_parent = _base.ensure_parent
dump_json = _base.dump_json
dump_text = _base.dump_text
normalize_space = _base.normalize_space
model_tag = _base.model_tag
load_tools = _base.load_tools
load_repair_loads = _base.load_repair_loads
load_dataframe = _base.load_dataframe
extract_relationships = _base.extract_relationships
build_examples = _base.build_examples
parse_llm_json = _base.parse_llm_json
heuristic_relation_regex = _base.heuristic_relation_regex
chunk_list = _base.chunk_list
is_acceptable_article_result = _base.is_acceptable_article_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="🧠 正式使用：生成 GRID KG 三元组三正则训练 parquet"
    )
    parser.add_argument("--input-parquet", default=str(DEFAULT_INPUT_PARQUET), help="输入文章级 KG parquet")
    parser.add_argument("--output-parquet", default=str(DEFAULT_OUTPUT_PARQUET), help="输出训练 parquet")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="调试输出根目录")
    parser.add_argument("--limit", type=int, default=500, help="处理前多少篇文章，默认 500")
    parser.add_argument("--api-model", default="gemini-2.5-flash", help="Model used for generation. Default: gemini-2.5-flash")
    parser.add_argument("--token", type=int, default=64 * 1024, help="单篇最大输出 token，默认 65536")
    parser.add_argument("--workers", type=int, default=8, help="首轮并发 worker 数，默认 8")
    parser.add_argument("--chunk-size", type=int, default=25, help="兼容参数，当前 asks 外层无分批")
    parser.add_argument("--retry-rounds", type=int, default=3, help="失败补跑轮数，默认 3")
    parser.add_argument("--retry-workers", type=int, default=1, help="补跑并发 worker 数，默认 1")
    parser.add_argument("--resume-run-dir", default="", help="从已有 run 目录断点续跑")
    return parser


def build_relation_regex_prompt(
    example: Dict[str, Any],
    retry_mode: bool = False,
    edge_indices: Optional[List[int]] = None,
    retry_notice: str = "",
) -> List[Dict[str, str]]:
    if edge_indices is None:
        edge_indices = list(range(len(example["relationships"])))
    indexed_relationships = []
    for edge_idx in edge_indices:
        edge_item = dict(example["relationships"][edge_idx])
        edge_item["edge_idx"] = edge_idx
        indexed_relationships.append(edge_item)

    relationships_json = json.dumps(
        indexed_relationships,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    input_edge_count = len(indexed_relationships)
    system_msg = (
        "You are a CTI knowledge-graph triple-alignment expert. "
        "Your job is to generate regexes that match semantically equivalent rollout-predicted triples, "
        "not to overfit to the exact article wording."
    )
    retry_hint = ""
    if retry_mode:
        retry_hint = """
补跑提醒:
- 上一轮失败通常是因为输出条数不对、JSON 不规范，或 regex 字段不完整。
- 这次务必只输出 JSON，且 `edge_regexes` 长度必须严格等于输入边数。
- 每条边必须同时给出 `sub_regex`、`rel_regex`、`obj_regex`。
- 不要写解释段落，不要加 markdown 围栏。
""".strip()
    chunk_hint = ""
    if input_edge_count != len(example["relationships"]):
        chunk_hint = """
分块提醒:
- 这次给你的只是整篇文章中的一个子集边列表，不是全部边。
- 输入边里的 `edge_idx` 是全局编号；输出时必须原样保留这些全局 `edge_idx`，绝对不能从 0 重新编号。
- 输出顺序仍必须与输入顺序一致。
""".strip()
    user_msg = f"""
你现在只处理 1 篇文章。

任务:
根据给定的“原文 + KG边列表”，为 **每一条 KG 边** 生成 **一组三正则**，用于后续给 rollout 模型输出的 KG 打分。
这些 regex 的目标不是苛刻还原原文，而是尽量匹配 **语义等价的预测三元组**，让 reward 更贴合实际。

⚠️ 强约束:
1. 必须逐边输出，**不能聚类、不能合并、不能减少条数**。
2. 输出数组长度必须与输入边数完全相同，顺序也必须完全一致。
3. 这些 regex 将用于匹配 rollout 预测三元组里的 `sub / rel / obj` 字段，不是匹配 JSON 原文串。
4. 目标是 **减少假阴性**:
   - 如果预测三元组和 groundtruth 在语义上明显一致，只是表面形式不同，regex 应尽量让它命中。
   - 宁可适度放宽以召回明显同义表达，也不要因为措辞、词形、代词、轻微泛化而错失本应命中的边。
5. `sub_regex` / `obj_regex` 的生成原则:
   - 覆盖实体表面形式的归一化 / 简化 / 别名 / 缩写 / 单复数 / 连字符差异。
   - 如果 rollout 常把具体实体写成更泛化的等价指称，也应覆盖，例如:
     - 具体恶意软件 -> `it / this malware / this trojan / the sample`
     - 具体 CVE / 命名漏洞 -> `the vulnerability / a vulnerability`
     - 更泛化类别 <-> 更具体实例（只在语义上明显同一对象时）
   - 但不要宽松到跨实体串类。
6. `rel_regex` 的生成原则:
   - 优先覆盖 rollout 常见的归一化谓词，而不是拘泥于原文某个单一短语。
   - 应覆盖词形变化、主动/被动、介词变化、轻微同义改写，例如:
     - `against` <-> `target|targets|targeted|targeting`
     - `discovered in` <-> `found in` <-> `vulnerability in` <-> `affects`
     - `detects` <-> `verifies` <-> `checks` <-> `used to verify`
     - `created by` <-> `authored by` <-> `written by`
   - 关系 regex 应优先写成“基型 + 常见词形 + 常见助动/包裹结构”：
     - 例如不要只写 `verify`，而应考虑 `verify|verifies|verified|verifying|used to verify|is used to verify`
     - 不要只写 `targeting|targeted`，要显式覆盖裸词 `target`
     - 对 `in / at / on / within / via / through` 这类常见介词改写，也应尽量兼容
   - 如果边有 `special_factuality`（如 possible / negated），也尽量覆盖这种语气变体。
7. 对象可允许有限度的功能性同义/具体化:
   - 例如 `spam filters` 可放宽到更具体的 `email threat filtering services`
   - 例如 `remote server` 可放宽到更具体的控制端/远端系统说法
   - 例如 `C2/C&C server` 可放宽到具体的远端主机 / 域名 / 控制端描述，只要明显还是同一功能角色
   - 但仍需保持在同一语义类别，不要过度泛化。
8. 如果 rollout 常把同一实体写成更短 head noun，也应尽量覆盖:
   - 例如 `Stagefright Detector app` <-> `the app`
   - 例如 `malicious cyber campaigns` <-> `attacks|operations`
   - 例如 `emails` <-> `messages|phishing emails`
9. 如果 groundtruth 的关系本质上是功能/用途/作用，也要容忍 rollout 加上的轻量壳层:
   - 例如 `detects` 允许匹配 `used to detect|used to verify|serves to detect`
   - 例如 `connects to` 允许匹配 `communicates with|contacts|beacons to`
   - 例如 `located at/in` 允许匹配 `in|stored in|written to|placed in`
10. 若某条边在原文中的词面证据很弱，也仍要输出一条 best-effort regex，并把 `confidence` 设低。
11. **输出尽量精简**，不要写长解释，避免因为输出过长而漏边。

{retry_hint}
{chunk_hint}

输出格式:
只能输出 JSON，不要加围栏外说明。

{{
  "article_local_idx": {example["local_idx"]},
  "stable_article_id": "{example.get("stable_article_id")}",
  "n_input_edges": {input_edge_count},
  "n_output_regexes": {input_edge_count},
  "edge_regexes": [
    {{
      "edge_idx": {edge_indices[0] if edge_indices else 0},
      "sub_regex": "(?is)...",
      "rel_regex": "(?is)...",
      "obj_regex": "(?is)...",
      "confidence": "high|medium|low"
    }}
  ]
}}

再次强调:
- `edge_regexes` 的长度必须等于 {input_edge_count}
- 第 i 个输出必须对应第 i 条输入边
- 即使多条边最终 regex 一样，也必须重复输出多次，不能省略
- 对某条 groundtruth 边，只有当 `sub_regex / rel_regex / obj_regex` 三个都命中 **同一条** 预测三元组时，这条边才算命中
- 你的 regex 应该优先匹配 **语义等价的预测三元组**，而不是机械复刻原文字面形式
- 不要把年份、冗余修饰语、非关键上下文硬编码进 regex，除非不用这些信息就无法区分实体
- 不要输出 `source_edge`、不要输出长解释、不要输出额外字段

原文:
{example["text"]}

KG边列表(JSON，顺序不可打乱，含全局 edge_idx):
{relationships_json}
""".strip()
    if retry_notice:
        user_msg += f"\n\n{retry_notice}"
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_regex_cache_retry_notice(attempt_label: str, retry_prompt_serial: int) -> str:
    return f"""
[Cache Retry Notice]
Attempt-Key: {attempt_label}__retry_prompt_serial_{retry_prompt_serial}
This is retry prompt serial {retry_prompt_serial} for this triple-regex generation task.
Last time you gave me an unusable result because the JSON, edge count, or sub/rel/obj regex fields could not be parsed correctly.
This time:
- output JSON only
- keep edge count strictly correct
- include sub_regex, rel_regex, and obj_regex for every edge
- do not output reasoning or markdown
- do not truncate
""".strip()


def validate_triple_regex_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    for key in ("sub_regex", "rel_regex", "obj_regex"):
        pattern = str(item.get(key, "")).strip()
        if not pattern:
            return False
        try:
            re.compile(pattern)
        except re.error:
            return False
    return True


def normalize_edge_regexes_by_position(
    edge_regexes: List[Any],
    expected_edge_indices: List[int],
) -> Tuple[Optional[List[Dict[str, Any]]], List[int]]:
    if len(edge_regexes) != len(expected_edge_indices):
        return None, []
    normalized: List[Dict[str, Any]] = []
    invalid_regex_indexes: List[int] = []
    for output_pos, (item, edge_idx) in enumerate(zip(edge_regexes, expected_edge_indices)):
        if not validate_triple_regex_item(item):
            invalid_regex_indexes.append(output_pos)
            continue
        normalized.append(
            {
                "edge_idx": edge_idx,
                "sub_regex": str(item.get("sub_regex", "")).strip(),
                "rel_regex": str(item.get("rel_regex", "")).strip(),
                "obj_regex": str(item.get("obj_regex", "")).strip(),
                "confidence": str(item.get("confidence", "medium") or "medium").strip() or "medium",
            }
        )
    if invalid_regex_indexes:
        return None, invalid_regex_indexes
    return normalized, []


def _strip_leading_inline_flags(pattern: str) -> str:
    return re.sub(r"^\(\?[a-zA-Z]+\)", "", str(pattern or "").strip(), count=1)


def heuristic_entity_regex(entity_text: Any) -> str:
    literal = normalize_space(entity_text)
    if not literal:
        return r"(?is)^$"
    escaped = re.escape(literal)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"[-\s]?")
    return rf"(?is){escaped}"


def ensure_regex_matches_literal(pattern: str, literal_text: Any) -> str:
    pattern = str(pattern or "").strip()
    literal = normalize_space(literal_text)
    literal_pattern = heuristic_entity_regex(literal)
    if not pattern:
        return literal_pattern
    try:
        if re.search(pattern, literal):
            return pattern
    except re.error:
        pass
    pattern_core = _strip_leading_inline_flags(pattern)
    literal_core = _strip_leading_inline_flags(literal_pattern)
    return rf"(?is)(?:{pattern_core}|{literal_core})"


def parse_article_result(
    example: Dict[str, Any],
    raw_output: Any,
    stage: str,
) -> Dict[str, Any]:
    raw_text = str(raw_output) if raw_output is not None else ""
    parsed, parser_name = parse_llm_json(raw_text)
    input_edge_count = len(example["relationships"])
    parsed_edge_count = None
    parsed_ok = False
    invalid_regex_indexes: List[int] = []

    if isinstance(parsed, dict) and isinstance(parsed.get("edge_regexes"), list):
        edge_regexes = parsed.get("edge_regexes", [])
        parsed_edge_count = len(edge_regexes)
        if parsed_edge_count == input_edge_count:
            normalized_edge_regexes, invalid_regex_indexes = normalize_edge_regexes_by_position(
                edge_regexes,
                list(range(input_edge_count)),
            )
            if normalized_edge_regexes is not None:
                parsed_ok = True
                parsed["edge_regexes"] = normalized_edge_regexes

    return {
        "local_idx": example["local_idx"],
        "stable_article_id": example.get("stable_article_id"),
        "source_file": example.get("source_file"),
        "input_edge_count": input_edge_count,
        "raw_text_chars": len(raw_text),
        "parsed_ok": parsed_ok,
        "parsed_edge_count": parsed_edge_count,
        "invalid_regex_indexes": invalid_regex_indexes,
        "parser": parser_name,
        "stage": stage,
        "raw_text": raw_text,
        "parsed": parsed,
    }


def choose_chunk_edge_limit(input_edge_count: int) -> int:
    if input_edge_count >= 48:
        return 6
    if input_edge_count >= 32:
        return 8
    if input_edge_count >= 20:
        return 10
    return 12


def try_chunked_article_generation(
    tools_mod,
    example: Dict[str, Any],
    api_model: str,
    token_limit: int,
    stage: str,
) -> Optional[Dict[str, Any]]:
    input_edge_count = len(example["relationships"])
    if input_edge_count <= 12:
        return None

    edge_indices = list(range(input_edge_count))
    chunk_edge_limit = choose_chunk_edge_limit(input_edge_count)
    edge_chunks = chunk_list(edge_indices, chunk_edge_limit)
    prompts = [
        build_relation_regex_prompt(
            example,
            retry_mode=True,
            edge_indices=edge_chunk,
            retry_notice="",
        )
        for edge_chunk in edge_chunks
    ]
    print(
        f"🧩 chunked rescue idx={example['local_idx']:03d} | "
        f"总边={input_edge_count} | chunk_size={chunk_edge_limit} | chunks={len(edge_chunks)}"
    )
    outputs = tools_mod.asks(
        prompt_list=prompts,
        model=api_model,
        token=token_limit,
        temp=0.0,
        streamprint=False,
        retry=1,
        adjust_token=True,
        check_history_cache=True,
        showthink=False,
        think="low",
        openai_verbosity="low",
        count=True,
        force_api_do_huge_input_Cloud=True,
        cloud_executor_workers=min(4, max(1, len(prompts))),
        note=f"grid_tripregex_{model_tag(api_model)}_{stage}_chunked_idx{example['local_idx']}",
    )

    merged_edge_regexes: List[Dict[str, Any]] = []
    raw_text_parts: List[str] = []
    parser_names: List[str] = []

    for edge_chunk, raw_output in zip(edge_chunks, outputs):
        raw_text = str(raw_output) if raw_output is not None else ""
        raw_text_parts.append(raw_text)
        parsed, parser_name = parse_llm_json(raw_text)
        if parser_name:
            parser_names.append(parser_name)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("edge_regexes"), list):
            return None
        edge_regexes = parsed.get("edge_regexes", [])
        normalized_edge_regexes, invalid_regex_indexes = normalize_edge_regexes_by_position(
            edge_regexes,
            edge_chunk,
        )
        if normalized_edge_regexes is None or invalid_regex_indexes:
            return None
        merged_edge_regexes.extend(normalized_edge_regexes)

    if len(merged_edge_regexes) != input_edge_count:
        return None

    merged_edge_regexes = sorted(merged_edge_regexes, key=lambda item: item["edge_idx"])
    parsed = {
        "article_local_idx": example["local_idx"],
        "stable_article_id": example.get("stable_article_id"),
        "n_input_edges": input_edge_count,
        "n_output_regexes": input_edge_count,
        "edge_regexes": merged_edge_regexes,
    }
    parser_label = "chunked_merge"
    if parser_names:
        parser_label += ":" + "+".join(sorted(set(parser_names)))
    return {
        "local_idx": example["local_idx"],
        "stable_article_id": example.get("stable_article_id"),
        "source_file": example.get("source_file"),
        "input_edge_count": input_edge_count,
        "raw_text_chars": sum(len(text) for text in raw_text_parts),
        "parsed_ok": True,
        "parsed_edge_count": input_edge_count,
        "invalid_regex_indexes": [],
        "parser": parser_label,
        "stage": f"{stage}_chunked",
        "raw_text": "\n\n===== CHUNK SPLIT =====\n\n".join(raw_text_parts),
        "parsed": parsed,
    }


def build_fallback_result(example: Dict[str, Any]) -> Dict[str, Any]:
    edge_regexes = []
    for edge_idx, edge in enumerate(example["relationships"]):
        edge_regexes.append(
            {
                "edge_idx": edge_idx,
                "sub_regex": heuristic_entity_regex(edge.get("sub", "")),
                "rel_regex": heuristic_relation_regex(
                    edge.get("rel", ""),
                    edge.get("special_factuality") or [],
                ),
                "obj_regex": heuristic_entity_regex(edge.get("obj", "")),
                "confidence": "low",
            }
        )
    parsed = {
        "article_local_idx": example["local_idx"],
        "stable_article_id": example.get("stable_article_id"),
        "n_input_edges": len(example["relationships"]),
        "n_output_regexes": len(edge_regexes),
        "edge_regexes": edge_regexes,
    }
    return {
        "local_idx": example["local_idx"],
        "stable_article_id": example.get("stable_article_id"),
        "source_file": example.get("source_file"),
        "input_edge_count": len(example["relationships"]),
        "raw_text_chars": 0,
        "parsed_ok": True,
        "parsed_edge_count": len(edge_regexes),
        "invalid_regex_indexes": [],
        "parser": "heuristic_fallback",
        "stage": "heuristic_fallback",
        "raw_text": "",
        "parsed": parsed,
    }


def save_progress(
    run_dir: Path,
    api_model_tag: str,
    article_result_map: Dict[int, Dict[str, Any]],
    examples: List[Dict[str, Any]],
) -> None:
    example_map = {item["local_idx"]: item for item in examples}
    ordered_results = [
        article_result_map[item["local_idx"]]
        for item in examples
        if item["local_idx"] in article_result_map
    ]
    flattened: List[Dict[str, Any]] = []
    for result in ordered_results:
        if not result.get("parsed_ok"):
            continue
        parsed = result.get("parsed") or {}
        edge_regexes = parsed.get("edge_regexes", []) if isinstance(parsed, dict) else []
        example = example_map[result["local_idx"]]
        for output_pos, item in enumerate(edge_regexes):
            edge_idx = item.get("edge_idx", output_pos)
            source_edge = (
                example["relationships"][edge_idx]
                if isinstance(edge_idx, int) and 0 <= edge_idx < len(example["relationships"])
                else None
            )
            flattened.append(
                {
                    "article_local_idx": example["local_idx"],
                    "stable_article_id": example.get("stable_article_id"),
                    "source_file": example.get("source_file"),
                    "edge_idx": edge_idx,
                    "source_edge": source_edge,
                    "sub_regex": item.get("sub_regex", ""),
                    "rel_regex": item.get("rel_regex", ""),
                    "obj_regex": item.get("obj_regex", ""),
                    "confidence": item.get("confidence", "low"),
                    "stage": result.get("stage"),
                    "parser": result.get("parser"),
                }
            )
    dump_json(run_dir / f"{api_model_tag}_tripregex_article_results.json", ordered_results)
    dump_json(run_dir / f"{api_model_tag}_tripregex_flattened_regexes.json", flattened)


def load_existing_progress(
    run_dir: Path,
    api_model_tag: str,
) -> Dict[int, Dict[str, Any]]:
    path = run_dir / f"{api_model_tag}_tripregex_article_results.json"
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    result_map: Dict[int, Dict[str, Any]] = {}
    if isinstance(loaded, list):
        for item in loaded:
            if isinstance(item, dict) and isinstance(item.get("local_idx"), int):
                result_map[item["local_idx"]] = item
    return result_map


def run_generation(
    tools_mod,
    examples: List[Dict[str, Any]],
    run_dir: Path,
    api_model: str,
    token_limit: int,
    workers: int,
    chunk_size: int,
    retry_rounds: int,
    retry_workers: int,
) -> Dict[str, Any]:
    api_model_tag = model_tag(api_model)
    manifest = [
        {
            "local_idx": example["local_idx"],
            "stable_article_id": example.get("stable_article_id"),
            "source_file": example.get("source_file"),
            "relationship_count": example["relationship_count"],
            "text_chars": example["text_chars"],
        }
        for example in examples
    ]
    dump_json(run_dir / f"{api_model_tag}_tripregex_prompts_manifest.json", manifest)

    article_result_map = load_existing_progress(run_dir, api_model_tag)

    def process_batch(
        batch_examples: List[Dict[str, Any]],
        stage: str,
        current_workers: int,
    ) -> Dict[str, Any]:
        current_temp = 0.1 if stage == "initial" else 0.0
        pending_examples = list(batch_examples)
        empty_like_count = 0
        acceptable_count = 0
        max_cache_retry_attempts = 3

        for cache_attempt_idx in range(max_cache_retry_attempts):
            if not pending_examples:
                break
            attempt_label = stage if cache_attempt_idx == 0 else f"{stage}_cache_retry_{cache_attempt_idx}"
            prompt_serials = [
                (
                    int((article_result_map.get(example["local_idx"], {}) or {}).get("retry_prompt_serial") or 0) + 1
                    if example["local_idx"] in article_result_map
                    else 0
                )
                for example in pending_examples
            ]
            prompts = [
                build_relation_regex_prompt(
                    example,
                    retry_mode=(stage != "initial") or (cache_attempt_idx > 0),
                    retry_notice=(
                        build_regex_cache_retry_notice(attempt_label, retry_prompt_serial)
                        if retry_prompt_serial > 0
                        else ""
                    ),
                )
                for example, retry_prompt_serial in zip(pending_examples, prompt_serials)
            ]
            print(
                f"🚀 阶段={attempt_label} | 批量文章数={len(pending_examples)} | "
                f"workers={current_workers} | token={token_limit} | temp={current_temp} | cache=on"
            )
            outputs = tools_mod.asks(
                prompt_list=prompts,
                model=api_model,
                token=token_limit,
                temp=current_temp,
                streamprint=False,
                retry=1,
                adjust_token=True,
                check_history_cache=True,
                showthink=False,
                think="low",
                openai_verbosity="low",
                count=True,
                force_api_do_huge_input_Cloud=True,
                cloud_executor_workers=current_workers,
                note=f"grid_tripregex_{api_model_tag}_{attempt_label}",
            )
            next_pending_examples: List[Dict[str, Any]] = []
            for example, retry_prompt_serial, raw_output in zip(pending_examples, prompt_serials, outputs):
                result = parse_article_result(example, raw_output, stage=attempt_label)
                if not result["parsed_ok"]:
                    chunked_result = try_chunked_article_generation(
                        tools_mod=tools_mod,
                        example=example,
                        api_model=api_model,
                        token_limit=token_limit,
                        stage=attempt_label,
                    )
                    if chunked_result is not None and chunked_result.get("parsed_ok", False):
                        result = chunked_result
                result["retry_prompt_serial"] = retry_prompt_serial
                article_result_map[example["local_idx"]] = result
                if is_acceptable_article_result(result):
                    acceptable_count += 1
                else:
                    next_pending_examples.append(example)
                if not str(raw_output or "").strip():
                    empty_like_count += 1
                print(
                    f"✅ {attempt_label} idx={example['local_idx']:03d} | "
                    f"输入边={result['input_edge_count']} | 解析边={result['parsed_edge_count']} | "
                    f"有效={result['parsed_ok']}"
                )
            pending_examples = next_pending_examples
            save_progress(run_dir, api_model_tag, article_result_map, examples)
            if pending_examples:
                print(
                    f"🧪 tripregex缓存验收后仍失败 {len(pending_examples)} 条 | "
                    f"下一轮将追加 retry notice 再 asks"
                )
        save_progress(run_dir, api_model_tag, article_result_map, examples)
        batch_size = len(batch_examples)
        failed_count = len(pending_examples)
        success_ratio = (acceptable_count / batch_size) if batch_size else 0.0
        return {
            "batch_size": batch_size,
            "acceptable_count": acceptable_count,
            "failed_count": failed_count,
            "empty_like_count": empty_like_count,
            "success_ratio": success_ratio,
            "failed_examples": list(pending_examples),
        }

    initial_pending = [
        example
        for example in examples
        if not is_acceptable_article_result(article_result_map.get(example["local_idx"], {}))
    ]
    if initial_pending:
        print(
            f"🚀 tripregex初轮采用 asks 无分批模式 | 待处理文章数={len(initial_pending)} | "
            f"workers={workers} | token={token_limit}"
        )
        process_batch(initial_pending, stage="initial", current_workers=workers)

    for retry_idx in range(retry_rounds):
        pending = [
            example
            for example in examples
            if not is_acceptable_article_result(article_result_map.get(example["local_idx"], {}))
        ]
        if not pending:
            break
        print(f"🔁 开始第 {retry_idx + 1}/{retry_rounds} 轮补跑 | 失败文章数={len(pending)}")
        process_batch(
            pending,
            stage=f"retry_round_{retry_idx + 1}",
            current_workers=max(1, retry_workers),
        )

    remaining = [
        example
        for example in examples
        if not is_acceptable_article_result(article_result_map.get(example["local_idx"], {}))
    ]
    if remaining:
        print(f"🩹 进入最终兜底 | 剩余失败文章数={len(remaining)}")
        for example in remaining:
            article_result_map[example["local_idx"]] = build_fallback_result(example)
            print(f"🛠️ heuristic fallback idx={example['local_idx']:03d} | 边数={len(example['relationships'])}")
        save_progress(run_dir, api_model_tag, article_result_map, examples)

    ordered_results = [article_result_map[item["local_idx"]] for item in examples]
    parsed_articles = sum(1 for item in ordered_results if is_acceptable_article_result(item))
    fallback_articles = sum(1 for item in ordered_results if item.get("stage") == "heuristic_fallback")
    total_input_edges = sum(item.get("input_edge_count", 0) for item in ordered_results)
    total_output_edges = sum(item.get("parsed_edge_count") or 0 for item in ordered_results)

    return {
        "api_model": api_model,
        "api_model_tag": api_model_tag,
        "article_count": len(ordered_results),
        "parsed_articles": parsed_articles,
        "fallback_articles": fallback_articles,
        "total_input_edges": total_input_edges,
        "total_output_edges": total_output_edges,
        "article_results": ordered_results,
        "article_results_path": str(run_dir / f"{api_model_tag}_tripregex_article_results.json"),
        "flattened_regexes_path": str(run_dir / f"{api_model_tag}_tripregex_flattened_regexes.json"),
    }


def build_regex_training_rows(
    df,
    examples: List[Dict[str, Any]],
    article_results: List[Dict[str, Any]],
    api_model: str,
    run_dir: Path,
):
    result_map = {item["local_idx"]: item for item in article_results}
    output_rows: List[Dict[str, Any]] = []

    for example in examples:
        row = df.iloc[example["local_idx"]].to_dict()
        result = result_map[example["local_idx"]]
        parsed = result.get("parsed") or {}
        edge_regexes = parsed.get("edge_regexes", []) if isinstance(parsed, dict) else []
        ordered_regexes: List[Dict[str, Any]] = []
        for output_pos, item in enumerate(edge_regexes):
            if not isinstance(item, dict):
                ordered_regexes.append(
                    {
                        "sub_regex": heuristic_entity_regex(""),
                        "rel_regex": heuristic_relation_regex(""),
                        "obj_regex": heuristic_entity_regex(""),
                        "confidence": "low",
                    }
                )
                continue
            source_edge = (
                example["relationships"][output_pos]
                if 0 <= output_pos < len(example["relationships"])
                else {}
            )
            ordered_regexes.append(
                {
                    "sub_regex": ensure_regex_matches_literal(
                        str(item.get("sub_regex", "")).strip(),
                        source_edge.get("sub", ""),
                    ),
                    "rel_regex": ensure_regex_matches_literal(
                        str(item.get("rel_regex", "")).strip(),
                        source_edge.get("rel", ""),
                    ),
                    "obj_regex": ensure_regex_matches_literal(
                        str(item.get("obj_regex", "")).strip(),
                        source_edge.get("obj", ""),
                    ),
                    "confidence": str(item.get("confidence", "medium") or "medium").strip() or "medium",
                }
            )
        if len(ordered_regexes) != len(example["relationships"]):
            raise RuntimeError(
                f"tripregex 数量与边数不一致: idx={example['local_idx']}, "
                f"regex={len(ordered_regexes)}, edges={len(example['relationships'])}"
            )

        extra_info = dict(row.get("extra_info") or {})
        extra_info["dataset_type"] = "train"
        extra_info["question_type"] = "regex match"
        extra_info["step6_task_type"] = "regex match"
        extra_info["regex_groundtruth_kind"] = "triple_regex_list"
        extra_info["regex_generation_model"] = api_model
        extra_info["regex_generation_run_dir"] = str(run_dir)
        extra_info["regex_generation_stage"] = result.get("stage")
        extra_info["regex_generation_parser"] = result.get("parser")
        extra_info["regex_generation_edge_count"] = len(example["relationships"])
        extra_info["regex_groundtruth_literal_selfmatch_guaranteed"] = True
        extra_info["regex_as_groundtruth"] = ordered_regexes
        row["extra_info"] = extra_info
        output_rows.append(row)
    return output_rows


def write_parquet(output_rows: List[Dict[str, Any]], output_parquet: Path) -> None:
    return _base.write_parquet(output_rows, output_parquet)


def write_overview_png(run_dir: Path, article_results: List[Dict[str, Any]]) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("⚠️ matplotlib 不可用，跳过 PNG 概览图")
        return None

    indices = [item["local_idx"] for item in article_results]
    edge_counts = [item.get("input_edge_count", 0) for item in article_results]
    status_values = [1 if item.get("stage") != "heuristic_fallback" else 0 for item in article_results]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), constrained_layout=True)
    axes[0].bar(indices, edge_counts, color="#2E8B57")
    axes[0].set_title("Triple Regex Edge Counts per Article")
    axes[0].set_xlabel("Article Local Index")
    axes[0].set_ylabel("KG Edge Count")

    axes[1].bar(indices, status_values, color="#1F77B4")
    axes[1].set_title("Triple Regex LLM Parse Success (1=LLM Success, 0=Fallback)")
    axes[1].set_xlabel("Article Local Index")
    axes[1].set_ylabel("Success")
    axes[1].set_ylim(-0.1, 1.1)

    png_path = run_dir / "triple_regex_overview.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return png_path


def write_summary(
    run_dir: Path,
    input_parquet: Path,
    output_parquet: Path,
    examples: List[Dict[str, Any]],
    generation_result: Dict[str, Any],
    png_path: Optional[Path],
) -> Path:
    article_results = generation_result["article_results"]
    llm_success_articles = sum(1 for item in article_results if item.get("stage") != "heuristic_fallback")
    total_edges = sum(item.get("input_edge_count", 0) for item in article_results)
    avg_edges = round(total_edges / max(len(article_results), 1), 2)
    lines = [
        f"运行目录: {run_dir}",
        f"输入 parquet: {input_parquet}",
        f"输出 parquet: {output_parquet}",
        f"生成时间: {datetime.now().isoformat()}",
        "",
        "=== 规模统计 ===",
        f"文章数: {len(examples)}",
        f"总边数: {total_edges}",
        f"平均每篇边数: {avg_edges}",
        "",
        "=== 生成结果 ===",
        f"api_model: {generation_result['api_model']}",
        f"article_count: {generation_result['article_count']}",
        f"parsed_articles: {generation_result['parsed_articles']}",
        f"llm_success_articles: {llm_success_articles}",
        f"fallback_articles: {generation_result['fallback_articles']}",
        f"total_input_edges: {generation_result['total_input_edges']}",
        f"total_output_edges: {generation_result['total_output_edges']}",
        f"article_results_path: {generation_result['article_results_path']}",
        f"flattened_regexes_path: {generation_result['flattened_regexes_path']}",
        f"overview_png: {png_path if png_path is not None else 'N/A'}",
        "regex_groundtruth_kind: triple_regex_list",
        "match_rule: sub_regex + rel_regex + obj_regex must all match the same predicted triple",
    ]
    summary_path = run_dir / "summary.txt"
    dump_text(summary_path, "\n".join(lines))
    return summary_path


def main() -> int:
    args = build_parser().parse_args()
    input_parquet = Path(args.input_parquet).expanduser().resolve()
    output_parquet = Path(args.output_parquet).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 输入 parquet: {input_parquet}")
    print(f"📁 调试输出目录: {run_dir}")
    print(f"🧾 目标 parquet: {output_parquet}")

    df = load_dataframe(input_parquet)
    print(f"📊 原始 parquet 行数: {len(df)}")

    examples = build_examples(df, args.limit)
    dump_json(run_dir / "examples_manifest.json", examples)
    print(
        f"📚 待处理文章数: {len(examples)} | "
        f"总边数={sum(item['relationship_count'] for item in examples)}"
    )

    tools_mod = load_tools()
    generation_result = run_generation(
        tools_mod=tools_mod,
        examples=examples,
        run_dir=run_dir,
        api_model=args.api_model,
        token_limit=args.token,
        workers=args.workers,
        chunk_size=args.chunk_size,
        retry_rounds=args.retry_rounds,
        retry_workers=args.retry_workers,
    )

    output_rows = build_regex_training_rows(
        df=df,
        examples=examples,
        article_results=generation_result["article_results"],
        api_model=args.api_model,
        run_dir=run_dir,
    )
    write_parquet(output_rows, output_parquet)
    png_path = write_overview_png(run_dir, generation_result["article_results"])
    summary_path = write_summary(
        run_dir=run_dir,
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        examples=examples,
        generation_result=generation_result,
        png_path=png_path,
    )
    print(f"✅ 已写出最终 tripregex parquet: {output_parquet}")
    print(f"📝 Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
