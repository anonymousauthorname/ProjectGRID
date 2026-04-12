#!/usr/bin/env python3

# Description: Generate per-edge relation-string regex ground truth for the first N articles of a GRID train parquet with gemini-2.5-flash, then package the results into a regex-for-recall training parquet for VERL reward. Keyword: GRID, parquet, gemini-2.5-flash, regex-for-recall, reward, VERL, KG.

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
DROPBOX_DIR = REPO_ROOT.parent
TRAIN_DATA_ROOT = REPO_ROOT / "train-data"
DEFAULT_INPUT_PARQUET = TRAIN_DATA_ROOT / "data" / "qa_selection" / "representative20_qa_selection_train.parquet"
DEFAULT_OUTPUT_PARQUET = TRAIN_DATA_ROOT / "data" / "triple_regex" / "representative20_regex_recall_train.parquet"
DEFAULT_OUTPUT_ROOT = TRAIN_DATA_ROOT / "data" / "triple_regex" / "regex_recall_outputs"
REL_BLOCK_RE = re.compile(
    r"#Relationship_List_Start#\s*(.*?)\s*#Relationship_List_End#",
    re.S,
)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="🧠 生成 GRID regex-for-recall 训练 parquet"
    )
    parser.add_argument(
        "--input-parquet",
        default=str(DEFAULT_INPUT_PARQUET),
        help="原始 train parquet 路径",
    )
    parser.add_argument(
        "--output-parquet",
        default=str(DEFAULT_OUTPUT_PARQUET),
        help="目标 regex-for-recall parquet 路径",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="调试输出根目录",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="处理前多少篇文章，默认 500",
    )
    parser.add_argument(
        "--api-model",
        default="gemini-2.5-flash",
        help="Model used for relation regex generation. Default: gemini-2.5-flash",
    )
    parser.add_argument(
        "--token",
        type=int,
        default=64 * 1024,
        help="每篇文章的最大输出 token，默认 65536",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="首轮并发 worker 数，默认 8",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25,
        help="每轮批量发多少篇文章，默认 25",
    )
    parser.add_argument(
        "--retry-rounds",
        type=int,
        default=3,
        help="失败补跑轮数，默认 3",
    )
    parser.add_argument(
        "--retry-workers",
        type=int,
        default=1,
        help="补跑并发 worker 数，默认 1",
    )
    parser.add_argument(
        "--resume-run-dir",
        default="",
        help="从已有 run 目录断点续跑；为空则新建",
    )
    return parser


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def dump_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def model_tag(model_name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(model_name or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown_model"


def load_tools():
    if str(DROPBOX_DIR) not in sys.path:
        sys.path.append(str(DROPBOX_DIR))
    import tools  # type: ignore

    importlib.reload(tools)
    return tools


def load_repair_loads():
    try:
        from json_repair import loads as repair_loads  # type: ignore

        return repair_loads
    except Exception:
        return None


def load_dataframe(parquet_path: Path):
    import pandas as pd  # type: ignore

    return pd.read_parquet(parquet_path, engine="pyarrow")


def extract_relationships(ground_truth: str) -> List[Dict[str, Any]]:
    match = REL_BLOCK_RE.search(str(ground_truth or ""))
    if not match:
        return []
    block = match.group(1).strip()
    return json.loads(block)


def build_examples(df, limit: int) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for local_idx in range(min(limit, len(df))):
        row = df.iloc[local_idx]
        extra_info = row.get("extra_info") or {}
        relationships = extract_relationships(row.get("ground_truth", ""))
        text = str(
            extra_info.get("text_fixed_by_revision")
            or extra_info.get("text_raw_from_file")
            or ""
        ).strip()
        examples.append(
            {
                "local_idx": local_idx,
                "stable_article_id": extra_info.get("stable_article_id"),
                "source_file": extra_info.get("source_file"),
                "text_chars": len(text),
                "relationship_count": len(relationships),
                "text": text,
                "relationships": relationships,
            }
        )
    return examples


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
        "You are a CTI knowledge-graph relation normalization expert. "
        "Generate one Python regex per KG edge to match semantically equivalent rollout relation strings. "
        "Output valid JSON only. Do not output reasoning, markdown, or any extra prose."
    )
    retry_hint = ""
    if retry_mode:
        retry_hint = """
补跑提醒:
- 上一轮失败通常是因为输出条数不对或 JSON 不规范。
- 这次务必只输出 JSON，且 `edge_regexes` 长度必须严格等于输入边数。
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
根据给定的“原文 + KG边列表”，为 **每一条 KG 边** 生成 **1 条 relation regex**，用于后续给 rollout 模型的 `Relationship_List[*].rel` 字符串打分。

这次只匹配 `rel` 字段字符串，不匹配实体，因此:
- 不要为 `sub` / `obj` 写 regex
- 你输出的每条规则只负责判断“这条边的关系短语有没有在 rollout 的 `rel` 中被表达出来”

⚠️ 强约束:
1. 必须逐边输出，不能合并，不能去重，不能少条数。
2. 输出数组长度必须与输入边数完全相同，顺序也必须完全一致。
3. 目标是减少假阴性，让 reward 更贴近“实际语义一致”。
4. regex 要尽量覆盖 rollout 常见变种:
   - 词形变化: `target / targets / targeted / targeting`
   - 主被动和助动词壳层: `used to detect / is used to detect`
   - 介词变化: `against / targeting`, `located in / stored in / written to`
   - 轻微同义改写: `created by / authored by / written by`
   - 归一化或简化: `query for / queries / querying / request`
5. 但不要过度泛化到完全不同的关系语义。
6. 如果多条边的 `rel` 一样或非常接近，也仍然必须重复输出多次，因为分母按边数计算。
7. 若边包含 `special_factuality`（如 `possible`、`negated`），在关系 regex 中尽量兼容这些语气词，但仍保持主关系可命中。
8. 输出必须是合法 Python regex，建议统一加 `(?is)` 前缀。
9. 不要把实体名写进 regex；这一步只匹配关系短语。
10. 不要输出任何多余字段或解释。

关系 regex 设计示例:
- `created by` -> `(?is)(created|authored|written|developed)\\s+by`
- `against` -> `(?is)(against|target(s|ed|ing)?)`
- `detects` -> `(?is)(detect(s|ed|ing)?|verif(y|ies|ied|ying)|check(s|ed|ing)?|used\\s+to\\s+(detect|verify|check))`
- `delivers` + possible -> `(?is)((may|might|could)\\s+)?deliver(s|ed|ing)?`

{retry_hint}
{chunk_hint}

输出格式:
只能输出 JSON，不要加围栏外说明，不要输出 reasoning，不要输出 ```json 围栏。

{{
  "article_local_idx": {example["local_idx"]},
  "stable_article_id": "{example.get("stable_article_id")}",
  "n_input_edges": {input_edge_count},
  "n_output_regexes": {input_edge_count},
  "edge_regexes": [
    {{
      "edge_idx": {edge_indices[0] if edge_indices else 0},
      "rel_regex": "(?is)...",
      "confidence": "high|medium|low"
    }}
  ]
}}

再次强调:
- `edge_regexes` 的长度必须等于 {input_edge_count}
- 第 i 个输出必须对应第 i 条输入边
- 即使 regex 一样也必须重复输出
- 不能漏边
- 不能输出说明文字

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
This is retry prompt serial {retry_prompt_serial} for this regex generation task.
Last time you gave me an unusable result because the JSON or edge count could not be parsed correctly.
This time:
- output JSON only
- keep edge count strictly correct
- do not output reasoning or markdown
- do not truncate
""".strip()


def parse_llm_json(raw_text: str) -> Tuple[Optional[Any], Optional[str]]:
    repair_loads = load_repair_loads()
    candidates = [str(raw_text or "").strip()]
    fence_match = FENCE_RE.search(raw_text or "")
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    raw_text = str(raw_text or "")
    first_obj_start = raw_text.find("{")
    last_obj_end = raw_text.rfind("}")
    if first_obj_start != -1 and last_obj_end != -1 and last_obj_end > first_obj_start:
        candidates.append(raw_text[first_obj_start:last_obj_end + 1].strip())
    for candidate in candidates:
        if not candidate:
            continue
        if repair_loads is not None:
            try:
                return repair_loads(candidate), "json_repair"
            except Exception:
                pass
        try:
            return json.loads(candidate), "json"
        except Exception:
            pass
    return None, None


def validate_rel_regex_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    pattern = str(item.get("rel_regex", "")).strip()
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
        if not validate_rel_regex_item(item):
            invalid_regex_indexes.append(output_pos)
            continue
        normalized.append(
            {
                "edge_idx": edge_idx,
                "rel_regex": str(item.get("rel_regex", "")).strip(),
                "confidence": str(item.get("confidence", "medium") or "medium").strip() or "medium",
            }
        )
    if invalid_regex_indexes:
        return None, invalid_regex_indexes
    return normalized, []


def heuristic_relation_regex(rel_text: str, special_factuality: Optional[List[str]] = None) -> str:
    rel_norm = normalize_space(rel_text).lower()
    factuality = [normalize_space(x).lower() for x in (special_factuality or [])]
    known_patterns = {
        "created by": r"(?is)(created|authored|written|developed)\s+by",
        "founded and maintained by": r"(?is)(founded|established|maintained)\s+(and\s+maintained\s+)?by",
        "written for": r"(?is)(written|built|designed)\s+for|target(s|ed|ing)?",
        "written in": r"(?is)(written|developed|implemented|coded)\s+in|uses?",
        "based on": r"(?is)(based|derived)\s+on|derived\s+from",
        "is part of": r"(?is)(is\s+)?(part\s+of|member\s+of|belong(s|ed|ing)?\s+to)",
        "query for": r"(?is)(query|queries|queried|querying|request|requests|requested|requesting)\s+(for\s+)?",
        "targeted": r"(?is)(target|targets|targeted|targeting|against|affect(s|ed|ing)?)",
        "affected": r"(?is)(affect|affects|affected|affecting|target|targets|targeted|targeting)",
        "infect": r"(?is)(infect|infects|infected|infecting|target|targets|targeted|targeting)",
        "delivers": r"(?is)deliver(s|ed|ing)?",
        "protected from": r"(?is)(protect(s|ed|ing)?\s+from|mitigate(s|d|ing)?|prevent(s|ed|ing)?)",
        "pretends to be a patch for": r"(?is)(pretend(s|ed|ing)?\s+to\s+be(\s+a\s+patch\s+for)?|impersonate(s|d|ing)?|masquerade(s|d|ing)?\s+as)",
    }
    pattern = known_patterns.get(rel_norm)
    if pattern is None:
        escaped = re.escape(rel_norm)
        escaped = escaped.replace(r"\ ", r"\s+")
        pattern = rf"(?is){escaped}"
    if "possible" in factuality and not re.search(r"\bmay\b|\bmight\b|\bcould\b", pattern):
        core = pattern[4:] if pattern.startswith("(?is)") else pattern
        pattern = rf"(?is)((may|might|could)\s+)?(?:{core})"
    if "negated" in factuality and not re.search(r"\bnot\b|\bnever\b|\bno\b", pattern):
        core = pattern[4:] if pattern.startswith("(?is)") else pattern
        pattern = rf"(?is)((not|never)\s+)?(?:{core})"
    return pattern


def _strip_leading_inline_flags(pattern: str) -> str:
    return re.sub(r"^\(\?[a-zA-Z]+\)", "", str(pattern or "").strip(), count=1)


def ensure_regex_matches_relation_literal(rel_regex: str, rel_text: str) -> str:
    pattern = str(rel_regex or "").strip()
    rel_literal = normalize_space(rel_text)
    if not pattern:
        escaped_literal = re.escape(rel_literal).replace(r"\ ", r"\s+")
        return rf"(?is){escaped_literal}"
    try:
        if re.search(pattern, rel_literal):
            return pattern
    except re.error:
        pass
    escaped_literal = re.escape(rel_literal).replace(r"\ ", r"\s+")
    pattern_core = _strip_leading_inline_flags(pattern)
    return rf"(?is)(?:{pattern_core}|{escaped_literal})"


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


def is_acceptable_article_result(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    if not result.get("parsed_ok", False):
        return False
    # 2026-03-25:
    
    
    return result.get("stage") != "heuristic_fallback"


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
        note=f"grid_relregex_{model_tag(api_model)}_{stage}_chunked_idx{example['local_idx']}",
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
                "rel_regex": heuristic_relation_regex(
                    edge.get("rel", ""),
                    edge.get("special_factuality") or [],
                ),
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
                    "rel_regex": item.get("rel_regex", ""),
                    "confidence": item.get("confidence", "low"),
                    "stage": result.get("stage"),
                    "parser": result.get("parser"),
                }
            )
    dump_json(run_dir / f"{api_model_tag}_relregex_article_results.json", ordered_results)
    dump_json(run_dir / f"{api_model_tag}_relregex_flattened_regexes.json", flattened)


def load_existing_progress(
    run_dir: Path,
    api_model_tag: str,
) -> Dict[int, Dict[str, Any]]:
    path = run_dir / f"{api_model_tag}_relregex_article_results.json"
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


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        chunk_size = 1
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


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
    dump_json(run_dir / f"{api_model_tag}_relregex_prompts_manifest.json", manifest)

    article_result_map = load_existing_progress(run_dir, api_model_tag)
    example_map = {item["local_idx"]: item for item in examples}

    def process_batch(
        batch_examples: List[Dict[str, Any]],
        stage: str,
        current_workers: int,
    ) -> Dict[str, Any]:
        force_json_retry_style = api_model.startswith("gemini-2.5-flash")
        current_temp = 0.0 if force_json_retry_style else (0.1 if stage == "initial" else 0.0)
        pending_examples = list(batch_examples)
        failed_examples: List[Dict[str, Any]] = []
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
                    retry_mode=(stage != "initial") or force_json_retry_style or (cache_attempt_idx > 0),
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
                # 2026-03-26:
                
                
                
                check_history_cache=True,
                showthink=False,
                think="low",
                openai_verbosity="low",
                count=True,
                force_api_do_huge_input_Cloud=True,
                cloud_executor_workers=current_workers,
                note=f"grid_relregex_{api_model_tag}_{attempt_label}",
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
                    f"🧪 regex缓存验收后仍失败 {len(pending_examples)} 条 | "
                    f"下一轮将追加 retry notice 再 asks"
                )
        failed_examples = list(pending_examples)
        save_progress(run_dir, api_model_tag, article_result_map, examples)
        batch_size = len(batch_examples)
        failed_count = len(failed_examples)
        success_ratio = (acceptable_count / batch_size) if batch_size else 0.0
        return {
            "batch_size": batch_size,
            "acceptable_count": acceptable_count,
            "failed_count": failed_count,
            "empty_like_count": empty_like_count,
            "success_ratio": success_ratio,
            "failed_examples": failed_examples,
        }

    initial_pending = [
        example
        for example in examples
        if not is_acceptable_article_result(article_result_map.get(example["local_idx"], {}))
    ]
    if initial_pending:
        print(
            f"🚀 regex初轮采用 asks 无分批模式 | 待处理文章数={len(initial_pending)} | "
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
        print(
            f"🔁 开始第 {retry_idx + 1}/{retry_rounds} 轮补跑 | 失败文章数={len(pending)}"
        )
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
            print(
                f"🛠️ heuristic fallback idx={example['local_idx']:03d} | "
                f"边数={len(example['relationships'])}"
            )
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
        "article_results_path": str(run_dir / f"{api_model_tag}_relregex_article_results.json"),
        "flattened_regexes_path": str(run_dir / f"{api_model_tag}_relregex_flattened_regexes.json"),
    }


def build_regex_training_rows(
    df,
    examples: List[Dict[str, Any]],
    article_results: List[Dict[str, Any]],
    api_model: str,
    run_dir: Path,
):
    example_map = {item["local_idx"]: item for item in examples}
    result_map = {item["local_idx"]: item for item in article_results}
    output_rows: List[Dict[str, Any]] = []

    for example in examples:
        row = df.iloc[example["local_idx"]].to_dict()
        result = result_map[example["local_idx"]]
        parsed = result.get("parsed") or {}
        edge_regexes = parsed.get("edge_regexes", []) if isinstance(parsed, dict) else []
        ordered_regexes: List[str] = []
        for output_pos, item in enumerate(edge_regexes):
            if not isinstance(item, dict):
                ordered_regexes.append("")
                continue
            edge_idx = item.get("edge_idx", output_pos)
            if not isinstance(edge_idx, int):
                edge_idx = output_pos
            source_edge = (
                example["relationships"][output_pos]
                if 0 <= output_pos < len(example["relationships"])
                else {}
            )
            rel_text = str(source_edge.get("rel", ""))
            widened_regex = ensure_regex_matches_relation_literal(
                str(item.get("rel_regex", "")).strip(),
                rel_text,
            )
            if edge_idx != output_pos:
                
                ordered_regexes.append(widened_regex)
            else:
                ordered_regexes.append(widened_regex)
        if len(ordered_regexes) != len(example["relationships"]):
            raise RuntimeError(
                f"regex 数量与边数不一致: idx={example['local_idx']}, "
                f"regex={len(ordered_regexes)}, edges={len(example['relationships'])}"
            )

        extra_info = dict(row.get("extra_info") or {})
        extra_info["dataset_type"] = "train"
        extra_info["question_type"] = "regex match"
        extra_info["step6_task_type"] = "regex match"
        extra_info["regex_groundtruth_kind"] = "relation_string_regex_list"
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
    import pandas as pd  # type: ignore

    ensure_parent(output_parquet)
    df = pd.DataFrame(output_rows)
    df.to_parquet(output_parquet, engine="pyarrow", index=False)


def write_overview_png(
    run_dir: Path,
    article_results: List[Dict[str, Any]],
) -> Optional[Path]:
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
    axes[0].set_title("Regex-for-Recall Edge Counts per Article")
    axes[0].set_xlabel("Article Local Index")
    axes[0].set_ylabel("KG Edge Count")

    axes[1].bar(indices, status_values, color="#1F77B4")
    axes[1].set_title("LLM Parse Success (1=LLM Success, 0=Fallback)")
    axes[1].set_xlabel("Article Local Index")
    axes[1].set_ylabel("Success")
    axes[1].set_ylim(-0.1, 1.1)

    png_path = run_dir / "regex_for_recall_500_overview.png"
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
    print(f"✅ 已写出 regex-for-recall parquet: {output_parquet}")

    png_path = write_overview_png(run_dir, generation_result["article_results"])
    summary_path = write_summary(
        run_dir=run_dir,
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        examples=examples,
        generation_result=generation_result,
        png_path=png_path,
    )
    print(f"📝 Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("💥 生成 regex-for-recall 训练集失败")
        traceback.print_exc()
        raise
