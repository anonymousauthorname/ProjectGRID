# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd
import tiktoken
import yaml
from json_repair import loads as json_repair_loads
from networkx.algorithms.approximation import treewidth_min_degree

try:
    import numpy as np
except Exception:  
    np = None


SCRIPT_VERSION = "20260319_v2"
DROPBOX_PATH = os.path.join(os.path.expanduser("~"), "Dropbox")

if DROPBOX_PATH not in sys.path:
    sys.path.insert(0, DROPBOX_PATH)

BASE_COLUMNS = [
    "prompt",
    "data_source",
    "ability",
    "ground_truth",
    "extra_info",
    "reward_model",
    "sft_ground_truth",
]

TASK_TYPE_ENTITY_ONLY = "entity_only"
TASK_TYPE_REL_ONLY = "relation_only_given_entities"
TASK_TYPE_FOCUS_SUBGRAPH = "focus_entity_subgraph"
TASK_TYPE_COREFERENCE = "coreference_only_for_entity"
TASK_TYPE_PAIR_REL = "relation_only_for_entity_pair"

TASK_SPECS: Dict[str, Dict[str, Any]] = {
    TASK_TYPE_ENTITY_ONLY: {
        "display_name": "只抽实体",
        "ability": "knowledge_graph_entity_only",
        "difficulty_offset": 0.08,
        "requires_relations": False,
        "requires_alias": False,
    },
    TASK_TYPE_COREFERENCE: {
        "display_name": "只抽共指",
        "ability": "knowledge_graph_coreference_only",
        "difficulty_offset": 0.22,
        "requires_relations": False,
        "requires_alias": True,
    },
    TASK_TYPE_PAIR_REL: {
        "display_name": "只抽实体对关系",
        "ability": "knowledge_graph_pair_relation_only",
        "difficulty_offset": 0.40,
        "requires_relations": True,
        "requires_alias": False,
    },
    TASK_TYPE_FOCUS_SUBGRAPH: {
        "display_name": "焦点实体一跳子图",
        "ability": "knowledge_graph_focus_entity_subgraph",
        "difficulty_offset": 0.58,
        "requires_relations": True,
        "requires_alias": False,
    },
    TASK_TYPE_REL_ONLY: {
        "display_name": "给定实体只抽关系",
        "ability": "knowledge_graph_relation_only",
        "difficulty_offset": 0.72,
        "requires_relations": True,
        "requires_alias": False,
    },
}

COMPLEXITY_RESEARCH_REFERENCES = [
    {
        "name": "TGCL (EMNLP Findings 2023)",
        "url": "https://aclanthology.org/2023.findings-emnlp.172.pdf",
        "summary": "把 degree、density、average clustering、degree assortativity、local bridges、treewidth(min-degree) 等指标用于图样本复杂度建模与 curriculum learning。",
    },
    {
        "name": "NetworkX density docs",
        "url": "https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.density.html",
        "summary": "给出图密度定义，可衡量连接稠密程度。",
    },
    {
        "name": "NetworkX average_clustering docs",
        "url": "https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.average_clustering.html",
        "summary": "给出平均聚类系数，可衡量局部闭包/三角结构。",
    },
    {
        "name": "NetworkX local_bridges docs",
        "url": "https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bridges.local_bridges.html",
        "summary": "local bridge 表示不属于任何三角形的边；桥越多，图越接近链式，通常交织程度越低。",
    },
    {
        "name": "NetworkX treewidth_min_degree docs",
        "url": "https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.treewidth.treewidth_min_degree.html",
        "summary": "treewidth 近似值可衡量图是否接近树结构；treewidth 越高，结构依赖通常越复杂。",
    },
    {
        "name": "Crossing Number Survey",
        "url": "https://www.combinatorics.org/files/Surveys/ds21/ds21v7-2022.pdf",
        "summary": "图交叉数与 fixed linear crossing number 提供了“固定点顺序下边交叉多少”的 formal basis，适合映射到文章线性顺序里的关系交叉复杂度。",
    },
]

FALLBACK_TEMPLATE_BANK: Dict[str, Any] = {
    "metadata": {
        "source": "fallback_local",
        "model": "manual_fallback",
        "version": SCRIPT_VERSION,
    },
    "task_templates": {
        TASK_TYPE_ENTITY_ONLY: {
            "instruction": "Read the CTI article and extract only the entities. Do not output any relations.",
            "output_rules": [
                "Use only article-grounded entities.",
                "Use the most natural article-grounded display name for each entity.",
                "Include type and alias/coreference when available.",
                "Return JSON only.",
            ],
        },
        TASK_TYPE_REL_ONLY: {
            "instruction": "You are given the article and the full entity inventory. Extract only the relations supported by the article among the provided entities. Do not add new entities.",
            "output_rules": [
                "Subject/object must come from the provided entity inventory.",
                "Only output article-grounded relations.",
                "Direction matters.",
                "Return JSON only.",
            ],
        },
        TASK_TYPE_FOCUS_SUBGRAPH: {
            "instruction": "You are given the article and one focus entity reference. Extract only the one-hop subgraph centered on that focus entity.",
            "output_rules": [
                "Only keep relations touching the focus entity.",
                "Only keep neighbor entities that participate in those relations.",
                "Do not output unrelated graph parts.",
                "Return JSON only.",
            ],
        },
        TASK_TYPE_COREFERENCE: {
            "instruction": "You are given the article and one focus entity reference. List only the coreferential mentions / aliases for that entity. Do not output any relations.",
            "output_rules": [
                "Only include mentions that refer to the same entity in this article.",
                "Do not include entity types or relations.",
                "Deduplicate repeated mentions.",
                "Return JSON only.",
            ],
        },
        TASK_TYPE_PAIR_REL: {
            "instruction": "You are given the article and one entity pair. Extract only the relation(s) between this entity pair.",
            "output_rules": [
                "Only consider the provided entity pair.",
                "Direction matters.",
                "If no relation exists, return an empty list.",
                "Return JSON only.",
            ],
        },
    },
    "judge_templates": {
        TASK_TYPE_ENTITY_ONLY: {
            "rubric": "Score entity precision/recall/F1, plus type accuracy on matched entities. Accept alias/coreference equivalence but reject hallucinations.",
        },
        TASK_TYPE_REL_ONLY: {
            "rubric": "Score relation precision/recall/F1, with alias/coreference resolution for endpoints and strict directionality.",
        },
        TASK_TYPE_FOCUS_SUBGRAPH: {
            "rubric": "Score both one-hop neighbor coverage and relation coverage around the focus entity. Reject unrelated edges.",
        },
        TASK_TYPE_COREFERENCE: {
            "rubric": "Score alias/coreference precision/recall/F1 for the specified entity only.",
        },
        TASK_TYPE_PAIR_REL: {
            "rubric": "Score only the relations between the provided entity pair, direction-aware.",
        },
    },
    "reward_design": {
        TASK_TYPE_ENTITY_ONLY: "0.70 * entity_name_f1 + 0.20 * type_accuracy + 0.10 * alias_accuracy",
        TASK_TYPE_REL_ONLY: "relation_f1",
        TASK_TYPE_FOCUS_SUBGRAPH: "0.80 * relation_f1 + 0.20 * neighbor_entity_f1",
        TASK_TYPE_COREFERENCE: "coreference_f1",
        TASK_TYPE_PAIR_REL: "relation_f1",
    },
}

SCHEMA_STRINGS: Dict[str, str] = {
    TASK_TYPE_ENTITY_ONLY: json.dumps(
        {
            "entity": [
                {"name": "article-grounded entity display name", "type": "entity type", "alias": ["optional alias 1", "optional alias 2"]}
            ]
        },
        ensure_ascii=False,
        indent=2,
    ),
    TASK_TYPE_REL_ONLY: json.dumps(
        {"relationship": [{"sub": "entity A", "rel": "relation label", "obj": "entity B"}]},
        ensure_ascii=False,
        indent=2,
    ),
    TASK_TYPE_FOCUS_SUBGRAPH: json.dumps(
        {
            "focus_entity": "focus entity display name",
            "entity": [{"name": "focus or one-hop neighbor display name", "type": "entity type", "alias": ["optional alias"]}],
            "relationship": [{"sub": "entity A display name", "rel": "relation label", "obj": "entity B display name"}],
        },
        ensure_ascii=False,
        indent=2,
    ),
    TASK_TYPE_COREFERENCE: json.dumps(
        {"focus_entity": "focus entity display name", "coreference": ["mention 1", "mention 2"]},
        ensure_ascii=False,
        indent=2,
    ),
    TASK_TYPE_PAIR_REL: json.dumps(
        {
            "entity_pair": ["entity A display name", "entity B display name"],
            "relationship": [{"sub": "entity A display name", "rel": "relation label", "obj": "entity B display name"}],
        },
        ensure_ascii=False,
        indent=2,
    ),
}

REWARD_MODULE_PATH = str(SCRIPT_DIR.parent / "reward" / "kg_reward.py")

LOW_QUALITY_ENTITY_NAMES = {
    "it",
    "its",
    "they",
    "them",
    "their",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "him",
    "her",
    "his",
    "hers",
    "the attacker",
    "the attackers",
    "attacker",
    "attackers",
    "the actor",
    "the actors",
    "actor",
    "actors",
    "the campaign",
    "campaign",
    "the malware",
    "malware",
    "the tool",
    "tool",
    "the software",
    "software",
    "the vulnerability",
    "vulnerability",
    "the exploit",
    "exploit",
    "the company",
    "company",
    "the organization",
    "organization",
    "the victim",
    "victim",
    "the group",
    "group",
    "the threat actor",
    "threat actor",
}


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n`\"'")


def _clean_alias_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    elif hasattr(value, "tolist") and not isinstance(value, dict):
        raw_items = list(value.tolist())
    elif isinstance(value, Sequence):
        raw_items = list(value)
    else:
        raw_items = [str(value)]

    cleaned: List[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if not text:
            continue
        if _norm_text(text) in {"none", "null", "n/a"}:
            continue
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def _dedupe_keep_order(items: Sequence[Any]) -> List[str]:
    result: List[str] = []
    seen: set = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        norm = _norm_text(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append(text)
    return result


def _find_exact_mention(text: str, candidate: str) -> Optional[int]:
    raw_text = str(text or "")
    cand = str(candidate or "").strip()
    if not raw_text or not cand:
        return None
    match = re.search(re.escape(cand), raw_text, flags=re.IGNORECASE)
    if match:
        return int(match.start())
    return None


def _score_display_name_candidate(article_text: str, candidate: str, canonical_name: str) -> Dict[str, Any]:
    text = str(candidate or "").strip()
    norm = _norm_text(text)
    first_pos = _find_exact_mention(article_text, text)
    token_count = len(text.split())
    score = 0.0

    if first_pos is not None:
        score += max(0.0, 40.0 - min(first_pos, 4000) / 100.0)
    else:
        score -= 8.0

    if any(ch.isupper() for ch in text):
        score += 8.0
    if any(ch.isdigit() for ch in text):
        score += 8.0
    if "-" in text or "_" in text or "/" in text:
        score += 2.0
    if 4 <= len(text) <= 48:
        score += 4.0
    elif len(text) <= 2:
        score -= 10.0

    if 2 <= token_count <= 6:
        score += 3.0
    elif token_count > 10:
        score -= 4.0

    if norm == _norm_text(canonical_name):
        score += 1.0
    if norm in LOW_QUALITY_ENTITY_NAMES:
        score -= 100.0
    if text.islower() and token_count == 1:
        score -= 3.0

    needs_llm = norm in LOW_QUALITY_ENTITY_NAMES or score < 12.0
    return {
        "candidate": text,
        "norm": norm,
        "score": float(score),
        "first_pos": first_pos,
        "needs_llm": bool(needs_llm),
    }


def _choose_display_name_heuristic(article_text: str, canonical_name: str, aliases: Sequence[str]) -> Dict[str, Any]:
    candidates = _dedupe_keep_order([canonical_name] + list(aliases))
    if not candidates:
        return {
            "display_name": canonical_name,
            "candidates": [canonical_name] if canonical_name else [],
            "name_selection_source": "heuristic",
            "name_selection_score": 0.0,
            "needs_llm": False,
        }

    scored = [_score_display_name_candidate(article_text, candidate, canonical_name) for candidate in candidates]
    scored.sort(
        key=lambda item: (
            -item["score"],
            item["first_pos"] if item["first_pos"] is not None else 10**9,
            -len(item["candidate"]),
            item["candidate"],
        )
    )
    best = scored[0]
    return {
        "display_name": best["candidate"],
        "candidates": candidates,
        "name_selection_source": "heuristic",
        "name_selection_score": float(best["score"]),
        "needs_llm": bool(best["needs_llm"]),
    }


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        return json_repair_loads(text)
    except Exception:
        return {}


def _to_plain_obj(value: Any) -> Any:
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [_to_plain_obj(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _to_plain_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_obj(v) for v in value]
    if isinstance(value, tuple):
        return [_to_plain_obj(v) for v in value]
    return value


def _ensure_tools():
    import tools  # type: ignore

    return tools


def _load_reward_module():
    spec = importlib.util.spec_from_file_location("grid_aug_reward", REWARD_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"❌ 无法加载 reward 模块: {REWARD_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="🚀 基于既有 GRID parquet 生成复杂度排序版 parquet 与新增强数据集"
    )
    parser.add_argument("--yaml", required=True, help="旧的 260311 实验 YAML 路径")
    parser.add_argument(
        "--output_root",
        default=os.path.join(
            DROPBOX_PATH,
            "项目GRID/src/Generate_Train_Data/temp/260318实验_Primus第一批次_复杂度排序与数据增强_v1_dataset",
        ),
        help="新的额外输出目录",
    )
    parser.add_argument(
        "--tokenizer",
        default="cl100k_base",
        help="用于估算文章 token 长度的 tokenizer 名称",
    )
    parser.add_argument("--max_rows", type=int, default=0, help="仅调试时使用；0 表示全量")
    parser.add_argument("--skip_llm_templates", action="store_true", help="跳过 gemini-2.5-flash 模板生成，仅使用本地 fallback 模板")
    parser.add_argument("--skip_validation", action="store_true", help="跳过 gemini-2.5-flash 在线验证")
    parser.add_argument("--template_model", type=str, default="gemini-2.5-flash", help="模板/验证使用的模型")
    parser.add_argument("--template_token", type=int, default=16384, help="模板/验证输出 token 上限")
    parser.add_argument(
        "--entity_name_mode",
        type=str,
        default="hybrid",
        choices=["heuristic", "llm", "hybrid"],
        help="增强任务里实体对外显示名的选择方式: heuristic / llm / hybrid",
    )
    parser.add_argument(
        "--entity_name_model",
        type=str,
        default="gemini-2.5-flash",
        help="当 entity_name_mode 需要 LLM 时使用的模型",
    )
    parser.add_argument(
        "--entity_name_token",
        type=int,
        default=8192,
        help="当 entity_name_mode 需要 LLM 时使用的输出 token 上限",
    )
    return parser


def _load_yaml(path: str) -> Dict[str, Any]:
    print(f"📄 读取 YAML: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _discover_variant_paths(config: Dict[str, Any]) -> Dict[str, str]:
    output_dir = str(config["output_dir"])
    output_name = str(config["output_name"])
    variant_paths: Dict[str, str] = {}
    for parquet_path in sorted(Path(output_dir).glob(f"{output_name}_*.parquet")):
        basename = parquet_path.name
        variant = basename[len(output_name) + 1 : -8]
        variant_paths[variant] = str(parquet_path)
    if not variant_paths:
        raise FileNotFoundError(f"❌ 未找到任何 parquet: output_dir={output_dir}, output_name={output_name}")
    print(f"📦 共发现 {len(variant_paths)} 个 parquet 变体")
    for variant, path in variant_paths.items():
        print(f"   - {variant}: {path}")
    return variant_paths


def _load_datasets(variant_paths: Dict[str, str], max_rows: int = 0) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for variant, path in variant_paths.items():
        print(f"📖 加载 {variant}")
        df = pd.read_parquet(path)
        if max_rows > 0:
            df = df.head(max_rows).copy()
        datasets[variant] = df
        print(f"   ✅ {variant}: {len(df)} rows")
    return datasets


def _build_tokenizer(tokenizer_name: str):
    try:
        return tiktoken.get_encoding(tokenizer_name)
    except Exception:
        print(f"⚠️ tokenizer={tokenizer_name} 不存在，回退到 cl100k_base")
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text.split())


def _parse_graph(graph_payload: Any) -> Dict[str, List[Dict[str, Any]]]:
    graph_obj = _safe_json_loads(graph_payload)
    if not isinstance(graph_obj, dict):
        return {"entity": [], "relationship": []}
    entity_list = graph_obj.get("entity", [])
    relationship_list = graph_obj.get("relationship", [])
    return {
        "entity": entity_list if isinstance(entity_list, list) else [],
        "relationship": relationship_list if isinstance(relationship_list, list) else [],
    }


def _normalize_graph(graph_payload: Any) -> Dict[str, List[Dict[str, Any]]]:
    raw_graph = _parse_graph(graph_payload)
    alias_to_canonical: Dict[str, str] = {}
    entity_map: Dict[str, Dict[str, Any]] = {}

    for entity in raw_graph["entity"]:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name") or "").strip()
        if not name:
            continue
        if name not in entity_map:
            entity_map[name] = {
                "name": name,
                "type": str(entity.get("type") or "").strip() or "unknown",
                "alias": [],
            }
        alias_to_canonical[_norm_text(name)] = name
        for alias in _clean_alias_list(entity.get("alias", [])):
            alias_to_canonical[_norm_text(alias)] = name
            if alias not in entity_map[name]["alias"]:
                entity_map[name]["alias"].append(alias)

    normalized_relationships: List[Dict[str, Any]] = []
    for rel in raw_graph["relationship"]:
        if not isinstance(rel, dict):
            continue
        raw_sub = str(rel.get("sub") or "").strip()
        raw_obj = str(rel.get("obj") or "").strip()
        sub = alias_to_canonical.get(_norm_text(raw_sub), raw_sub)
        obj = alias_to_canonical.get(_norm_text(raw_obj), raw_obj)
        rel_label = str(rel.get("rel") or "").strip()
        rel_types = [str(x).strip() for x in rel.get("rel_type", []) if str(x).strip()] if isinstance(rel.get("rel_type", []), list) else []
        if not rel_label and rel_types:
            rel_label = rel_types[0]
        if not sub or not obj or not rel_label:
            continue
        if sub not in entity_map:
            entity_map[sub] = {"name": sub, "type": "unknown", "alias": []}
        if obj not in entity_map:
            entity_map[obj] = {"name": obj, "type": "unknown", "alias": []}
        normalized_relationships.append(
            {
                "sub": sub,
                "rel": rel_label,
                "obj": obj,
                "rel_type": rel_types,
            }
        )

    return {
        "entity": list(entity_map.values()),
        "relationship": normalized_relationships,
    }


def _find_first_mention(text: str, candidates: Sequence[str]) -> Optional[int]:
    lowered = text.lower()
    best: Optional[int] = None
    for cand in candidates:
        cand = str(cand or "").strip()
        if not cand:
            continue
        pos = lowered.find(cand.lower())
        if pos == -1:
            continue
        if best is None or pos < best:
            best = pos
    return best


def _build_entity_order(text: str, graph: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict[str, int], Dict[str, Optional[int]], Dict[str, int]]:
    entity_positions: Dict[str, Optional[int]] = {}
    ordered_entities: List[Tuple[int, str, int]] = []

    for idx, entity in enumerate(graph["entity"]):
        name = entity["name"]
        candidates = [name] + _clean_alias_list(entity.get("alias", []))
        pos = _find_first_mention(text, candidates)
        entity_positions[name] = pos
        if pos is not None:
            ordered_entities.append((pos, name, idx))

    ordered_entities.sort(key=lambda item: (item[0], item[2], item[1]))
    rank_map: Dict[str, int] = {}
    next_rank = 0
    for _, name, _ in ordered_entities:
        rank_map[name] = next_rank
        next_rank += 1

    for idx, entity in enumerate(graph["entity"]):
        name = entity["name"]
        if name not in rank_map:
            rank_map[name] = next_rank
            next_rank += 1

    return rank_map, entity_positions, {entity["name"]: idx for idx, entity in enumerate(graph["entity"])}


def _select_display_names_with_api_model(
    article_text: str,
    graph: Dict[str, List[Dict[str, Any]]],
    model: str,
    token: int,
) -> Dict[str, str]:
    entity_payload: List[Dict[str, Any]] = []
    for entity in graph["entity"]:
        canonical_name = str(entity.get("name") or "").strip()
        candidates = _dedupe_keep_order([canonical_name] + _clean_alias_list(entity.get("alias", [])))
        if not canonical_name or len(candidates) <= 1:
            continue
        entity_payload.append(
            {
                "canonical_name": canonical_name,
                "type": str(entity.get("type") or "unknown"),
                "candidates": candidates,
            }
        )

    if not entity_payload:
        return {}

    tools = _ensure_tools()
    prompt = (
        "You are selecting the best user-facing entity display names for training prompts.\n"
        "For each entity, choose exactly one display_name from its provided candidate list.\n"
        "The chosen display_name must be article-grounded, natural when shown to a model in an instruction, "
        "and as specific as possible.\n"
        "Avoid pure pronouns or overly generic mentions if a more specific candidate exists.\n"
        "Never invent a new string; each display_name must be copied exactly from the candidates list.\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "choices": [\n'
        '    {"canonical_name": "...", "display_name": "..."}\n'
        "  ]\n"
        "}\n\n"
        f"Article:\n<<<ARTICLE>>>\n{article_text}\n<<<END_ARTICLE>>>\n\n"
        f"Entities:\n{json.dumps(entity_payload, ensure_ascii=False, indent=2)}"
    )
    responses = tools.ask_group_link(
        prompt_list=[[{"role": "user", "content": prompt}]],
        model=model,
        token=max(4096, int(token)),
        temp=0.1,
        streamprint=False,
        check_history_cache=True,
        retry=True,
        force_api_do_huge_input_Cloud=True,
        flex=True,
        note="grid-aug-display-name",
    )
    if hasattr(tools, "cleanthinkans"):
        responses = tools.cleanthinkans(responses)
    parsed = _safe_json_loads(responses[0] if responses else "")
    if not isinstance(parsed, dict):
        return {}

    candidate_lookup: Dict[str, Dict[str, str]] = {}
    for entity in entity_payload:
        canonical_name = entity["canonical_name"]
        candidate_lookup[canonical_name] = {_norm_text(name): name for name in entity["candidates"]}

    result: Dict[str, str] = {}
    for item in parsed.get("choices", []):
        if not isinstance(item, dict):
            continue
        canonical_name = str(item.get("canonical_name") or "").strip()
        display_name = str(item.get("display_name") or "").strip()
        normalized_display = _norm_text(display_name)
        if canonical_name not in candidate_lookup:
            continue
        display_name = candidate_lookup[canonical_name].get(normalized_display, "")
        if display_name:
            result[canonical_name] = display_name
    return result


def _build_entity_catalog(
    article_text: str,
    graph: Dict[str, List[Dict[str, Any]]],
    name_mode: str,
    name_model: str,
    name_token: int,
) -> List[Dict[str, Any]]:
    rank_map, _, entity_order = _build_entity_order(article_text, graph)
    sorted_entities = sorted(
        graph["entity"],
        key=lambda entity: (
            rank_map.get(entity["name"], 10**9),
            entity_order.get(entity["name"], 10**9),
            entity["name"],
        ),
    )

    heuristic_choices: Dict[str, Dict[str, Any]] = {}
    for entity in sorted_entities:
        canonical_name = entity["name"]
        heuristic_choices[canonical_name] = _choose_display_name_heuristic(
            article_text=article_text,
            canonical_name=canonical_name,
            aliases=_clean_alias_list(entity.get("alias", [])),
        )

    should_call_llm = False
    if name_mode == "llm":
        should_call_llm = True
    elif name_mode == "hybrid":
        should_call_llm = any(
            choice.get("needs_llm")
            for choice in heuristic_choices.values()
            if len(choice.get("candidates", [])) > 1
        )

    llm_choices: Dict[str, str] = {}
    if should_call_llm:
        try:
            llm_choices = _select_display_names_with_api_model(
                article_text=article_text,
                graph=graph,
                model=name_model,
                token=name_token,
            )
        except Exception as exc:
            print(f"⚠️ display name LLM 选择失败，回退 heuristic: {exc}")
            llm_choices = {}

    width = max(2, len(str(max(1, len(sorted_entities)))))
    catalog: List[Dict[str, Any]] = []
    for idx, entity in enumerate(sorted_entities, start=1):
        canonical_name = entity["name"]
        heur = heuristic_choices[canonical_name]
        display_name = str(llm_choices.get(canonical_name) or heur["display_name"]).strip() or canonical_name
        alias_names = _dedupe_keep_order([canonical_name] + _clean_alias_list(entity.get("alias", [])))
        filtered_aliases = [name for name in alias_names if _norm_text(name) != _norm_text(display_name)]
        source = heur.get("name_selection_source", "heuristic")
        if canonical_name in llm_choices:
            source = "llm" if name_mode == "llm" else "hybrid_llm"
        elif name_mode == "hybrid" and heur.get("needs_llm"):
            source = "hybrid_heuristic_fallback"

        catalog.append(
            {
                "entity_id": f"E{idx:0{width}d}",
                "canonical_name": canonical_name,
                "display_name": display_name,
                "type": str(entity.get("type") or "unknown"),
                "alias": filtered_aliases,
                "all_names": alias_names,
                "article_mention_rank": int(rank_map.get(canonical_name, idx - 1) + 1),
                "name_selection_source": source,
                "name_selection_score": float(heur.get("name_selection_score", 0.0)),
            }
        )
    return catalog


def _build_entity_catalog_map(entity_catalog: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("canonical_name") or ""): item
        for item in entity_catalog
        if str(item.get("canonical_name") or "").strip()
    }


def _compute_order_metrics(text: str, graph: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    rank_map, entity_positions, entity_order = _build_entity_order(text, graph)

    relation_rank_gaps: List[int] = []
    unique_edge_intervals: List[Tuple[str, str, int, int]] = []
    seen_pairs: set = set()

    for rel in graph["relationship"]:
        sub = rel["sub"]
        obj = rel["obj"]
        if sub not in rank_map or obj not in rank_map:
            continue
        gap = abs(rank_map[sub] - rank_map[obj])
        relation_rank_gaps.append(gap)
        if sub == obj:
            continue
        pair_key = tuple(sorted((sub, obj)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        left = min(rank_map[sub], rank_map[obj])
        right = max(rank_map[sub], rank_map[obj])
        unique_edge_intervals.append((pair_key[0], pair_key[1], left, right))

    crossing_count = 0
    comparable_pairs = 0
    for i in range(len(unique_edge_intervals)):
        _, _, a, b = unique_edge_intervals[i]
        u1, v1, _, _ = unique_edge_intervals[i]
        for j in range(i + 1, len(unique_edge_intervals)):
            _, _, c, d = unique_edge_intervals[j]
            u2, v2, _, _ = unique_edge_intervals[j]
            if len({u1, v1, u2, v2}) < 4:
                continue
            comparable_pairs += 1
            if (a < c < b < d) or (c < a < d < b):
                crossing_count += 1

    relation_span_mean = (sum(relation_rank_gaps) / len(relation_rank_gaps)) if relation_rank_gaps else 0.0
    relation_span_max = max(relation_rank_gaps) if relation_rank_gaps else 0
    crossing_density = crossing_count / comparable_pairs if comparable_pairs else 0.0

    return {
        "entity_rank_map": rank_map,
        "entity_positions": entity_positions,
        "entity_order_index": entity_order,
        "mean_relation_rank_gap": relation_span_mean,
        "max_relation_rank_gap": relation_span_max,
        "fixed_linear_crossing_count": crossing_count,
        "fixed_linear_crossing_density": crossing_density,
    }


def _compute_row_complexity(
    row_index: int,
    variant: str,
    row: pd.Series,
    tokenizer,
) -> Dict[str, Any]:
    info = _to_plain_obj(row["extra_info"] if "extra_info" in row else {})
    info = info if isinstance(info, dict) else {}

    text = str(info.get("text_fixed_by_revision") or info.get("text_raw_from_file") or "")
    graph = _normalize_graph(info.get("graph_from_text_raw_from_file", {}))
    order_metrics = _compute_order_metrics(text, graph)

    entity_count = len(graph["entity"])
    relation_count = len(graph["relationship"])
    article_token_length = _count_tokens(text, tokenizer)
    entity_per_1k_tokens = 1000.0 * entity_count / max(article_token_length, 1)
    relation_per_1k_tokens = 1000.0 * relation_count / max(article_token_length, 1)

    alias_counts = [len(_clean_alias_list(entity.get("alias", []))) for entity in graph["entity"]]
    avg_alias_count = (sum(alias_counts) / len(alias_counts)) if alias_counts else 0.0
    entity_with_alias_ratio = (sum(1 for x in alias_counts if x > 0) / len(alias_counts)) if alias_counts else 0.0

    undirected_edges = {
        tuple(sorted((rel["sub"], rel["obj"])))
        for rel in graph["relationship"]
        if rel["sub"] and rel["obj"] and rel["sub"] != rel["obj"]
    }
    g = nx.Graph()
    g.add_nodes_from(entity["name"] for entity in graph["entity"])
    g.add_edges_from(undirected_edges)

    node_count = g.number_of_nodes()
    edge_count = g.number_of_edges()
    avg_degree = (2.0 * edge_count / node_count) if node_count else 0.0
    density = nx.density(g) if node_count > 1 else 0.0
    avg_clustering = nx.average_clustering(g) if edge_count > 0 else 0.0
    local_bridge_ratio = (len(list(nx.local_bridges(g, with_span=False))) / edge_count) if edge_count > 0 else 0.0
    non_local_bridge_ratio = 1.0 - local_bridge_ratio if edge_count > 0 else 0.0
    treewidth = treewidth_min_degree(g)[0] if node_count > 0 else 0
    core_numbers = nx.core_number(g) if edge_count > 0 else {node: 0 for node in g.nodes}
    max_k_core = max(core_numbers.values()) if core_numbers else 0
    component_count = nx.number_connected_components(g) if node_count > 0 else 0

    return {
        "__row_index": row_index,
        "variant": variant,
        "stable_article_id": str(info.get("stable_article_id") or f"{variant}::row_{row_index}"),
        "sample_order": int(info.get("sample_order", row_index)),
        "article_token_length": int(article_token_length),
        "entity_count": int(entity_count),
        "relation_count": int(relation_count),
        "entity_per_1k_tokens": float(entity_per_1k_tokens),
        "relation_per_1k_tokens": float(relation_per_1k_tokens),
        "avg_alias_count": float(avg_alias_count),
        "entity_with_alias_ratio": float(entity_with_alias_ratio),
        "unique_node_count": int(node_count),
        "undirected_edge_count": int(edge_count),
        "avg_degree": float(avg_degree),
        "density": float(density),
        "avg_clustering": float(avg_clustering),
        "treewidth_min_degree": int(treewidth),
        "max_k_core": int(max_k_core),
        "non_local_bridge_ratio": float(non_local_bridge_ratio),
        "component_count": int(component_count),
        "mean_relation_rank_gap": float(order_metrics["mean_relation_rank_gap"]),
        "max_relation_rank_gap": int(order_metrics["max_relation_rank_gap"]),
        "fixed_linear_crossing_count": int(order_metrics["fixed_linear_crossing_count"]),
        "fixed_linear_crossing_density": float(order_metrics["fixed_linear_crossing_density"]),
    }


def _attach_complexity_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metrics_df = metrics_df.copy()

    positive_metrics = [
        "article_token_length",
        "entity_count",
        "relation_count",
        "entity_per_1k_tokens",
        "relation_per_1k_tokens",
        "avg_alias_count",
        "entity_with_alias_ratio",
        "avg_degree",
        "density",
        "avg_clustering",
        "treewidth_min_degree",
        "max_k_core",
        "non_local_bridge_ratio",
        "mean_relation_rank_gap",
        "fixed_linear_crossing_density",
    ]

    for col in positive_metrics:
        metrics_df[f"rank__{col}"] = metrics_df[col].rank(method="average", pct=True, ascending=True)

    base_cols = [
        "rank__article_token_length",
        "rank__entity_count",
        "rank__relation_count",
        "rank__entity_per_1k_tokens",
        "rank__relation_per_1k_tokens",
    ]
    graph_cols = [
        "rank__avg_alias_count",
        "rank__entity_with_alias_ratio",
        "rank__avg_degree",
        "rank__density",
        "rank__avg_clustering",
        "rank__treewidth_min_degree",
        "rank__max_k_core",
        "rank__non_local_bridge_ratio",
        "rank__mean_relation_rank_gap",
        "rank__fixed_linear_crossing_density",
    ]
    metrics_df["base_complexity_score"] = metrics_df[base_cols].mean(axis=1)
    metrics_df["graph_structure_complexity_score"] = metrics_df[graph_cols].mean(axis=1)
    metrics_df["final_complexity_score"] = 0.65 * metrics_df["base_complexity_score"] + 0.35 * metrics_df["graph_structure_complexity_score"]

    metrics_df = metrics_df.sort_values(
        by=["final_complexity_score", "sample_order", "stable_article_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    total = len(metrics_df)
    metrics_df["curriculum_rank"] = metrics_df.index + 1
    metrics_df["curriculum_percent"] = metrics_df["curriculum_rank"].apply(lambda x: (x - 1) / max(total - 1, 1))
    metrics_df["curriculum_decile"] = metrics_df["curriculum_percent"].apply(lambda x: min(10, int(math.floor(x * 10)) + 1))
    return metrics_df


def _save_dataframe(df: pd.DataFrame, parquet_path: str, json_path: str) -> None:
    df = df.copy()
    for col in BASE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_to_plain_obj)
    df.to_parquet(parquet_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"✅ 保存 {len(df)} rows -> {parquet_path}")


def _augment_extra_info_with_complexity(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    variant: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_row = {int(row["__row_index"]): row for row in metrics_df.to_dict("records")}
    sorted_indices = [int(row["__row_index"]) for row in metrics_df.to_dict("records")]
    sorted_df = df.iloc[sorted_indices].copy().reset_index(drop=True)

    updated_extra_infos: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    total = len(sorted_df)

    for new_index, old_row_index in enumerate(sorted_indices):
        original_extra = _to_plain_obj(sorted_df.iloc[new_index]["extra_info"])
        original_extra = copy.deepcopy(original_extra if isinstance(original_extra, dict) else {})
        metric_row = copy.deepcopy(by_row[int(old_row_index)])
        flat_rows.append(metric_row)

        complexity_payload = {
            "version": SCRIPT_VERSION,
            "variant": variant,
            "article_token_length": metric_row["article_token_length"],
            "entity_count": metric_row["entity_count"],
            "relation_count": metric_row["relation_count"],
            "entity_per_1k_tokens": metric_row["entity_per_1k_tokens"],
            "relation_per_1k_tokens": metric_row["relation_per_1k_tokens"],
            "avg_alias_count": metric_row["avg_alias_count"],
            "entity_with_alias_ratio": metric_row["entity_with_alias_ratio"],
            "avg_degree": metric_row["avg_degree"],
            "density": metric_row["density"],
            "avg_clustering": metric_row["avg_clustering"],
            "treewidth_min_degree": metric_row["treewidth_min_degree"],
            "max_k_core": metric_row["max_k_core"],
            "non_local_bridge_ratio": metric_row["non_local_bridge_ratio"],
            "mean_relation_rank_gap": metric_row["mean_relation_rank_gap"],
            "max_relation_rank_gap": metric_row["max_relation_rank_gap"],
            "fixed_linear_crossing_count": metric_row["fixed_linear_crossing_count"],
            "fixed_linear_crossing_density": metric_row["fixed_linear_crossing_density"],
            "base_complexity_score": metric_row["base_complexity_score"],
            "graph_structure_complexity_score": metric_row["graph_structure_complexity_score"],
            "final_complexity_score": metric_row["final_complexity_score"],
            "curriculum_rank": int(new_index + 1),
            "curriculum_decile": int(metric_row["curriculum_decile"]),
        }

        original_extra["complexity_metrics"] = complexity_payload
        original_extra["complexity_references"] = COMPLEXITY_RESEARCH_REFERENCES
        original_extra["index"] = int(new_index)
        original_extra["totalnum"] = int(total)
        original_extra["curriculum_rank"] = int(new_index + 1)
        updated_extra_infos.append(original_extra)

    sorted_df["extra_info"] = updated_extra_infos
    return sorted_df, pd.DataFrame(flat_rows)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _generate_template_bank_with_api_model(
    output_dir: str,
    model: str,
    token: int,
) -> Dict[str, Any]:
    tools = _ensure_tools()
    prompt = f"""
You are designing training-time data augmentation prompt templates and LLM judge prompt templates for a cybersecurity article -> knowledge graph dataset called GRID.

We need exactly these 5 task types:
1. {TASK_TYPE_ENTITY_ONLY}: article -> output only entities, no relations
2. {TASK_TYPE_REL_ONLY}: article + fixed entity inventory -> output only relations
3. {TASK_TYPE_FOCUS_SUBGRAPH}: article + one focus entity -> output only the one-hop subgraph for that entity
4. {TASK_TYPE_COREFERENCE}: article + one focus entity -> output only aliases/coreferential mentions, no relations
5. {TASK_TYPE_PAIR_REL}: article + one entity pair -> output only relations for that pair

Return strict JSON only with this schema:
{{
  "metadata": {{"source": "api_generated", "model": "{model}", "version": "{SCRIPT_VERSION}"}},
  "task_templates": {{
    "{TASK_TYPE_ENTITY_ONLY}": {{"instruction": "...", "output_rules": ["...", "..."]}},
    "{TASK_TYPE_REL_ONLY}": {{"instruction": "...", "output_rules": ["...", "..."]}},
    "{TASK_TYPE_FOCUS_SUBGRAPH}": {{"instruction": "...", "output_rules": ["...", "..."]}},
    "{TASK_TYPE_COREFERENCE}": {{"instruction": "...", "output_rules": ["...", "..."]}},
    "{TASK_TYPE_PAIR_REL}": {{"instruction": "...", "output_rules": ["...", "..."]}}
  }},
  "judge_templates": {{
    "{TASK_TYPE_ENTITY_ONLY}": {{"rubric": "..."}},
    "{TASK_TYPE_REL_ONLY}": {{"rubric": "..."}},
    "{TASK_TYPE_FOCUS_SUBGRAPH}": {{"rubric": "..."}},
    "{TASK_TYPE_COREFERENCE}": {{"rubric": "..."}},
    "{TASK_TYPE_PAIR_REL}": {{"rubric": "..."}}
  }},
  "reward_design": {{
    "{TASK_TYPE_ENTITY_ONLY}": "...",
    "{TASK_TYPE_REL_ONLY}": "...",
    "{TASK_TYPE_FOCUS_SUBGRAPH}": "...",
    "{TASK_TYPE_COREFERENCE}": "...",
    "{TASK_TYPE_PAIR_REL}": "..."
  }}
}}

Constraints:
- Prompts should be in English.
- They should be concise but strict.
- The actual article / entity inventory / focus entity / pair will be appended later by a script.
- If a user-facing entity name is needed, assume the script will provide a natural display name instead of a raw canonical name.
- Reward design descriptions should be short formula-style descriptions.
- No markdown fences. JSON only.
""".strip()

    print(f"🤖 使用 {model} 生成模板银行 ...")
    responses = tools.ask_group_link(
        prompt_list=[[{"role": "user", "content": prompt}]],
        model=model,
        token=token,
        temp=0.3,
        streamprint=False,
        check_history_cache=True,
        retry=True,
        force_api_do_huge_input_Cloud=True,
        flex=True,
        note="grid-aug-template-bank",
    )
    if hasattr(tools, "cleanthinkans"):
        responses = tools.cleanthinkans(responses)
    raw_response = responses[0] if responses else ""
    raw_path = os.path.join(output_dir, "template_bank_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_response)

    try:
        parsed = json_repair_loads(raw_response)
        if not isinstance(parsed, dict):
            parsed = {}
    except Exception:
        parsed = {}

    merged = _deep_merge(FALLBACK_TEMPLATE_BANK, parsed)
    with open(os.path.join(output_dir, "模板银行_merged.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


def _format_json(obj: Any) -> str:
    return json.dumps(_to_plain_obj(obj), ensure_ascii=False, indent=2)


def _pick_focus_entity(graph: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    degree_counter: Counter = Counter()
    for rel in graph["relationship"]:
        degree_counter[rel["sub"]] += 1
        degree_counter[rel["obj"]] += 1
    alias_counter = {entity["name"]: len(_clean_alias_list(entity.get("alias", []))) for entity in graph["entity"]}
    ranked = sorted(
        [entity["name"] for entity in graph["entity"]],
        key=lambda name: (-degree_counter.get(name, 0), -alias_counter.get(name, 0), name),
    )
    return ranked[0] if ranked else None


def _pick_coref_entity(graph: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    candidates = []
    degree_counter: Counter = Counter()
    for rel in graph["relationship"]:
        degree_counter[rel["sub"]] += 1
        degree_counter[rel["obj"]] += 1
    for entity in graph["entity"]:
        aliases = _clean_alias_list(entity.get("alias", []))
        if aliases:
            candidates.append((len(aliases), degree_counter.get(entity["name"], 0), entity["name"]))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return candidates[0][2]


def _pick_relation_pair(graph: Dict[str, List[Dict[str, Any]]], text: str) -> Optional[Tuple[List[str], List[Dict[str, Any]]]]:
    if not graph["relationship"]:
        return None
    order_metrics = _compute_order_metrics(text, graph)
    rank_map = order_metrics["entity_rank_map"]

    def rel_key(rel: Dict[str, Any]) -> Tuple[int, str, str, str]:
        gap = abs(rank_map.get(rel["sub"], 0) - rank_map.get(rel["obj"], 0))
        return (-gap, rel["sub"], rel["obj"], rel["rel"])

    best_rel = sorted(graph["relationship"], key=rel_key)[0]
    pair_set = {best_rel["sub"], best_rel["obj"]}
    pair_relations = [
        rel
        for rel in graph["relationship"]
        if {rel["sub"], rel["obj"]} == pair_set
    ]
    ordered_pair = [best_rel["sub"], best_rel["obj"]]
    return ordered_pair, pair_relations


def _build_focus_prompt_payload(
    canonical_name: str,
    entity_catalog_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    item = entity_catalog_map.get(canonical_name, {})
    return {
        "entity_id": item.get("entity_id", ""),
        "name": item.get("display_name", canonical_name),
    }


def _simplify_entities(
    entity_items: Sequence[Dict[str, Any]],
    entity_catalog_map: Dict[str, Dict[str, Any]],
    include_ids: bool,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for entity in entity_items:
        catalog_item = entity_catalog_map.get(entity["name"], {})
        payload = {
            "name": catalog_item.get("display_name", entity["name"]),
            "type": entity.get("type", "unknown"),
        }
        if include_ids and catalog_item.get("entity_id"):
            payload["entity_id"] = catalog_item["entity_id"]
        aliases = _dedupe_keep_order(catalog_item.get("alias", _clean_alias_list(entity.get("alias", []))))
        if aliases:
            payload["alias"] = aliases
        result.append(payload)
    return result


def _simplify_relations(
    relation_items: Sequence[Dict[str, Any]],
    entity_catalog_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "sub": entity_catalog_map.get(rel["sub"], {}).get("display_name", rel["sub"]),
            "rel": rel["rel"],
            "obj": entity_catalog_map.get(rel["obj"], {}).get("display_name", rel["obj"]),
        }
        for rel in relation_items
    ]


def _make_prompt_messages(
    task_type: str,
    template_bank: Dict[str, Any],
    article_text: str,
    graph: Dict[str, List[Dict[str, Any]]],
    entity_catalog: Sequence[Dict[str, Any]],
    focus_entity: Optional[str] = None,
    entity_pair: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    template = template_bank["task_templates"][task_type]
    rules = "\n".join(f"- {rule}" for rule in template.get("output_rules", []))
    entity_catalog_map = _build_entity_catalog_map(entity_catalog)

    extra_blocks: List[str] = []
    if task_type == TASK_TYPE_REL_ONLY:
        extra_blocks.append(
            f"Entity Inventory:\n{_format_json({'entity': _simplify_entities(graph['entity'], entity_catalog_map=entity_catalog_map, include_ids=True)})}"
        )
    if task_type == TASK_TYPE_FOCUS_SUBGRAPH and focus_entity:
        extra_blocks.append(f"Focus Entity:\n{_format_json(_build_focus_prompt_payload(focus_entity, entity_catalog_map))}")
    if task_type == TASK_TYPE_COREFERENCE and focus_entity:
        extra_blocks.append(f"Focus Entity:\n{_format_json(_build_focus_prompt_payload(focus_entity, entity_catalog_map))}")
    if task_type == TASK_TYPE_PAIR_REL and entity_pair:
        extra_blocks.append(
            f"Entity Pair:\n{_format_json([_build_focus_prompt_payload(name, entity_catalog_map) for name in entity_pair])}"
        )

    prompt_text = (
        f"{template['instruction']}\n\n"
        f"Rules:\n{rules}\n\n"
        f"Target JSON Schema:\n{SCHEMA_STRINGS[task_type]}\n\n"
    )
    if extra_blocks:
        prompt_text += f"{os.linesep.join(extra_blocks)}\n\n"
    prompt_text += f"Article:\n<<<ARTICLE>>>\n{article_text}\n<<<END_ARTICLE>>>\n\nReturn JSON only."
    return [{"role": "user", "content": prompt_text}]


def _build_task_rows_for_dataset(
    variant: str,
    sorted_df: pd.DataFrame,
    template_bank: Dict[str, Any],
    entity_name_mode: str,
    entity_name_model: str,
    entity_name_token: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(sorted_df.itertuples(index=False), start=1):
        info = _to_plain_obj(getattr(row, "extra_info"))
        info = info if isinstance(info, dict) else {}
        graph = _normalize_graph(info.get("graph_from_text_raw_from_file", {}))
        article_text = str(info.get("text_fixed_by_revision") or info.get("text_raw_from_file") or "")
        if not article_text or not graph["entity"]:
            continue

        parent_complexity = float((info.get("complexity_metrics") or {}).get("final_complexity_score", 0.0))
        parent_rank = int((info.get("complexity_metrics") or {}).get("curriculum_rank", idx))
        dataset_type = str(info.get("dataset_type") or "")
        entity_catalog = _build_entity_catalog(
            article_text=article_text,
            graph=graph,
            name_mode=entity_name_mode,
            name_model=entity_name_model,
            name_token=entity_name_token,
        )
        entity_catalog_map = _build_entity_catalog_map(entity_catalog)

        entity_lookup = {entity["name"]: entity for entity in graph["entity"]}

        candidate_tasks: List[Dict[str, Any]] = []
        candidate_tasks.append(
            {
                "task_type": TASK_TYPE_ENTITY_ONLY,
                "focus_entity": None,
                "entity_pair": None,
                "answer": {"entity": _simplify_entities(graph["entity"], entity_catalog_map=entity_catalog_map, include_ids=False)},
            }
        )

        if graph["relationship"]:
            candidate_tasks.append(
                {
                    "task_type": TASK_TYPE_REL_ONLY,
                    "focus_entity": None,
                    "entity_pair": None,
                    "answer": {"relationship": _simplify_relations(graph["relationship"], entity_catalog_map=entity_catalog_map)},
                }
            )

            focus_entity = _pick_focus_entity(graph)
            if focus_entity:
                focus_rels = [
                    rel for rel in graph["relationship"]
                    if rel["sub"] == focus_entity or rel["obj"] == focus_entity
                ]
                neighbor_names = {focus_entity}
                for rel in focus_rels:
                    neighbor_names.add(rel["sub"])
                    neighbor_names.add(rel["obj"])
                candidate_tasks.append(
                    {
                        "task_type": TASK_TYPE_FOCUS_SUBGRAPH,
                        "focus_entity": focus_entity,
                        "entity_pair": None,
                        "answer": {
                            "focus_entity": entity_catalog_map.get(focus_entity, {}).get("display_name", focus_entity),
                            "entity": _simplify_entities(
                                [entity_lookup[name] for name in neighbor_names if name in entity_lookup],
                                entity_catalog_map=entity_catalog_map,
                                include_ids=False,
                            ),
                            "relationship": _simplify_relations(focus_rels, entity_catalog_map=entity_catalog_map),
                        },
                    }
                )

            pair_payload = _pick_relation_pair(graph, article_text)
            if pair_payload is not None:
                entity_pair, pair_relations = pair_payload
                candidate_tasks.append(
                    {
                        "task_type": TASK_TYPE_PAIR_REL,
                        "focus_entity": None,
                        "entity_pair": entity_pair,
                        "answer": {
                            "entity_pair": [
                                entity_catalog_map.get(name, {}).get("display_name", name)
                                for name in entity_pair
                            ],
                            "relationship": _simplify_relations(pair_relations, entity_catalog_map=entity_catalog_map),
                        },
                    }
                )

        coref_entity = _pick_coref_entity(graph)
        if coref_entity:
            aliases = _clean_alias_list(entity_lookup[coref_entity].get("alias", []))
            candidate_tasks.append(
                {
                    "task_type": TASK_TYPE_COREFERENCE,
                    "focus_entity": coref_entity,
                    "entity_pair": None,
                    "answer": {
                        "focus_entity": entity_catalog_map.get(coref_entity, {}).get("display_name", coref_entity),
                        "coreference": aliases,
                    },
                }
            )

        for task in candidate_tasks:
            task_type = task["task_type"]
            task_spec = TASK_SPECS[task_type]
            answer_str = _format_json(task["answer"])
            prompt_messages = _make_prompt_messages(
                task_type=task_type,
                template_bank=template_bank,
                article_text=article_text,
                graph=graph,
                entity_catalog=entity_catalog,
                focus_entity=task["focus_entity"],
                entity_pair=task["entity_pair"],
            )
            task_complexity_score = min(
                1.0,
                0.80 * parent_complexity + 0.20 * float(task_spec["difficulty_offset"]),
            )

            extra_info = copy.deepcopy(info)
            extra_info["augmentation_task_type"] = task_type
            extra_info["augmentation_task_display_name"] = task_spec["display_name"]
            extra_info["augmentation_focus_entity"] = task["focus_entity"]
            extra_info["augmentation_focus_pair"] = task["entity_pair"]
            extra_info["augmentation_focus_entity_display"] = (
                entity_catalog_map.get(task["focus_entity"], {}).get("display_name")
                if task["focus_entity"] else None
            )
            extra_info["augmentation_focus_pair_display"] = (
                [entity_catalog_map.get(name, {}).get("display_name", name) for name in task["entity_pair"]]
                if task["entity_pair"] else None
            )
            extra_info["augmentation_template_source"] = template_bank.get("metadata", {}).get("source", "unknown")
            extra_info["augmentation_template_version"] = template_bank.get("metadata", {}).get("version", SCRIPT_VERSION)
            extra_info["augmentation_parent_variant"] = variant
            extra_info["augmentation_parent_complexity_score"] = parent_complexity
            extra_info["augmentation_complexity_score"] = task_complexity_score
            extra_info["augmentation_parent_rank"] = parent_rank
            extra_info["augmentation_entity_catalog"] = copy.deepcopy(entity_catalog)
            extra_info["augmentation_entity_name_mode"] = entity_name_mode
            extra_info["augmentation_reward_mode"] = "llm_judge"
            extra_info["augmentation_reward_model"] = "gemini-2.5-flash"
            extra_info["dataset_type"] = dataset_type

            rows.append(
                {
                    "prompt": prompt_messages,
                    "data_source": "grid_dataset_kg_augmented_sft",
                    "ability": task_spec["ability"],
                    "ground_truth": answer_str,
                    "extra_info": extra_info,
                    "reward_model": {
                        "style": "llm_judge_with_rule_fallback",
                        "model": "gemini-2.5-flash",
                        "ground_truth": answer_str,
                    },
                    "sft_ground_truth": answer_str,
                    "__task_type": task_type,
                    "__task_complexity_score": task_complexity_score,
                    "__parent_rank": parent_rank,
                }
            )

    aug_df = pd.DataFrame(rows)
    if aug_df.empty:
        return aug_df, pd.DataFrame()

    aug_df = aug_df.sort_values(
        by=["__task_complexity_score", "__parent_rank", "__task_type"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    updated_extras: List[Dict[str, Any]] = []
    total = len(aug_df)
    for idx, row in enumerate(aug_df.itertuples(index=False)):
        info = copy.deepcopy(_to_plain_obj(getattr(row, "extra_info")))
        info["index"] = idx
        info["totalnum"] = total
        updated_extras.append(info)
    aug_df["extra_info"] = updated_extras

    summary_df = (
        aug_df.groupby("__task_type")
        .size()
        .reset_index(name="count")
        .sort_values(by=["__task_type"])
        .reset_index(drop=True)
    )

    aug_df = aug_df[BASE_COLUMNS].copy()
    return aug_df, summary_df


def _write_complexity_markdown(output_path: str, per_variant_summaries: Dict[str, pd.DataFrame]) -> None:
    lines: List[str] = []
    lines.append("# GRID 复杂度排序说明")
    lines.append("")
    lines.append("本次排序分为两层：")
    lines.append("")
    lines.append("1. 基础复杂度分数（用户指定的 5 个维度）")
    lines.append("   - 文章 token 长度")
    lines.append("   - 实体数量")
    lines.append("   - 关系数量")
    lines.append("   - 每 1000 token 的实体密度")
    lines.append("   - 每 1000 token 的关系密度")
    lines.append("")
    lines.append("2. 图结构复杂度分数（额外加入）")
    lines.append("   - 平均度数")
    lines.append("   - 图密度")
    lines.append("   - 平均聚类系数")
    lines.append("   - treewidth(min-degree) 近似值")
    lines.append("   - max k-core")
    lines.append("   - 非 local-bridge 比例（越多三角/闭包，通常越交织）")
    lines.append("   - 平均关系跨度（按文章线性顺序）")
    lines.append("   - fixed linear crossing density（文章顺序下边交叉比例）")
    lines.append("   - 共指/别名密度（avg_alias_count, entity_with_alias_ratio）")
    lines.append("")
    lines.append("最终分数 = 0.65 * 基础复杂度 + 0.35 * 图结构复杂度。")
    lines.append("")
    lines.append("## 为什么要加 fixed linear crossing density")
    lines.append("")
    lines.append("文章是线性顺序；知识图谱把远距离实体跨位置连起来。")
    lines.append("如果文章实体顺序为 a b c d e f：")
    lines.append("- `a-b, c-d, e-f` 基本是局部相邻连接，较简单。")
    lines.append("- `a-c, b-d, e-f` 会出现更多跨越与交叉，模型需要做更远距离的依赖绑定，通常更难。")
    lines.append("")
    lines.append("## 参考互联网资料")
    lines.append("")
    for ref in COMPLEXITY_RESEARCH_REFERENCES:
        lines.append(f"- [{ref['name']}]({ref['url']}): {ref['summary']}")
    lines.append("")
    lines.append("## 各变体统计")
    lines.append("")
    for variant, df in per_variant_summaries.items():
        if df.empty:
            continue
        lines.append(f"### {variant}")
        lines.append("")
        lines.append(f"- 样本数: {len(df)}")
        lines.append(f"- easiest final_complexity_score: {df['final_complexity_score'].min():.4f}")
        lines.append(f"- hardest final_complexity_score: {df['final_complexity_score'].max():.4f}")
        lines.append(f"- 平均 token 长度: {df['article_token_length'].mean():.2f}")
        lines.append(f"- 平均实体数: {df['entity_count'].mean():.2f}")
        lines.append(f"- 平均关系数: {df['relation_count'].mean():.2f}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_augmentation_design_markdown(output_path: str, template_bank: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# GRID 新增强任务设计")
    lines.append("")
    lines.append(f"- 模板来源: {template_bank.get('metadata', {}).get('source', 'unknown')}")
    lines.append(f"- 模板版本: {template_bank.get('metadata', {}).get('version', SCRIPT_VERSION)}")
    lines.append("- 实体命名策略: 内部保留 canonical_name；给模型看的 prompt/题面优先使用 display_name；必要时用 `entity_id` 做歧义消解。")
    lines.append("- display_name 选择模式: `heuristic / llm / hybrid`，其中 `hybrid` 会在启发式结果明显不自然时调用 `gemini-2.5-flash`。")
    lines.append("- reward 策略: 默认 `LLM judge(gemini-2.5-flash)` 打分，规则版仅作为兜底与离线调试。")
    lines.append("")
    lines.append("## 任务类型")
    lines.append("")
    for task_type in [
        TASK_TYPE_ENTITY_ONLY,
        TASK_TYPE_COREFERENCE,
        TASK_TYPE_PAIR_REL,
        TASK_TYPE_FOCUS_SUBGRAPH,
        TASK_TYPE_REL_ONLY,
    ]:
        task = template_bank["task_templates"][task_type]
        judge = template_bank["judge_templates"][task_type]
        lines.append(f"### {task_type}")
        lines.append("")
        lines.append(f"- 中文名: {TASK_SPECS[task_type]['display_name']}")
        lines.append(f"- ability: `{TASK_SPECS[task_type]['ability']}`")
        lines.append(f"- 难度偏移: {TASK_SPECS[task_type]['difficulty_offset']}")
        lines.append(f"- 用户 Prompt 主指令: {task.get('instruction', '')}")
        lines.append("- 输出规则:")
        for rule in task.get("output_rules", []):
            lines.append(f"  - {rule}")
        lines.append(f"- JSON schema: `{SCHEMA_STRINGS[task_type]}`")
        lines.append(f"- Judge rubric: {judge.get('rubric', '')}")
        lines.append(f"- Reward 设计: {template_bank.get('reward_design', {}).get(task_type, '')}")
        lines.append("")
    lines.append("## 裁判与 Reward 代码")
    lines.append("")
    lines.append(f"- Python 模块: `{REWARD_MODULE_PATH}`")
    lines.append("- 核心接口: `compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float`")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _run_validation(
    aug_df: pd.DataFrame,
    output_dir: str,
    model: str,
    token: int,
) -> List[Dict[str, Any]]:
    if aug_df.empty:
        return []
    tools = _ensure_tools()
    reward_module = _load_reward_module()

    samples: List[pd.Series] = []
    task_order = [
        TASK_TYPE_ENTITY_ONLY,
        TASK_TYPE_COREFERENCE,
        TASK_TYPE_PAIR_REL,
        TASK_TYPE_FOCUS_SUBGRAPH,
        TASK_TYPE_REL_ONLY,
    ]
    task_series = aug_df["extra_info"].apply(lambda info: (info or {}).get("augmentation_task_type", ""))
    for task_type in task_order:
        subset = aug_df.loc[task_series == task_type]
        if subset.empty:
            continue
        samples.append(subset.iloc[min(len(subset) // 2, len(subset) - 1)])

    if not samples:
        return []

    print(f"🧪 使用 {model} 对 {len(samples)} 条增强样本做在线验证 ...")
    prompt_list = [sample["prompt"] for sample in samples]
    responses = tools.ask_group_link(
        prompt_list=prompt_list,
        model=model,
        token=token,
        temp=0.2,
        streamprint=False,
        check_history_cache=True,
        retry=True,
        force_api_do_huge_input_Cloud=True,
        flex=True,
        note="grid-aug-validation",
    )
    if hasattr(tools, "cleanthinkans"):
        responses = tools.cleanthinkans(responses)

    results: List[Dict[str, Any]] = []
    for sample, response in zip(samples, responses):
        info = _to_plain_obj(sample["extra_info"])
        task_type = info.get("augmentation_task_type", "")
        detail = reward_module.compute_detailed_score(
            data_source=str(sample["data_source"]),
            solution_str=response,
            ground_truth=sample["ground_truth"],
            extra_info=info,
        )
        reward_score = reward_module.compute_score(
            data_source=str(sample["data_source"]),
            solution_str=response,
            ground_truth=sample["ground_truth"],
            extra_info=info,
        )
        llm_judge_result = None
        if hasattr(reward_module, "compute_llm_judge_result"):
            try:
                llm_judge_result = reward_module.compute_llm_judge_result(
                    data_source=str(sample["data_source"]),
                    solution_str=response,
                    ground_truth=sample["ground_truth"],
                    extra_info=info,
                )
            except Exception as exc:
                llm_judge_result = {"error": str(exc)}
        judge_prompt = reward_module.build_llm_judge_prompt(
            task_type=task_type,
            article_text=str(info.get("text_fixed_by_revision") or info.get("text_raw_from_file") or ""),
            ground_truth=sample["ground_truth"],
            prediction=response,
            extra_info=info,
        )
        results.append(
            {
                "task_type": task_type,
                "score_detail": _to_plain_obj(detail),
                "reward_score": reward_score,
                "llm_judge_result": _to_plain_obj(llm_judge_result),
                "response": response,
                "ground_truth": sample["ground_truth"],
                "prompt": _to_plain_obj(sample["prompt"]),
                "judge_prompt": judge_prompt,
                "stable_article_id": info.get("stable_article_id", ""),
            }
        )

    with open(os.path.join(output_dir, "api_validation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def _write_output_index(
    output_root: str,
    variant_paths: Dict[str, str],
    complexity_dir: str,
    augmentation_dir: str,
    validation_dir: str,
) -> None:
    lines = [
        "# 输出索引",
        "",
        "## 原始输入 parquet",
        "",
    ]
    for variant, path in variant_paths.items():
        lines.append(f"- {variant}: `{path}`")
    lines.extend(
        [
            "",
            "## 新输出目录",
            "",
            f"- complexity: `{complexity_dir}`",
            f"- augmentation: `{augmentation_dir}`",
            f"- validation: `{validation_dir}`",
            "",
            "## 关键文件",
            "",
            "- `复杂度指标总表.csv`: 全变体复杂度扁平指标总表",
            "- `复杂度排序说明.md`: 复杂度定义与互联网参考",
            "- `增强任务设计.md`: 新 prompt / judge / reward 设计",
            "- `模板银行_merged.json`: gemini-2.5-flash + fallback 合并后的模板银行",
            "- `api_validation_results.json`: 少量在线验证样本与打分结果",
        ]
    )
    with open(os.path.join(output_root, "README_输出索引.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = _load_yaml(args.yaml)
    variant_paths = _discover_variant_paths(config)
    datasets = _load_datasets(variant_paths, max_rows=args.max_rows)
    tokenizer = _build_tokenizer(args.tokenizer)

    output_root = args.output_root
    complexity_dir = os.path.join(output_root, "complexity")
    augmentation_dir = os.path.join(output_root, "augmentation")
    validation_dir = os.path.join(output_root, "validation")
    os.makedirs(complexity_dir, exist_ok=True)
    os.makedirs(augmentation_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    if args.skip_llm_templates:
        print("⚙️ 跳过 gemini-2.5-flash 模板生成，使用 fallback 模板")
        template_bank = copy.deepcopy(FALLBACK_TEMPLATE_BANK)
    else:
        template_bank = _generate_template_bank_with_api_model(
            output_dir=augmentation_dir,
            model=args.template_model,
            token=max(16384, int(args.template_token)),
        )

    per_variant_metrics: Dict[str, pd.DataFrame] = {}
    all_metrics_frames: List[pd.DataFrame] = []
    sorted_datasets: Dict[str, pd.DataFrame] = {}

    for variant, df in datasets.items():
        print(f"🧮 计算复杂度: {variant}")
        metric_rows: List[Dict[str, Any]] = []
        for row_index, (_, row) in enumerate(df.iterrows()):
            if row_index > 0 and row_index % 200 == 0:
                print(f"   ⏳ {variant}: 已处理 {row_index}/{len(df)}")
            metric_rows.append(_compute_row_complexity(row_index=row_index, variant=variant, row=row, tokenizer=tokenizer))

        metrics_df = _attach_complexity_scores(pd.DataFrame(metric_rows))
        sorted_df, flat_metrics_df = _augment_extra_info_with_complexity(df=df, metrics_df=metrics_df, variant=variant)
        per_variant_metrics[variant] = flat_metrics_df
        all_metrics_frames.append(flat_metrics_df)
        sorted_datasets[variant] = sorted_df

        parquet_path = os.path.join(complexity_dir, f"{Path(variant_paths[variant]).stem}__curriculum_sorted.parquet")
        json_path = os.path.join(complexity_dir, f"{Path(variant_paths[variant]).stem}__curriculum_sorted.json")
        _save_dataframe(sorted_df, parquet_path, json_path)
        flat_metrics_df.to_csv(
            os.path.join(complexity_dir, f"{Path(variant_paths[variant]).stem}__complexity_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    all_metrics_df = pd.concat(all_metrics_frames, ignore_index=True) if all_metrics_frames else pd.DataFrame()
    if not all_metrics_df.empty:
        all_metrics_df.to_csv(
            os.path.join(complexity_dir, "复杂度指标总表.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    _write_complexity_markdown(
        output_path=os.path.join(complexity_dir, "复杂度排序说明.md"),
        per_variant_summaries=per_variant_metrics,
    )

    aug_summary_rows: List[Dict[str, Any]] = []
    validation_seed_df: Optional[pd.DataFrame] = None
    for variant, sorted_df in sorted_datasets.items():
        print(f"🧪 生成增强数据: {variant}")
        aug_df, summary_df = _build_task_rows_for_dataset(
            variant=variant,
            sorted_df=sorted_df,
            template_bank=template_bank,
            entity_name_mode=args.entity_name_mode,
            entity_name_model=args.entity_name_model,
            entity_name_token=max(4096, int(args.entity_name_token)),
        )
        if aug_df.empty:
            print(f"   ⚠️ {variant} 没有生成任何增强样本")
            continue

        parquet_path = os.path.join(augmentation_dir, f"{Path(variant_paths[variant]).stem}__augmentation.parquet")
        json_path = os.path.join(augmentation_dir, f"{Path(variant_paths[variant]).stem}__augmentation.json")
        _save_dataframe(aug_df, parquet_path, json_path)
        summary_df["variant"] = variant
        aug_summary_rows.extend(summary_df.to_dict("records"))
        if validation_seed_df is None and "train(full_nosplit)" in variant:
            validation_seed_df = aug_df

    if aug_summary_rows:
        pd.DataFrame(aug_summary_rows).to_csv(
            os.path.join(augmentation_dir, "增强任务汇总.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    _write_augmentation_design_markdown(
        output_path=os.path.join(augmentation_dir, "增强任务设计.md"),
        template_bank=template_bank,
    )

    if not args.skip_validation and validation_seed_df is not None:
        _run_validation(
            aug_df=validation_seed_df,
            output_dir=validation_dir,
            model=args.template_model,
            token=max(16384, int(args.template_token)),
        )
    else:
        print("⏭️ 跳过在线验证")

    _write_output_index(
        output_root=output_root,
        variant_paths=variant_paths,
        complexity_dir=complexity_dir,
        augmentation_dir=augmentation_dir,
        validation_dir=validation_dir,
    )

    print(f"🎉 全部完成，输出目录: {output_root}")


if __name__ == "__main__":
    main()
