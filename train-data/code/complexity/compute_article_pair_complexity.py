# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import multiprocessing
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


SCRIPT_VERSION = "20260321_v2"
DROPBOX_PATH = os.path.join(os.path.expanduser("~"), "Dropbox")
SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_SCRIPT_PATH = SCRIPT_DIR / "legacy_complexity_scoring.py"

if DROPBOX_PATH not in sys.path:
    sys.path.insert(0, DROPBOX_PATH)

_LEGACY_MODULE = None
_TOOLS_MODULE = None
_PAIR_WORKER_EMBEDDING_CACHE: Dict[str, Optional[List[float]]] = {}


@dataclass
class ArticleRecord:
    stable_article_id: str
    variant: str
    sample_order: int
    text: str
    graph_payload: Any
    graph: Dict[str, List[Dict[str, Any]]]


def _load_legacy_module():
    global _LEGACY_MODULE
    if _LEGACY_MODULE is not None:
        return _LEGACY_MODULE
    if not LEGACY_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"❌ 找不到旧版复杂度脚本: {LEGACY_SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("grid_legacy_complexity", str(LEGACY_SCRIPT_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"❌ 无法加载旧版复杂度脚本: {LEGACY_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LEGACY_MODULE = module
    return module


def _ensure_tools():
    global _TOOLS_MODULE
    if _TOOLS_MODULE is not None:
        return _TOOLS_MODULE
    import tools  # type: ignore

    _TOOLS_MODULE = tools
    return tools


def _norm_text(value: Any) -> str:
    legacy = _load_legacy_module()
    return legacy._norm_text(value)


def _clean_alias_list(value: Any) -> List[str]:
    legacy = _load_legacy_module()
    return legacy._clean_alias_list(value)


def _dedupe_keep_order(items: Sequence[Any]) -> List[str]:
    legacy = _load_legacy_module()
    return legacy._dedupe_keep_order(items)


def _normalize_graph(graph_payload: Any) -> Dict[str, List[Dict[str, Any]]]:
    legacy = _load_legacy_module()
    return legacy._normalize_graph(graph_payload)


def _build_tokenizer(tokenizer_name: str):
    legacy = _load_legacy_module()
    return legacy._build_tokenizer(tokenizer_name)


def _compute_order_metrics(text: str, graph: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    legacy = _load_legacy_module()
    return legacy._compute_order_metrics(text, graph)


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


def _safe_json_loads(value: Any) -> Any:
    legacy = _load_legacy_module()
    return legacy._safe_json_loads(value)


def _extract_article_record(record: Any, row_index: int, default_variant: str = "input") -> ArticleRecord:
    if isinstance(record, pd.Series):
        payload = record.to_dict()
    elif isinstance(record, dict):
        payload = dict(record)
    else:
        payload = dict(record)

    extra_info = _safe_json_loads(payload.get("extra_info", {}))
    extra_info = extra_info if isinstance(extra_info, dict) else {}

    text = str(
        payload.get("text_fixed_by_revision")
        or extra_info.get("text_fixed_by_revision")
        or payload.get("text_raw_from_file")
        or extra_info.get("text_raw_from_file")
        or payload.get("text")
        or ""
    )
    graph_payload = (
        payload.get("graph_from_text_raw_from_file")
        or extra_info.get("graph_from_text_raw_from_file")
        or payload.get("graph")
        or {}
    )
    stable_article_id = str(
        payload.get("stable_article_id")
        or extra_info.get("stable_article_id")
        or f"{default_variant}::row_{row_index}"
    )
    variant = str(payload.get("variant") or extra_info.get("variant") or default_variant)
    sample_order = int(payload.get("sample_order") or extra_info.get("sample_order") or row_index)
    graph = _normalize_graph(graph_payload)
    return ArticleRecord(
        stable_article_id=stable_article_id,
        variant=variant,
        sample_order=sample_order,
        text=text,
        graph_payload=graph_payload,
        graph=graph,
    )


def _build_article_metrics_dataframe(records: Sequence[ArticleRecord], tokenizer_name: str) -> pd.DataFrame:
    legacy = _load_legacy_module()
    tokenizer = _build_tokenizer(tokenizer_name)
    metric_rows: List[Dict[str, Any]] = []
    for row_index, item in enumerate(records):
        series = pd.Series(
            {
                "extra_info": {
                    "text_fixed_by_revision": item.text,
                    "graph_from_text_raw_from_file": item.graph_payload,
                    "stable_article_id": item.stable_article_id,
                    "sample_order": item.sample_order,
                }
            }
        )
        metric_rows.append(
            legacy._compute_row_complexity(
                row_index=row_index,
                variant=item.variant,
                row=series,
                tokenizer=tokenizer,
            )
        )
    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        return metrics_df
    return legacy._attach_complexity_scores(metrics_df)


def _split_sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    pattern = re.compile(r".+?(?:[.!?。！？]+(?:\s+|$)|\n+|$)", flags=re.S)
    for match in pattern.finditer(text):
        start, end = match.span()
        if text[start:end].strip():
            spans.append((start, end))
    if not spans and text:
        spans.append((0, len(text)))
    return spans


def _sentence_index_for_pos(sentence_spans: Sequence[Tuple[int, int]], pos: int) -> int:
    for idx, (start, end) in enumerate(sentence_spans):
        if start <= pos < end:
            return idx
    return max(0, len(sentence_spans) - 1)


def _make_exactish_pattern(candidate: str) -> re.Pattern[str]:
    escaped = re.escape(candidate)
    prefix = r"(?<!\w)" if candidate and candidate[0].isalnum() else ""
    suffix = r"(?!\w)" if candidate and candidate[-1].isalnum() else ""
    return re.compile(prefix + escaped + suffix, flags=re.IGNORECASE)


def _find_all_mentions(text: str, candidate_names: Sequence[str], sentence_spans: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    mentions: List[Dict[str, Any]] = []
    seen = set()
    for candidate in _dedupe_keep_order(candidate_names):
        cand = str(candidate or "").strip()
        if not cand:
            continue
        pattern = _make_exactish_pattern(cand)
        for match in pattern.finditer(text):
            start, end = match.span()
            key = (start, end, _norm_text(match.group(0)))
            if key in seen:
                continue
            seen.add(key)
            mentions.append(
                {
                    "text": match.group(0),
                    "start": start,
                    "end": end,
                    "sentence_idx": _sentence_index_for_pos(sentence_spans, start),
                }
            )
    mentions.sort(key=lambda item: (item["start"], item["end"], item["text"]))
    return mentions


def _average_vectors(vectors: Sequence[Sequence[float]]) -> Optional[List[float]]:
    usable = [list(v) for v in vectors if v is not None]
    if not usable:
        return None
    if np is not None:
        arr = np.asarray(usable, dtype=float)
        return arr.mean(axis=0).tolist()
    dim = len(usable[0])
    sums = [0.0] * dim
    for vec in usable:
        if len(vec) != dim:
            continue
        for idx, value in enumerate(vec):
            sums[idx] += float(value)
    return [value / len(usable) for value in sums]


def _cosine_similarity(vec_a: Optional[Sequence[float]], vec_b: Optional[Sequence[float]]) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    if np is not None:
        a = np.asarray(vec_a, dtype=float)
        b = np.asarray(vec_b, dtype=float)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for va, vb in zip(vec_a, vec_b):
        va = float(va)
        vb = float(vb)
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom <= 0.0:
        return 0.0
    return dot / denom


def _pairwise_dispersion(vectors: Sequence[Sequence[float]]) -> float:
    usable = [list(v) for v in vectors if v is not None]
    if len(usable) < 2:
        return 0.0
    sims: List[float] = []
    for idx in range(len(usable)):
        for jdx in range(idx + 1, len(usable)):
            sims.append(_cosine_similarity(usable[idx], usable[jdx]))
    if not sims:
        return 0.0
    return max(0.0, 1.0 - (sum(sims) / len(sims)))


class EmbeddingManager:

    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 0,
        max_processes: int = 256,
        db_path: str = "",
        feedback: bool = True,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        
        
        self.batch_size = int(batch_size)
        self.max_processes = max(1, int(max_processes))
        self.db_path = str(db_path or "").strip()
        self.feedback = feedback
        self._cache: Dict[str, Optional[List[float]]] = {}

    def prefetch(self, texts: Sequence[str]) -> None:
        if not self.enabled:
            return
        unique_texts = []
        seen = set()
        for text in texts:
            text = str(text or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            if text not in self._cache:
                unique_texts.append(text)
        if not unique_texts:
            return

        tools = _ensure_tools()
        if self.feedback:
            print(f"🧠 准备拉取 embedding: {len(unique_texts)} 个文本, model={self.model_name}")

        effective_batch_size = len(unique_texts) if self.batch_size <= 0 else self.batch_size
        for start in range(0, len(unique_texts), effective_batch_size):
            chunk = unique_texts[start : start + effective_batch_size]
            kwargs = {
                "model_name": self.model_name,
                "max_processes": min(self.max_processes, max(1, len(chunk))),
                "feedback": self.feedback,
            }
            if self.db_path:
                kwargs["db_path"] = self.db_path
            try:
                vectors = tools.embeddings(chunk, **kwargs)
            except Exception as exc:
                print(f"⚠️ embedding 调用失败，后续降级为纯结构版: {exc}")
                self.enabled = False
                return
            for text, vector in zip(chunk, vectors):
                self._cache[text] = list(vector) if vector is not None else None

    def get(self, text: str) -> Optional[List[float]]:
        text = str(text or "").strip()
        if not text:
            return None
        if text not in self._cache and self.enabled:
            self.prefetch([text])
        return self._cache.get(text)


class PreloadedEmbeddingLookup:

    def __init__(self, cache: Optional[Dict[str, Optional[List[float]]]] = None) -> None:
        self.enabled = bool(cache)
        self._cache = cache or {}

    def prefetch(self, texts: Sequence[str]) -> None:
        return None

    def get(self, text: str) -> Optional[List[float]]:
        text = str(text or "").strip()
        if not text:
            return None
        return self._cache.get(text)


def _init_pair_worker(embedding_cache: Optional[Dict[str, Optional[List[float]]]]) -> None:
    global _PAIR_WORKER_EMBEDDING_CACHE
    _PAIR_WORKER_EMBEDDING_CACHE = embedding_cache or {}


def _compute_entity_pair_complexity_worker(
    article_payload: Dict[str, Any],
    article_complexity_row: Optional[Dict[str, Any]],
    local_padding_chars: int,
    far_span_char_threshold: int,
    cosine_threshold: float,
) -> pd.DataFrame:
    article = ArticleRecord(
        stable_article_id=str(article_payload["stable_article_id"]),
        variant=str(article_payload["variant"]),
        sample_order=int(article_payload["sample_order"]),
        text=str(article_payload["text"]),
        graph_payload=article_payload["graph_payload"],
        graph=article_payload["graph"],
    )
    embedding_lookup = PreloadedEmbeddingLookup(_PAIR_WORKER_EMBEDDING_CACHE)
    return compute_entity_pair_complexity_for_article(
        article=article,
        article_complexity_row=article_complexity_row,
        embedding_manager=embedding_lookup,
        local_padding_chars=local_padding_chars,
        far_span_char_threshold=far_span_char_threshold,
        cosine_threshold=cosine_threshold,
    )


def _build_entity_profiles(
    article: ArticleRecord,
    sentence_spans: Sequence[Tuple[int, int]],
    embedding_manager: Optional[EmbeddingManager],
) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    candidate_texts_to_prefetch: List[str] = []

    for entity in article.graph["entity"]:
        name = entity["name"]
        all_names = _dedupe_keep_order([name] + _clean_alias_list(entity.get("alias", [])))
        profiles[name] = {
            "name": name,
            "type": str(entity.get("type") or "unknown"),
            "all_names": all_names,
            "alias_only": [item for item in all_names if _norm_text(item) != _norm_text(name)],
            "mentions": _find_all_mentions(article.text, all_names, sentence_spans),
        }
        candidate_texts_to_prefetch.extend(all_names)

    if embedding_manager is not None and embedding_manager.enabled:
        embedding_manager.prefetch(candidate_texts_to_prefetch)

    for name, profile in profiles.items():
        text_vectors = [
            embedding_manager.get(text_item) if embedding_manager is not None and embedding_manager.enabled else None
            for text_item in profile["all_names"]
        ]
        profile["name_vectors"] = text_vectors
        profile["entity_vector"] = _average_vectors([vec for vec in text_vectors if vec is not None])
        profile["alias_embedding_dispersion"] = _pairwise_dispersion([vec for vec in text_vectors if vec is not None])
        profile["alias_count"] = len(profile["alias_only"])

    return profiles


def _build_pair_relation_lookup(graph: Dict[str, List[Dict[str, Any]]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    lookup: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rel in graph["relationship"]:
        key = tuple(sorted((rel["sub"], rel["obj"])))
        lookup[key].append(rel)
    return lookup


def _build_graph_role_context(graph: Dict[str, List[Dict[str, Any]]], order_metrics: Dict[str, Any]) -> Dict[str, Any]:
    undirected_edges = {
        tuple(sorted((rel["sub"], rel["obj"])))
        for rel in graph["relationship"]
        if rel["sub"] and rel["obj"] and rel["sub"] != rel["obj"]
    }
    g = nx.Graph()
    g.add_nodes_from(entity["name"] for entity in graph["entity"])
    g.add_edges_from(undirected_edges)

    edge_betweenness = nx.edge_betweenness_centrality(g, normalized=True) if g.number_of_edges() > 0 else {}
    local_bridge_edges = set(nx.local_bridges(g, with_span=False)) if g.number_of_edges() > 0 else set()

    rank_map = order_metrics["entity_rank_map"]
    intervals: List[Tuple[Tuple[str, str], int, int]] = []
    for edge in undirected_edges:
        left = min(rank_map.get(edge[0], 0), rank_map.get(edge[1], 0))
        right = max(rank_map.get(edge[0], 0), rank_map.get(edge[1], 0))
        intervals.append((edge, left, right))

    crossing_counts: Dict[Tuple[str, str], int] = {edge: 0 for edge in undirected_edges}
    for idx in range(len(intervals)):
        edge_a, a, b = intervals[idx]
        for jdx in range(idx + 1, len(intervals)):
            edge_b, c, d = intervals[jdx]
            if len({edge_a[0], edge_a[1], edge_b[0], edge_b[1]}) < 4:
                continue
            if (a < c < b < d) or (c < a < d < b):
                crossing_counts[edge_a] += 1
                crossing_counts[edge_b] += 1

    return {
        "graph": g,
        "edge_betweenness": edge_betweenness,
        "local_bridge_edges": local_bridge_edges,
        "crossing_counts": crossing_counts,
        "undirected_edges": undirected_edges,
    }


def _merge_windows(windows: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    usable = [(max(0, int(start)), max(0, int(end))) for start, end in windows if end > start]
    if not usable:
        return []
    usable.sort(key=lambda item: (item[0], item[1]))
    merged = [usable[0]]
    for start, end in usable[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _find_case_insensitive(text: str, snippet: str, start_pos: int = 0) -> Optional[Tuple[int, int]]:
    snippet = str(snippet or "").strip()
    if not snippet:
        return None
    match = re.search(re.escape(snippet), text[start_pos:], flags=re.IGNORECASE)
    if not match:
        return None
    return start_pos + match.start(), start_pos + match.end()


def _locate_relation_windows(text: str, relations: Sequence[Dict[str, Any]], padding_chars: int) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    for rel in relations:
        raw_start = str(rel.get("raw_text_start") or "").strip()
        raw_end = str(rel.get("raw_text_end") or "").strip()
        located_start = _find_case_insensitive(text, raw_start) if raw_start else None
        if not located_start:
            continue
        start_pos = located_start[0]
        end_pos = located_start[1]
        if raw_end:
            located_end = _find_case_insensitive(text, raw_end, start_pos)
            if located_end:
                end_pos = located_end[1]
        windows.append((max(0, start_pos - padding_chars), min(len(text), end_pos + padding_chars)))
    return _merge_windows(windows)


def _build_fallback_pair_window(
    text_length: int,
    mentions_u: Sequence[Dict[str, Any]],
    mentions_v: Sequence[Dict[str, Any]],
    padding_chars: int,
) -> List[Tuple[int, int]]:
    if mentions_u and mentions_v:
        best_pair = None
        best_distance = None
        for mention_u in mentions_u:
            for mention_v in mentions_v:
                distance = abs(int(mention_u["start"]) - int(mention_v["start"]))
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_pair = (mention_u, mention_v)
        if best_pair is not None:
            start = min(int(best_pair[0]["start"]), int(best_pair[1]["start"]))
            end = max(int(best_pair[0]["end"]), int(best_pair[1]["end"]))
            return [(max(0, start - padding_chars), min(text_length, end + padding_chars))]
    return [(0, min(text_length, max(1, padding_chars * 2)))] if text_length > 0 else []


def _count_entities_in_windows(
    profiles: Dict[str, Dict[str, Any]],
    windows: Sequence[Tuple[int, int]],
) -> List[str]:
    names: List[str] = []
    for entity_name, profile in profiles.items():
        for mention in profile["mentions"]:
            mention_start = int(mention["start"])
            if any(start <= mention_start < end for start, end in windows):
                names.append(entity_name)
                break
    return sorted(set(names))


def _pair_distance_metrics(
    text: str,
    mentions_u: Sequence[Dict[str, Any]],
    mentions_v: Sequence[Dict[str, Any]],
    sentence_count: int,
    far_span_char_threshold: int,
) -> Dict[str, float]:
    text_len = max(1, len(text))
    if not mentions_u or not mentions_v:
        return {
            "min_span_chars": float(text_len),
            "mean_span_chars": float(text_len),
            "cross_sentence_ratio": 1.0,
            "far_span_ratio": 1.0,
        }

    span_values: List[int] = []
    cross_sentence_hits = 0
    far_hits = 0
    total_pairs = 0
    for mention_u in mentions_u:
        for mention_v in mentions_v:
            total_pairs += 1
            span = abs(int(mention_u["start"]) - int(mention_v["start"]))
            span_values.append(span)
            if int(mention_u["sentence_idx"]) != int(mention_v["sentence_idx"]):
                cross_sentence_hits += 1
            if span >= far_span_char_threshold:
                far_hits += 1

    return {
        "min_span_chars": float(min(span_values) if span_values else text_len),
        "mean_span_chars": float(sum(span_values) / len(span_values) if span_values else text_len),
        "cross_sentence_ratio": float(cross_sentence_hits / max(total_pairs, 1)),
        "far_span_ratio": float(far_hits / max(total_pairs, 1)),
    }


def _semantic_neighbor_overlap(
    entity_u: str,
    entity_v: str,
    graph_role_context: Dict[str, Any],
    profiles: Dict[str, Dict[str, Any]],
) -> float:
    graph = graph_role_context["graph"]
    neighbors_u = list(graph.neighbors(entity_u)) if entity_u in graph else []
    neighbors_v = list(graph.neighbors(entity_v)) if entity_v in graph else []
    centroid_u = _average_vectors(
        [profiles[name]["entity_vector"] for name in neighbors_u if profiles.get(name, {}).get("entity_vector") is not None]
    )
    centroid_v = _average_vectors(
        [profiles[name]["entity_vector"] for name in neighbors_v if profiles.get(name, {}).get("entity_vector") is not None]
    )
    return max(0.0, _cosine_similarity(centroid_u, centroid_v))


def _semantic_competitor_density(
    entity_u: str,
    entity_v: str,
    local_entities: Sequence[str],
    profiles: Dict[str, Dict[str, Any]],
    cosine_threshold: float,
) -> float:
    vec_u = profiles.get(entity_u, {}).get("entity_vector")
    vec_v = profiles.get(entity_v, {}).get("entity_vector")
    if vec_u is None and vec_v is None:
        return 0.0

    competitors = 0
    for candidate in local_entities:
        if candidate in {entity_u, entity_v}:
            continue
        candidate_vec = profiles.get(candidate, {}).get("entity_vector")
        if candidate_vec is None:
            continue
        sim = max(_cosine_similarity(candidate_vec, vec_u), _cosine_similarity(candidate_vec, vec_v))
        if sim >= cosine_threshold:
            competitors += 1
    return float(competitors)


def _percent_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(numeric) <= 1 or numeric.nunique(dropna=False) <= 1:
        return pd.Series([0.5] * len(numeric), index=numeric.index)
    return numeric.rank(method="average", pct=True, ascending=True)


def compute_entity_pair_complexity_for_article(
    article: ArticleRecord,
    article_complexity_row: Optional[Dict[str, Any]] = None,
    embedding_manager: Optional[EmbeddingManager] = None,
    local_padding_chars: int = 160,
    far_span_char_threshold: int = 400,
    cosine_threshold: float = 0.75,
) -> pd.DataFrame:
    
    graph = _normalize_graph(article.graph_payload if article.graph_payload is not None else article.graph)
    entity_names = [entity["name"] for entity in graph["entity"]]
    if len(entity_names) < 2:
        return pd.DataFrame()

    sentence_spans = _split_sentence_spans(article.text)
    profiles = _build_entity_profiles(article, sentence_spans, embedding_manager)
    order_metrics = _compute_order_metrics(article.text, graph)
    graph_role_context = _build_graph_role_context(graph, order_metrics)
    pair_relation_lookup = _build_pair_relation_lookup(graph)

    rows: List[Dict[str, Any]] = []
    for entity_u, entity_v in combinations(sorted(entity_names), 2):
        pair_key = tuple(sorted((entity_u, entity_v)))
        relations = pair_relation_lookup.get(pair_key, [])
        mentions_u = profiles[entity_u]["mentions"]
        mentions_v = profiles[entity_v]["mentions"]

        windows = _locate_relation_windows(article.text, relations, padding_chars=local_padding_chars)
        if not windows:
            windows = _build_fallback_pair_window(
                text_length=len(article.text),
                mentions_u=mentions_u,
                mentions_v=mentions_v,
                padding_chars=local_padding_chars,
            )
        local_entities = _count_entities_in_windows(profiles, windows)

        graph_obj = graph_role_context["graph"]
        neighbors_u = set(graph_obj.neighbors(entity_u)) if entity_u in graph_obj else set()
        neighbors_v = set(graph_obj.neighbors(entity_v)) if entity_v in graph_obj else set()
        union_neighbors = neighbors_u | neighbors_v
        common_neighbors = neighbors_u & neighbors_v

        distance_metrics = _pair_distance_metrics(
            text=article.text,
            mentions_u=mentions_u,
            mentions_v=mentions_v,
            sentence_count=len(sentence_spans),
            far_span_char_threshold=far_span_char_threshold,
        )

        edge_betweenness = float(
            graph_role_context["edge_betweenness"].get(pair_key, 0.0)
            or graph_role_context["edge_betweenness"].get((pair_key[1], pair_key[0]), 0.0)
        )
        local_bridge_score = 1.0 if pair_key in graph_role_context["local_bridge_edges"] or (pair_key[1], pair_key[0]) in graph_role_context["local_bridge_edges"] else 0.0

        row = {
            "stable_article_id": article.stable_article_id,
            "variant": article.variant,
            "entity_u": entity_u,
            "entity_v": entity_v,
            "pair_key": f"{entity_u} || {entity_v}",
            "is_positive_pair": bool(relations),
            "pair_relation_count": int(len(relations)),
            "common_neighbor_count": float(len(common_neighbors)),
            "neighbor_jaccard": float(len(common_neighbors) / max(len(union_neighbors), 1)),
            "semantic_neighbor_overlap": float(_semantic_neighbor_overlap(entity_u, entity_v, graph_role_context, profiles)),
            "alias_count_sum": float(profiles[entity_u]["alias_count"] + profiles[entity_v]["alias_count"]),
            "alias_embedding_dispersion_mean": float(
                (profiles[entity_u]["alias_embedding_dispersion"] + profiles[entity_v]["alias_embedding_dispersion"]) / 2.0
            ),
            "min_span_chars": distance_metrics["min_span_chars"],
            "mean_span_chars": distance_metrics["mean_span_chars"],
            "cross_sentence_ratio": distance_metrics["cross_sentence_ratio"],
            "far_span_ratio": distance_metrics["far_span_ratio"],
            "local_entity_density": float(len(local_entities)),
            "local_candidate_pair_count": float(len(local_entities) * (len(local_entities) - 1) / 2.0),
            "local_semantic_competitor_density": float(
                _semantic_competitor_density(entity_u, entity_v, local_entities, profiles, cosine_threshold=cosine_threshold)
            ),
            "edge_betweenness": edge_betweenness,
            "local_bridge_score": float(local_bridge_score),
            "pair_crossing_participation": float(graph_role_context["crossing_counts"].get(pair_key, 0.0)),
            "window_count": int(len(windows)),
            "window_char_coverage": float(sum(end - start for start, end in windows)),
        }
        if article_complexity_row:
            row["article_final_complexity_score"] = float(article_complexity_row.get("final_complexity_score", 0.0))
            row["article_curriculum_rank"] = int(article_complexity_row.get("curriculum_rank", 0))
        rows.append(row)

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df

    rank_columns = [
        "common_neighbor_count",
        "neighbor_jaccard",
        "semantic_neighbor_overlap",
        "alias_count_sum",
        "alias_embedding_dispersion_mean",
        "min_span_chars",
        "mean_span_chars",
        "cross_sentence_ratio",
        "far_span_ratio",
        "local_entity_density",
        "local_candidate_pair_count",
        "local_semantic_competitor_density",
        "pair_relation_count",
        "edge_betweenness",
        "local_bridge_score",
        "pair_crossing_participation",
    ]
    for column in rank_columns:
        pair_df[f"rank__{column}"] = _percent_rank(pair_df[column])

    
    pair_df["S_share"] = pair_df[
        [
            "rank__common_neighbor_count",
            "rank__neighbor_jaccard",
            "rank__semantic_neighbor_overlap",
        ]
    ].mean(axis=1)
    pair_df["S_coref"] = pair_df[
        [
            "rank__alias_count_sum",
            "rank__alias_embedding_dispersion_mean",
        ]
    ].mean(axis=1)
    pair_df["S_dist"] = pair_df[
        [
            "rank__min_span_chars",
            "rank__mean_span_chars",
            "rank__cross_sentence_ratio",
            "rank__far_span_ratio",
        ]
    ].mean(axis=1)
    pair_df["S_context"] = pair_df[
        [
            "rank__local_entity_density",
            "rank__local_candidate_pair_count",
            "rank__local_semantic_competitor_density",
        ]
    ].mean(axis=1)
    pair_df["S_role"] = pair_df[
        [
            "rank__pair_relation_count",
            "rank__edge_betweenness",
            "rank__local_bridge_score",
            "rank__pair_crossing_participation",
        ]
    ].mean(axis=1)
    pair_df["final_pair_complexity_score"] = pair_df[
        ["S_share", "S_coref", "S_dist", "S_context", "S_role"]
    ].mean(axis=1)
    pair_df = pair_df.sort_values(
        by=["stable_article_id", "final_pair_complexity_score", "entity_u", "entity_v"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    pair_df["pair_rank_within_article"] = pair_df.groupby("stable_article_id")["final_pair_complexity_score"].rank(
        method="dense", ascending=False
    )
    return pair_df


def compute_article_and_entity_pair_complexity(
    records: Sequence[ArticleRecord],
    tokenizer_name: str = "cl100k_base",
    embedding_enabled: bool = True,
    embedding_model: str = "text-embedding-3-small",
    embedding_batch_size: int = 0,
    embedding_max_processes: int = 256,
    embedding_db_path: str = "",
    embedding_feedback: bool = False,
    article_workers: int = 64,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    article_df = _build_article_metrics_dataframe(records, tokenizer_name=tokenizer_name)
    article_lookup = {
        str(row["stable_article_id"]): row
        for row in article_df.to_dict("records")
    }

    embedding_manager = EmbeddingManager(
        enabled=embedding_enabled,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        max_processes=embedding_max_processes,
        db_path=embedding_db_path,
        feedback=embedding_feedback,
    )
    embedding_cache: Dict[str, Optional[List[float]]] = {}
    if embedding_enabled:
        prefetch_texts: List[str] = []
        for article in records:
            graph = _normalize_graph(article.graph_payload if article.graph_payload is not None else article.graph)
            for entity in graph["entity"]:
                prefetch_texts.extend(_dedupe_keep_order([entity["name"]] + _clean_alias_list(entity.get("alias", []))))
        prefetch_texts = sorted(set(text for text in prefetch_texts if str(text or "").strip()))
        print(f"🧠 批量预取 embedding: unique_texts={len(prefetch_texts)}")
        embedding_manager.prefetch(prefetch_texts)
        embedding_cache = dict(embedding_manager._cache)

    pair_tables: List[pd.DataFrame] = []
    worker_count = max(1, min(int(article_workers), max(1, len(records))))
    if worker_count == 1:
        for article in records:
            pair_df = compute_entity_pair_complexity_for_article(
                article=article,
                article_complexity_row=article_lookup.get(article.stable_article_id),
                embedding_manager=embedding_manager,
            )
            if not pair_df.empty:
                pair_tables.append(pair_df)
    else:
        print(f"⚙️ 多进程处理文章级实体对复杂度: workers={worker_count}")
        mp_context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_context,
            initializer=_init_pair_worker,
            initargs=(embedding_cache,),
        ) as executor:
            futures = []
            for article in records:
                futures.append(
                    executor.submit(
                        _compute_entity_pair_complexity_worker,
                        {
                            "stable_article_id": article.stable_article_id,
                            "variant": article.variant,
                            "sample_order": article.sample_order,
                            "text": article.text,
                            "graph_payload": article.graph_payload,
                            "graph": _normalize_graph(article.graph_payload if article.graph_payload is not None else article.graph),
                        },
                        article_lookup.get(article.stable_article_id),
                        160,
                        400,
                        0.75,
                    )
                )
            for future in futures:
                pair_df = future.result()
                if not pair_df.empty:
                    pair_tables.append(pair_df)

    all_pair_df = pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()
    return article_df, all_pair_df


def _load_records_from_parquet(parquet_path: str, max_rows: int = 0) -> Tuple[pd.DataFrame, List[ArticleRecord]]:
    print(f"📦 读取 parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if max_rows > 0:
        df = df.head(max_rows).copy()
    records = [_extract_article_record(row, idx, default_variant=Path(parquet_path).stem) for idx, row in enumerate(df.to_dict("records"))]
    print(f"✅ 成功解析文章数: {len(records)}")
    return df, records


def _load_single_article_record(
    article_text_file: str,
    graph_file: str,
    stable_article_id: str,
) -> Tuple[pd.DataFrame, List[ArticleRecord]]:
    with open(article_text_file, "r", encoding="utf-8") as f:
        text = f.read()
    with open(graph_file, "r", encoding="utf-8") as f:
        graph_payload = f.read()
    single_df = pd.DataFrame(
        [
            {
                "prompt": [],
                "data_source": "single_input",
                "ability": "",
                "ground_truth": "",
                "extra_info": {
                    "stable_article_id": stable_article_id,
                    "text_fixed_by_revision": text,
                    "graph_from_text_raw_from_file": graph_payload,
                },
                "reward_model": {},
                "sft_ground_truth": "",
            }
        ]
    )
    record = ArticleRecord(
        stable_article_id=stable_article_id,
        variant="single_input",
        sample_order=0,
        text=text,
        graph_payload=graph_payload,
        graph=_normalize_graph(graph_payload),
    )
    return single_df, [record]


def _compact_article_complexity_payload(article_row: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
        "stable_article_id",
        "variant",
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
        "max_relation_rank_gap",
        "fixed_linear_crossing_count",
        "fixed_linear_crossing_density",
        "base_complexity_score",
        "graph_structure_complexity_score",
        "final_complexity_score",
        "curriculum_rank",
        "curriculum_percent",
        "curriculum_decile",
    ]
    return {key: _to_plain_obj(article_row.get(key)) for key in keep_keys if key in article_row}


def _compact_pair_complexity_payload(pair_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    compact_pairs: List[Dict[str, Any]] = []
    scores: List[float] = []
    for row in pair_rows:
        score = float(row.get("final_pair_complexity_score", 0.0))
        scores.append(score)
        compact_pairs.append(
            {
                "entity_u": row.get("entity_u", ""),
                "entity_v": row.get("entity_v", ""),
                "is_positive_pair": bool(row.get("is_positive_pair", False)),
                "pair_relation_count": int(row.get("pair_relation_count", 0)),
                "S_share": float(row.get("S_share", 0.0)),
                "S_coref": float(row.get("S_coref", 0.0)),
                "S_dist": float(row.get("S_dist", 0.0)),
                "S_context": float(row.get("S_context", 0.0)),
                "S_role": float(row.get("S_role", 0.0)),
                "实体对复杂度": score,
                "pair_rank_within_article": int(float(row.get("pair_rank_within_article", 0))),
            }
        )
    return {
        "实体对数量": int(len(compact_pairs)),
        "平均实体对复杂度": float(sum(scores) / len(scores)) if scores else 0.0,
        "最高实体对复杂度": float(max(scores)) if scores else 0.0,
        "实体对复杂度列表": compact_pairs,
    }


def _build_augmented_output_dataframe(
    original_df: pd.DataFrame,
    article_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    article_lookup = {
        str(row["stable_article_id"]): _compact_article_complexity_payload(row)
        for row in article_df.to_dict("records")
    }
    pair_lookup: Dict[str, Dict[str, Any]] = {}
    if not pair_df.empty:
        grouped = pair_df.groupby("stable_article_id", sort=False)
        for stable_article_id, group_df in grouped:
            pair_lookup[str(stable_article_id)] = _compact_pair_complexity_payload(group_df.to_dict("records"))

    output_df = original_df.copy()
    updated_extra_infos: List[Dict[str, Any]] = []
    for row_index, row in enumerate(output_df.to_dict("records")):
        extra_info = _safe_json_loads(row.get("extra_info", {}))
        extra_info = copy.deepcopy(extra_info if isinstance(extra_info, dict) else {})
        stable_article_id = str(row.get("stable_article_id") or extra_info.get("stable_article_id") or f"row_{row_index}")
        extra_info["文章复杂度"] = article_lookup.get(stable_article_id, {})
        extra_info["KG复杂度"] = pair_lookup.get(
            stable_article_id,
            {"实体对数量": 0, "平均实体对复杂度": 0.0, "最高实体对复杂度": 0.0, "实体对复杂度列表": []},
        )
        updated_extra_infos.append(extra_info)
    output_df["extra_info"] = updated_extra_infos
    return output_df


def _save_augmented_parquet(df: pd.DataFrame, parquet_path: str) -> None:
    df_to_save = df.copy()
    for column in df_to_save.columns:
        df_to_save[column] = df_to_save[column].apply(_to_plain_obj)
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df_to_save.to_parquet(parquet_path, index=False)
    print(f"✅ 保存增强 parquet: {parquet_path}")


def _render_complexity_formula_png(output_path: str) -> str:
    
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import subprocess

    font_props = None
    try:
        font_probe = subprocess.run(
            ["fc-match", "-f", "%{file}\n", "Noto Sans CJK SC"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        font_path = str(font_probe.stdout or "").strip().splitlines()[0]
        if font_path and Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            font_props = font_manager.FontProperties(fname=font_path)
    except Exception:
        font_props = None

    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    article_formula = (
        r"$C_{\mathrm{article}}(a)=0.65\cdot\frac{1}{5}\sum_{m\in\mathcal{M}_{\mathrm{base}}} r_m(a)"
        r"+0.35\cdot\frac{1}{10}\sum_{m\in\mathcal{M}_{\mathrm{graph}}} r_m(a)$"
    )
    edge_formula = (
        r"$C_{\mathrm{edge}}(u,v)=\frac{1}{5}\left("
        r"S_{\mathrm{share}}+S_{\mathrm{coref}}+S_{\mathrm{dist}}+S_{\mathrm{context}}+S_{\mathrm{role}}"
        r"\right)$"
    )

    article_box = "\n".join(
        [
            "Article-level score for article a.",
            "r_m(a): percentile rank of metric m across the article set.",
            "M_base = {L_tok, |V|, |R|, 1000|V|/L_tok, 1000|R|/L_tok}.",
            "M_graph = {A_bar, rho_alias, d_bar, delta, c_bar, tw, k_max, 1-rho_bridge, Delta_bar_rank, rho_cross}.",
            "L_tok: article token count; |V| / |R|: entity / relation count.",
            "A_bar: average alias count; rho_alias: ratio of entities with aliases.",
            "d_bar: average degree; delta: graph density; c_bar: average clustering.",
            "tw: min-degree treewidth; k_max: maximum k-core index.",
            "1-rho_bridge: non-local-bridge ratio.",
            "Delta_bar_rank: mean subject-object rank gap in article order.",
            "rho_cross: fixed-order edge-crossing density.",
        ]
    )
    edge_box = "\n".join(
        [
            "Entity-pair score for candidate edge (u, v).",
            "All terms are percentile ranks computed among candidate pairs within the same article.",
            "S_share = mean(common-neighbor count, neighbor Jaccard, semantic neighbor overlap).",
            "S_coref = mean(alias-count sum, alias-embedding dispersion).",
            "S_dist = mean(min-span, mean-span, cross-sentence ratio, far-span ratio).",
            "S_context = mean(local entity density, local candidate-pair count, semantic competitor density).",
            "S_role = mean(pair relation count, edge betweenness, local bridge score, crossing participation).",
        ]
    )

    fig = plt.figure(figsize=(18, 11), dpi=240, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    fig.text(
        0.5,
        0.95,
        "GRID Complexity Formulas",
        ha="center",
        va="top",
        fontsize=28,
        fontweight="bold",
        color="#0F172A",
        fontproperties=font_props,
    )

    fig.text(0.5, 0.845, article_formula, ha="center", va="center", fontsize=28, color="#111827")
    fig.text(
        0.07,
        0.685,
        article_box,
        ha="left",
        va="top",
        fontsize=14,
        linespacing=1.45,
        color="#1F2937",
        fontproperties=font_props,
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "#F8FAFC",
            "edgecolor": "#CBD5E1",
            "linewidth": 1.6,
        },
    )

    fig.text(0.5, 0.39, edge_formula, ha="center", va="center", fontsize=28, color="#111827")
    fig.text(
        0.07,
        0.26,
        edge_box,
        ha="left",
        va="top",
        fontsize=14,
        linespacing=1.45,
        color="#1F2937",
        fontproperties=font_props,
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "#F8FAFC",
            "edgecolor": "#CBD5E1",
            "linewidth": 1.6,
        },
    )

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"🖼️ 保存复杂度公式 PNG: {output}")
    return str(output)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="🧮 计算 GRID 文章复杂度与实体对复杂贡献度")
    parser.add_argument("--input_parquet", type=str, default="", help="输入 parquet；优先从 extra_info 中读取 text_fixed_by_revision / graph_from_text_raw_from_file")
    parser.add_argument("--article_text_file", type=str, default="", help="单篇文章文本文件")
    parser.add_argument("--graph_file", type=str, default="", help="单篇文章对应的 KG JSON 文件")
    parser.add_argument("--stable_article_id", type=str, default="single_article", help="单篇模式下的 stable_article_id")
    parser.add_argument("--output_parquet", type=str, default="", help="输出 parquet 路径；建议文件名包含“临时用未来删除”")
    parser.add_argument("--formula_png", type=str, default="", help="可选：把英文论文风格的文章/edge 复杂度公式渲染为 PNG")
    parser.add_argument("--formula_only", action="store_true", help="只渲染复杂度公式 PNG，不执行 parquet / 单篇复杂度计算")
    parser.add_argument("--tokenizer", type=str, default="cl100k_base", help="文章复杂度沿用旧版脚本时使用的 tokenizer")
    parser.add_argument("--max_rows", type=int, default=0, help="只读取前 X 篇文章；0 表示全量")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small", help="tools.py embeddings() 使用的模型名")
    parser.add_argument("--embedding_batch_size", type=int, default=0, help="embedding 预取分块大小；<=0 表示把当前批次文本尽量合并成一次调用")
    parser.add_argument("--embedding_max_processes", type=int, default=256, help="传给 tools.py embeddings() 的最大并行进程数")
    parser.add_argument("--embedding_db_path", type=str, default="", help="embedding 缓存 DB；留空则沿用 tools.py 默认值")
    parser.add_argument("--disable_embeddings", action="store_true", help="关闭 embedding 增强，仅保留纯结构版 S_share / S_coref / S_context")
    parser.add_argument("--embedding_feedback", action="store_true", help="打印 tools.py embedding 的详细进度")
    parser.add_argument("--article_workers", type=int, default=max(1, min(64, os.cpu_count() or 1)), help="并行处理文章的 worker 数")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.formula_only:
        formula_png = args.formula_png or str(SCRIPT_DIR / "给文章和edge复杂度公式__English.png")
        formula_path = _render_complexity_formula_png(formula_png)
        print(json.dumps({"script_version": SCRIPT_VERSION, "formula_png": formula_path}, ensure_ascii=False, indent=2))
        return

    if args.input_parquet:
        original_df, records = _load_records_from_parquet(args.input_parquet, max_rows=args.max_rows)
    else:
        if not args.article_text_file or not args.graph_file:
            parser.error("❌ 请提供 --input_parquet，或同时提供 --article_text_file 与 --graph_file")
        original_df, records = _load_single_article_record(
            article_text_file=args.article_text_file,
            graph_file=args.graph_file,
            stable_article_id=args.stable_article_id,
        )

    if len(records) == 1:
        print("⚠️ 当前只输入了 1 篇文章；文章复杂度虽然仍沿用旧版 rank 逻辑，但 rank-based 分数会退化为单样本情形。")

    article_df, pair_df = compute_article_and_entity_pair_complexity(
        records=records,
        tokenizer_name=args.tokenizer,
        embedding_enabled=not args.disable_embeddings,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_processes=args.embedding_max_processes,
        embedding_db_path=args.embedding_db_path,
        embedding_feedback=args.embedding_feedback,
        article_workers=args.article_workers,
    )

    output_df = _build_augmented_output_dataframe(original_df=original_df, article_df=article_df, pair_df=pair_df)

    if args.output_parquet:
        output_parquet = args.output_parquet
    elif args.input_parquet:
        input_path = Path(args.input_parquet)
        output_parquet = str(input_path.with_name(f"{input_path.stem}__临时用未来删除__文章KG复杂度.parquet"))
    else:
        output_parquet = str(SCRIPT_DIR / "temp_文章与实体对复杂度__临时用未来删除__.parquet")

    _save_augmented_parquet(output_df, output_parquet)

    summary = {
        "script_version": SCRIPT_VERSION,
        "article_count": int(len(output_df)),
        "pair_count_total": int(len(pair_df)),
        "embedding_enabled": bool(not args.disable_embeddings),
        "embedding_model": args.embedding_model,
        "article_workers": int(args.article_workers),
        "output_parquet": output_parquet,
        "legacy_article_complexity_path": str(LEGACY_SCRIPT_PATH),
    }
    with open(Path(output_parquet).with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.formula_png:
        _render_complexity_formula_png(args.formula_png)

    print("🎉 复杂度计算完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
