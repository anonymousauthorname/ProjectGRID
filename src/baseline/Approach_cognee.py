# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from baseline_shared_utils import (
    SharedPromptBaseline,
    load_text,
    normalize_relations,
    parse_json_payload,
    render_jinja_like_template,
    sentence_chunk_text,
)

try:
    from article_io_cache_parser import KGCacheManager
except ImportError:
    KGCacheManager = None


REPO_ROOT = Path(__file__).resolve().parents[2]
COGNEE_PROMPT_DIR = REPO_ROOT / "baseline" / "resources" / "cognee_prompts"


class CogneeMethod(SharedPromptBaseline):

    def __init__(
        self,
        model: str = "gpt-5-nano",
        token: int = 8192,
        temp: float = 0.7,
        chunk_target_chars: int = 10000,
        chunk_overlap_chars: int = 800,
        n_rounds: int = 2,
        **kwargs: Any,
    ):
        super().__init__(
            name="Cognee",
            model=model,
            token=token,
            temp=temp,
            **kwargs,
        )
        self.chunk_target_chars = max(2500, int(chunk_target_chars or 10000))
        self.chunk_overlap_chars = max(0, int(chunk_overlap_chars or 0))
        self.n_rounds = max(1, int(n_rounds or 1))
        self.cache = (
            KGCacheManager(os.path.basename(__file__), method_file_path=os.path.abspath(__file__))
            if KGCacheManager
            else None
        )
        self.node_system_prompt = load_text(COGNEE_PROMPT_DIR / "extract_graph_nodes_prompt_system.txt").strip()
        self.node_input_template = load_text(COGNEE_PROMPT_DIR / "extract_graph_nodes_prompt_input.txt").strip()
        self.rel_system_prompt = load_text(COGNEE_PROMPT_DIR / "extract_graph_relationship_names_prompt_system.txt").strip()
        self.rel_input_template = load_text(COGNEE_PROMPT_DIR / "extract_graph_relationship_names_prompt_input.txt").strip()
        self.edge_system_prompt = load_text(COGNEE_PROMPT_DIR / "extract_graph_edge_triplets_prompt_system.txt").strip()
        self.edge_input_template = load_text(COGNEE_PROMPT_DIR / "extract_graph_edge_triplets_prompt_input.txt").strip()

    def _call_json_stage(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        sample_content: str,
        stage: str,
    ) -> Any:
        raw = self._call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            sample_content=sample_content,
            stage=stage,
        )
        return parse_json_payload(raw)

    def _extract_nodes(self, chunk_text: str, sample_content: str, chunk_idx: int) -> List[str]:
        all_nodes: List[str] = []
        seen = set()
        for round_idx in range(1, self.n_rounds + 1):
            print(f"    🔹 [Cognee] chunk {chunk_idx} node round {round_idx}/{self.n_rounds}")
            context = {
                "text": chunk_text,
                "previous_entities": json.dumps(all_nodes, ensure_ascii=False),
                "previous_nodes": json.dumps(all_nodes, ensure_ascii=False),
                "round_number": round_idx,
                "total_rounds": self.n_rounds,
            }
            user_prompt = render_jinja_like_template(self.node_input_template, context)
            user_prompt += (
                "\n\nReturn JSON only in this exact schema:\n"
                '{"nodes": ["entity_a", "entity_b"]}'
            )
            payload = self._call_json_stage(
                system_prompt=self.node_system_prompt,
                user_prompt=user_prompt,
                sample_content=sample_content,
                stage=f"cognee_nodes_chunk_{chunk_idx}_round_{round_idx}",
            )
            items = payload.get("nodes") if isinstance(payload, dict) else payload
            for node in items or []:
                name = str(node or "").strip()
                if not name:
                    continue
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                all_nodes.append(name)
        return all_nodes

    def _extract_relationship_names(
        self,
        *,
        chunk_text: str,
        candidate_nodes: List[str],
        sample_content: str,
        chunk_idx: int,
    ) -> Tuple[List[str], List[str]]:
        all_nodes = list(candidate_nodes)
        node_seen = {node.lower() for node in all_nodes}
        all_relationship_names: List[str] = []
        rel_seen = set()

        for round_idx in range(1, self.n_rounds + 1):
            print(f"    🔹 [Cognee] chunk {chunk_idx} relation-name round {round_idx}/{self.n_rounds}")
            context = {
                "text": chunk_text,
                "potential_nodes": json.dumps(candidate_nodes, ensure_ascii=False),
                "previous_nodes": json.dumps(all_nodes, ensure_ascii=False),
                "previous_relationship_names": json.dumps(all_relationship_names, ensure_ascii=False),
                "round_number": round_idx,
                "total_rounds": self.n_rounds,
            }
            user_prompt = render_jinja_like_template(self.rel_input_template, context)
            user_prompt += (
                "\n\nReturn JSON only in this exact schema:\n"
                '{"nodes": ["entity_a"], "relationship_names": ["uses", "targets"]}'
            )
            payload = self._call_json_stage(
                system_prompt=self.rel_system_prompt,
                user_prompt=user_prompt,
                sample_content=sample_content,
                stage=f"cognee_relationship_names_chunk_{chunk_idx}_round_{round_idx}",
            )
            if not isinstance(payload, dict):
                continue
            for node in payload.get("nodes", []) or []:
                name = str(node or "").strip()
                if not name:
                    continue
                key = name.lower()
                if key in node_seen:
                    continue
                node_seen.add(key)
                all_nodes.append(name)
            for rel in payload.get("relationship_names", []) or []:
                rel_name = str(rel or "").strip()
                if not rel_name:
                    continue
                key = rel_name.lower()
                if key in rel_seen:
                    continue
                rel_seen.add(key)
                all_relationship_names.append(rel_name)

        return all_nodes, all_relationship_names

    def _extract_edges(
        self,
        *,
        chunk_text: str,
        nodes: List[str],
        relationship_names: List[str],
        sample_content: str,
        chunk_idx: int,
    ) -> List[Dict[str, str]]:
        final_relations: List[Dict[str, str]] = []
        previous_triplets: List[Dict[str, str]] = []
        node_name_to_id: Dict[str, str] = {}

        for round_idx in range(1, self.n_rounds + 1):
            print(f"    🔹 [Cognee] chunk {chunk_idx} edge round {round_idx}/{self.n_rounds}")
            context = {
                "text": chunk_text,
                "potential_nodes": json.dumps(nodes, ensure_ascii=False),
                "potential_relationship_names": json.dumps(relationship_names, ensure_ascii=False),
                "previous_nodes": json.dumps(
                    [{"id": node_name_to_id[name], "name": name} for name in node_name_to_id],
                    ensure_ascii=False,
                ),
                "previous_edge_triplets": json.dumps(previous_triplets, ensure_ascii=False),
                "round_number": round_idx,
                "total_rounds": self.n_rounds,
            }
            user_prompt = render_jinja_like_template(self.edge_input_template, context)
            user_prompt += (
                "\n\nReturn JSON only in this exact schema:\n"
                "{"
                '"nodes": [{"id": "n1", "name": "Entity", "type": "Entity", "description": "Brief description"}], '
                '"edges": [{"source_node_id": "n1", "target_node_id": "n2", "relationship_name": "uses"}]'
                "}"
            )
            payload = self._call_json_stage(
                system_prompt=self.edge_system_prompt,
                user_prompt=user_prompt,
                sample_content=sample_content,
                stage=f"cognee_edges_chunk_{chunk_idx}_round_{round_idx}",
            )
            if not isinstance(payload, dict):
                continue

            round_nodes = payload.get("nodes", []) or []
            for idx, node in enumerate(round_nodes, start=1):
                if not isinstance(node, dict):
                    continue
                name = str(node.get("name") or "").strip()
                node_id = str(node.get("id") or f"chunk{chunk_idx}_r{round_idx}_n{idx}").strip()
                if name and name not in node_name_to_id:
                    node_name_to_id[name] = node_id

            for edge in payload.get("edges", []) or []:
                if not isinstance(edge, dict):
                    continue
                source_id = str(edge.get("source_node_id") or "").strip()
                target_id = str(edge.get("target_node_id") or "").strip()
                relationship_name = str(edge.get("relationship_name") or "").strip()
                if not (source_id and target_id and relationship_name):
                    continue
                source_name = next((name for name, node_id in node_name_to_id.items() if node_id == source_id), "")
                target_name = next((name for name, node_id in node_name_to_id.items() if node_id == target_id), "")
                if source_name and target_name and source_name != target_name:
                    triplet = {"sub": source_name, "rel": relationship_name, "obj": target_name}
                    final_relations.append(triplet)
                    previous_triplets.append(triplet)

        return normalize_relations(final_relations)

    def generate(self, content: str, **_: Any) -> Dict[str, Any]:
        if self.cache:
            cached = self.cache.load(content)
            if isinstance(cached, dict):
                print(f"✨ [Cognee] 命中缓存: chars={len(content)}")
                return cached

        chunks = sentence_chunk_text(
            content,
            target_chars=self.chunk_target_chars,
            overlap_chars=self.chunk_overlap_chars,
            min_chunk_chars=1200,
        )
        print(f"🚀 [Cognee] 全文 chunk 数: {len(chunks)}")

        all_nodes: Dict[str, Dict[str, str]] = {}
        all_relations: List[Dict[str, str]] = []

        for chunk_idx, chunk_text in enumerate(chunks, start=1):
            candidate_nodes = self._extract_nodes(chunk_text, content, chunk_idx)
            refined_nodes, relationship_names = self._extract_relationship_names(
                chunk_text=chunk_text,
                candidate_nodes=candidate_nodes,
                sample_content=content,
                chunk_idx=chunk_idx,
            )
            relations = self._extract_edges(
                chunk_text=chunk_text,
                nodes=refined_nodes,
                relationship_names=relationship_names,
                sample_content=content,
                chunk_idx=chunk_idx,
            )
            for node in refined_nodes:
                key = node.lower()
                if key not in all_nodes:
                    all_nodes[key] = {"name": node, "type": "Entity"}
            all_relations.extend(relations)
            print(
                f"  ✅ [Cognee] chunk {chunk_idx}/{len(chunks)} "
                f"nodes={len(refined_nodes)} rel_names={len(relationship_names)} edges={len(relations)}"
            )

        result = {
            "entities": list(all_nodes.values()),
            "relations": normalize_relations(all_relations),
        }
        if self.cache:
            self.cache.save(content, result)
        return result


Method = CogneeMethod
