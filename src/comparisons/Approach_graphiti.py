# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from baseline_shared_utils import (
    SharedPromptBaseline,
    normalize_relations,
    parse_json_payload,
    sentence_chunk_text,
)

try:
    from article_io_cache_parser import KGCacheManager
except ImportError:
    KGCacheManager = None


GRAPHITI_ENTITY_TYPES: List[Dict[str, Any]] = [
    {"entity_type_id": 1, "name": "ThreatActor", "description": "APT groups, operators, and named adversaries."},
    {"entity_type_id": 2, "name": "Malware", "description": "Malware families, implants, payloads, droppers, and ransomware."},
    {"entity_type_id": 3, "name": "Tool", "description": "Attack tools, frameworks, scripts, and utilities used by attackers."},
    {"entity_type_id": 4, "name": "Infrastructure", "description": "Domains, IPs, servers, C2 infrastructure, and hosting assets."},
    {"entity_type_id": 5, "name": "Vulnerability", "description": "CVE identifiers, exploits, and vulnerable conditions."},
    {"entity_type_id": 6, "name": "Campaign", "description": "Named campaigns, operations, or intrusion sets in a concrete activity."},
    {"entity_type_id": 7, "name": "Organization", "description": "Victim organizations, vendors, research teams, and institutions."},
    {"entity_type_id": 8, "name": "Location", "description": "Countries, regions, geolocations, and physical places."},
    {"entity_type_id": 9, "name": "AttackPattern", "description": "ATT&CK techniques, tactics, and concrete attack behaviors."},
    {"entity_type_id": 10, "name": "Indicator", "description": "Indicators such as hashes, file names, file paths, mutexes, URLs, and registry keys."},
    {"entity_type_id": 11, "name": "Software", "description": "Targeted or abused software, platforms, systems, and applications."},
    {"entity_type_id": 12, "name": "Person", "description": "Named people such as researchers, victims, or operators."},
]

GRAPHITI_TYPE_BY_ID = {item["entity_type_id"]: item["name"] for item in GRAPHITI_ENTITY_TYPES}


def _canonical_relation_type(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "_" in text:
        return text.lower().replace("_", " ")
    return text


class GraphitiMethod(SharedPromptBaseline):

    def __init__(
        self,
        model: str = "gpt-5-nano",
        token: int = 8192,
        temp: float = 0.7,
        chunk_target_chars: int = 12000,
        chunk_overlap_chars: int = 900,
        previous_episode_window: int = 2,
        reference_time: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="Graphiti",
            model=model,
            token=token,
            temp=temp,
            **kwargs,
        )
        self.chunk_target_chars = max(2500, int(chunk_target_chars or 12000))
        self.chunk_overlap_chars = max(0, int(chunk_overlap_chars or 0))
        self.previous_episode_window = max(0, int(previous_episode_window or 0))
        self.reference_time = reference_time or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.cache = (
            KGCacheManager(os.path.basename(__file__), method_file_path=os.path.abspath(__file__))
            if KGCacheManager
            else None
        )

    def _entity_types_block(self) -> str:
        return json.dumps(GRAPHITI_ENTITY_TYPES, ensure_ascii=False, indent=2)

    def _node_messages(self, chunk_text: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are an AI assistant that extracts entity nodes from text. "
            "Your primary task is to extract and classify significant entities mentioned in the provided text."
        )
        user_prompt = f"""
<ENTITY TYPES>
{self._entity_types_block()}
</ENTITY TYPES>

<TEXT>
{chunk_text}
</TEXT>

Given the above text, extract entities from the TEXT that are explicitly or implicitly mentioned.
For each entity extracted, also determine its entity type based on the provided ENTITY TYPES and their descriptions.
Indicate the classified entity type by providing its entity_type_id.

Additional extraction instructions:
- Focus on cybersecurity threat intelligence entities.
- Prefer canonical names over pronouns or vague phrases.
- Do not extract dates, years, or pure temporal expressions as nodes.
- Do not create nodes for actions or relations.

Return JSON only in this exact schema:
{{
  "extracted_entities": [
    {{
      "name": "Entity name",
      "entity_type_id": 1
    }}
  ]
}}
"""
        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

    def _edge_messages(
        self,
        *,
        chunk_text: str,
        current_nodes: List[Dict[str, str]],
        previous_chunks: List[str],
    ) -> List[Dict[str, str]]:
        system_prompt = (
            "You are an expert fact extractor that extracts fact triples from text. "
            "Extracted fact triples should include relevant date information when explicitly available, "
            "and CURRENT TIME is the reference used to resolve relative dates."
        )
        previous_payload = [{"content": item} for item in previous_chunks]
        user_prompt = f"""
<PREVIOUS_MESSAGES>
{json.dumps(previous_payload, ensure_ascii=False, indent=2)}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{chunk_text}
</CURRENT_MESSAGE>

<ENTITIES>
{json.dumps(current_nodes, ensure_ascii=False, indent=2)}
</ENTITIES>

<REFERENCE_TIME>
{self.reference_time}
</REFERENCE_TIME>

# TASK
Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
Only extract facts that:
- involve two DISTINCT ENTITIES from the ENTITIES list,
- are clearly stated or unambiguously implied in the CURRENT MESSAGE,
- can be represented as edges in a knowledge graph,
- use entity names rather than pronouns.

You may use information from PREVIOUS_MESSAGES only to disambiguate references or support continuity.

# EXTRACTION RULES
1. source_entity_name and target_entity_name must exactly match entity names from ENTITIES.
2. Each fact must involve two distinct entities.
3. Do not emit duplicate or semantically redundant facts.
4. The fact should paraphrase the source text and not copy long text spans verbatim.
5. relation_type should be in SCREAMING_SNAKE_CASE when possible.
6. Use ISO-8601 timestamps only if the time is explicit or clearly resolvable; otherwise use null.

Return JSON only in this exact schema:
{{
  "edges": [
    {{
      "source_entity_name": "Entity A",
      "target_entity_name": "Entity B",
      "relation_type": "USES",
      "fact": "Entity A uses Entity B",
      "valid_at": null,
      "invalid_at": null
    }}
  ]
}}
"""
        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

    def _extract_nodes(self, chunk_text: str, sample_content: str, chunk_idx: int) -> List[Dict[str, str]]:
        raw = self._call_llm(
            self._node_messages(chunk_text),
            sample_content=sample_content,
            stage=f"graphiti_nodes_chunk_{chunk_idx}",
        )
        payload = parse_json_payload(raw)
        items = payload.get("extracted_entities") if isinstance(payload, dict) else payload
        nodes: List[Dict[str, str]] = []
        seen = set()
        for item in items or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            entity_type_id = item.get("entity_type_id")
            entity_type = GRAPHITI_TYPE_BY_ID.get(entity_type_id, str(item.get("entity_type") or "Entity").strip() or "Entity")
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            nodes.append({"name": name, "type": entity_type})
        return nodes

    def _extract_edges(
        self,
        *,
        chunk_text: str,
        current_nodes: List[Dict[str, str]],
        previous_chunks: List[str],
        sample_content: str,
        chunk_idx: int,
    ) -> List[Dict[str, str]]:
        if len(current_nodes) < 2:
            return []
        raw = self._call_llm(
            self._edge_messages(
                chunk_text=chunk_text,
                current_nodes=current_nodes,
                previous_chunks=previous_chunks,
            ),
            sample_content=sample_content,
            stage=f"graphiti_edges_chunk_{chunk_idx}",
        )
        payload = parse_json_payload(raw)
        items = payload.get("edges") if isinstance(payload, dict) else payload
        relations: List[Dict[str, str]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            sub = str(item.get("source_entity_name") or item.get("source_node_name") or "").strip()
            obj = str(item.get("target_entity_name") or item.get("target_node_name") or "").strip()
            rel = _canonical_relation_type(str(item.get("relation_type") or "")) or str(item.get("fact") or "").strip()
            if sub and rel and obj and sub != obj:
                relations.append({"sub": sub, "rel": rel, "obj": obj})
        return normalize_relations(relations)

    def generate(self, content: str, **_: Any) -> Dict[str, Any]:
        if self.cache:
            cached = self.cache.load(content)
            if isinstance(cached, dict):
                print(f"✨ [Graphiti] 命中缓存: chars={len(content)}")
                return cached

        chunks = sentence_chunk_text(
            content,
            target_chars=self.chunk_target_chars,
            overlap_chars=self.chunk_overlap_chars,
            min_chunk_chars=1200,
        )
        print(f"🚀 [Graphiti] 全文 chunk 数: {len(chunks)}")

        all_entities: Dict[str, Dict[str, str]] = {}
        all_relations: List[Dict[str, str]] = []

        for chunk_idx, chunk_text in enumerate(chunks, start=1):
            nodes = self._extract_nodes(chunk_text, content, chunk_idx)
            for node in nodes:
                key = node["name"].lower()
                if key not in all_entities:
                    all_entities[key] = node
            previous_chunks = chunks[max(0, chunk_idx - 1 - self.previous_episode_window): chunk_idx - 1]
            relations = self._extract_edges(
                chunk_text=chunk_text,
                current_nodes=nodes,
                previous_chunks=previous_chunks,
                sample_content=content,
                chunk_idx=chunk_idx,
            )
            all_relations.extend(relations)
            print(
                f"  ✅ [Graphiti] chunk {chunk_idx}/{len(chunks)} "
                f"nodes={len(nodes)} edges={len(relations)}"
            )

        result = {
            "entities": [{"name": node["name"], "type": node["type"]} for node in all_entities.values()],
            "relations": normalize_relations(all_relations),
        }
        if self.cache:
            self.cache.save(content, result)
        return result


Method = GraphitiMethod
