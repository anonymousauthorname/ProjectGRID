# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List

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


ENTITY_EXTRACTION_PROMPT = """You are a cybersecurity threat intelligence information extractor. Your goal is to extract all entities related to attack activities from a given cyber threat intelligence report or text and return only a list of entities. The information that must be extracted includes (but is not limited to):
Malware names, backdoor programs, Dropper files, or scripts
URLs, IP addresses, domain names, C2 infrastructure
Filenames, file paths, malicious scripts involved in the attack process
Tools, scripts, encoding methods (e.g., PowerShell, shellcode, macros)
Threat actors or organizations (APT or others), and any role descriptions (e.g., "attacker," "victim")
Attack vectors (e.g., phishing emails, USB devices, etc.)
Vulnerabilities, exploits, CVE numbers
Registry keys, scheduled tasks, persistence mechanisms
Technical identifiers for the same vulnerability (e.g., "CVE-2021-44228", "Log4Shell vulnerability")

Please note:
1. Do not provide any additional explanations or groupings; return only a flat list of entity strings.
2. Do not omit any technical indicators, abstract roles, or descriptions of attack stages.
3. For references like "the malware" or "it," try to resolve them into specific entity names based on context; if unable to determine, retain the original wording.

Text:
{text}

Return JSON only:
["entity1", "entity2", "..."]"""


RELATION_EXTRACTION_PROMPT = """You are a cybersecurity threat intelligence extractor. Your goal is to extract all "subject-predicate-object" triples from a given threat intelligence text, based on a known entity list, to represent key relationships in attack activities. You need to:
Convert all inferred or explicit relationships between entities in the text into the structure of (subject, action, object);
Use precise verbs/actions for predicates, such as "uses," "downloads," "connects to," "exploits," "injects," etc.;
Convert passive or indirect expressions into explicit triples; for example, "a file was downloaded and executed" should be broken down into multiple relationships;
Resolve references like "the file," "this payload," etc., by tracing back to the original entity name; if unclear, retain the original wording;
For nested structures, infer entity relationships reasonably based on context or naming logic.

Note:
1. Only output relationships involving entities from the known entity list; if the subject or object is not in the known entity list, do not output that triple.
2. For sentences with nested or multiple actions, split them into multiple independent triples to preserve the complete attack chain logic.
3. Do not repeat triples with the same meaning unless the text explicitly repeats them with slight variations in meaning.

Known entity list:
{entity_list}

Text:
{text}

Return JSON only:
[{{"subject": "entity1", "predicate": "action_verb", "object": "entity2"}}]"""


def _normalize_entity_surface(entity: str) -> str:
    text = str(entity or "").strip()
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


class LLMAKGMethod(SharedPromptBaseline):

    def __init__(
        self,
        model: str = "gpt-5-nano",
        token: int = 8192,
        temp: float = 0.7,
        chunk_target_chars: int = 12000,
        chunk_overlap_chars: int = 700,
        **kwargs: Any,
    ):
        super().__init__(
            name="LLM-CAKG",
            model=model,
            token=token,
            temp=temp,
            **kwargs,
        )
        self.chunk_target_chars = max(2500, int(chunk_target_chars or 12000))
        self.chunk_overlap_chars = max(0, int(chunk_overlap_chars or 0))
        self.cache = (
            KGCacheManager(os.path.basename(__file__), method_file_path=os.path.abspath(__file__))
            if KGCacheManager
            else None
        )

    def _extract_entities_from_chunk(self, chunk_text: str, sample_content: str, chunk_idx: int) -> List[str]:
        payload = parse_json_payload(
            self._call_llm(
                [{"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(text=chunk_text)}],
                sample_content=sample_content,
                stage=f"llm_akg_entities_chunk_{chunk_idx}",
            )
        )
        raw_entities = payload if isinstance(payload, list) else []
        entities: List[str] = []
        seen = set()
        for item in raw_entities:
            name = _normalize_entity_surface(item)
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            entities.append(name)
        return entities

    def _extract_relations_from_chunk(
        self,
        *,
        chunk_text: str,
        entity_list: List[str],
        sample_content: str,
        chunk_idx: int,
    ) -> List[Dict[str, str]]:
        payload = parse_json_payload(
            self._call_llm(
                [
                    {
                        "role": "user",
                        "content": RELATION_EXTRACTION_PROMPT.format(
                            entity_list=json.dumps(entity_list, ensure_ascii=False),
                            text=chunk_text,
                        ),
                    }
                ],
                sample_content=sample_content,
                stage=f"llm_akg_relations_chunk_{chunk_idx}",
            )
        )
        items = payload if isinstance(payload, list) else []
        relations: List[Dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            sub = _normalize_entity_surface(item.get("subject"))
            rel = str(item.get("predicate") or "").strip()
            obj = _normalize_entity_surface(item.get("object"))
            if sub and rel and obj and sub != obj:
                relations.append({"sub": sub, "rel": rel, "obj": obj})
        return normalize_relations(relations)

    def generate(self, content: str, **_: Any) -> Dict[str, Any]:
        if self.cache:
            cached = self.cache.load(content)
            if isinstance(cached, dict):
                print(f"✨ [LLM-CAKG] 命中缓存: chars={len(content)}")
                return cached

        chunks = sentence_chunk_text(
            content,
            target_chars=self.chunk_target_chars,
            overlap_chars=self.chunk_overlap_chars,
            min_chunk_chars=1200,
        )
        print(f"🚀 [LLM-CAKG] 全文 chunk 数: {len(chunks)}")

        merged_entities: List[str] = []
        seen_entities = set()
        for chunk_idx, chunk_text in enumerate(chunks, start=1):
            chunk_entities = self._extract_entities_from_chunk(chunk_text, content, chunk_idx)
            for entity in chunk_entities:
                key = entity.lower()
                if key in seen_entities:
                    continue
                seen_entities.add(key)
                merged_entities.append(entity)
            print(f"  ✅ [LLM-CAKG] chunk {chunk_idx}/{len(chunks)} entities={len(chunk_entities)}")

        all_relations: List[Dict[str, str]] = []
        for chunk_idx, chunk_text in enumerate(chunks, start=1):
            chunk_relations = self._extract_relations_from_chunk(
                chunk_text=chunk_text,
                entity_list=merged_entities,
                sample_content=content,
                chunk_idx=chunk_idx,
            )
            all_relations.extend(chunk_relations)
            print(f"  ✅ [LLM-CAKG] chunk {chunk_idx}/{len(chunks)} relations={len(chunk_relations)}")

        result = {
            "entities": [{"name": entity, "type": "Entity"} for entity in merged_entities],
            "relations": normalize_relations(all_relations),
        }
        if self.cache:
            self.cache.save(content, result)
        return result


Method = LLMAKGMethod
