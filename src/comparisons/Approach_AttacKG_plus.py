# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from baseline_shared_utils import SharedPromptBaseline, normalize_relations, parse_json_payload

try:
    from article_io_cache_parser import KGCacheManager
except ImportError:
    KGCacheManager = None


TACTIC_ORDER = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact",
]

ENTITIES = [
    "Report Type", "Author", "Vender", "Date", "Infrastructure", "Location",
    "Malware", "Organization", "Security Tool", "Vulnerability",
    "Vulnerable Software", "System", "Platform", "Threat Actor",
    "Tactic", "Technique", "Filename", "Filepath", "URL", "Registry",
    "Domain", "Hash", "IP",
]

RELATIONS = [
    "Modify", "Construct", "Use", "Detect", "Write", "Operate",
    "Has vulnerability", "Has geolocation", "Belong to", "Locate at",
    "Has URL", "Has value", "Has hostname",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
MITRE_JSON_PATH = str(REPO_ROOT / "baseline" / "resources" / "attackg_plus" / "mitre.json")

REWRITE_SYSTEM_PROMPT = """You are an assistant to process CTI (Cyber Threat Intelligence) reports for attack graph extraction.

Process of section summary: For each tactic, if there is relevant content in the CTI report, extract and rewrite the related content in chronological order as detailed as possible. If there is no relevant content, skip that tactic and output "None" as the summary. Key information, including the names of entities and the relationships of relations related to the subordinate techniques for given tactic, needs to be preserved in the rewriting."""

REWRITE_USER_TEMPLATE = """Given the following CTI article, for each of the 14 MITRE ATT&CK tactics listed below, extract and rewrite relevant sentences in chronological order. Output "None" if no relevant content for that tactic.

{tactic_description}

article:
{text}

Return JSON only, with tactic name as key and rewritten content (or "None") as value."""

EXTRACT_SYSTEM_PROMPT = """You are an assistant to perform structured entity extraction and relationship extraction from article, especially in the domain of Cyber Threat Intelligence (CTI) report, according to the following rules:

Rule 1: The output format MUST follow: Subject(EntityType) ; Relation ; Object(EntityType)
Rule 2: Extract entities only from these types: {entities}
Rule 3: Extract relations only from these candidate relation types (you can add other relations if needed): {relations}
Rule 4: Do not use pronouns and always replace them with actual entity names
Rule 5: The description of relationships should use active voice in the present simple tense"""

EXTRACT_USER_TEMPLATE = """Please extract the security related triplets in the article below.
article {tactic}:
{text}

Extracted triplets are:"""


def _parse_triplet_line(line: str) -> Dict[str, str] | None:
    text = str(line or "").strip()
    if not text or ";" not in text:
        return None
    parts = [part.strip() for part in text.split(";")]
    if len(parts) < 3:
        return None
    sub = re.sub(r"\([^)]*\)", "", parts[0]).strip()
    rel = parts[1].strip()
    obj = re.sub(r"\([^)]*\)", "", parts[2]).strip()
    if sub and rel and obj:
        return {"sub": sub, "rel": rel, "obj": obj}
    return None


def _build_mitre_tactic_technique_description(mitre_data: Dict[str, Any]) -> str:
    rule_description = (
        "There are 14 tactics in cyber attacks, and these 14 tactics are provided "
        "with their names, descriptions, and corresponding techniques in the logic order of cyber attacks:\n"
    )
    tactics = list(mitre_data.get("tactics", []))
    tactics = sorted(
        tactics,
        key=lambda item: TACTIC_ORDER.index(item.get("name")) if item.get("name") in TACTIC_ORDER else 999,
    )
    lines: List[str] = []
    for idx, tactic in enumerate(tactics, start=1):
        lines.append(f"{idx}. {tactic.get('name')}: {tactic.get('description', '')}")
        for tech_idx, technique in enumerate(tactic.get("techniques", []), start=1):
            lines.append(f"{idx}.{tech_idx} {technique.get('name')}")
    return rule_description + "\n".join(lines)


class AttacKGPlusMethod(SharedPromptBaseline):

    def __init__(
        self,
        model: str = "gpt-5-nano",
        token: int = 8192,
        temp: float = 0.7,
        enable_method_cache: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name="AttacKG+",
            model=model,
            token=token,
            temp=temp,
            **kwargs,
        )
        self._entities_str = ", ".join(ENTITIES)
        self._relations_str = ", ".join(RELATIONS)
        
        
        
        self.cache = (
            KGCacheManager(os.path.basename(__file__), method_file_path=os.path.abspath(__file__))
            if KGCacheManager and enable_method_cache
            else None
        )
        self._mitre = None

    def _load_mitre(self) -> Dict[str, Any]:
        if self._mitre is None:
            with open(MITRE_JSON_PATH, "r", encoding="utf-8") as f:
                self._mitre = json.load(f)
        return self._mitre

    def _step1_rewrite(self, content: str) -> Dict[str, str]:
        mitre = self._load_mitre()
        tactic_description = _build_mitre_tactic_technique_description(mitre)
        raw = self._call_llm(
            [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": REWRITE_USER_TEMPLATE.format(
                        tactic_description=tactic_description,
                        text=content,
                    ),
                },
            ],
            sample_content=content,
            stage="attackg_plus_rewrite",
        )
        payload = parse_json_payload(raw)
        tactic_texts: Dict[str, str] = {}
        if isinstance(payload, dict):
            for tactic in TACTIC_ORDER:
                value = payload.get(tactic)
                if value and str(value).strip().lower() not in {"none", "null", ""}:
                    tactic_texts[tactic] = str(value).strip()
        if tactic_texts:
            return tactic_texts

        
        for tactic in TACTIC_ORDER:
            pattern = rf"{re.escape(tactic)}\s*[:：]\s*(.+?)(?=\n(?:{'|'.join(map(re.escape, TACTIC_ORDER))})\s*[:：]|\Z)"
            match = re.search(pattern, raw, flags=re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value.lower() not in {"none", "null", ""}:
                    tactic_texts[tactic] = value
        return tactic_texts

    def _step2_extract(self, tactic_texts: Dict[str, str], sample_content: str) -> List[Dict[str, str]]:
        if not tactic_texts:
            return []
        system_prompt = EXTRACT_SYSTEM_PROMPT.format(
            entities=self._entities_str,
            relations=self._relations_str,
        )
        messages_list: List[List[Dict[str, str]]] = []
        tactic_order: List[str] = []
        for tactic, text in tactic_texts.items():
            messages_list.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": EXTRACT_USER_TEMPLATE.format(tactic=tactic, text=text)},
                ]
            )
            tactic_order.append(tactic)

        responses = []
        for idx, messages in enumerate(messages_list, start=1):
            responses.append(
                self._call_llm(
                    messages,
                    sample_content=sample_content,
                    stage=f"attackg_plus_extract_{idx}",
                )
            )

        relations: List[Dict[str, str]] = []
        for tactic, response in zip(tactic_order, responses):
            for line in str(response or "").splitlines():
                triplet = _parse_triplet_line(line)
                if triplet:
                    triplet["tactic"] = tactic
                    relations.append(triplet)
        return relations

    def generate(self, content: str, **_: Any) -> Dict[str, Any]:
        if self.cache:
            cached = self.cache.load(content)
            if isinstance(cached, dict):
                print(f"✨ [AttacKG+] 命中缓存: chars={len(content)}")
                return cached

        print("🚀 [AttacKG+] Step 1: rewrite")
        tactic_texts = self._step1_rewrite(content)
        print(f"  ✅ [AttacKG+] rewrite tactics={len(tactic_texts)}")

        print("🚀 [AttacKG+] Step 2: extract")
        relations = self._step2_extract(tactic_texts, content)
        result = {"relations": normalize_relations(relations)}
        if self.cache:
            self.cache.save(content, result)
        return result


Method = AttacKGPlusMethod
