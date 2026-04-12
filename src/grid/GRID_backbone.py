# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from single_pass_template import ToolsAskMethod, VLLMServerMethod
from shared_eval_backend import build_default_shared_backend, DEFAULT_SHARED_QWEN_MODEL_PATH


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RL_MODEL_PATH = str(_REPO_ROOT / "models" / "task_bank_reward")
PROMPT_VERSION = "20260319"

ENTITY_TYPES = (
    "user-account, identity, threat-actor-or-intrusion-set, "
    "malware, hacker-tool, general-software, detailed-part-of-malware-or-hackertool, detailed-part-of-general-software, "
    "vulnerability, attack-pattern, campaign, "
    "file, process, windows-registry-key, "
    "ipv4-addr, ipv6-addr, domain-name, url, email-address, network-traffic, mac-address, infrastructure, "
    "credential-value, x509-certificate, "
    "indicator, course-of-action, security-product, malware-analysis-document-or-publication-or-conference, "
    "location, abstract-concept, generic-noun, other, noise"
)

REL_TYPES = (
    "exploits, bypasses, malicious-investigates-track-detects, impersonates, targets, compromises, leads-to, "
    "drops, downloads, executes, delivers, beacons-to, exfiltrate-to, leaks, communicates-with, "
    "resolves-to, hosts, provides, "
    "authored-by, owns, controls, attributed-to, affiliated-with, cooperates-with, "
    "is-part-of, consists-of, has, depends-on, creates-or-generates, modifies-or-removes-or-replaces, uses, "
    "variant-of, derived-from, alias-of, compares-to, categorized-as, "
    "located-at, originates-from, "
    "indicates, mitigates, based-on, research-describes-analysis-of-characterizes-detects, "
    "negation, other"
)

_LATEST_GRID_PROMPT_BODY = """
                [FIX-260303-1] ⚠️ OVERRIDING CONSTRAINT: Text-Provable Truth
                All extraction MUST be grounded in direct textual evidence. FORBIDDEN:
                - External knowledge completion (inferring from domain knowledge not in text)
                - Subject elevation (attributing tool/component behavior to controller/parent)
                - Behavioral-to-structural conversion ("A uses B" does NOT mean "B is-part-of A")
                - Chain deduction with subject change ("A uses B" + "B does C" does NOT mean "A does C"; extract both edges separately)

                [Definitions & Constraints]
                1. Definition of Entity: Any unit of information that can be independently identified, described, and analyzed. Includes Explicit Entities (named objects like "get-logon-history.ps1") and Implicit Entities (contextual objects like "RAR file", "the registry key", "embedded payload") even without fixed names. Key properties: independence, relevance, contextual significance.
                2. Anti-Hallucination Rule: Only extract entities explicitly or implicitly present in the input text. Do NOT hallucinate entities not supported by text.
                3. Type Lists:
                Entities: [{entity_types}]
                Relationships: [{rel_types}]
                4. Glossary for Complex Types:
                Entities: threat-actor-or-intrusion-set (malicious groups/clusters); hacker-tool (offensive) vs general-software (legitimate but abused); detailed-part-of (internal components); indicator (known malicious pattern for detection); course-of-action (defense); generic-noun (countable class names).
                Relationships: malicious-investigates-track-detects (recon/spy/anti-analysis); delivers (abstract vector); leads-to (causal chain); research-describes-analysis-of-characterizes-detects (document analyzing threat or tool detecting threat); authored-by (creator); attributed-to (responsibility).
                5. Crucial Distinctions: malware vs file (identity vs technical object); threat-actor vs identity (malicious vs neutral); drops vs downloads (local creation vs remote transfer); exfiltrate-to vs leaks (directed theft vs public disclosure); consists-of vs is-part-of (list/set vs component); owns vs controls (human/org vs software).

                [Stage 1: Entity Extraction]
                1. Identification: Scan for all Explicit and Implicit entities. If attacker/campaign is unnamed, create entity named "the attacker" with type threat-actor-or-intrusion-set. Use 'other' type ONLY when no predefined type fits.
                2. Naming & Alias Logic: 'name' MUST be the string used at the FIRST appearance for the entity list only. 'alias' MUST include formal aliases, generic/pronominal references ("the malware"), and abstract co-references ("the real thing"). If no alias exists, OMIT the 'alias' field entirely. If no parent/lineage exists, OMIT the 'mother entity' field entirely.
                2.1. Relationship Mention Rule for Co-references: When the SAME entity is mentioned in different ways across different sentences/clauses, keep the entity list canonical naming unchanged (`name` = first appearance; later forms go to `alias`), BUT in each relationship triple, `sub` and `obj` should preferentially use the entity wording that appears in the ORIGINAL supporting sentence/clause for that specific relation, rather than always rewriting it to the first-appearance `name`. Example: if "a xxx b" and later "c xxx d" refer to the same entity for `a`/`c`, then extract [a, xxx, b] from the first sentence and [c, xxx, d] from the later sentence; do NOT rewrite the second triple into [a, xxx, d]. Prefer concrete sentence-local mentions over bare pronouns when possible.
                3. Recheck: Did I miss explicitly named entities? Did I miss contextually important implicit entities (file, image, script)?

                [Stage 2: Relationship Extraction]
                Pass 1 Primary Action Logic:
                Case 1 (Explicit Verb): If clear verb exists, set 'rel' to phrase and map 'rel_type' to closest category.
                Case 2 (Implied Match): If implied relation matches type perfectly, set BOTH 'rel' and 'rel_type' to that type name.
                Case 3 (Fallback): Summarize action as 'rel' in own words, set 'rel_type' to ['other'].
                Pass 2 Structural & Definitional Extraction: Re-scan for relationships in structures like Possessive (owns), Preposition (located-at), Appositives (alias-of), Lists (consists-of).
                Factuality Annotation: Output 'special_factuality' as List (e.g., ['possible', 'future'], ['negated']) ONLY when the relation is non-standard factuality. If the relation is standard factual, OMIT the 'special_factuality' field entirely.
                [FIX-260303-2][FIX-260303-3] Connectivity Recheck: Every entity must participate in at least one relationship that is DIRECTLY SUPPORTED by the text. If an entity has no text-provable relationship, REMOVE that entity from the output rather than inferring a relationship using external knowledge. Do NOT fabricate indirect relationships to "main entities" based on co-occurrence in the same report.

                [Stage 3: Output Format]
                First, output the full reasoning section (Entity Reasoning and Relationship Reasoning) between #Reasoning_Start# and #Reasoning_End#.
                Second, output strictly the two JSON lists between #Entity_List_Start# / #Entity_List_End# and #Relationship_List_Start# / #Relationship_List_End#.
                Entity list and relationship list use DIFFERENT naming rules: entity list uses canonical first-appearance `name`; relationship triples use the sentence-local mention from the supporting evidence clause when that mention is a valid name/alias for the same entity.
                Do not include any extra text outside these markers.

                #Reasoning_Start#
                ... Entity Reasoning ...
                ... Relationship Reasoning ...
                #Reasoning_End#

                #Entity_List_Start#
                [
                  {{\\"name\\": \\"Entity_With_Alias_And_Parent\\", \\"type\\": \\"Category\\", \\"alias\\": [\\"Alias1\\", \\"Alias2\\"], \\"mother entity\\": [\\"Parent_Name\\"]}},
                  {{\\"name\\": \\"Entity_With_Alias_Only\\", \\"type\\": \\"Category\\", \\"alias\\": [\\"Alias1\\"]}},
                  {{\\"name\\": \\"Entity_With_Parent_Only\\", \\"type\\": \\"Category\\", \\"mother entity\\": [\\"Parent_Name\\"]}},
                  {{\\"name\\": \\"Entity_With_No_Alias_Or_Parent\\", \\"type\\": \\"Category\\"}}
                ]
                #Entity_List_End#

                #Relationship_List_Start#
                [
                  {{\\"sub\\": \\"Sentence_Local_Mention\\", \\"rel\\": \\"Verb_Phrase\\", \\"rel_type\\": [\\"Category\\"], \\"obj\\": \\"Sentence_Local_Mention\\"}},
                  {{\\"sub\\": \\"Sentence_Local_Mention\\", \\"rel\\": \\"Verb_Phrase\\", \\"rel_type\\": [\\"Category\\"], \\"obj\\": \\"Sentence_Local_Mention\\", \\"special_factuality\\": [\\"possible\\"]}},
                  {{\\"sub\\": \\"Sentence_Local_Mention\\", \\"rel\\": \\"Verb_Phrase\\", \\"rel_type\\": [\\"Category\\"], \\"obj\\": \\"Sentence_Local_Mention\\", \\"special_factuality\\": [\\"negated\\"]}}
                ]
                #Relationship_List_End#
""".format(entity_types=ENTITY_TYPES, rel_types=REL_TYPES)

_STEP1_ENTITY_PROMPT_TEMPLATE = """You are an NLP model specialized in CTI knowledge graph extraction. This is Step 1 of a mandatory two-step extraction workflow.

                [TWO-STEP STEP1 OVERRIDE]
                Preserve the latest GRID 20260319 prompt below as the base instruction. In this step, keep all entity-related rules unchanged, but temporarily do the following minimal overrides:
                1. Output ONLY the reasoning section and the entity list.
                2. Do NOT output a relationship list in Step 1.
                3. Do NOT apply the final connectivity pruning in Step 1; that pruning is deferred to Step 2 after relationships are extracted.
                4. Do NOT simplify, summarize, or weaken any other original rule.

__GRID_LATEST_PROMPT_BODY__

                [STEP1 OUTPUT OVERRIDE]
                Replace the Stage 3 output requirement with the following for this step only:
                First, output the reasoning section between #Reasoning_Start# and #Reasoning_End#.
                Second, output strictly one JSON list between #Entity_List_Start# and #Entity_List_End#.
                Do not output #Relationship_List_Start# / #Relationship_List_End# in Step 1.
                Do not include any extra text outside these markers.

                Input Text: __GRID_INPUT_TEXT__
""".replace("__GRID_LATEST_PROMPT_BODY__", _LATEST_GRID_PROMPT_BODY)

_STEP2_RELATION_PROMPT_TEMPLATE = """You are an NLP model specialized in CTI knowledge graph extraction. This is Step 2 of a mandatory two-step extraction workflow.

                [TWO-STEP STEP2 OVERRIDE]
                Preserve the latest GRID 20260319 prompt below as the base instruction. In addition to all original rules, you are given a Step 1 candidate entity list. Use it as a strong starting point, but still verify everything against the original text:
                1. Keep directly supported Step 1 entities.
                2. Add directly supported missing entities if Step 1 missed them.
                3. Remove unsupported/noise entities.
                4. Then apply the original Stage 2 and connectivity rules normally.
                5. Do NOT simplify, summarize, or weaken any other original rule.

__GRID_LATEST_PROMPT_BODY__

                [STEP2 ADDITIONAL INPUT]
                Step 1 Candidate Entity List (JSON only, no reasoning):
                __GRID_STEP1_ENTITIES_JSON__

                Input Text: __GRID_INPUT_TEXT__
""".replace("__GRID_LATEST_PROMPT_BODY__", _LATEST_GRID_PROMPT_BODY)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _extract_reasoning_block(raw_text: str) -> str:
    if not raw_text:
        return ""
    match = re.search(r"#Reasoning_Start#\s*([\s\S]*?)\s*#Reasoning_End#", raw_text)
    if not match:
        return ""
    return match.group(1).strip()


def _build_final_split_output(
    step1_raw: str,
    step2_raw: str,
    final_entities: List[Dict[str, Any]],
    final_relations: List[Dict[str, Any]],
) -> str:
    step1_reasoning = _extract_reasoning_block(step1_raw)
    step2_reasoning = _extract_reasoning_block(step2_raw)
    reasoning_parts = []
    if step1_reasoning:
        reasoning_parts.append("[Step1 Entity Reasoning]\n" + step1_reasoning)
    if step2_reasoning:
        reasoning_parts.append("[Step2 Entity+Relation Reasoning]\n" + step2_reasoning)
    merged_reasoning = "\n\n".join(reasoning_parts).strip()

    blocks = []
    if merged_reasoning:
        blocks.append("#Reasoning_Start#")
        blocks.append(merged_reasoning)
        blocks.append("#Reasoning_End#")
        blocks.append("")

    blocks.extend(
        [
            "#Entity_List_Start#",
            _json_dumps(final_entities),
            "#Entity_List_End#",
            "",
            "#Relationship_List_Start#",
            _json_dumps(final_relations),
            "#Relationship_List_End#",
        ]
    )
    return "\n".join(blocks).strip()


class _GRIDBackboneMixin:

    def _create_entity_step_prompt(self, content: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "user",
                
                "content": _STEP1_ENTITY_PROMPT_TEMPLATE.replace("__GRID_INPUT_TEXT__", str(content)),
            }
        ]

    def _create_relation_step_prompt(
        self,
        content: str,
        candidate_entities: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, str]]:
        candidate_entities_json = _json_dumps(candidate_entities or [])
        return [
            {
                "role": "user",
                
                "content": (
                    _STEP2_RELATION_PROMPT_TEMPLATE
                    .replace("__GRID_STEP1_ENTITIES_JSON__", candidate_entities_json)
                    .replace("__GRID_INPUT_TEXT__", str(content))
                ),
            }
        ]

    def _create_prompt(self, content: str) -> Dict[str, Any]:
        preview_entities = [
            {"name": "Preview_Entity", "type": "other", "alias": ["Preview Alias"]},
            {"name": "Preview_Target", "type": "other"},
        ]
        return {
            "prompt_type": "GRID_backbone",
            "prompt_version": PROMPT_VERSION,
            "step1_entity_prompt": self._create_entity_step_prompt(content),
            "step2_entity_relation_prompt_template": self._create_relation_step_prompt(
                content,
                preview_entities,
            ),
        }

    def _call_stage(self, contents: List[str], prompt_list: List[List[Dict[str, str]]], stage: str) -> List[str]:
        self._grid_prompt_metadata = self._build_prompt_metadata(contents, attempt=0, stage=stage)
        try:
            responses = self._call_llm(prompt_list)
        finally:
            self._grid_prompt_metadata = None
        return [resp if resp else "" for resp in responses]

    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        clean_text = self._clean_response(raw_text) if hasattr(self, "_clean_response") else (raw_text or "")
        return self._robust_json_parse(clean_text)

    def _compose_result(
        self,
        step1_raw: str,
        step1_parsed: Dict[str, Any],
        step2_raw: str,
        step2_parsed: Dict[str, Any],
    ) -> Dict[str, Any]:
        final_entities = list(step2_parsed.get("entities") or step1_parsed.get("entities") or [])
        final_relations = list(step2_parsed.get("relations") or [])
        merged_raw_output = _build_final_split_output(
            step1_raw=step1_raw,
            step2_raw=step2_raw,
            final_entities=final_entities,
            final_relations=final_relations,
        )
        result = {
            "entities": final_entities,
            "relations": final_relations,
            "raw_output": merged_raw_output,
            "step1_entities": list(step1_parsed.get("entities") or []),
            "step1_raw_output": step1_raw,
            "step2_raw_output": step2_raw,
        }
        if step2_parsed.get("error"):
            result["error"] = step2_parsed["error"]
        elif step1_parsed.get("error") and not final_entities and not final_relations:
            result["error"] = step1_parsed["error"]
        return result

    def generate(self, content: str, max_retries: int = 3) -> Dict[str, Any]:
        del max_retries  
        step1_raw = ""
        step2_raw = ""
        try:
            step1_prompt = self._create_entity_step_prompt(content)
            step1_raw = self._call_stage([content], [step1_prompt], stage="generate_step1_entities")[0]
            step1_parsed = self._parse_response(step1_raw)

            step2_prompt = self._create_relation_step_prompt(
                content,
                candidate_entities=list(step1_parsed.get("entities") or []),
            )
            step2_raw = self._call_stage([content], [step2_prompt], stage="generate_step2_entity_relations")[0]
            step2_parsed = self._parse_response(step2_raw)

            result = self._compose_result(step1_raw, step1_parsed, step2_raw, step2_parsed)
            self._save_check_log(content, result.get("raw_output", ""), result)
            return result
        except Exception as exc:
            print(f"  ❌ 两步 GRID 调用失败: {exc}")
            result = {
                "entities": [],
                "relations": [],
                "raw_output": _build_final_split_output(step1_raw, step2_raw, [], []),
                "error": str(exc),
                "step1_raw_output": step1_raw,
                "step2_raw_output": step2_raw,
            }
            self._save_check_log(content, result.get("raw_output", ""), result)
            return result

    def batch_generate(self, contents: List[str], max_retries: int = 3, **kwargs) -> List[Dict[str, Any]]:
        del max_retries, kwargs
        if not contents:
            return []

        print(f"🚀 [{self.name}] 两步批量生成开始: samples={len(contents)}")

        step1_prompts = [self._create_entity_step_prompt(content) for content in contents]
        step1_responses = self._call_stage(contents, step1_prompts, stage="generate_step1_entities")
        step1_parsed_list = [self._parse_response(raw_text) for raw_text in step1_responses]

        step2_prompts = [
            self._create_relation_step_prompt(content, candidate_entities=list(step1_parsed.get("entities") or []))
            for content, step1_parsed in zip(contents, step1_parsed_list)
        ]
        step2_responses = self._call_stage(contents, step2_prompts, stage="generate_step2_entity_relations")
        step2_parsed_list = [self._parse_response(raw_text) for raw_text in step2_responses]

        results: List[Dict[str, Any]] = []
        for idx, content in enumerate(contents):
            result = self._compose_result(
                step1_raw=step1_responses[idx],
                step1_parsed=step1_parsed_list[idx],
                step2_raw=step2_responses[idx],
                step2_parsed=step2_parsed_list[idx],
            )
            self._save_check_log(content, result.get("raw_output", ""), result)
            results.append(result)

        success_count = sum(1 for item in results if item.get("entities") or item.get("relations"))
        print(f"✅ [{self.name}] 两步批量生成完成: {success_count}/{len(contents)} 有结构化结果")
        return results


class _GRIDBackboneToolsAskMethod(_GRIDBackboneMixin, ToolsAskMethod):
    pass


class _GRIDBackboneVLLMServerMethod(_GRIDBackboneMixin, VLLMServerMethod):
    pass


class GRIDBackboneMethod:

    def __init__(
        self,
        llm_backend: str = "cloud_api",
        model: Optional[str] = "gpt-5-nano",
        token: int = 64 * 1024,
        temp: float = 0.7,
        think: int = 2,
        check_cache: bool = True,
        flex: bool = False,
        shared_llm_backend: Optional[Dict[str, Any]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.llm_backend = str(llm_backend or "cloud_api").strip().lower()
        self.runtime_context = runtime_context or {}
        self.name = "GRIDBackbone"

        
        
        effective_think = max(0, int(think or 0))

        shared_backend = dict(shared_llm_backend or {})
        if self.llm_backend == "shared_vllm" and not shared_backend:
            shared_backend = build_default_shared_backend(
                model_path=model or DEFAULT_SHARED_QWEN_MODEL_PATH,
            )

        if self.llm_backend == "shared_vllm":
            self.impl = _GRIDBackboneVLLMServerMethod(
                model="local",
                token=token,
                temp=temp,
                think=effective_think,
                check_cache=check_cache,
                shared_llm_backend=shared_backend,
                runtime_context=self.runtime_context,
                **kwargs,
            )
        elif self.llm_backend == "dedicated_vllm":
            self.impl = _GRIDBackboneVLLMServerMethod(
                model="local",
                model_path=model or DEFAULT_RL_MODEL_PATH,
                token=token,
                temp=temp,
                think=effective_think,
                check_cache=check_cache,
                runtime_context=self.runtime_context,
                **kwargs,
            )
        else:
            self.impl = _GRIDBackboneToolsAskMethod(
                model=model or "gpt-5-nano",
                token=max(int(token or 0), 64 * 1024),
                temp=temp,
                think=effective_think,
                check_cache=check_cache,
                flex=flex,
                runtime_context=self.runtime_context,
                **kwargs,
            )
        self.impl.name = self.name

    def __getattr__(self, item):
        return getattr(self.impl, item)

    def cleanup(self):
        if hasattr(self.impl, "cleanup"):
            self.impl.cleanup()


Method = GRIDBackboneMethod
