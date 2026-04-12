# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

import GRID_backbone as v1
from single_pass_template import ToolsAskMethod, VLLMServerMethod
from shared_eval_backend import build_default_shared_backend, DEFAULT_SHARED_QWEN_MODEL_PATH


DEFAULT_RL_MODEL_PATH = v1.DEFAULT_RL_MODEL_PATH
PROMPT_VERSION = "20260328_step1_relaxed"


_STEP1_ENTITY_PROMPT_TEMPLATE_V2 = """You are an NLP model specialized in CTI knowledge graph extraction. This is Step 1 of a mandatory two-step extraction workflow.

                [TWO-STEP STEP1 V2 OVERRIDE | HIGH-RECALL CANDIDATE ENTITY MODE]
                Preserve the latest GRID 20260319 prompt below as the base instruction. In this step, keep all entity-related rules unchanged, but temporarily apply the following overrides:
                1. Output ONLY the reasoning section and the entity list.
                2. Do NOT output a relationship list in Step 1.
                3. Do NOT apply the final connectivity pruning in Step 1; all pruning is deferred to Step 2 after relationships are extracted.
                4. Step 1 is a HIGH-RECALL candidate mining stage, not a minimal final entity stage. When unsure between KEEP and DROP for a text-grounded candidate, KEEP it.
                5. Prefer over-inclusion for entities that may be needed by relations involving aliases, dates/times, quantities/money, victims/targets, research/analysis/disclosure, patches/releases, lures/social engineering, PoCs/exploits, infrastructure, commands/scripts/files, emails/domains/URLs, documents/publications, and other relation anchors.
                6. Keep relation-dependent supporting entities even if they seem secondary, one-off, abstract, or only weakly central. Step 2 will remove real noise later.
                7. Do NOT optimize for brevity or a “clean small entity list” in Step 1. A slightly noisy superset is preferred over a prematurely pruned list.
                8. Do NOT simplify, summarize, or weaken any other original rule.

__GRID_LATEST_PROMPT_BODY__

                [STEP1 OUTPUT OVERRIDE]
                Replace the Stage 3 output requirement with the following for this step only:
                First, output the reasoning section between #Reasoning_Start# and #Reasoning_End#.
                Second, output strictly one JSON list between #Entity_List_Start# and #Entity_List_End#.
                Do not output #Relationship_List_Start# / #Relationship_List_End# in Step 1.
                Do not include any extra text outside these markers.

                Input Text: __GRID_INPUT_TEXT__
""".replace("__GRID_LATEST_PROMPT_BODY__", v1._LATEST_GRID_PROMPT_BODY)


class _GRIDOursMixin(v1._GRIDBackboneMixin):

    def _create_entity_step_prompt(self, content: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "user",
                
                "content": _STEP1_ENTITY_PROMPT_TEMPLATE_V2.replace("__GRID_INPUT_TEXT__", str(content)),
            }
        ]

    def _create_prompt(self, content: str) -> Dict[str, Any]:
        preview_entities = [
            {"name": "Preview_Entity", "type": "other", "alias": ["Preview Alias"]},
            {"name": "Preview_Target", "type": "other"},
        ]
        return {
            "prompt_type": "GRID_Ours",
            "prompt_version": PROMPT_VERSION,
            "step1_entity_prompt": self._create_entity_step_prompt(content),
            "step2_entity_relation_prompt_template": self._create_relation_step_prompt(
                content,
                preview_entities,
            ),
        }


class _GRIDOursToolsAskMethod(_GRIDOursMixin, ToolsAskMethod):
    pass


class _GRIDOursVLLMServerMethod(_GRIDOursMixin, VLLMServerMethod):
    pass


class GRIDOursMethod:

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
        self.name = "GRID_Ours"

        
        effective_think = max(0, int(think or 0))

        shared_backend = dict(shared_llm_backend or {})
        if self.llm_backend == "shared_vllm" and not shared_backend:
            shared_backend = build_default_shared_backend(
                model_path=model or DEFAULT_SHARED_QWEN_MODEL_PATH,
            )

        if self.llm_backend == "shared_vllm":
            self.impl = _GRIDOursVLLMServerMethod(
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
            self.impl = _GRIDOursVLLMServerMethod(
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
            self.impl = _GRIDOursToolsAskMethod(
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


Method = GRIDOursMethod
