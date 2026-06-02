# -*- coding: utf-8 -*-
"""
Filename: verl_taskbank_reward.py
Description: Scripted task-bank reward used by the Docker VERL smoke path.
Keywords: GRID, VERL, reward, task-bank, Docker

The full paper also contains graph-level LLM-judge evaluation. This file is
intentionally narrower: it verifies that VERL can call a reusable local
task-bank reward during RL without an online LLM judge.
"""

from __future__ import annotations

import json
import re
from typing import Any


VALID_OPTIONS = {"A", "B", "C", "D"}


def _parse_option_list(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = value.get("ground_truth", value.get("expected_options", []))
    if isinstance(value, list):
        return [str(item).strip().upper() for item in value if str(item).strip().upper() in VALID_OPTIONS]
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return _parse_option_list(parsed)
    except Exception:
        pass
    return [item.upper() for item in re.findall(r"\b[A-D]\b", text.upper()) if item.upper() in VALID_OPTIONS]


def _parse_prediction(solution_str: str) -> list[str]:
    text = "" if solution_str is None else str(solution_str)
    marker_matches = list(re.finditer(r"####\s*(\[[^\]]*\])", text, flags=re.IGNORECASE | re.DOTALL))
    candidates = [match.group(1) for match in marker_matches]
    bracket_matches = list(re.finditer(r"(\[[^\]]*\])", text, flags=re.IGNORECASE | re.DOTALL))
    candidates.extend(match.group(1) for match in bracket_matches)
    for candidate in reversed(candidates):
        parsed = _parse_option_list(candidate)
        if parsed or candidate.strip() == "[]":
            return parsed
    return _parse_option_list(text)


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs) -> float:
    del data_source, kwargs
    expected = _parse_option_list(ground_truth)
    if not expected and isinstance(extra_info, dict):
        expected = _parse_option_list(extra_info.get("expected_options"))
    predicted = _parse_prediction("" if solution_str is None else str(solution_str))
    return 1.0 if set(predicted) == set(expected) else 0.0


def compute_score_kg(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs) -> float:
    return compute_score(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info, **kwargs)
