# kg_reward.py
import json
import re
import sys
import os
import argparse
import hashlib
import threading
import requests
import time
import random
import subprocess
import unicodedata
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio  
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import tools_prompt_nano as tools_prompt

try:
    import pytz
    PHOENIX_TZ = pytz.timezone('America/Phoenix')
except ImportError:
    pytz = None
    PHOENIX_TZ = None
    print("Warning: pytz not found. Timestamps will use local time.")
try:
    import json_repair
except ImportError:
    json_repair = None
    print("Warning: json_repair not found. Install it for better JSON parsing robustness.")


def _resolve_tools_prompt_attr(*candidate_names):
    if tools_prompt is None:
        return ""
    for name in candidate_names:
        if hasattr(tools_prompt, name):
            return getattr(tools_prompt, name)
    return ""


def _env_flag(name: str, default: str = "0") -> bool:
    value = str(os.environ.get(name, default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_json_maybe(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if json_repair is not None:
            try:
                return json_repair.loads(text)
            except Exception:
                pass
        try:
            return json.loads(text)
        except Exception:
            return None
    return None


def _normalize_extra_info(extra_info):
    if isinstance(extra_info, dict):
        return extra_info
    parsed = _parse_json_maybe(extra_info)
    return parsed if isinstance(parsed, dict) else {}


def _is_easyreward_non_llm_sample(extra_info: dict | None) -> bool:
    extra_info = _normalize_extra_info(extra_info)
    top_level_ability = str(extra_info.get("top_level_ability", "")).strip().lower()
    reward_mode = str(extra_info.get("step6_reward_mode", "")).strip().lower()
    top_level_data_source = str(extra_info.get("top_level_data_source", "")).strip().lower()
    rq3_step5a_reward_mode = str(extra_info.get("rq3_step5a_reward_mode", "")).strip().lower()
    return (
        top_level_ability == "step6_easyreward"
        or reward_mode == "mcq_xml"
        or rq3_step5a_reward_mode == "json_option_array_abcd"
        or ("easyreward" in top_level_data_source and "step6" in top_level_ability)
    )


def _is_regex_match_sample(extra_info: dict | None) -> bool:
    extra_info = _normalize_extra_info(extra_info)
    question_type = str(
        extra_info.get("question_type")
        or extra_info.get("step6_task_type")
        or extra_info.get("task_type")
        or ""
    ).strip().lower()
    regex_as_groundtruth = extra_info.get("regex_as_groundtruth")
    return question_type == "regex match" or regex_as_groundtruth not in (None, "", [], {})


def _is_easyreward_mcq_sample(extra_info: dict | None) -> bool:
    # Backward-compatible alias: step6 easyreward items are no longer only MCQ.
    return _is_easyreward_non_llm_sample(extra_info)


def _resolve_dataset_type(extra_info) -> str:
    extra_info = _normalize_extra_info(extra_info)
    if _is_easyreward_non_llm_sample(extra_info):
        return "train"
    dataset_type = str(extra_info.get("dataset_type", "")).strip().lower()
    if dataset_type in {"train", "test"}:
        return dataset_type
    if str(extra_info.get("test_variant", "")).strip().lower() == "real":
        return "test"
    return "train"


def _extract_easyreward_spec(ground_truth, extra_info: dict | None):
    extra_info = _normalize_extra_info(extra_info)
    rq3_step5a_reward_mode = str(extra_info.get("rq3_step5a_reward_mode", "")).strip().lower()
    if rq3_step5a_reward_mode == "json_option_array_abcd":
        parsed_ground_truth = _parse_json_maybe(ground_truth)
        if isinstance(parsed_ground_truth, list):
            normalized_options = []
            for item in parsed_ground_truth:
                option = str(item or "").strip().upper()
                if option in {"A", "B", "C", "D"} and option not in normalized_options:
                    normalized_options.append(option)
            return {
                "reward_mode": "json_option_array_abcd",
                "task_type": extra_info.get("rq3_question_group") or "step5a_mcq",
                "prompt_version": extra_info.get("rq3_prompt_version") or "step5a_legacy_abcd",
                "correct_option_list": normalized_options,
            }
    candidates = [
        ground_truth,
        extra_info.get("top_level_ground_truth"),
    ]
    for candidate in candidates:
        parsed = _parse_json_maybe(candidate)
        if isinstance(parsed, dict) and (
            parsed.get("correct_option")
            or parsed.get("reward_mode")
            or parsed.get("answer_verifier")
            or parsed.get("triple_regex_list")
            or parsed.get("entity_regex_list")
        ):
            return {
                "correct_option": str(parsed.get("correct_option", "")).strip().upper(),
                "reward_mode": parsed.get("reward_mode"),
                "task_type": parsed.get("task_type"),
                "prompt_version": parsed.get("prompt_version"),
                "result_wrapper_tag": parsed.get("result_wrapper_tag", "myresult"),
                "inner_result_tag": parsed.get("inner_result_tag", "mychoice"),
                "answer_verifier": parsed.get("answer_verifier"),
                "triple_regex_list": parsed.get("triple_regex_list") or [],
                "entity_regex_list": parsed.get("entity_regex_list") or [],
            }
    return None


def _extract_easyreward_choice(solution_str, result_wrapper_tag="myresult", inner_result_tag="mychoice"):
    solution_text = _strip_outer_think_blocks(solution_str)
    patterns = []
    if inner_result_tag:
        inner_tag = re.escape(str(inner_result_tag))
        patterns.append(rf"<{inner_tag}>\s*([A-Ja-j])\s*</{inner_tag}>")
    if result_wrapper_tag:
        wrapper_tag = re.escape(str(result_wrapper_tag))
        patterns.append(rf"<{wrapper_tag}>\s*([A-Ja-j])\s*</{wrapper_tag}>")
        if inner_result_tag:
            inner_tag = re.escape(str(inner_result_tag))
            patterns.append(rf"<{wrapper_tag}>.*?<{inner_tag}>\s*([A-Ja-j])\s*</{inner_tag}>.*?</{wrapper_tag}>")
    for pattern in patterns:
        match = re.search(pattern, solution_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).upper()
    stripped = solution_text.strip()
    if re.fullmatch(r"[A-Ja-j]", stripped):
        return stripped.upper()
    return None


def _extract_legacy_step5a_choice_list(solution_str):
    solution_text = "" if solution_str is None else str(solution_str)
    matches = list(re.finditer(r"####\s*(\[[^\]]*\])", solution_text, flags=re.IGNORECASE | re.DOTALL))
    if not matches:
        return []
    candidate = str(matches[-1].group(1) or "").strip()
    try:
        parsed = json.loads(candidate)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    normalized = []
    for item in parsed:
        option = str(item or "").strip().upper()
        if option in {"A", "B", "C", "D"} and option not in normalized:
            normalized.append(option)
    return normalized


def _extract_xml_tag_block(text, tag_name):
    if not tag_name:
        return None
    text = "" if text is None else str(text)
    tag_name = re.escape(str(tag_name))
    match = re.search(rf"<{tag_name}\b[^>]*>(.*?)</{tag_name}>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _extract_easyreward_result_payload(solution_str, result_wrapper_tag="myresult"):
    solution_text = _strip_outer_think_blocks(solution_str)
    wrapped_payload = _extract_xml_tag_block(solution_text, result_wrapper_tag)
    if wrapped_payload is not None:
        return wrapped_payload
    return solution_text.strip()


def _normalize_easyreward_answer_text(text):
    normalized = unicodedata.normalize("NFKC", "" if text is None else str(text))
    normalized = normalized.strip()
    while len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in "\"'`":
        normalized = normalized[1:-1].strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _build_easyreward_regex_flags(regex_flags: dict | None):
    flags = 0
    regex_flags = regex_flags if isinstance(regex_flags, dict) else {}
    if regex_flags.get("case_insensitive"):
        flags |= re.IGNORECASE
    if regex_flags.get("multiline"):
        flags |= re.MULTILINE
    if regex_flags.get("dotall"):
        flags |= re.DOTALL
    return flags


def _easyreward_regex_match_score(text, patterns, match_policy="match_any", regex_flags=None):
    text = _normalize_easyreward_answer_text(text)
    regex_flags_value = _build_easyreward_regex_flags(regex_flags)
    valid_patterns = [str(pattern) for pattern in (patterns or []) if str(pattern).strip()]
    if not valid_patterns:
        return 0.0, 0, 0

    matched_count = 0
    for pattern in valid_patterns:
        if re.search(pattern, text, flags=regex_flags_value):
            matched_count += 1

    normalized_policy = str(match_policy or "match_any").strip().lower()
    if normalized_policy == "match_all":
        score = 1.0 if matched_count == len(valid_patterns) else 0.0
    else:
        score = 1.0 if matched_count > 0 else 0.0
    return score, matched_count, len(valid_patterns)


def _easyreward_structured_regex_score(text, regex_specs, field_names):
    text = _normalize_easyreward_answer_text(text)
    specs = [spec for spec in (regex_specs or []) if isinstance(spec, dict)]
    if not specs:
        return 0.0, 0, 0

    matched_items = 0
    for spec in specs:
        item_ok = True
        for field_name in field_names:
            pattern = str(spec.get(field_name, "")).strip()
            if not pattern or re.search(pattern, text, flags=re.DOTALL) is None:
                item_ok = False
                break
        if item_ok:
            matched_items += 1

    return matched_items / len(specs), matched_items, len(specs)


def _prompt_requires_easyreward_format_bonus(extra_info: dict | None) -> bool:
    extra_info = _normalize_extra_info(extra_info)
    prompt_text = ""
    for key in ("top_level_prompt", "prompt"):
        value = extra_info.get(key)
        if value:
            prompt_text = str(value)
            break
    prompt_text_lower = prompt_text.lower()
    return "<mythink>" in prompt_text_lower and "<myresult>" in prompt_text_lower


def _solution_has_easyreward_format_tags(solution_str):
    solution_text = "" if solution_str is None else str(solution_str)
    has_mythink = re.search(r"<mythink\b[^>]*>.*?</mythink>", solution_text, flags=re.IGNORECASE | re.DOTALL) is not None
    has_myresult = re.search(
        r"<myresult\b[^>]*>.*?</myresult>",
        solution_text,
        flags=re.IGNORECASE | re.DOTALL,
    ) is not None
    return has_mythink, has_myresult


def _compute_easyreward_non_llm_score(solution_str, ground_truth, extra_info: dict | None):
    extra_info = _normalize_extra_info(extra_info)
    spec = _extract_easyreward_spec(ground_truth, extra_info)
    reward_mode = str((spec or {}).get("reward_mode", "")).strip().lower()
    result_wrapper_tag = (spec or {}).get("result_wrapper_tag", "myresult")
    inner_result_tag = (spec or {}).get("inner_result_tag", "mychoice")
    result_payload = _extract_easyreward_result_payload(solution_str, result_wrapper_tag=result_wrapper_tag)
    normalized_result_payload = _normalize_easyreward_answer_text(result_payload)

    chosen_option = None
    correct_option = None
    base_score = 0.0
    regex_matched_count = None
    regex_total_count = None
    structured_matched_count = None
    structured_total_count = None

    if spec is not None and reward_mode == "mcq_xml":
        chosen_option = _extract_easyreward_choice(
            solution_str,
            result_wrapper_tag=result_wrapper_tag,
            inner_result_tag=inner_result_tag,
        )
        correct_option = str(spec.get("correct_option", "")).strip().upper()
        if chosen_option and correct_option and chosen_option == correct_option:
            base_score = 1.0
    elif spec is not None and reward_mode == "json_option_array_abcd":
        predicted_option_list = _extract_legacy_step5a_choice_list(solution_str)
        gold_option_list = [
            str(item or "").strip().upper()
            for item in list(spec.get("correct_option_list") or [])
            if str(item or "").strip().upper() in {"A", "B", "C", "D"}
        ]
        predicted_option_set = set(predicted_option_list)
        gold_option_set = set(gold_option_list)
        if predicted_option_set == gold_option_set:
            base_score = 1.0
        elif predicted_option_set & gold_option_set:
            base_score = 0.5
        else:
            base_score = 0.0
    elif spec is not None and reward_mode == "regex_answer_verifier":
        verifier = spec.get("answer_verifier") if isinstance(spec.get("answer_verifier"), dict) else {}
        base_score, regex_matched_count, regex_total_count = _easyreward_regex_match_score(
            text=result_payload,
            patterns=verifier.get("regex_patterns"),
            match_policy=verifier.get("regex_match_policy", "match_any"),
            regex_flags=verifier.get("regex_flags"),
        )
    elif spec is not None and reward_mode == "triple_regex_extraction":
        base_score, structured_matched_count, structured_total_count = _easyreward_structured_regex_score(
            text=result_payload,
            regex_specs=spec.get("triple_regex_list"),
            field_names=("sub_regex", "rel_regex", "obj_regex"),
        )
    elif spec is not None and reward_mode == "entity_regex_extraction":
        base_score, structured_matched_count, structured_total_count = _easyreward_structured_regex_score(
            text=result_payload,
            regex_specs=spec.get("entity_regex_list"),
            field_names=("entity_regex",),
        )

    prompt_requires_format_bonus = _prompt_requires_easyreward_format_bonus(extra_info)
    has_mythink, has_myresult = _solution_has_easyreward_format_tags(solution_str)
    format_bonus = 0.1 if prompt_requires_format_bonus and has_mythink and has_myresult else 0.0
    score = min(1.0, max(0.0, float(base_score) + format_bonus))

    reward_eval_started_at, eval_details = _init_eval_details("easyreward_non_llm")
    eval_details["precision_score"] = score
    eval_details["recall_score"] = score
    eval_details["f1_score"] = score
    eval_details["easyreward_non_llm"] = True
    eval_details["easyreward_reward_mode"] = reward_mode or None
    eval_details["easyreward_task_type"] = (spec or {}).get("task_type")
    eval_details["easyreward_prompt_version"] = (spec or {}).get("prompt_version")
    eval_details["easyreward_result_wrapper_tag"] = result_wrapper_tag
    eval_details["easyreward_inner_result_tag"] = inner_result_tag
    eval_details["easyreward_chosen_option"] = chosen_option
    eval_details["easyreward_correct_option"] = correct_option
    if spec is not None and reward_mode == "json_option_array_abcd":
        eval_details["easyreward_predicted_option_list"] = _extract_legacy_step5a_choice_list(solution_str)
        eval_details["easyreward_gold_option_list"] = list(spec.get("correct_option_list") or [])
    eval_details["easyreward_base_score_before_format_bonus"] = float(base_score)
    eval_details["easyreward_format_bonus"] = format_bonus
    eval_details["easyreward_prompt_requires_format_bonus"] = prompt_requires_format_bonus
    eval_details["easyreward_solution_has_mythink_tag"] = has_mythink
    eval_details["easyreward_solution_has_myresult_tag"] = has_myresult
    eval_details["easyreward_solution_has_required_format_tags"] = has_mythink and has_myresult
    eval_details["easyreward_result_payload_preview"] = normalized_result_payload[:512]
    if regex_matched_count is not None:
        eval_details["easyreward_regex_matched_count"] = regex_matched_count
        eval_details["easyreward_regex_total_count"] = regex_total_count
    if structured_matched_count is not None:
        eval_details["easyreward_structured_matched_count"] = structured_matched_count
        eval_details["easyreward_structured_total_count"] = structured_total_count
    return score, eval_details, reward_eval_started_at


def _normalize_regex_groundtruth_list(extra_info: dict | None):
    extra_info = _normalize_extra_info(extra_info)
    regex_list = extra_info.get("regex_as_groundtruth")
    if hasattr(regex_list, "tolist"):
        try:
            regex_list = regex_list.tolist()
        except Exception:
            pass
    parsed = _parse_json_maybe(regex_list)
    if isinstance(parsed, list):
        regex_list = parsed
    if isinstance(regex_list, tuple):
        regex_list = list(regex_list)
    if not isinstance(regex_list, list):
        return []
    normalized = []
    for item in regex_list:
        if item is None:
            continue
        pattern = str(item).strip()
        if pattern:
            normalized.append(pattern)
    return normalized


def _normalize_triple_regex_groundtruth_list(extra_info: dict | None):
    extra_info = _normalize_extra_info(extra_info)
    regex_list = extra_info.get("regex_as_groundtruth")
    if hasattr(regex_list, "tolist"):
        try:
            regex_list = regex_list.tolist()
        except Exception:
            pass
    parsed = _parse_json_maybe(regex_list)
    if isinstance(parsed, list):
        regex_list = parsed
    if isinstance(regex_list, tuple):
        regex_list = list(regex_list)
    if not isinstance(regex_list, list):
        return []

    normalized = []
    for item in regex_list:
        if not isinstance(item, dict):
            continue
        sub_regex = str(item.get("sub_regex", "")).strip()
        rel_regex = str(item.get("rel_regex", "")).strip()
        obj_regex = str(item.get("obj_regex", "")).strip()
        if not sub_regex or not rel_regex or not obj_regex:
            continue
        normalized.append(
            {
                "sub_regex": sub_regex,
                "rel_regex": rel_regex,
                "obj_regex": obj_regex,
                "confidence": str(item.get("confidence", "")).strip(),
            }
        )
    return normalized


def _extract_predicted_triples(solution_str):
    raw_pred_text = _strip_outer_think_blocks(solution_str)
    relation_block = _extract_marker_block(
        raw_pred_text,
        "#Relationship_List_Start#",
        "#Relationship_List_End#",
    )
    if relation_block is None:
        return []
    try:
        relation_items = parse_kg_block_json(relation_block)
    except Exception:
        return []
    triples = []
    if isinstance(relation_items, list):
        for item in relation_items:
            if isinstance(item, dict):
                sub_text = str(item.get("sub", "")).strip()
                rel_text = str(item.get("rel", "")).strip()
                obj_text = str(item.get("obj", "")).strip()
                if sub_text or rel_text or obj_text:
                    triples.append(
                        {
                            "sub": sub_text,
                            "rel": rel_text,
                            "obj": obj_text,
                        }
                    )
    return triples


def _extract_predicted_relation_strings(solution_str):
    triples = _extract_predicted_triples(solution_str)
    relation_strings = []
    for item in triples:
        rel_text = str(item.get("rel", "")).strip()
        if rel_text:
            relation_strings.append(rel_text)
    return relation_strings


def _safe_regex_search(pattern, text):
    try:
        return re.search(pattern, text) is not None, False
    except re.error:
        return False, True


def _compute_tripregex_match_score(solution_str, extra_info: dict | None):
    extra_info = _normalize_extra_info(extra_info)
    regex_specs = _normalize_triple_regex_groundtruth_list(extra_info)
    predicted_triples = _extract_predicted_triples(solution_str)

    hits = 0
    invalid_regex_count = 0
    matched_regex_indexes = []
    invalid_regex_indexes = []
    matched_predicted_triple_indexes = []

    for idx, spec in enumerate(regex_specs):
        matched = False
        spec_invalid = False
        matched_triple_idx = None
        for triple_idx, triple in enumerate(predicted_triples):
            sub_ok, sub_invalid = _safe_regex_search(spec.get("sub_regex", ""), str(triple.get("sub", "")))
            rel_ok, rel_invalid = _safe_regex_search(spec.get("rel_regex", ""), str(triple.get("rel", "")))
            obj_ok, obj_invalid = _safe_regex_search(spec.get("obj_regex", ""), str(triple.get("obj", "")))
            if sub_invalid or rel_invalid or obj_invalid:
                spec_invalid = True
                continue
            if sub_ok and rel_ok and obj_ok:
                matched = True
                matched_triple_idx = triple_idx
                break
        if spec_invalid:
            invalid_regex_count += 1
            invalid_regex_indexes.append(idx)
        if matched:
            hits += 1
            matched_regex_indexes.append(idx)
            if matched_triple_idx is not None:
                matched_predicted_triple_indexes.append(matched_triple_idx)

    total = len(regex_specs)
    score = (hits / total) if total else 0.0

    reward_eval_started_at, eval_details = _init_eval_details("regex_match_triple_list")
    eval_details["precision_score"] = score
    eval_details["recall_score"] = score
    eval_details["f1_score"] = score
    eval_details["regex_match_non_llm"] = True
    eval_details["regex_match_question_type"] = str(
        extra_info.get("question_type") or extra_info.get("step6_task_type") or "regex match"
    )
    eval_details["regex_groundtruth_kind"] = "triple_regex_list"
    eval_details["regex_match_groundtruth_count"] = total
    eval_details["regex_match_pred_triple_count"] = len(predicted_triples)
    eval_details["regex_match_hit_count"] = hits
    eval_details["regex_match_invalid_regex_count"] = invalid_regex_count
    eval_details["regex_match_invalid_regex_indexes"] = invalid_regex_indexes[:128]
    eval_details["regex_match_matched_regex_indexes"] = matched_regex_indexes[:256]
    eval_details["regex_match_matched_pred_triple_indexes"] = matched_predicted_triple_indexes[:256]
    eval_details["regex_match_pred_triple_preview"] = predicted_triples[:16]
    eval_details["regex_match_groundtruth_preview"] = regex_specs[:16]
    return score, eval_details, reward_eval_started_at


def _compute_regex_match_score(solution_str, extra_info: dict | None):
    extra_info = _normalize_extra_info(extra_info)
    groundtruth_kind = str(extra_info.get("regex_groundtruth_kind") or "").strip()
    raw_regex_list = extra_info.get("regex_as_groundtruth")
    parsed_raw_regex_list = _parse_json_maybe(raw_regex_list)
    first_non_null = None
    if isinstance(parsed_raw_regex_list, list):
        raw_regex_list = parsed_raw_regex_list
    if isinstance(raw_regex_list, (list, tuple)):
        for item in raw_regex_list:
            if item is not None:
                first_non_null = item
                break
    if (
        groundtruth_kind == "triple_regex_list"
        or isinstance(first_non_null, dict)
        and any(key in first_non_null for key in ("sub_regex", "rel_regex", "obj_regex"))
    ):
        return _compute_tripregex_match_score(solution_str, extra_info)

    regex_list = _normalize_regex_groundtruth_list(extra_info)
    predicted_relations = _extract_predicted_relation_strings(solution_str)

    hits = 0
    invalid_regex_count = 0
    matched_regex_indexes = []
    invalid_regex_indexes = []
    for idx, pattern in enumerate(regex_list):
        try:
            matched = any(re.search(pattern, rel_text) for rel_text in predicted_relations)
        except re.error:
            matched = False
            invalid_regex_count += 1
            invalid_regex_indexes.append(idx)
        if matched:
            hits += 1
            matched_regex_indexes.append(idx)

    total = len(regex_list)
    score = (hits / total) if total else 0.0

    reward_eval_started_at, eval_details = _init_eval_details("regex_match_relation_list")
    eval_details["precision_score"] = score
    eval_details["recall_score"] = score
    eval_details["f1_score"] = score
    eval_details["regex_match_non_llm"] = True
    eval_details["regex_match_question_type"] = str(
        extra_info.get("question_type") or extra_info.get("step6_task_type") or "regex match"
    )
    eval_details["regex_match_groundtruth_count"] = total
    eval_details["regex_match_pred_relation_count"] = len(predicted_relations)
    eval_details["regex_match_hit_count"] = hits
    eval_details["regex_match_invalid_regex_count"] = invalid_regex_count
    eval_details["regex_match_invalid_regex_indexes"] = invalid_regex_indexes[:128]
    eval_details["regex_match_matched_regex_indexes"] = matched_regex_indexes[:256]
    eval_details["regex_match_pred_relation_preview"] = predicted_relations[:32]
    eval_details["regex_match_groundtruth_preview"] = regex_list[:32]
    return score, eval_details, reward_eval_started_at


def _phoenix_now():
    return datetime.now(PHOENIX_TZ) if PHOENIX_TZ else datetime.now()


def _phoenix_iso(dt_obj):
    return dt_obj.isoformat() if dt_obj is not None else None


def _init_eval_details(eval_mode):
    reward_eval_started_at = _phoenix_now()
    eval_details = {
        "precision_prompt": None,
        "precision_response": None,
        "precision_score": None,
        "recall_prompt": None,
        "recall_response": None,
        "recall_score": None,
        "f1_score": None,
        "eval_mode": eval_mode,
        "reward_eval_date_phoenix": reward_eval_started_at.date().isoformat(),
        "reward_eval_started_at_phoenix": _phoenix_iso(reward_eval_started_at),
        "reward_eval_finished_at_phoenix": None,
        "reward_eval_wall_seconds": None,
        "precision_llm_request_started_at_phoenix": None,
        "precision_llm_response_received_at_phoenix": None,
        "precision_score_calc_started_at_phoenix": None,
        "precision_llm_latency_seconds": None,
        "recall_llm_request_started_at_phoenix": None,
        "recall_llm_response_received_at_phoenix": None,
        "recall_score_calc_started_at_phoenix": None,
        "recall_llm_latency_seconds": None,
    }
    return reward_eval_started_at, eval_details


def _record_eval_branch_timing(eval_details, branch_name, request_started_at, response_received_at):
    eval_details[f"{branch_name}_llm_request_started_at_phoenix"] = _phoenix_iso(request_started_at)
    eval_details[f"{branch_name}_llm_response_received_at_phoenix"] = _phoenix_iso(response_received_at)
    eval_details[f"{branch_name}_score_calc_started_at_phoenix"] = _phoenix_iso(response_received_at)
    if request_started_at is not None and response_received_at is not None:
        eval_details[f"{branch_name}_llm_latency_seconds"] = round(
            (response_received_at - request_started_at).total_seconds(), 6
        )


def _finalize_eval_details(eval_details, reward_eval_started_at):
    reward_eval_finished_at = _phoenix_now()
    eval_details["reward_eval_finished_at_phoenix"] = _phoenix_iso(reward_eval_finished_at)
    if reward_eval_started_at is not None:
        eval_details["reward_eval_wall_seconds"] = round(
            (reward_eval_finished_at - reward_eval_started_at).total_seconds(), 6
        )


# 2026-03-13:


KG_REWARD_SAVE_DEBUG_PARQUET = _env_flag("KG_REWARD_SAVE_DEBUG_PARQUET", "0")
KG_REWARD_VERBOSE_DEBUG = _env_flag("KG_REWARD_VERBOSE_DEBUG", "0")
VERL_INPUT_PARQUET_SNAPSHOT_KEY = "__verl_input_parquet_snapshot__"
VERL_RUNTIME_GENERATED_KEY = "__verl_runtime_generated__"
VERL_INTERNAL_EXTRA_INFO_KEYS = {
    VERL_INPUT_PARQUET_SNAPSHOT_KEY,
    VERL_RUNTIME_GENERATED_KEY,
}
VERL_RUNTIME_ONLY_EXTRA_INFO_KEYS = {
    "num_turns",
    "rollout_reward_scores",
}
FIELD_SOURCE_SUFFIXES = {
    "input_parquet": "__input_parquet",
    "rollout_model": "__rollout_model",
    "llm_judge_model": "__llm_judge_model",
    "verl": "__verl",
}


_DEFAULT_JUDGE_PROMPT_BUNDLE = tools_prompt.get_judge_prompt_bundle("grid_judge_fav")
GRID_PRECISION_PROMPT_TEXT = _DEFAULT_JUDGE_PROMPT_BUNDLE.get("precision_prompt") or ""
GRID_RECALL_PROMPT_TEXT = _DEFAULT_JUDGE_PROMPT_BUNDLE.get("recall_prompt") or ""
GRID_PRECISION_PROMPT_TEXT_WITH_INDEX = GRID_PRECISION_PROMPT_TEXT
GRID_RECALL_PROMPT_TEXT_WITH_INDEX = GRID_RECALL_PROMPT_TEXT


def _resolve_grid_prompt_pair_from_env():
    prompt_mode = str(
        os.environ.get("KG_REWARD_TEST_PROMPT_MODE")
        or os.environ.get("KG_REWARD_PROMPT_MODE")
        or ""
    ).strip().lower()
    if prompt_mode == "grid_judge_fav":
        precision_text = GRID_PRECISION_PROMPT_TEXT_WITH_INDEX or GRID_PRECISION_PROMPT_TEXT
        recall_text = GRID_RECALL_PROMPT_TEXT_WITH_INDEX or GRID_RECALL_PROMPT_TEXT
        return precision_text, recall_text
    return GRID_PRECISION_PROMPT_TEXT, GRID_RECALL_PROMPT_TEXT


GRID_PRECISION_PROMPT_TEXT, GRID_RECALL_PROMPT_TEXT = _resolve_grid_prompt_pair_from_env()

# Import openai for model name checking
try:
    from openai import OpenAI
except ImportError:
    print("Warning: OpenAI library not found. dynamic model name fetching might fail.")
    OpenAI = None

# Import the shared API helper dynamically
import importlib.util


try:
    api_keys_dir = os.path.join(grandparent_dir, "APIKeys")
    tools_mini_path = os.path.join(api_keys_dir, "tools_mini_ver_in_verl.py")
    if os.path.exists(tools_mini_path):
        spec = importlib.util.spec_from_file_location("tools_mini", tools_mini_path)
        tools_mini = importlib.util.module_from_spec(spec)
        sys.modules["tools_mini"] = tools_mini
        spec.loader.exec_module(tools_mini)
        print("✅ Successfully imported tools_mini_ver_in_verl.py (unified API)")
        
        use_xy_api = tools_mini
        use_nv_api = tools_mini
    else:
        print(f"⚠️ Warning: tools_mini_ver_in_verl.py not found at {tools_mini_path}")
        tools_mini = None
        use_xy_api = None
        use_nv_api = None
except Exception as e:
    print(f"❌ Error importing tools_mini_ver_in_verl.py: {e}")
    tools_mini = None
    use_xy_api = None
    use_nv_api = None

# ==========================================

# ==========================================
_MACHINE_CODE_CACHE = None

def get_machine_code():
    global _MACHINE_CODE_CACHE
    if _MACHINE_CODE_CACHE is not None:
        return _MACHINE_CODE_CACHE
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            _MACHINE_CODE_CACHE = "Unknown"
            return _MACHINE_CODE_CACHE
        
        gpu_names = result.stdout.strip().split('\n')
        gpu_count = len(gpu_names)
        gpu_model = gpu_names[0] if gpu_names else ""
        
        if gpu_count == 4 and "RTX 6000 Ada" in gpu_model:
            _MACHINE_CODE_CACHE = "Ultra"
        elif gpu_count == 2 and "RTX 6000 Ada" in gpu_model:
            _MACHINE_CODE_CACHE = "Super"
        elif gpu_count == 2 and "RTX A6000" in gpu_model:
            _MACHINE_CODE_CACHE = "Normal"
        else:
            _MACHINE_CODE_CACHE = f"Unknown_{gpu_count}x{gpu_model[:20]}"
    except Exception:
        _MACHINE_CODE_CACHE = "Unknown"
    
    return _MACHINE_CODE_CACHE

# ==========================================

# ==========================================
try:
    import tiktoken
    _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKEN_ENCODING = None


def _tokenlen_for_log(text):
    text = "" if text is None else str(text)
    if _TOKEN_ENCODING is not None:
        try:
            return len(_TOKEN_ENCODING.encode(text, disallowed_special=()))
        except Exception:
            pass
    return len(text)


def _md5_text(text):
    text = "" if text is None else str(text)
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _normalize_jsonable(obj):
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [_normalize_jsonable(x) for x in obj.tolist()]
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _normalize_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_normalize_jsonable(v) for v in obj]
    return obj


def _prompt_messages_to_text(prompt_obj):
    normalized = _normalize_jsonable(prompt_obj)
    if isinstance(normalized, list):
        parts = []
        for item in normalized:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts), len(normalized)
    return str(normalized), 1 if normalized else 0


def _extract_source_dataset(extra_info):
    if not isinstance(extra_info, dict):
        return None
    source_file = str(extra_info.get("source_file", "") or "")
    if ":" in source_file:
        return source_file.split(":", 1)[0]
    return source_file or None


def _clean_extra_info_for_input_snapshot(extra_info):
    if not isinstance(extra_info, dict):
        return {}
    cleaned = {}
    for key, value in extra_info.items():
        if key in VERL_INTERNAL_EXTRA_INFO_KEYS or key in VERL_RUNTIME_ONLY_EXTRA_INFO_KEYS:
            continue
        cleaned[str(key)] = _normalize_jsonable(value)
    return cleaned


def _build_fallback_input_parquet_snapshot(data_source, ground_truth, extra_info):
    extra_info = extra_info if isinstance(extra_info, dict) else {}
    reward_model_snapshot = extra_info.get("top_level_reward_model")
    if reward_model_snapshot is None:
        reward_model_snapshot = {"ground_truth": extra_info.get("top_level_ground_truth", ground_truth)}
    return {
        "prompt": _normalize_jsonable(extra_info.get("top_level_prompt")),
        "data_source": extra_info.get("top_level_data_source", data_source),
        "ability": extra_info.get("top_level_ability"),
        "ground_truth": extra_info.get("top_level_ground_truth", ground_truth),
        "extra_info": _clean_extra_info_for_input_snapshot(extra_info),
        "reward_model": _normalize_jsonable(reward_model_snapshot),
        "sft_ground_truth": extra_info.get("top_level_sft_ground_truth"),
    }


def _extract_reward_debug_payloads(data_source, ground_truth, extra_info):
    extra_info = extra_info if isinstance(extra_info, dict) else {}
    input_parquet_snapshot = _normalize_jsonable(extra_info.get(VERL_INPUT_PARQUET_SNAPSHOT_KEY))
    runtime_generated_snapshot = _normalize_jsonable(extra_info.get(VERL_RUNTIME_GENERATED_KEY))

    if not isinstance(input_parquet_snapshot, dict):
        input_parquet_snapshot = _build_fallback_input_parquet_snapshot(data_source, ground_truth, extra_info)

    if not isinstance(runtime_generated_snapshot, dict):
        runtime_generated_snapshot = {}
        if "num_turns" in extra_info:
            runtime_generated_snapshot["num_turns"] = _normalize_jsonable(extra_info.get("num_turns"))
        if "rollout_reward_scores" in extra_info:
            runtime_generated_snapshot["rollout_reward_scores"] = _normalize_jsonable(extra_info.get("rollout_reward_scores"))

    return input_parquet_snapshot, runtime_generated_snapshot


def _suffix_key(key, source_name):
    return f"{key}{FIELD_SOURCE_SUFFIXES[source_name]}"


def _suffix_flat_dict(flat_dict, source_name):
    return {
        _suffix_key(str(key), source_name): _normalize_jsonable(value)
        for key, value in (flat_dict or {}).items()
    }


def _build_origin_payloads(mode, solution_str, eval_details, runtime_generated_snapshot):
    eval_details = eval_details or {}
    rollout_model_payload = {
        "solution_str": "" if solution_str is None else str(solution_str),
    }
    llm_judge_model_payload = {
        "precision_llm_response": eval_details.get("precision_response"),
        "recall_llm_response": eval_details.get("recall_response"),
    }
    verl_payload = {
        "dataset_type": mode,
        "eval_mode_json": _json_compact(eval_details.get("eval_mode")),
        "precision_prompt": eval_details.get("precision_prompt"),
        "precision_score": eval_details.get("precision_score"),
        "recall_prompt": eval_details.get("recall_prompt"),
        "recall_score": eval_details.get("recall_score"),
        "f1_score": eval_details.get("f1_score"),
        "reward_eval_date_phoenix": eval_details.get("reward_eval_date_phoenix"),
        "reward_eval_started_at_phoenix": eval_details.get("reward_eval_started_at_phoenix"),
        "reward_eval_finished_at_phoenix": eval_details.get("reward_eval_finished_at_phoenix"),
        "reward_eval_wall_seconds": eval_details.get("reward_eval_wall_seconds"),
        "precision_llm_request_started_at_phoenix": eval_details.get("precision_llm_request_started_at_phoenix"),
        "precision_llm_response_received_at_phoenix": eval_details.get("precision_llm_response_received_at_phoenix"),
        "precision_score_calc_started_at_phoenix": eval_details.get("precision_score_calc_started_at_phoenix"),
        "precision_llm_latency_seconds": eval_details.get("precision_llm_latency_seconds"),
        "recall_llm_request_started_at_phoenix": eval_details.get("recall_llm_request_started_at_phoenix"),
        "recall_llm_response_received_at_phoenix": eval_details.get("recall_llm_response_received_at_phoenix"),
        "recall_score_calc_started_at_phoenix": eval_details.get("recall_score_calc_started_at_phoenix"),
        "recall_llm_latency_seconds": eval_details.get("recall_llm_latency_seconds"),
        "verl_runtime_generated": _normalize_jsonable(runtime_generated_snapshot),
    }
    return rollout_model_payload, llm_judge_model_payload, verl_payload


def _build_field_origin_dictionary(
    input_parquet_snapshot,
    rollout_model_payload,
    llm_judge_model_payload,
    verl_payload,
):
    input_parquet_snapshot = input_parquet_snapshot if isinstance(input_parquet_snapshot, dict) else {}
    rollout_model_payload = rollout_model_payload if isinstance(rollout_model_payload, dict) else {}
    llm_judge_model_payload = llm_judge_model_payload if isinstance(llm_judge_model_payload, dict) else {}
    verl_payload = verl_payload if isinstance(verl_payload, dict) else {}

    input_extra_info = input_parquet_snapshot.get("extra_info")
    reward_model_snapshot = input_parquet_snapshot.get("reward_model")
    verl_runtime_snapshot = verl_payload.get("verl_runtime_generated")
    return {
        "input_parquet": {
            "suffix": FIELD_SOURCE_SUFFIXES["input_parquet"],
            "top_level_keys": sorted(str(key) for key, value in input_parquet_snapshot.items() if value is not None),
            "nested_keys": {
                "extra_info": (
                    sorted(str(key) for key in input_extra_info.keys()) if isinstance(input_extra_info, dict) else []
                ),
                "reward_model": (
                    sorted(str(key) for key in reward_model_snapshot.keys())
                    if isinstance(reward_model_snapshot, dict)
                    else []
                ),
            },
        },
        "rollout_model": {
            "suffix": FIELD_SOURCE_SUFFIXES["rollout_model"],
            "keys": sorted(str(key) for key, value in rollout_model_payload.items() if value is not None),
        },
        "llm_judge_model": {
            "suffix": FIELD_SOURCE_SUFFIXES["llm_judge_model"],
            "keys": sorted(str(key) for key, value in llm_judge_model_payload.items() if value is not None),
        },
        "verl": {
            "suffix": FIELD_SOURCE_SUFFIXES["verl"],
            "keys": sorted(
                str(key)
                for key, value in verl_payload.items()
                if key != "verl_runtime_generated" and value is not None
            ),
            "nested_keys": {
                "verl_runtime_generated": (
                    sorted(str(key) for key in verl_runtime_snapshot.keys())
                    if isinstance(verl_runtime_snapshot, dict)
                    else []
                ),
            },
        },
    }


def _strip_outer_think_blocks(solution_str):
    solution_str = "" if solution_str is None else str(solution_str)
    if "</mythink>" in solution_str:
        return solution_str.split("</mythink>", 1)[-1].strip()
    if "</think>" in solution_str:
        return solution_str.split("</think>", 1)[-1].strip()
    return solution_str


def _extract_marker_block(text, start_marker, end_marker):
    text = "" if text is None else str(text)
    pattern = re.compile(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker), re.DOTALL)
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _extract_structure_stats(full_text, is_solution=False):
    raw_text = _strip_outer_think_blocks(full_text) if is_solution else ("" if full_text is None else str(full_text))

    reasoning_block = _extract_marker_block(raw_text, "#Reasoning_Start#", "#Reasoning_End#")
    entity_block = _extract_marker_block(raw_text, "#Entity_List_Start#", "#Entity_List_End#")
    relation_block = _extract_marker_block(raw_text, "#Relationship_List_Start#", "#Relationship_List_End#")

    entity_count = 0
    relation_count = 0
    entity_parse_ok = False
    relation_parse_ok = False

    if entity_block is not None:
        try:
            entity_data = parse_kg_block_json(entity_block)
            if isinstance(entity_data, list):
                entity_count = len(entity_data)
                entity_parse_ok = True
        except Exception:
            entity_parse_ok = False

    if relation_block is not None:
        try:
            relation_data = parse_kg_block_json(relation_block)
            if isinstance(relation_data, list):
                relation_count = len(relation_data)
                relation_parse_ok = True
        except Exception:
            relation_parse_ok = False

    return {
        "full_chars": len(raw_text),
        "full_tokens": _tokenlen_for_log(raw_text),
        "full_md5": _md5_text(raw_text),
        "has_reasoning_block": reasoning_block is not None,
        "has_entity_block": entity_block is not None,
        "has_relationship_block": relation_block is not None,
        "reasoning_chars": len(reasoning_block or ""),
        "reasoning_tokens": _tokenlen_for_log(reasoning_block or ""),
        "entity_block_chars": len(entity_block or ""),
        "entity_block_tokens": _tokenlen_for_log(entity_block or ""),
        "relationship_block_chars": len(relation_block or ""),
        "relationship_block_tokens": _tokenlen_for_log(relation_block or ""),
        "entity_parse_ok": entity_parse_ok,
        "relationship_parse_ok": relation_parse_ok,
        "entity_count": entity_count,
        "relationship_count": relation_count,
    }


def _summarize_large_extra_info_value(key, value):
    normalized = _normalize_jsonable(value)
    if key == "top_level_prompt":
        prompt_text, message_count = _prompt_messages_to_text(normalized)
        return {
            "kind": "prompt_messages",
            "message_count": message_count,
            "chars": len(prompt_text),
            "tokens": _tokenlen_for_log(prompt_text),
            "md5": _md5_text(prompt_text),
        }
    if key == "graph_from_text_raw_from_file":
        summary = {
            "kind": "graph",
            "chars": len(json.dumps(normalized, ensure_ascii=False)),
            "tokens": _tokenlen_for_log(json.dumps(normalized, ensure_ascii=False)),
            "md5": _md5_text(json.dumps(normalized, ensure_ascii=False)),
        }
        if isinstance(normalized, dict):
            ents = normalized.get("entity") or []
            rels = normalized.get("relationship") or []
            summary["entity_count"] = len(ents) if isinstance(ents, list) else None
            summary["relationship_count"] = len(rels) if isinstance(rels, list) else None
        return summary
    text = json.dumps(normalized, ensure_ascii=False) if not isinstance(normalized, str) else normalized
    return {
        "kind": type(normalized).__name__,
        "chars": len(text),
        "tokens": _tokenlen_for_log(text),
        "md5": _md5_text(text),
    }


def _build_compact_extra_info(extra_info):
    if not isinstance(extra_info, dict):
        return {}, {}, []

    compact = {}
    large_field_summaries = {}
    omitted_fields = []

    for key, value in extra_info.items():
        normalized = _normalize_jsonable(value)
        is_large_key = key in {
            "top_level_prompt",
            "top_level_ground_truth",
            "top_level_sft_ground_truth",
            "graph_from_text_raw_from_file",
            "text_fixed_by_revision",
            "text_raw_from_file",
            VERL_INPUT_PARQUET_SNAPSHOT_KEY,
            VERL_RUNTIME_GENERATED_KEY,
        }
        is_large_value = (
            isinstance(normalized, (list, dict))
            or (isinstance(normalized, str) and len(normalized) > 512)
        )
        if is_large_key or is_large_value:
            omitted_fields.append(key)
            large_field_summaries[key] = _summarize_large_extra_info_value(key, normalized)
        else:
            compact[key] = normalized

    return compact, large_field_summaries, omitted_fields


def _build_analysis_snapshot(data_source, solution_str, ground_truth, extra_info, mode, eval_details):
    extra_info = extra_info if isinstance(extra_info, dict) else {}
    compact_extra_info, large_field_summaries, omitted_fields = _build_compact_extra_info(extra_info)
    input_parquet_snapshot, runtime_generated_snapshot = _extract_reward_debug_payloads(
        data_source=data_source,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )

    article_text = extra_info.get("text_fixed_by_revision", "") or ""
    raw_prompt_obj = extra_info.get("top_level_prompt")
    prompt_text, prompt_message_count = _prompt_messages_to_text(raw_prompt_obj) if raw_prompt_obj is not None else ("", 0)

    precision_prompt = (eval_details or {}).get("precision_prompt") or ""
    precision_response = (eval_details or {}).get("precision_response") or ""
    recall_prompt = (eval_details or {}).get("recall_prompt") or ""
    recall_response = (eval_details or {}).get("recall_response") or ""
    rollout_model_payload, llm_judge_model_payload, verl_payload = _build_origin_payloads(
        mode=mode,
        solution_str=solution_str,
        eval_details=eval_details or {},
        runtime_generated_snapshot=runtime_generated_snapshot,
    )
    reward_output_payload = {
        **rollout_model_payload,
        **llm_judge_model_payload,
        **{key: value for key, value in verl_payload.items() if key != "verl_runtime_generated"},
    }
    field_origin_dictionary = _build_field_origin_dictionary(
        input_parquet_snapshot=input_parquet_snapshot,
        rollout_model_payload=rollout_model_payload,
        llm_judge_model_payload=llm_judge_model_payload,
        verl_payload=verl_payload,
    )

    return {
        "routing": {
            "dataset_type": mode,
            "source_dataset": _extract_source_dataset(extra_info),
            "data_source": data_source,
            "eval_mode": (eval_details or {}).get("eval_mode"),
        },
        "lookup_keys": {
            "stable_article_id": extra_info.get("stable_article_id"),
            "source_file": extra_info.get("source_file"),
            "original_id": extra_info.get("original_id"),
            "index": _normalize_jsonable(extra_info.get("index")),
            "sample_order": _normalize_jsonable(extra_info.get("sample_order")),
            "test_variant": extra_info.get("test_variant"),
        },
        "prompt_stats": {
            "message_count": prompt_message_count,
            "chars": len(prompt_text),
            "tokens": _tokenlen_for_log(prompt_text),
            "md5": _md5_text(prompt_text),
        },
        "article_text_stats": {
            "chars": len(article_text),
            "tokens": _tokenlen_for_log(article_text),
            "md5": _md5_text(article_text),
        },
        "ground_truth_stats": _extract_structure_stats(ground_truth, is_solution=False),
        "prediction_stats": _extract_structure_stats(solution_str, is_solution=True),
        "judge_stats": {
            "precision_prompt_chars": len(precision_prompt),
            "precision_prompt_tokens": _tokenlen_for_log(precision_prompt),
            "precision_response_chars": len(str(precision_response or "")),
            "precision_response_tokens": _tokenlen_for_log(precision_response or ""),
            "recall_prompt_chars": len(recall_prompt),
            "recall_prompt_tokens": _tokenlen_for_log(recall_prompt),
            "recall_response_chars": len(str(recall_response or "")),
            "recall_response_tokens": _tokenlen_for_log(recall_response or ""),
            "precision_llm_latency_seconds": (eval_details or {}).get("precision_llm_latency_seconds"),
            "recall_llm_latency_seconds": (eval_details or {}).get("recall_llm_latency_seconds"),
        },
        "raw_payloads": {
            "input_parquet": input_parquet_snapshot,
            "rollout_model": rollout_model_payload,
            "llm_judge_model": llm_judge_model_payload,
            "verl": verl_payload,
            "reward_outputs_combined": reward_output_payload,
        },
        "field_origin_dictionary": field_origin_dictionary,
        "extra_info_compact": compact_extra_info,
        "extra_info_large_field_summaries": large_field_summaries,
        "extra_info_omitted_large_fields": omitted_fields,
    }


def _json_compact(obj):
    return json.dumps(_normalize_jsonable(obj), ensure_ascii=False, separators=(",", ":"))


def _flatten_dict(data, parent_key="", sep="__"):
    items = {}
    for key, value in (data or {}).items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(_flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = _normalize_jsonable(value)
    return items


def _build_parquet_record(data_source, solution_str, ground_truth, extra_info, mode, timestamp, machine_code, eval_details, analysis_snapshot):
    routing = analysis_snapshot.get("routing", {})
    lookup = analysis_snapshot.get("lookup_keys", {})
    prompt_stats = analysis_snapshot.get("prompt_stats", {})
    article_stats = analysis_snapshot.get("article_text_stats", {})
    gt_stats = analysis_snapshot.get("ground_truth_stats", {})
    pred_stats = analysis_snapshot.get("prediction_stats", {})
    judge_stats = analysis_snapshot.get("judge_stats", {})
    raw_payloads = analysis_snapshot.get("raw_payloads", {})
    input_parquet_snapshot = raw_payloads.get("input_parquet", {})
    rollout_model_payload = raw_payloads.get("rollout_model", {})
    llm_judge_model_payload = raw_payloads.get("llm_judge_model", {})
    verl_payload = raw_payloads.get("verl", {})
    field_origin_dictionary = analysis_snapshot.get("field_origin_dictionary", {})
    compact_extra = analysis_snapshot.get("extra_info_compact", {})
    large_summaries = analysis_snapshot.get("extra_info_large_field_summaries", {})

    extra_info = extra_info if isinstance(extra_info, dict) else {}
    prompt_text, prompt_message_count = _prompt_messages_to_text(extra_info.get("top_level_prompt"))
    article_text = extra_info.get("text_fixed_by_revision", "") or ""

    input_top_level = {
        "data_source": data_source,
        "source_dataset": routing.get("source_dataset"),
        "stable_article_id": lookup.get("stable_article_id"),
        "source_file": lookup.get("source_file"),
        "original_id": lookup.get("original_id"),
        "sample_index": lookup.get("index"),
        "sample_order": lookup.get("sample_order"),
        "test_variant": lookup.get("test_variant"),
        "prompt_message_count": prompt_message_count,
        "prompt_text": prompt_text,
        "article_text": article_text,
        "ground_truth": "" if ground_truth is None else str(ground_truth),
        "input_parquet_raw_json": _json_compact(input_parquet_snapshot),
        "input_parquet_top_level_keys_json": _json_compact(
            field_origin_dictionary.get("input_parquet", {}).get("top_level_keys", [])
        ),
        "input_parquet_nested_keys_json": _json_compact(
            field_origin_dictionary.get("input_parquet", {}).get("nested_keys", {})
        ),
        "extra_info_compact_json": _json_compact(compact_extra),
        "extra_info_large_field_summaries_json": _json_compact(large_summaries),
        "extra_info_omitted_large_fields_json": _json_compact(
            analysis_snapshot.get("extra_info_omitted_large_fields", [])
        ),
    }
    rollout_top_level = {
        "solution_str": "" if solution_str is None else str(solution_str),
        "rollout_model_raw_json": _json_compact(rollout_model_payload),
        "rollout_model_keys_json": _json_compact(field_origin_dictionary.get("rollout_model", {}).get("keys", [])),
    }
    llm_judge_top_level = {
        "precision_llm_response": (eval_details or {}).get("precision_response"),
        "recall_llm_response": (eval_details or {}).get("recall_response"),
        "llm_judge_model_raw_json": _json_compact(llm_judge_model_payload),
        "llm_judge_model_keys_json": _json_compact(
            field_origin_dictionary.get("llm_judge_model", {}).get("keys", [])
        ),
    }
    verl_top_level = {
        "record_schema_version": "reward_parquet_v3_source_suffix",
        "dataset_type": mode,
        "timestamp": timestamp,
        "machine_code": machine_code,
        "eval_mode_json": _json_compact(routing.get("eval_mode")),
        "precision_prompt": (eval_details or {}).get("precision_prompt"),
        "precision_score": (eval_details or {}).get("precision_score"),
        "recall_prompt": (eval_details or {}).get("recall_prompt"),
        "recall_score": (eval_details or {}).get("recall_score"),
        "f1_score": (eval_details or {}).get("f1_score"),
        "reward_eval_date_phoenix": (eval_details or {}).get("reward_eval_date_phoenix"),
        "reward_eval_started_at_phoenix": (eval_details or {}).get("reward_eval_started_at_phoenix"),
        "reward_eval_finished_at_phoenix": (eval_details or {}).get("reward_eval_finished_at_phoenix"),
        "reward_eval_wall_seconds": (eval_details or {}).get("reward_eval_wall_seconds"),
        "precision_llm_request_started_at_phoenix": (eval_details or {}).get("precision_llm_request_started_at_phoenix"),
        "precision_llm_response_received_at_phoenix": (eval_details or {}).get(
            "precision_llm_response_received_at_phoenix"
        ),
        "precision_score_calc_started_at_phoenix": (eval_details or {}).get(
            "precision_score_calc_started_at_phoenix"
        ),
        "precision_llm_latency_seconds": (eval_details or {}).get("precision_llm_latency_seconds"),
        "recall_llm_request_started_at_phoenix": (eval_details or {}).get("recall_llm_request_started_at_phoenix"),
        "recall_llm_response_received_at_phoenix": (eval_details or {}).get(
            "recall_llm_response_received_at_phoenix"
        ),
        "recall_score_calc_started_at_phoenix": (eval_details or {}).get("recall_score_calc_started_at_phoenix"),
        "recall_llm_latency_seconds": (eval_details or {}).get("recall_llm_latency_seconds"),
        "verl_generated_raw_json": _json_compact(verl_payload),
        "verl_keys_json": _json_compact(field_origin_dictionary.get("verl", {}).get("keys", [])),
        "verl_nested_keys_json": _json_compact(field_origin_dictionary.get("verl", {}).get("nested_keys", {})),
        "field_origin_dictionary_json": _json_compact(field_origin_dictionary),
    }

    record = {}
    record.update(_suffix_flat_dict(input_top_level, "input_parquet"))
    record.update(_suffix_flat_dict(rollout_top_level, "rollout_model"))
    record.update(_suffix_flat_dict(llm_judge_top_level, "llm_judge_model"))
    record.update(_suffix_flat_dict(verl_top_level, "verl"))

    record.update(_suffix_flat_dict(_flatten_dict({"routing": routing}), "verl"))
    record.update(_suffix_flat_dict(_flatten_dict({"lookup": lookup}), "input_parquet"))
    record.update(_suffix_flat_dict(_flatten_dict({"prompt_stats": prompt_stats}), "input_parquet"))
    record.update(_suffix_flat_dict(_flatten_dict({"article_text_stats": article_stats}), "input_parquet"))
    record.update(_suffix_flat_dict(_flatten_dict({"ground_truth_stats": gt_stats}), "input_parquet"))
    record.update(_suffix_flat_dict(_flatten_dict({"prediction_stats": pred_stats}), "rollout_model"))

    judge_flat = _flatten_dict({"judge_stats": judge_stats})
    for key, value in judge_flat.items():
        source_name = "llm_judge_model" if "response" in key else "verl"
        record[_suffix_key(key, source_name)] = _normalize_jsonable(value)
    return record


def _save_debug_reward_record(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    dataset_type,
    eval_details,
    reward_eval_started_at=None,
):
    _finalize_eval_details(eval_details, reward_eval_started_at)
    if dataset_type in ["test", "train"]:
        save_to_json(data_source, solution_str, ground_truth, extra_info, dataset_type, eval_details)

# ==========================================

# ==========================================
def save_to_json(data_source, solution_str, ground_truth, extra_info, mode: str, eval_details: dict = None):
    if not KG_REWARD_SAVE_DEBUG_PARQUET:
        return

    machine_code = get_machine_code()
    base_dir = Path(f"/workspace/verl/GRID_dataset/temp_result_check/reward_detail_parquet/{machine_code}/{mode}")
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return  
    
    if PHOENIX_TZ:
        timestamp = datetime.now(PHOENIX_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    
    suffix = 1
    while True:
        filename = f"{timestamp}_{os.getpid()}_{threading.get_ident()}_{suffix}.parquet"
        filepath = base_dir / filename
        if not filepath.exists():
            break
        suffix += 1
    
    analysis_snapshot = _build_analysis_snapshot(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        mode=mode,
        eval_details=eval_details or {},
    )

    record = _build_parquet_record(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        mode=mode,
        timestamp=timestamp,
        machine_code=machine_code,
        eval_details=eval_details or {},
        analysis_snapshot=analysis_snapshot,
    )
    try:
        pd.DataFrame([record]).to_parquet(filepath, index=False, compression="zstd")
    except Exception as e:
        if KG_REWARD_VERBOSE_DEBUG:
            print(f"❌ [Parquet] 保存失败: {filepath} -> {e}")  

# --- Configuration ---
# VLLM Servers for Load Balancing
VLLM_SERVERS = {
    'super': "http://192.168.100.1:8000/v1",
    'normal': "http://192.168.100.3:8001/v1",
    'ultra': "http://192.168.100.2:8002/v1"
}

CONFIG = {
    'super': {
        'api_key': "EMPTY",
        'model_name': "Qwen3-Next-80B-A3B-Instruct-AWQ-4bit",
        'max_tokens': 16*1024,
        'proxy': None
    },
    'normal': {
        'api_key': "EMPTY",
        'model_name': "Qwen3-Next-80B-A3B-Instruct-AWQ-4bit",
        'max_tokens': 16*1024,
        'proxy': None
    },
    'ultra': {
        'api_key': "EMPTY",
        'model_name': "Qwen3-Next-80B-A3B-Instruct-AWQ-4bit",
        'max_tokens': 16*1024,
        'proxy': None
    },
    'api': {
        'api_url': "https://api.openai.com/v1/chat/completions",
        'api_key': "managed-by-env",
        'model_name': "gpt-5.4-mini",
        'max_tokens': 32 * 1024,
        'proxy': None
    },
    'gemini': {
        'api_url': "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        'api_key': "managed-by-env",
        'model_name': "gemini-2.5-flash",
        'max_tokens': 32 * 1024,
        'proxy': None
    }
}

# ==========================================

# ==========================================


EVAL_MODE_CONFIG = {
    'train': {
        'mode': ['api'],
        'model_override': 'gpt-5.4-mini',
    },
    'test': {
        'mode': ['gemini'],
        'model_override': 'gemini-2.5-flash',
    },
    'default': {
        'mode': ['api'],
        'model_override': 'gpt-5.4-mini',
    }
}

def _parse_env_mode_list(env_value: str | None):
    if env_value is None:
        return None
    parts = [part.strip() for part in str(env_value).split(",") if part.strip()]
    return parts or None


def _resolve_eval_mode_from_env(dataset_type: str):
    normalized = (dataset_type or "default").strip().lower()
    key_prefix = {
        "train": "KG_REWARD_TRAIN",
        "test": "KG_REWARD_TEST",
    }.get(normalized, "KG_REWARD_DEFAULT")

    env_mode = _parse_env_mode_list(os.environ.get(f"{key_prefix}_MODE"))
    env_model = os.environ.get(f"{key_prefix}_MODEL")
    if env_mode is None and env_model is None:
        return None

    return {
        "mode": env_mode,
        "model_override": env_model,
    }


def get_eval_mode_for_dataset(dataset_type: str, extra_info: dict | None = None) -> tuple:
    
    
    
    

    env_cfg = _resolve_eval_mode_from_env(dataset_type)

    if dataset_type in EVAL_MODE_CONFIG:
        cfg = EVAL_MODE_CONFIG[dataset_type].copy()
    else:
        cfg = EVAL_MODE_CONFIG.get('default', {'mode': ['api'], 'model_override': 'gpt-5.4-mini'}).copy()

    
    
    if env_cfg is not None:
        if env_cfg.get("mode") is not None:
            cfg["mode"] = env_cfg["mode"]
        if env_cfg.get("model_override") is not None:
            cfg["model_override"] = env_cfg["model_override"]
    
    return cfg['mode'], cfg.get('model_override')


def _get_reward_request_env(dataset_type: str, provider: str, suffix: str):
    dataset_key = (dataset_type or "default").strip().upper()
    provider_key = (provider or "default").strip().upper()
    suffix_key = str(suffix).strip().upper()
    for env_key in (
        f"KG_REWARD_{dataset_key}_{provider_key}_{suffix_key}",
        f"KG_REWARD_{dataset_key}_{suffix_key}",
        f"KG_REWARD_{provider_key}_{suffix_key}",
        f"KG_REWARD_{suffix_key}",
    ):
        env_value = os.environ.get(env_key)
        if env_value not in (None, ""):
            return env_value
    return None


def _parse_positive_int_env(raw_value, default_value):
    if raw_value in (None, ""):
        return default_value
    try:
        return max(1, int(raw_value))
    except Exception:
        return default_value


def _build_reward_request_kwargs(dataset_type: str, provider: str) -> dict:
    provider_norm = (provider or "").strip().lower()
    request_kwargs = {}

    if provider_norm in {"api", "gemini"}:
        request_kwargs["retry"] = _parse_positive_int_env(
            _get_reward_request_env(dataset_type, provider_norm, "RETRY"),
            4,
        )
        return request_kwargs

    return request_kwargs

# Cache for resolved model names to avoid frequent API calls
_RESOLVED_MODEL_NAMES = {}

def llmname(vllm_server_name='local', shortname=True):
    """
    Get current VLLM model name from the specified server.
    """
    if vllm_server_name not in VLLM_SERVERS:
        # For non-vllm servers or unknown ones, return None
        return None
        
    api_base = VLLM_SERVERS.get(vllm_server_name)
    if not api_base:
        print(f"llmname: 无效的 vllm_server_name '{vllm_server_name}', "
              f"可用选项: {list(VLLM_SERVERS.keys())}")
        return None
    
    # Check cache first
    if vllm_server_name in _RESOLVED_MODEL_NAMES:
        return _RESOLVED_MODEL_NAMES[vllm_server_name]

    # Try using requests first (no dependency)
    try:
        # Use default key if config missing
        cfg = CONFIG.get(vllm_server_name, {})
        key = cfg.get('api_key', "mfodkgmkfjgnoijtrkytlmy")
        response = requests.get(f"{api_base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=2.0)
        
        if response.status_code == 200:
            data = response.json()
            # Compatible with both list and dict response
            data_list = data.get('data', []) if isinstance(data, dict) else data
            if data_list and len(data_list) > 0:
                 model_id = data_list[0]['id']
                 _RESOLVED_MODEL_NAMES[vllm_server_name] = model_id
                 return model_id
    except Exception as e:
        # print(f"Requests-based model fetch failed: {e}")
        pass

    if OpenAI is None:
        return None

    try:
        client = OpenAI(api_key="managed-by-env", base_url=api_base)
        # We use a short timeout to avoid hanging if server is down
        model_list = client.models.list(timeout=2.0)
        setmodel = re.search(r"id='(.*?)'", str(model_list))
        if not setmodel:
            setmodel = re.search(r'id="(.*?)"', str(model_list))
            
        model_id = setmodel.group(1) if setmodel else None
        
        if model_id:
            _RESOLVED_MODEL_NAMES[vllm_server_name] = model_id
            
        return model_id
    except Exception as e:
        print(f"Error fetching model name from {vllm_server_name}: {e}")
        return None

def update_config_with_real_model_names(mode_list):
    """
    Update CONFIG with actual model names from running VLLM servers in mode_list.
    """
    for server in mode_list:
        if server in VLLM_SERVERS:
            real_name = llmname(server)
            if real_name:
                CONFIG[server]['model_name'] = real_name
                #print(f"Updated {server} model name to: {real_name}")

def check_vllm_realtime_overload(server_name, max_waiting=2, max_kv_usage=0.8):
    # Only check overload for local VLLM servers
    if server_name not in VLLM_SERVERS:
        return False

    
    api_base = VLLM_SERVERS.get(server_name)
    if not api_base:
        return True # Should not happen if check above passes
    
    
    
    metrics_url = api_base.replace("/v1", "") + "/metrics"
    if metrics_url.endswith("//metrics"): 
        metrics_url = metrics_url.replace("//metrics", "/metrics")

    try:
        
        response = requests.get(metrics_url, timeout=1.5)
        if response.status_code != 200:
            return True 
        
        content = response.text
        
        
        
        
        
        waiting_match = re.search(r'vllm:num_requests_waiting\{[^}]*\}\s+(\d+\.?\d*)', content)
        waiting_val = float(waiting_match.group(1)) if waiting_match else 0.0
        
        
        kv_match = re.search(r'vllm:kv_cache_usage_perc\{[^}]*\}\s+(\d+\.?\d*)', content)
        if not kv_match:
            kv_match = re.search(r'vllm:gpu_cache_usage_perc\{[^}]*\}\s+(\d+\.?\d*)', content)
        
        kv_val = float(kv_match.group(1)) if kv_match else 0.0
        
        
        is_overloaded = False
        
        if waiting_val > max_waiting:
            is_overloaded = True
            
        if kv_val > max_kv_usage:
            is_overloaded = True
            
        if is_overloaded:
            return True
            
        return False 

    except Exception as e:
        return True

def call_llm_api(
    prompt_messages,
    mode_list=['super', 'normal'],
    dataset_type='train',
    model_override=None,
    max_tokens_override=None,
    temperature_override=None,
    service_tier_override=None,
):
    """
    Call the LLM API based on provided mode list.
    Supports load balancing for 'super', 'normal', 'ultra'.
    """
    if not mode_list:
        print("Error: No mode specified.")
        return None

    # Shuffle servers to distribute load
    candidate_servers = list(mode_list)
    random.shuffle(candidate_servers)
    
    # Retry loop
    max_retries = 3
    retry_count = 0
    
    while True:
        target_server = None
        
        # 1. Select a server
        # Prioritize non-overloaded local servers
        local_candidates = [s for s in candidate_servers if s in VLLM_SERVERS]
        other_candidates = [s for s in candidate_servers if s not in VLLM_SERVERS]
        
        # Check local servers for overload
        available_locals = []
        for server in local_candidates:
            if not check_vllm_realtime_overload(server):
                available_locals.append(server)
        
        if available_locals:
            target_server = random.choice(available_locals)
        elif other_candidates:
            target_server = random.choice(other_candidates)
        elif local_candidates:
             # All local servers overloaded and no other candidates
            # Wait and retry (or pick one anyway if we want to force it, but better to wait)
            print("⚠️ All selected local servers are overloaded. Waiting 2 seconds...")
            time.sleep(2)
            continue
        else:
             # Should not happen if lists are not empty
             print("Error: No available servers.")
             return None

        
        # 2. Prepare Request
        config = CONFIG.get(target_server)
        if not config:
            print(f"Error: Configuration not found for {target_server}")
            return None

        api_url = config.get('api_url')
        if not api_url:
            # Construct from VLLM_SERVERS if not explicitly set in CONFIG (for local ones)
            if target_server in VLLM_SERVERS:
                api_url = f"{VLLM_SERVERS[target_server]}/chat/completions"
            else:
                print(f"Error: API URL not found for {target_server}")
                return None
        
        # Ensure model_name is fresh if it's one of ours
        if target_server in VLLM_SERVERS and target_server not in _RESOLVED_MODEL_NAMES:
             # Try to fetch once if not done
             real_name = llmname(target_server)
             if real_name:
                 CONFIG[target_server]['model_name'] = real_name

        api_key = config['api_key']
        model_name = model_override or config['model_name']
        max_tokens = max_tokens_override or config['max_tokens']
        proxy = config['proxy']
        temperature = 0.2 if temperature_override is None else temperature_override

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_name,
            "messages": prompt_messages,
            "stream": False
        }

        # 2026-03-23:
        
        
        
        service_tier = service_tier_override
        if service_tier in (None, ""):
            service_tier = _get_reward_request_env(dataset_type, target_server, "SERVICE_TIER")
        if service_tier:
            payload["service_tier"] = str(service_tier).strip()

        if "gpt-5" in model_name or "o1" in model_name or "o3" in model_name:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
        
        proxies = {}
        if proxy:
            proxies = {
                'http': proxy,
                'https': proxy
            }
        
        # 3. Send Request
        try:
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=3600, proxies=proxies)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API Call failed on {target_server}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            
            # Simple retry logic
            time.sleep(1)
            retry_count += 1
            if retry_count > max_retries * len(candidate_servers):
                 print("Max retries reached.")
                 return None
            continue


async def call_llm_api_async(
    prompt_messages,
    mode_list=['api'],
    max_empty_retries=3,
    dataset_type='train',
    progress_scope=None,
    progress_label=None,
    progress_interval_seconds=None,
    model_override=None,
    max_tokens_override=None,
    temperature_override=None,
    service_tier_override=None,
):
    if not mode_list:
        print("Error: No mode specified.")
        return None
    
    def _is_empty_response(content):
        if content is None:
            return True
        if isinstance(content, str):
            stripped = content.strip()
            if stripped == "" or stripped == "[]" or stripped == "{}":
                return True
        return False
    
    
    return await asyncio.to_thread(
        call_llm_api,
        prompt_messages,
        mode_list,
        dataset_type,
        model_override,
        max_tokens_override,
        temperature_override,
        service_tier_override,
    )

def parse_llm_response(response_json):
    """
    Extract content from LLM response, handling </think>.
    """
    if not response_json or 'choices' not in response_json:
        return ""
    
    content = response_json['choices'][0]['message']['content']
    
    if "</think>" in content:
        parts = content.split("</think>")
        final_answer = parts[-1].strip()
    else:
        final_answer = content.strip()
        
    return final_answer

def parse_eval_json(json_str):
    """
    Robust JSON parser for evaluation results using json_repair.
    """
    if not json_str:
        return []
        
    if json_repair:
        try:
            # json_repair handles markdown blocks, truncated json, etc.
            decoded = json_repair.loads(json_str)
            return decoded
        except Exception as e:
            # Fallback to empty list if repair fails, though json_repair is usually very aggressive
            # print(f"json_repair failed: {e}")
            pass
            
    # Fallback legacy logic if json_repair not available or failed
    # 1. Try markdown json block
    match = re.search(r"```json\s*(\[.*?\])\s*```", json_str, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass

    # 2. Try generic code block
    match = re.search(r"```\s*(\[.*?\])\s*```", json_str, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass

    # 3. Try finding list brackets
    try:
        start = json_str.find('[')
        end = json_str.rfind(']')
        if start != -1 and end != -1 and end > start:
            return json.loads(json_str[start:end+1])
    except:
        pass

    # 4. Direct load
    try:
        return json.loads(json_str)
    except:
        pass
        
    return []


def parse_kg_block_json(json_str):
    if not json_str:
        return []

    if json_repair:
        try:
            decoded = json_repair.loads(json_str)
            if isinstance(decoded, list):
                return decoded
        except Exception:
            pass

    try:
        decoded = json.loads(json_str)
        if isinstance(decoded, list):
            return decoded
    except Exception:
        pass

    return []


def _looks_like_record_list(value, required_keys):
    if not isinstance(value, list) or not value:
        return False
    dict_count = 0
    matched = 0
    for item in value[:64]:
        if not isinstance(item, dict):
            continue
        dict_count += 1
        if set(item.keys()) & set(required_keys):
            matched += 1
    return dict_count > 0 and matched > 0


def _find_record_list_in_obj(obj, key_candidates, required_keys):
    queue = [obj]
    seen = set()
    while queue:
        current = queue.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)

        if isinstance(current, dict):
            for key in key_candidates:
                value = current.get(key)
                if _looks_like_record_list(value, required_keys):
                    return value
            for value in current.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(current, list):
            if _looks_like_record_list(current, required_keys):
                return current
            for value in current[:64]:
                if isinstance(value, (dict, list)):
                    queue.append(value)
    return []


def _extract_balanced_json_spans(text, max_spans=24):
    spans = []
    start_idx = None
    stack = []
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if start_idx is None:
            if ch == "{":
                start_idx = idx
                stack = ["}"]
                in_string = False
                escape = False
            elif ch == "[":
                start_idx = idx
                stack = ["]"]
                in_string = False
                escape = False
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                spans.append(text[start_idx:idx + 1])
                start_idx = None
                if len(spans) >= max_spans:
                    break
        elif ch in "}]":
            start_idx = None
            stack = []
            in_string = False
            escape = False

    return spans


def _iter_relaxed_json_candidates(text):
    stripped = str(text or "").strip()
    if not stripped:
        return

    yielded = set()

    def _yield_once(candidate):
        candidate = str(candidate or "").strip()
        if not candidate or candidate in yielded:
            return
        yielded.add(candidate)
        yield candidate

    for candidate in _yield_once(stripped):
        yield candidate

    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE)
    for block in fenced_blocks:
        for candidate in _yield_once(block):
            yield candidate

    for block in re.findall(r"#Entity_List_Start#(.*?)#Entity_List_End#", stripped, flags=re.DOTALL):
        for candidate in _yield_once(block):
            yield candidate

    for block in re.findall(r"#Relationship_List_Start#(.*?)#Relationship_List_End#", stripped, flags=re.DOTALL):
        for candidate in _yield_once(block):
            yield candidate

    for span in _extract_balanced_json_spans(stripped):
        for candidate in _yield_once(span):
            yield candidate


def _load_relaxed_json_candidates(text):
    parsed_objects = []
    for candidate in _iter_relaxed_json_candidates(text):
        if json_repair:
            try:
                parsed_objects.append(("json_repair", candidate, json_repair.loads(candidate)))
                continue
            except Exception:
                pass
        try:
            parsed_objects.append(("json", candidate, json.loads(candidate)))
        except Exception:
            continue
    return parsed_objects


def _extract_entities_from_structured_obj(obj):
    return _find_record_list_in_obj(
        obj,
        key_candidates=["entity", "entities", "node", "nodes", "entity_list", "Entity_List"],
        required_keys={"name", "type", "alias", "mother entity"},
    )


def _extract_relationships_from_structured_obj(obj):
    return _find_record_list_in_obj(
        obj,
        key_candidates=[
            "relationship",
            "relationships",
            "relation",
            "relations",
            "relationship_list",
            "Relationship_List",
        ],
        required_keys={"sub", "rel", "obj", "rel_type"},
    )


def extract_entities(kg_text):
    if not kg_text:
        return []

    try:
        start_marker = "#Entity_List_Start#"
        end_marker = "#Entity_List_End#"
        start_idx = kg_text.find(start_marker)
        end_idx = kg_text.find(end_marker)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = kg_text[start_idx + len(start_marker):end_idx].strip()
            parsed = parse_kg_block_json(json_str)
            if parsed:
                return parsed
    except Exception:
        pass

    for _, _, parsed_obj in _load_relaxed_json_candidates(kg_text):
        entities = _extract_entities_from_structured_obj(parsed_obj)
        if entities:
            return entities
    return []


def _extract_prediction_kg_lists_relaxed(raw_pred_text):
    cleaned_text = str(raw_pred_text or "").strip()
    if not cleaned_text:
        return {
            "entities": [],
            "relationships": [],
            "parse_success": False,
            "parse_strategy": "empty",
        }

    cleaned_text = re.sub(r"<think>[\s\S]*?</think>", "", cleaned_text, flags=re.IGNORECASE).strip()
    cleaned_text = re.sub(r"<mythink>[\s\S]*?</mythink>", "", cleaned_text, flags=re.IGNORECASE).strip()
    cleaned_text = re.sub(r"#Reasoning_Start#[\s\S]*?#Reasoning_End#", "", cleaned_text, flags=re.DOTALL).strip()

    entities = extract_entities(cleaned_text)
    relationships = extract_relationships(cleaned_text)
    if entities or relationships:
        return {
            "entities": entities,
            "relationships": relationships,
            "parse_success": True,
            "parse_strategy": "marker_or_direct",
        }

    parsed_candidates = _load_relaxed_json_candidates(cleaned_text)
    for parser_name, _, parsed_obj in parsed_candidates:
        entities = _extract_entities_from_structured_obj(parsed_obj)
        relationships = _extract_relationships_from_structured_obj(parsed_obj)
        if entities or relationships:
            return {
                "entities": entities,
                "relationships": relationships,
                "parse_success": True,
                "parse_strategy": f"relaxed_{parser_name}",
            }

    return {
        "entities": [],
        "relationships": [],
        "parse_success": False,
        "parse_strategy": "no_entity_or_relation_found",
    }

def extract_relationships(kg_text):
    """
    Extract relationship list from KG text/JSON.
    """
    # Try parsing as full JSON first
    try:
        if json_repair:
            data = json_repair.loads(kg_text)
        else:
            data = json.loads(kg_text)
            
        if isinstance(data, dict) and "relationship" in data:
            return data["relationship"]
        # Some models might return list directly
        if isinstance(data, list):
            # Check if it's a list of lists (e.g. [[entities], [relationships]])
            if data and isinstance(data[0], list):
                # Search for the list containing relationship objects
                for sublist in data:
                    if isinstance(sublist, list) and len(sublist) > 0:
                        first_item = sublist[0]
                        if isinstance(first_item, dict) and 'sub' in first_item and 'obj' in first_item:
                            return sublist
                # If no clear relationship list found, maybe return the second one if exists?
                # Or just fall through to other parsing methods
            else:
                # It's a single list. Check if it looks like relationships.
                if data and isinstance(data[0], dict):
                    # If it has specific keys, good. If not, we might accept it anyway or check for 'sub'.
                    # For now, let's assume if it is a list of dicts, it might be the right one, 
                    # UNLESS it is an entity list.
                    first_item = data[0]
                    if 'sub' in first_item or 'rel' in first_item:
                        return data
                    # If it looks like entity list (has 'name', 'type' but no 'sub'), maybe ignoring it is safer?
                    # But the previous code just returned 'data'.
                    # Let's be stricter if possible.
                    if 'sub' in first_item and 'obj' in first_item:
                         return data
                    # If it's ambiguous, fall through to marker extraction which is more specific.
                    if 'name' in first_item and 'type' in first_item and 'sub' not in first_item:
                        # Likely entity list, so we should NOT return this.
                        pass
                    else:
                        # Unknown list of dicts, return it to be safe (backward compatibility)
                        return data
                elif not data:
                    return []
    except:
        pass

    
    for _, _, parsed_obj in _load_relaxed_json_candidates(kg_text):
        rels = _extract_relationships_from_structured_obj(parsed_obj)
        if rels:
            return rels

    # Try markers
    try:
        start_marker = "#Relationship_List_Start#"
        end_marker = "#Relationship_List_End#"
        start_idx = kg_text.find(start_marker)
        end_idx = kg_text.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            json_str = kg_text[start_idx + len(start_marker):end_idx].strip()
            return parse_eval_json(json_str)
    except:
        pass
        
    return []

def add_indices_to_kg(kg_data, prefix):
    if not isinstance(kg_data, list):
        return kg_data
    indexed_data = []
    for i, item in enumerate(kg_data):
        if not isinstance(item, dict):
            if KG_REWARD_VERBOSE_DEBUG:
                print(f"[DEBUG] add_indices_to_kg item {i} type: {type(item)}, content: {item}")
            continue
        new_item = item.copy()
        new_item['index'] = f"{prefix}_{i+1}"
        indexed_data.append(new_item)
    return indexed_data


def build_kg_control_block_text(relations, entities=None, include_reasoning=False, reasoning_text=""):
    relation_list = parse_kg_block_json(json.dumps(relations or [], ensure_ascii=False))
    entity_list = parse_kg_block_json(json.dumps(entities or [], ensure_ascii=False))

    parts = []
    if include_reasoning:
        reasoning_payload = str(reasoning_text or "").strip() or "N/A"
        parts.append(f"#Reasoning_Start#\n{reasoning_payload}\n#Reasoning_End#")
    parts.append(
        "#Entity_List_Start#\n"
        f"{json.dumps(entity_list, ensure_ascii=False)}\n"
        "#Entity_List_End#"
    )
    parts.append(
        "#Relationship_List_Start#\n"
        f"{json.dumps(relation_list, ensure_ascii=False)}\n"
        "#Relationship_List_End#"
    )
    return "\n\n".join(parts)


def _normalize_eval_result_items(eval_data):
    if isinstance(eval_data, dict):
        eval_data = [eval_data]
    if not isinstance(eval_data, list):
        return []

    normalized = []
    for idx, item in enumerate(eval_data):
        if isinstance(item, dict):
            result = str(item.get("result", "")).strip().upper()
            if result:
                payload = dict(item)
                payload["result"] = result
                normalized.append(payload)
            continue
        if isinstance(item, str):
            result = item.strip().upper()
            if result in {"TP", "FP", "FN"}:
                normalized.append({"index": idx, "result": result})
    return normalized


def calculate_metrics_details(eval_data, eval_type=None):
    normalized_items = _normalize_eval_result_items(eval_data)
    if not normalized_items:
        return {
            "score": 0.0,
            "TP": 0,
            "total": 0,
            "raw_results": [],
            "parse_success": False,
            "error": "No valid TP/FP/FN items found",
        }

    tp = 0
    negative = 0
    for item in normalized_items:
        res = str(item.get("result", "")).upper()
        if res == "TP":
            tp += 1
        elif res in {"FP", "FN"}:
            negative += 1

    total = tp + negative
    score = tp / total if total > 0 else 0.0
    output = {
        "score": score,
        "TP": tp,
        "total": total,
        "raw_results": normalized_items,
        "parse_success": total > 0,
        "error": "" if total > 0 else "No TP/FP/FN labels found",
    }
    if eval_type == "precision":
        output["FP"] = negative
    elif eval_type == "recall":
        output["FN"] = negative
    return output

def calculate_metrics(eval_data):
    """
    Calculate Precision/Recall score from evaluation list.
    Returns: score (0.0-1.0)
    """
    details = calculate_metrics_details(eval_data)
    return float(details.get("score", 0.0) or 0.0)


def _coerce_mode_list(mode_value):
    if mode_value is None:
        return None
    if isinstance(mode_value, (list, tuple, set)):
        return [str(item).strip() for item in mode_value if str(item).strip()]
    text = str(mode_value).strip()
    if not text:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def _extract_llm_response_text(response_payload):
    if response_payload is None:
        return ""
    if isinstance(response_payload, str):
        return response_payload.strip()
    try:
        return parse_llm_response(response_payload)
    except Exception:
        return str(response_payload).strip()


def _build_metric_failure_details(eval_type, error_text):
    details = calculate_metrics_details([], eval_type=eval_type)
    details["error"] = str(error_text or "No valid TP/FP/FN items found")
    return details


def _build_detailed_score_payload(score, eval_details=None):
    eval_details = eval_details if isinstance(eval_details, dict) else {}
    precision_details = dict(
        eval_details.get("precision_details") or _build_metric_failure_details("precision", "No precision details recorded")
    )
    recall_details = dict(
        eval_details.get("recall_details") or _build_metric_failure_details("recall", "No recall details recorded")
    )
    return {
        "score": float(score or 0.0),
        "precision": float(eval_details.get("precision_score") or 0.0),
        "recall": float(eval_details.get("recall_score") or 0.0),
        "f1": float(eval_details.get("f1_score") or score or 0.0),
        "precision_details": precision_details,
        "recall_details": recall_details,
        "precision_prompt": eval_details.get("precision_prompt") or "",
        "recall_prompt": eval_details.get("recall_prompt") or "",
        "precision_response": (
            eval_details.get("precision_response_text") or _extract_llm_response_text(eval_details.get("precision_response"))
        ),
        "recall_response": (
            eval_details.get("recall_response_text") or _extract_llm_response_text(eval_details.get("recall_response"))
        ),
        "eval_mode": eval_details.get("eval_mode"),
        "prediction_parse_mode": eval_details.get("prediction_parse_mode"),
        "prediction_parse_success": eval_details.get("prediction_parse_success"),
        "prediction_parse_strategy": eval_details.get("prediction_parse_strategy"),
        "reward_eval_started_at_phoenix": eval_details.get("reward_eval_started_at_phoenix"),
        "reward_eval_finished_at_phoenix": eval_details.get("reward_eval_finished_at_phoenix"),
        "reward_eval_wall_seconds": eval_details.get("reward_eval_wall_seconds"),
    }


def _finalize_and_build_result(
    *,
    score,
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    dataset_type,
    eval_details,
    reward_eval_started_at,
    return_details=False,
    async_mode=False,
    save_debug_record=True,
    extra_payload=None,
):
    _finalize_eval_details(eval_details, reward_eval_started_at)
    if _coerce_bool(save_debug_record) and dataset_type in ["test", "train"]:
        save_to_json(data_source, solution_str, ground_truth, extra_info, dataset_type, eval_details)

    if return_details:
        payload = _build_detailed_score_payload(score, eval_details)
        if extra_payload:
            payload.update(extra_payload)
        return payload

    if async_mode:
        payload = {"score": float(score or 0.0)}
        if extra_payload:
            payload.update(extra_payload)
        return payload

    return float(score or 0.0)

def compute_score_kg(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    mode=['super', 'normal'],
    skip_reward_and_return_zero_for_all_inputs=False,
    **kwargs,
):
    extra_info = _normalize_extra_info(extra_info)
    dataset_type = _resolve_dataset_type(extra_info)
    return_details = _coerce_bool(kwargs.get("return_details"))
    save_debug_record = _coerce_bool(kwargs.get("save_debug_record", True))

    force_mode_raw = kwargs.get("force_mode")
    if force_mode_raw is None:
        force_mode_raw = kwargs.get("force_mode_list")
    if force_mode_raw is None:
        force_mode_raw = kwargs.get("mode_override")
    force_mode = _coerce_mode_list(force_mode_raw)

    force_model_name = kwargs.get("force_model_name")
    if force_model_name in (None, ""):
        force_model_name = kwargs.get("model_name_override")
    if force_model_name in (None, ""):
        force_model_name = kwargs.get("model_override")

    force_max_tokens = kwargs.get("force_max_tokens")
    if force_max_tokens is None:
        force_max_tokens = kwargs.get("max_tokens_override")

    force_temperature = kwargs["force_temperature"] if "force_temperature" in kwargs else kwargs.get("temperature_override")
    force_service_tier = (
        kwargs["force_service_tier"] if "force_service_tier" in kwargs else kwargs.get("service_tier_override")
    )

    precision_prompt_text = kwargs.get("precision_prompt_text") or GRID_PRECISION_PROMPT_TEXT
    recall_prompt_text = kwargs.get("recall_prompt_text") or GRID_RECALL_PROMPT_TEXT

    if _coerce_bool(skip_reward_and_return_zero_for_all_inputs):
        reward_eval_started_at, eval_details = _init_eval_details("skip_reward_and_return_zero_for_all_inputs")
        eval_details["skip_reward_and_return_zero_for_all_inputs"] = True
        eval_details["f1_score"] = 0.0
        return _finalize_and_build_result(
            score=0.0,
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            dataset_type=dataset_type,
            eval_details=eval_details,
            reward_eval_started_at=reward_eval_started_at,
            return_details=return_details,
            async_mode=False,
            save_debug_record=save_debug_record,
        )
    if dataset_type == "train" and _is_regex_match_sample(extra_info):
        score, eval_details, reward_eval_started_at = _compute_regex_match_score(
            solution_str=solution_str,
            extra_info=extra_info,
        )
        return _finalize_and_build_result(
            score=score,
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            dataset_type=dataset_type,
            eval_details=eval_details,
            reward_eval_started_at=reward_eval_started_at,
            return_details=return_details,
            async_mode=False,
            save_debug_record=save_debug_record,
        )
    if dataset_type == "train" and _is_easyreward_non_llm_sample(extra_info):
        score, eval_details, reward_eval_started_at = _compute_easyreward_non_llm_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        return _finalize_and_build_result(
            score=score,
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            dataset_type=dataset_type,
            eval_details=eval_details,
            reward_eval_started_at=reward_eval_started_at,
            return_details=return_details,
            async_mode=False,
            save_debug_record=save_debug_record,
        )

    mode, model_override = get_eval_mode_for_dataset(dataset_type, extra_info)
    mode = force_mode or mode
    if force_model_name not in (None, ""):
        model_override = force_model_name

    reward_eval_started_at, eval_details = _init_eval_details(f"{mode} ({model_override or 'default'})")

    try:
        progress_scope = None
        progress_label = None
        progress_interval_seconds = None
        if extra_info and isinstance(extra_info, dict):
            progress_scope = extra_info.get("verl_progress_scope")
            progress_label = extra_info.get("verl_progress_label")
            progress_interval_seconds = extra_info.get("verl_progress_interval_seconds")

        dataset_type = _resolve_dataset_type(extra_info)
        update_config_with_real_model_names(mode)

        if not ground_truth:
            eval_details["f1_score"] = 0.0
            return _finalize_and_build_result(
                score=0.0,
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                dataset_type=dataset_type,
                eval_details=eval_details,
                reward_eval_started_at=reward_eval_started_at,
                return_details=return_details,
                async_mode=False,
                save_debug_record=save_debug_record,
            )

        article_text = ""
        if extra_info and isinstance(extra_info, dict):
            article_text = extra_info.get("text_fixed_by_revision", "") or ""

        gt_entity_pattern = re.compile(r'#Entity_List_Start#(.*?)#Entity_List_End#', re.DOTALL)
        gt_relation_pattern = re.compile(r'#Relationship_List_Start#(.*?)#Relationship_List_End#', re.DOTALL)

        gt_entity_match = gt_entity_pattern.search(ground_truth)
        gt_relation_match = gt_relation_pattern.search(ground_truth)

        if gt_entity_match and gt_relation_match:
            gt_entity_json = gt_entity_match.group(1).strip()
            gt_relation_json = gt_relation_match.group(1).strip()
            gt_entity_json = json.dumps(parse_kg_block_json(gt_entity_json), ensure_ascii=False)
            gt_relation_json = json.dumps(parse_kg_block_json(gt_relation_json), ensure_ascii=False)
            gt_kg_text = (
                f"#Entity_List_Start#\n{gt_entity_json}\n#Entity_List_End#\n\n"
                f"#Relationship_List_Start#\n{gt_relation_json}\n#Relationship_List_End#"
            )
        else:
            split_marker = "###我是分割线###"
            if split_marker in ground_truth:
                parts = ground_truth.split(split_marker)
                if not article_text:
                    article_text = parts[0].strip()
                gt_kg_text = parts[1].strip() if len(parts) > 1 else ground_truth
            else:
                gt_kg_text = ground_truth

        solution_str = solution_str or ""
        if "</mythink>" in solution_str:
            raw_pred_text = solution_str.split("</mythink>")[-1].strip()
        elif "</think>" in solution_str:
            raw_pred_text = solution_str.split("</think>")[-1].strip()
        else:
            raw_pred_text = solution_str

        reasoning_pattern = re.compile(r'(#Reasoning_Start#.*?#Reasoning_End#)', re.DOTALL)
        pred_entity_pattern = re.compile(r'#Entity_List_Start#(.*?)#Entity_List_End#', re.DOTALL)
        pred_relation_pattern = re.compile(r'#Relationship_List_Start#(.*?)#Relationship_List_End#', re.DOTALL)

        if dataset_type == "test":
            relaxed_parse = _extract_prediction_kg_lists_relaxed(raw_pred_text)
            eval_details["prediction_parse_mode"] = "relaxed_test"
            eval_details["prediction_parse_success"] = bool(relaxed_parse.get("parse_success"))
            eval_details["prediction_parse_strategy"] = relaxed_parse.get("parse_strategy")
            pred_entity_json = json.dumps(relaxed_parse.get("entities", []), ensure_ascii=False)
            pred_relation_json = json.dumps(relaxed_parse.get("relationships", []), ensure_ascii=False)
        else:
            reasoning_match = reasoning_pattern.search(raw_pred_text)
            pred_entity_match = pred_entity_pattern.search(raw_pred_text)
            pred_relation_match = pred_relation_pattern.search(raw_pred_text)

            if not (reasoning_match and pred_entity_match and pred_relation_match):
                print("Format Error: Missing one or more required blocks (Reasoning, Entity List, Relationship List). Returning 0.0.")
                eval_details['f1_score'] = 0.0
                eval_details["prediction_parse_mode"] = "strict_train"
                eval_details["prediction_parse_success"] = False
                eval_details["prediction_parse_strategy"] = "missing_required_blocks"
                return _finalize_and_build_result(
                    score=0.0,
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    dataset_type=dataset_type,
                    eval_details=eval_details,
                    reward_eval_started_at=reward_eval_started_at,
                    return_details=return_details,
                    async_mode=False,
                    save_debug_record=save_debug_record,
                )

            pred_entity_json = pred_entity_match.group(1).strip()
            pred_relation_json = pred_relation_match.group(1).strip()
            pred_entity_json = json.dumps(parse_kg_block_json(pred_entity_json), ensure_ascii=False)
            pred_relation_json = json.dumps(parse_kg_block_json(pred_relation_json), ensure_ascii=False)
            eval_details["prediction_parse_mode"] = "strict_train"
            eval_details["prediction_parse_success"] = True
            eval_details["prediction_parse_strategy"] = "strict_blocks"

        pred_kg_text = (
            f"#Entity_List_Start#\n{pred_entity_json}\n#Entity_List_End#\n\n"
            f"#Relationship_List_Start#\n{pred_relation_json}\n#Relationship_List_End#"
        )

        gt_rels = extract_relationships(gt_kg_text)
        pred_rels = extract_relationships(pred_kg_text)

        if not gt_rels and not pred_rels:
            eval_details["f1_score"] = 1.0
            return _finalize_and_build_result(
                score=1.0,
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                dataset_type=dataset_type,
                eval_details=eval_details,
                reward_eval_started_at=reward_eval_started_at,
                return_details=return_details,
                async_mode=False,
                save_debug_record=save_debug_record,
            )
        if not gt_rels or not pred_rels:
            eval_details["f1_score"] = 0.0
            return _finalize_and_build_result(
                score=0.0,
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                dataset_type=dataset_type,
                eval_details=eval_details,
                reward_eval_started_at=reward_eval_started_at,
                return_details=return_details,
                async_mode=False,
                save_debug_record=save_debug_record,
            )

        gt_rels_indexed = add_indices_to_kg(gt_rels, "truth_relationship")
        pred_rels_indexed = add_indices_to_kg(pred_rels, "predict_relationship")

        gt_kg_str = json.dumps(gt_rels_indexed, indent=2, ensure_ascii=False)
        pred_kg_str = json.dumps(pred_rels_indexed, indent=2, ensure_ascii=False)

        cot_instruction = ""
        check_server = None
        for s in mode:
            if s in VLLM_SERVERS:
                check_server = s
                break
        if check_server:
            local_model_name = model_override or CONFIG[check_server]['model_name']
            if "Instruct" in local_model_name:
                cot_instruction = "\\n\\n要求: 请先进行逐步思考 (Chain of Thought), 将思考过程包裹在 <think>...</think> 标签中, 最后再输出符合格式的 JSON 结果."

        precision = 0.0
        precision_details = _build_metric_failure_details("precision", "Precision branch not executed")
        prec_resp = None
        try:
            precision_prompt = (
                f"{precision_prompt_text}{cot_instruction}\n\n"
                f"--- 待评估的预测值 (Prediction to Evaluate) ---\n{pred_kg_str}\n\n"
                f"--- 参照标准 (Ground Truth) ---\n{gt_kg_str}\n\n"
                f"--- 附带正文 (Context) ---\n{article_text}"
            )
            eval_details["precision_prompt"] = precision_prompt
            precision_request_started_at = _phoenix_now()
            prec_resp = call_llm_api(
                [{"role": "user", "content": precision_prompt}],
                mode_list=mode,
                dataset_type=dataset_type,
                model_override=model_override,
                max_tokens_override=force_max_tokens,
                temperature_override=force_temperature,
                service_tier_override=force_service_tier,
            )
            precision_response_received_at = _phoenix_now()
            _record_eval_branch_timing(eval_details, "precision", precision_request_started_at, precision_response_received_at)
            eval_details["precision_response"] = prec_resp
            eval_details["precision_response_text"] = _extract_llm_response_text(prec_resp)

            if prec_resp:
                prec_data = parse_eval_json(eval_details["precision_response_text"])
                precision_details = calculate_metrics_details(prec_data, eval_type="precision")
            else:
                precision_details = _build_metric_failure_details("precision", "Empty LLM response")
            precision = float(precision_details.get("score", 0.0) or 0.0)
        except Exception as e:
            print(f"Error calculating precision: {e}")
            precision = 0.0
            precision_details = _build_metric_failure_details("precision", e)
        eval_details["precision_score"] = precision
        eval_details["precision_details"] = precision_details

        recall = 0.0
        recall_details = _build_metric_failure_details("recall", "Recall branch not executed")
        rec_resp = None
        try:
            recall_prompt = (
                f"{recall_prompt_text}{cot_instruction}\n\n"
                f"--- 真实值 (Ground Truth) ---\n{gt_kg_str}\n\n"
                f"--- 待查候选集 (Prediction Pool) ---\n{pred_kg_str}\n\n"
                f"--- 附带正文 (Context) ---\n{article_text}"
            )
            eval_details["recall_prompt"] = recall_prompt
            recall_request_started_at = _phoenix_now()
            rec_resp = call_llm_api(
                [{"role": "user", "content": recall_prompt}],
                mode_list=mode,
                dataset_type=dataset_type,
                model_override=model_override,
                max_tokens_override=force_max_tokens,
                temperature_override=force_temperature,
                service_tier_override=force_service_tier,
            )
            recall_response_received_at = _phoenix_now()
            _record_eval_branch_timing(eval_details, "recall", recall_request_started_at, recall_response_received_at)
            eval_details["recall_response"] = rec_resp
            eval_details["recall_response_text"] = _extract_llm_response_text(rec_resp)

            if rec_resp:
                rec_data = parse_eval_json(eval_details["recall_response_text"])
                recall_details = calculate_metrics_details(rec_data, eval_type="recall")
            else:
                recall_details = _build_metric_failure_details("recall", "Empty LLM response")
            recall = float(recall_details.get("score", 0.0) or 0.0)
        except Exception as e:
            print(f"Error calculating recall: {e}")
            recall = 0.0
            recall_details = _build_metric_failure_details("recall", e)
        eval_details["recall_score"] = recall
        eval_details["recall_details"] = recall_details

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        eval_details["f1_score"] = f1
        return _finalize_and_build_result(
            score=f1,
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            dataset_type=dataset_type,
            eval_details=eval_details,
            reward_eval_started_at=reward_eval_started_at,
            return_details=return_details,
            async_mode=False,
            save_debug_record=save_debug_record,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR in compute_score_kg: {e}")
        return _finalize_and_build_result(
            score=0.0,
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            dataset_type=dataset_type,
            eval_details=eval_details,
            reward_eval_started_at=reward_eval_started_at,
            return_details=return_details,
            async_mode=False,
            save_debug_record=save_debug_record,
            extra_payload={"critical_error": str(e)},
        )


async def compute_score_kg_async(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    reward_router_address=None,  
    reward_model_tokenizer=None,  
    **kwargs
) -> dict:
    extra_info = _normalize_extra_info(extra_info)
    dataset_type = _resolve_dataset_type(extra_info)
    return_details = _coerce_bool(kwargs.get("return_details"))
    save_debug_record = _coerce_bool(kwargs.get("save_debug_record", True))

    force_mode_raw = kwargs.get("force_mode")
    if force_mode_raw is None:
        force_mode_raw = kwargs.get("force_mode_list")
    if force_mode_raw is None:
        force_mode_raw = kwargs.get("mode_override")
    force_mode = _coerce_mode_list(force_mode_raw)

    force_model_name = kwargs.get("force_model_name")
    if force_model_name in (None, ""):
        force_model_name = kwargs.get("model_name_override")
    if force_model_name in (None, ""):
        force_model_name = kwargs.get("model_override")

    force_max_tokens = kwargs.get("force_max_tokens")
    if force_max_tokens is None:
        force_max_tokens = kwargs.get("max_tokens_override")

    force_temperature = kwargs["force_temperature"] if "force_temperature" in kwargs else kwargs.get("temperature_override")
    force_service_tier = (
        kwargs["force_service_tier"] if "force_service_tier" in kwargs else kwargs.get("service_tier_override")
    )

    precision_prompt_text = kwargs.get("precision_prompt_text") or GRID_PRECISION_PROMPT_TEXT
    recall_prompt_text = kwargs.get("recall_prompt_text") or GRID_RECALL_PROMPT_TEXT

    if _coerce_bool(kwargs.get("skip_reward_and_return_zero_for_all_inputs")):
        reward_eval_started_at, eval_details = _init_eval_details("skip_reward_and_return_zero_for_all_inputs")
        eval_details["skip_reward_and_return_zero_for_all_inputs"] = True
        eval_details["f1_score"] = 0.0
        return _finalize_and_build_result(
            score=0.0,
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            dataset_type=dataset_type,
            eval_details=eval_details,
            reward_eval_started_at=reward_eval_started_at,
            return_details=return_details,
            async_mode=True,
            save_debug_record=save_debug_record,
            extra_payload={"skip_reward_and_return_zero_for_all_inputs": True},
        )
    if dataset_type == "train" and _is_regex_match_sample(extra_info):
        score, eval_details, reward_eval_started_at = _compute_regex_match_score(
            solution_str=solution_str,
            extra_info=extra_info,
        )
        return _finalize_and_build_result(
            score=score,
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            dataset_type=dataset_type,
            eval_details=eval_details,
            reward_eval_started_at=reward_eval_started_at,
            return_details=return_details,
            async_mode=True,
            save_debug_record=save_debug_record,
            extra_payload={
                "regex_match_reward": True,
                "easyreward_non_llm": False,
            },
        )
    if dataset_type == "train" and _is_easyreward_non_llm_sample(extra_info):
        score, eval_details, reward_eval_started_at = _compute_easyreward_non_llm_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        return _finalize_and_build_result(
            score=score,
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            dataset_type=dataset_type,
            eval_details=eval_details,
            reward_eval_started_at=reward_eval_started_at,
            return_details=return_details,
            async_mode=True,
            save_debug_record=save_debug_record,
            extra_payload={
                "regex_match_reward": False,
                "easyreward_non_llm": True,
            },
        )

    mode, model_override = get_eval_mode_for_dataset(dataset_type, extra_info)
    mode = force_mode or mode
    if force_model_name not in (None, ""):
        model_override = force_model_name

    reward_eval_started_at, eval_details = _init_eval_details(f"{mode} ({model_override or 'default'})")

    try:
        progress_scope = None
        progress_label = None
        progress_interval_seconds = None
        if extra_info and isinstance(extra_info, dict):
            progress_scope = extra_info.get("verl_progress_scope")
            progress_label = extra_info.get("verl_progress_label")
            progress_interval_seconds = extra_info.get("verl_progress_interval_seconds")

        dataset_type = _resolve_dataset_type(extra_info)

        if not ground_truth:
            eval_details["f1_score"] = 0.0
            return _finalize_and_build_result(
                score=0.0,
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                dataset_type=dataset_type,
                eval_details=eval_details,
                reward_eval_started_at=reward_eval_started_at,
                return_details=return_details,
                async_mode=True,
                save_debug_record=save_debug_record,
            )

        article_text = ""
        if extra_info and isinstance(extra_info, dict):
            article_text = extra_info.get("text_fixed_by_revision", "") or ""

        gt_entity_pattern = re.compile(r'#Entity_List_Start#(.*?)#Entity_List_End#', re.DOTALL)
        gt_relation_pattern = re.compile(r'#Relationship_List_Start#(.*?)#Relationship_List_End#', re.DOTALL)

        gt_entity_match = gt_entity_pattern.search(ground_truth)
        gt_relation_match = gt_relation_pattern.search(ground_truth)

        if gt_entity_match and gt_relation_match:
            gt_entity_json = gt_entity_match.group(1).strip()
            gt_relation_json = gt_relation_match.group(1).strip()
            gt_entity_json = json.dumps(parse_kg_block_json(gt_entity_json), ensure_ascii=False)
            gt_relation_json = json.dumps(parse_kg_block_json(gt_relation_json), ensure_ascii=False)
            gt_kg_text = (
                f"#Entity_List_Start#\n{gt_entity_json}\n#Entity_List_End#\n\n"
                f"#Relationship_List_Start#\n{gt_relation_json}\n#Relationship_List_End#"
            )
        else:
            split_marker = "###我是分割线###"
            if split_marker in ground_truth:
                parts = ground_truth.split(split_marker)
                if not article_text:
                    article_text = parts[0].strip()
                gt_kg_text = parts[1].strip() if len(parts) > 1 else ground_truth
            else:
                gt_kg_text = ground_truth

        solution_str = solution_str or ""
        if "</mythink>" in solution_str:
            raw_pred_text = solution_str.split("</mythink>")[-1].strip()
        elif "</think>" in solution_str:
            raw_pred_text = solution_str.split("</think>")[-1].strip()
        else:
            raw_pred_text = solution_str

        reasoning_pattern = re.compile(r'(#Reasoning_Start#.*?#Reasoning_End#)', re.DOTALL)
        pred_entity_pattern = re.compile(r'#Entity_List_Start#(.*?)#Entity_List_End#', re.DOTALL)
        pred_relation_pattern = re.compile(r'#Relationship_List_Start#(.*?)#Relationship_List_End#', re.DOTALL)

        if dataset_type == "test":
            relaxed_parse = _extract_prediction_kg_lists_relaxed(raw_pred_text)
            eval_details["prediction_parse_mode"] = "relaxed_test"
            eval_details["prediction_parse_success"] = bool(relaxed_parse.get("parse_success"))
            eval_details["prediction_parse_strategy"] = relaxed_parse.get("parse_strategy")
            pred_entity_json = json.dumps(relaxed_parse.get("entities", []), ensure_ascii=False)
            pred_relation_json = json.dumps(relaxed_parse.get("relationships", []), ensure_ascii=False)
        else:
            reasoning_match = reasoning_pattern.search(raw_pred_text)
            pred_entity_match = pred_entity_pattern.search(raw_pred_text)
            pred_relation_match = pred_relation_pattern.search(raw_pred_text)

            if not (reasoning_match and pred_entity_match and pred_relation_match):
                print("Format Error: Missing required blocks. Returning 0.0.")
                eval_details['f1_score'] = 0.0
                eval_details["prediction_parse_mode"] = "strict_train"
                eval_details["prediction_parse_success"] = False
                eval_details["prediction_parse_strategy"] = "missing_required_blocks"
                return _finalize_and_build_result(
                    score=0.0,
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    dataset_type=dataset_type,
                    eval_details=eval_details,
                    reward_eval_started_at=reward_eval_started_at,
                    return_details=return_details,
                    async_mode=True,
                    save_debug_record=save_debug_record,
                )

            pred_entity_json = pred_entity_match.group(1).strip()
            pred_relation_json = pred_relation_match.group(1).strip()
            pred_entity_json = json.dumps(parse_kg_block_json(pred_entity_json), ensure_ascii=False)
            pred_relation_json = json.dumps(parse_kg_block_json(pred_relation_json), ensure_ascii=False)
            eval_details["prediction_parse_mode"] = "strict_train"
            eval_details["prediction_parse_success"] = True
            eval_details["prediction_parse_strategy"] = "strict_blocks"

        pred_kg_text = (
            f"#Entity_List_Start#\n{pred_entity_json}\n#Entity_List_End#\n\n"
            f"#Relationship_List_Start#\n{pred_relation_json}\n#Relationship_List_End#"
        )

        gt_rels = extract_relationships(gt_kg_text)
        pred_rels = extract_relationships(pred_kg_text)

        if not gt_rels and not pred_rels:
            eval_details["f1_score"] = 1.0
            return _finalize_and_build_result(
                score=1.0,
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                dataset_type=dataset_type,
                eval_details=eval_details,
                reward_eval_started_at=reward_eval_started_at,
                return_details=return_details,
                async_mode=True,
                save_debug_record=save_debug_record,
            )
        if not gt_rels or not pred_rels:
            eval_details["f1_score"] = 0.0
            return _finalize_and_build_result(
                score=0.0,
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                dataset_type=dataset_type,
                eval_details=eval_details,
                reward_eval_started_at=reward_eval_started_at,
                return_details=return_details,
                async_mode=True,
                save_debug_record=save_debug_record,
            )

        gt_rels_indexed = add_indices_to_kg(gt_rels, "truth_relationship")
        pred_rels_indexed = add_indices_to_kg(pred_rels, "predict_relationship")

        gt_kg_str = json.dumps(gt_rels_indexed, indent=2, ensure_ascii=False)
        pred_kg_str = json.dumps(pred_rels_indexed, indent=2, ensure_ascii=False)

        cot_instruction = ""

        async def _run_eval_branch(branch_name, prompt_text, eval_type):
            response_payload = None
            response_text = ""
            score_value = 0.0
            detail = _build_metric_failure_details(eval_type, f"{branch_name} branch not executed")
            request_started_at = _phoenix_now()
            response_received_at = None
            try:
                response_payload = await call_llm_api_async(
                    [{"role": "user", "content": prompt_text}],
                    mode_list=mode,
                    dataset_type=dataset_type,
                    progress_scope=progress_scope,
                    progress_label=progress_label,
                    progress_interval_seconds=progress_interval_seconds,
                    model_override=model_override,
                    max_tokens_override=force_max_tokens,
                    temperature_override=force_temperature,
                    service_tier_override=force_service_tier,
                )
                response_received_at = _phoenix_now()
                response_text = _extract_llm_response_text(response_payload)
                if response_text:
                    parsed_data = parse_eval_json(response_text)
                    detail = calculate_metrics_details(parsed_data, eval_type=eval_type)
                else:
                    detail = _build_metric_failure_details(eval_type, "Empty LLM response")
                score_value = float(detail.get("score", 0.0) or 0.0)
            except Exception as branch_exc:
                print(f"Error calculating {branch_name}: {branch_exc}")
                detail = _build_metric_failure_details(eval_type, branch_exc)
            if response_received_at is None:
                response_received_at = _phoenix_now()
            return response_payload, response_text, score_value, request_started_at, response_received_at, detail

        precision_prompt = (
            f"{precision_prompt_text}{cot_instruction}\n\n"
            f"--- 待评估的预测值 (Prediction to Evaluate) ---\n{pred_kg_str}\n\n"
            f"--- 参照标准 (Ground Truth) ---\n{gt_kg_str}\n\n"
            f"--- 附带正文 (Context) ---\n{article_text}"
        )
        recall_prompt = (
            f"{recall_prompt_text}{cot_instruction}\n\n"
            f"--- 真实值 (Ground Truth) ---\n{gt_kg_str}\n\n"
            f"--- 待查候选集 (Prediction Pool) ---\n{pred_kg_str}\n\n"
            f"--- 附带正文 (Context) ---\n{article_text}"
        )
        eval_details["precision_prompt"] = precision_prompt
        eval_details["recall_prompt"] = recall_prompt

        precision_result, recall_result = await asyncio.gather(
            _run_eval_branch("precision", precision_prompt, "precision"),
            _run_eval_branch("recall", recall_prompt, "recall"),
        )
        (
            prec_resp,
            prec_resp_text,
            precision,
            precision_request_started_at,
            precision_response_received_at,
            precision_details,
        ) = precision_result
        (
            rec_resp,
            rec_resp_text,
            recall,
            recall_request_started_at,
            recall_response_received_at,
            recall_details,
        ) = recall_result
        _record_eval_branch_timing(eval_details, "precision", precision_request_started_at, precision_response_received_at)
        _record_eval_branch_timing(eval_details, "recall", recall_request_started_at, recall_response_received_at)
        eval_details["precision_response"] = prec_resp
        eval_details["precision_response_text"] = prec_resp_text
        eval_details["precision_score"] = precision
        eval_details["precision_details"] = precision_details
        eval_details["recall_response"] = rec_resp
        eval_details["recall_response_text"] = rec_resp_text
        eval_details["recall_score"] = recall
        eval_details["recall_details"] = recall_details

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        eval_details["f1_score"] = f1
        return _finalize_and_build_result(
            score=f1,
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            dataset_type=dataset_type,
            eval_details=eval_details,
            reward_eval_started_at=reward_eval_started_at,
            return_details=return_details,
            async_mode=True,
            save_debug_record=save_debug_record,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR in compute_score_kg_async: {e}")
        return _finalize_and_build_result(
            score=0.0,
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            dataset_type=dataset_type,
            eval_details=eval_details,
            reward_eval_started_at=reward_eval_started_at,
            return_details=return_details,
            async_mode=True,
            save_debug_record=save_debug_record,
            extra_payload={"critical_error": str(e)},
        )


def read_file_content(filename):
    """Helper to read file content from current directory."""
    filepath = os.path.join(current_dir, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return None
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def _compute_single_sample(idx, data_source, solution_str, ground_truth, extra_info, mode):
    
    score = compute_score_kg(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        mode=mode,
    )
    return idx, score


def compute_score_kg_batch(
    solution_str=None,
    ground_truth=None,
    data_source=None,
    extra_info=None,
    mode=['api'],
    max_workers=1280,
    **kwargs
):
    

    
    single_sample_mode = False
    
    
    if isinstance(solution_str, list):
        solution_strs = solution_str
    else:
        solution_strs = [solution_str] if solution_str is not None else []
        single_sample_mode = True  
        
    if isinstance(ground_truth, list):
        ground_truths = ground_truth
    else:
        ground_truths = [ground_truth] if ground_truth is not None else []
        
    if isinstance(extra_info, list):
        extra_infos = extra_info
    else:
        extra_infos = [extra_info] if extra_info is not None else []

    if isinstance(data_source, list):
        data_sources = data_source
    else:
        data_sources = [data_source] * len(solution_strs) if solution_strs else []
    
    
    if not solution_strs:
        print("Warning: compute_score_kg_batch called with empty solution_str")
        return 0.0 if single_sample_mode else []

    n = len(solution_strs)

    
    if not (len(data_sources) == len(ground_truths) == n):
        print("Warning: data_sources / solution_strs / ground_truths 长度不一致")
        n = min(len(data_sources), len(solution_strs), len(ground_truths))

    
    if extra_infos is None:
        extra_infos = [None] * n
    elif len(extra_infos) != n:
        print("Warning: extra_infos 长度与其它字段不一致，截断对齐")
        extra_infos = list(extra_infos)[:n]

    results = [0.0] * n

    
    worker_num = max(1, min(max_workers, n))

    with ThreadPoolExecutor(max_workers=worker_num) as executor:
        futures = []
        for i in range(n):
            futures.append(
                executor.submit(
                    _compute_single_sample,
                    i,
                    data_sources[i],
                    solution_strs[i],
                    ground_truths[i],
                    extra_infos[i],
                    mode,
                )
            )

        for fut in as_completed(futures):
            try:
                idx, score = fut.result()
                
                try:
                    score_val = float(score)
                except Exception:
                    score_val = 0.0
                if 0 <= idx < n:
                    results[idx] = score_val
            except Exception as e:
                
                print(f"Error in compute_score_kg_batch worker: {e}")
                

    
    if single_sample_mode:
        return results[0] if results else 0.0
    return results



SELF_CHECK_DEFAULT_OUTPUT_JSON = "/tmp/kg_reward_self_check_result.json"
SELF_CHECK_TRAIN_PARQUET_CANDIDATES = [
    "/workspace/verl/GRID_dataset/KG-gen/default_train_run/default_train_KG_extraction_train(full_nosplit).parquet",
    "/workspace/verl/GRID_dataset/KG-gen/reference_train_run/reference_train_KG_extraction_train(full_nosplit).parquet",
]
SELF_CHECK_TEST_PARQUET_CANDIDATES = [
    "/workspace/verl/GRID_dataset/KG-gen/default_train_run/default_train_KG_extraction_test(real).parquet",
    "/workspace/verl/GRID_dataset/KG-gen/reference_train_run/reference_train_KG_extraction_test(real).parquet",
]


def _resolve_self_check_parquet(cli_value, env_key, candidates):
    if cli_value:
        return cli_value
    env_value = os.environ.get(env_key)
    if env_value:
        return env_value
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return candidates[0]


def _normalize_prompt_to_messages_for_self_check(prompt_obj):
    if hasattr(prompt_obj, "tolist"):
        prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, dict):
        return [prompt_obj]
    if isinstance(prompt_obj, list):
        normalized = []
        for item in prompt_obj:
            if hasattr(item, "tolist"):
                item = item.tolist()
            if isinstance(item, dict):
                normalized.append(
                    {
                        "role": item.get("role", "user"),
                        "content": item.get("content", ""),
                    }
                )
            else:
                normalized.append({"role": "user", "content": str(item)})
        return normalized
    return [{"role": "user", "content": str(prompt_obj)}]


def _prompt_size_score_for_self_check(prompt_obj):
    messages = _normalize_prompt_to_messages_for_self_check(prompt_obj)
    return len(json.dumps(messages, ensure_ascii=False))


def _parse_json_like_for_self_check(raw_value):
    if isinstance(raw_value, (dict, list)):
        return raw_value
    if raw_value in (None, ""):
        return None
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return None
        if json_repair is not None:
            try:
                return json_repair.loads(stripped)
            except Exception:
                pass
        try:
            return json.loads(stripped)
        except Exception:
            return None
    return None


def _is_minimally_valid_row_for_self_check(row_dict):
    prompt_messages = _normalize_prompt_to_messages_for_self_check(row_dict.get("prompt"))
    if not prompt_messages:
        return False
    prompt_text = json.dumps(prompt_messages, ensure_ascii=False)
    if "Input Text: None" in prompt_text or "Input Text: none" in prompt_text:
        return False
    ground_truth = str(row_dict.get("ground_truth", "") or "").strip()
    return ground_truth not in {"", "None", "none"}


def _is_strictly_valid_row_for_self_check(row_dict):
    if not _is_minimally_valid_row_for_self_check(row_dict):
        return False
    extra_info = _normalize_extra_info(row_dict.get("extra_info"))
    revised_text = str(extra_info.get("text_fixed_by_revision", "") or "").strip()
    if revised_text and revised_text.lower() != "none":
        return True
    graph_obj = _parse_json_like_for_self_check(extra_info.get("graph_from_text_raw_from_file"))
    if isinstance(graph_obj, dict):
        if graph_obj.get("entity") or graph_obj.get("relationship"):
            return True
    return False


def _select_rows_for_self_check(df, label, samples_per_split):
    candidates = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        if not _is_minimally_valid_row_for_self_check(row_dict):
            continue
        prompt_chars = _prompt_size_score_for_self_check(row_dict.get("prompt"))
        strict_rank = 0 if _is_strictly_valid_row_for_self_check(row_dict) else 1
        row_dict["_selected_index"] = int(idx)
        row_dict["_selected_prompt_chars"] = int(prompt_chars)
        row_dict["_selected_validity"] = "strict" if strict_rank == 0 else "fallback"
        candidates.append((strict_rank, prompt_chars, int(idx), row_dict))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = [item[3] for item in candidates[:samples_per_split]]
    if len(selected) < samples_per_split:
        raise RuntimeError(f"{label} parquet 中仅找到 {len(selected)} 条可用样本，少于要求的 {samples_per_split} 条")
    return selected


def _extract_openai_message_text(message_obj):
    content = getattr(message_obj, "content", "") or ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None) or ""
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def _generate_with_gpt5mini_for_self_check(messages, max_completion_tokens):
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK 未安装，无法执行 gpt-5-mini 自检")
    api_key = CONFIG.get("gpt", {}).get("api_key")
    if not api_key:
        raise RuntimeError("CONFIG['gpt']['api_key'] 缺失，无法执行 gpt-5-mini 自检")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        max_completion_tokens=max_completion_tokens,
    )
    return _extract_openai_message_text(response.choices[0].message)


def _run_single_self_check_case(row_dict, dataset_type, max_completion_tokens, use_ground_truth_as_solution):
    prompt_messages = _normalize_prompt_to_messages_for_self_check(row_dict["prompt"])
    extra_info = _normalize_extra_info(row_dict.get("extra_info"))
    extra_info["dataset_type"] = dataset_type
    extra_info["verl_progress_scope"] = "kg_reward_self_check"
    extra_info["verl_progress_label"] = f"{dataset_type}-idx{row_dict['_selected_index']}"
    mode_list, model_override = get_eval_mode_for_dataset(dataset_type, extra_info)

    if use_ground_truth_as_solution:
        response_text = str(row_dict.get("ground_truth") or "")
        generation_seconds = 0.0
        print(
            f"🧪 [{dataset_type}] 使用 ground_truth 作为控制组 solution_str: "
            f"idx={row_dict['_selected_index']}, chars={len(response_text)}"
        )
    else:
        print(
            f"🧪 [{dataset_type}] gpt-5-mini 开始生成: "
            f"idx={row_dict['_selected_index']}, prompt_chars={row_dict['_selected_prompt_chars']}, "
            f"validity={row_dict['_selected_validity']}"
        )
        generation_started_at = time.time()
        response_text = _generate_with_gpt5mini_for_self_check(prompt_messages, max_completion_tokens)
        generation_seconds = round(time.time() - generation_started_at, 3)
        print(
            f"✅ [{dataset_type}] gpt-5-mini 生成完成: response_chars={len(response_text)}, "
            f"seconds={generation_seconds}"
        )

    score_started_at = time.time()
    score = compute_score_kg(
        data_source=row_dict.get("data_source", f"kg_reward_self_check_{dataset_type}"),
        solution_str=response_text,
        ground_truth=row_dict["ground_truth"],
        extra_info=extra_info,
    )
    score_seconds = round(time.time() - score_started_at, 3)
    effective_route_mode = mode_list
    effective_route_model_override = model_override
    if dataset_type == "train" and _is_regex_match_sample(extra_info):
        groundtruth_kind = str(extra_info.get("regex_groundtruth_kind") or "").strip() or "regex_match"
        effective_route_mode = ["local_regex"]
        effective_route_model_override = groundtruth_kind
    print(
        f"✅ [{dataset_type}] reward 完成: idx={row_dict['_selected_index']}, "
        f"score={score}, seconds={score_seconds}, route={effective_route_mode}/{effective_route_model_override}"
    )

    return {
        "dataset_type": dataset_type,
        "selected_index": row_dict["_selected_index"],
        "selected_prompt_chars": row_dict["_selected_prompt_chars"],
        "selected_validity": row_dict["_selected_validity"],
        "reward_route_mode": effective_route_mode,
        "reward_route_model_override": effective_route_model_override,
        "prompt_message_count": len(prompt_messages),
        "generation_seconds": generation_seconds,
        "score_seconds": score_seconds,
        "response_chars": len(response_text),
        "has_reasoning_block": "#Reasoning_Start#" in response_text and "#Reasoning_End#" in response_text,
        "has_entity_block": "#Entity_List_Start#" in response_text and "#Entity_List_End#" in response_text,
        "has_relationship_block": "#Relationship_List_Start#" in response_text and "#Relationship_List_End#" in response_text,
        "score": float(score) if score is not None else None,
        "response_preview": response_text[:800],
    }


def run_self_check(
    train_parquet=None,
    test_parquet=None,
    samples_per_split=2,
    parallel_workers=4,
    max_completion_tokens=16384,
    output_json=SELF_CHECK_DEFAULT_OUTPUT_JSON,
    use_ground_truth_as_solution=False,
):
    train_parquet = _resolve_self_check_parquet(
        train_parquet,
        "KG_REWARD_SELF_CHECK_TRAIN_PARQUET",
        SELF_CHECK_TRAIN_PARQUET_CANDIDATES,
    )
    test_parquet = _resolve_self_check_parquet(
        test_parquet,
        "KG_REWARD_SELF_CHECK_TEST_PARQUET",
        SELF_CHECK_TEST_PARQUET_CANDIDATES,
    )

    print("🚀 开始 kg_reward.py 自检模式")
    print(f"📦 train_parquet={train_parquet}")
    print(f"📦 test_parquet={test_parquet}")
    print(
        f"🔢 samples_per_split={samples_per_split}, parallel_workers={parallel_workers}, "
        f"max_completion_tokens={max_completion_tokens}"
    )

    if not Path(train_parquet).exists():
        raise FileNotFoundError(f"train parquet 不存在: {train_parquet}")
    if not Path(test_parquet).exists():
        raise FileNotFoundError(f"test parquet 不存在: {test_parquet}")

    train_df = pd.read_parquet(train_parquet)
    test_df = pd.read_parquet(test_parquet)
    train_rows = _select_rows_for_self_check(train_df, "train", samples_per_split)
    test_rows = _select_rows_for_self_check(test_df, "test", samples_per_split)

    for row in train_rows:
        print(
            f"📍 选中 train idx={row['_selected_index']}, prompt_chars={row['_selected_prompt_chars']}, "
            f"validity={row['_selected_validity']}"
        )
    for row in test_rows:
        print(
            f"📍 选中 test idx={row['_selected_index']}, prompt_chars={row['_selected_prompt_chars']}, "
            f"validity={row['_selected_validity']}"
        )

    task_specs = [("train", row) for row in train_rows] + [("test", row) for row in test_rows]
    max_workers = max(1, min(int(parallel_workers), len(task_specs)))
    case_results = [None] * len(task_specs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                _run_single_self_check_case,
                row,
                dataset_type,
                max_completion_tokens,
                use_ground_truth_as_solution,
            ): idx
            for idx, (dataset_type, row) in enumerate(task_specs)
        }
        for future in as_completed(future_to_index):
            case_results[future_to_index[future]] = future.result()

    success = all(
        isinstance(item.get("score"), float)
        and 0.0 <= item["score"] <= 1.0
        and item.get("response_chars", 0) > 0
        for item in case_results
    )
    result = {
        "meta": {
            "train_parquet": train_parquet,
            "test_parquet": test_parquet,
            "samples_per_split": samples_per_split,
            "parallel_workers": max_workers,
            "generator_model": "gpt-5-mini",
            "max_completion_tokens": max_completion_tokens,
            "use_ground_truth_as_solution": bool(use_ground_truth_as_solution),
            "reward_routes": {
                "train": {
                    "mode": get_eval_mode_for_dataset("train")[0],
                    "model_override": get_eval_mode_for_dataset("train")[1],
                },
                "test": {
                    "mode": get_eval_mode_for_dataset("test")[0],
                    "model_override": get_eval_mode_for_dataset("test")[1],
                },
            },
            "success": success,
        },
        "cases": case_results,
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print("🎯 kg_reward.py 自检完成")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"💾 自检结果已保存到: {output_path}")
    return result


def _run_legacy_local_file_test():
    print("Running kg_reward.py")
    print("开始旧版本地文件自我测试, 读取文本文件来代替实际输入")
    article_text = read_file_content("1根据知识图谱修改后的文章.txt")
    ground_truth_kg = read_file_content("2原始文章对应知识图谱.json")
    prediction_kg = read_file_content("3结果预测值.txt")

    if article_text and ground_truth_kg and prediction_kg:
        print("Successfully loaded input files.")
        ground_truth_combined = f"{article_text}###我是分割线###{ground_truth_kg}"
        score = compute_score_kg(None, prediction_kg, ground_truth_combined, mode=['api'])
        print(f"Final F1 Score: {score}")
    else:
        print("Failed to load one or more input files.")


def main():
    parser = argparse.ArgumentParser(description="kg_reward.py CLI")
    parser.add_argument("--self-check", action="store_true", help="运行 parquet -> gpt-5-mini -> reward 的容器内自检")
    parser.add_argument("--train-parquet", type=str, default=None, help="自检使用的 train parquet 路径")
    parser.add_argument("--test-parquet", type=str, default=None, help="自检使用的 test parquet 路径")
    parser.add_argument("--samples-per-split", type=int, default=2, help="train/test 各取多少条样本，默认 2")
    parser.add_argument("--parallel-workers", type=int, default=4, help="并行执行多少个生成+reward case，默认 4")
    parser.add_argument("--max-completion-tokens", type=int, default=16384, help="gpt-5-mini 最大输出 token")
    parser.add_argument("--output-json", type=str, default=SELF_CHECK_DEFAULT_OUTPUT_JSON, help="自检结果 JSON 输出路径")
    parser.add_argument("--use-ground-truth-as-solution", action="store_true", help="跳过 gpt-5-mini 生成，直接用 ground_truth 作为控制组")
    args = parser.parse_args()

    if args.self_check:
        run_self_check(
            train_parquet=args.train_parquet,
            test_parquet=args.test_parquet,
            samples_per_split=max(1, int(args.samples_per_split)),
            parallel_workers=max(1, int(args.parallel_workers)),
            max_completion_tokens=max(1, int(args.max_completion_tokens)),
            output_json=args.output_json,
            use_ground_truth_as_solution=args.use_ground_truth_as_solution,
        )
        return

    _run_legacy_local_file_test()


if __name__ == '__main__':
    main()
