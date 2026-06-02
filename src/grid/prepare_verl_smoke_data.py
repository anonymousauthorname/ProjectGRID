# -*- coding: utf-8 -*-
"""
Filename: prepare_verl_smoke_data.py
Description: Build compact GRID task-bank Parquet files for Docker VERL smoke training.
Keywords: GRID, Docker, VERL, task-bank, Parquet

This script keeps the reviewer-facing VERL path small while still using rows
derived from the packaged GRID task-bank file. It rewrites long task prompts
into compact multiple-choice prompts and preserves the task-bank answer as the
scripted reward target.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _parse_json_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    value = _parse_json_maybe(value) if isinstance(value, str) else value
    if isinstance(value, dict):
        return [{"role": str(value.get("role", "user")), "content": str(value.get("content", ""))}]
    if isinstance(value, list):
        rows: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, dict):
                rows.append({"role": str(item.get("role", "user")), "content": str(item.get("content", ""))})
        if rows:
            return rows
    return [{"role": "user", "content": str(value)}]


def _first_user_text(prompt_value: Any) -> str:
    for message in _normalize_messages(prompt_value):
        if message.get("role", "user") == "user":
            return str(message.get("content", ""))
    messages = _normalize_messages(prompt_value)
    return str(messages[0].get("content", "")) if messages else ""


def _extract_block(text: str, start_marker: str, end_markers: Iterable[str]) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end = len(text)
    for marker in end_markers:
        marker_pos = text.find(marker, start)
        if marker_pos >= 0:
            end = min(end, marker_pos)
    return text[start:end].strip()


def _compact_prompt(text: str, *, max_chars: int) -> str:
    context = _extract_block(text, "**Context:**", ["**Question:**", "**Options:**", "**Your Task"])
    question = _extract_block(text, "**Question:**", ["**Options:**", "**Your Task"])
    options = _extract_block(text, "**Options:**", ["**Your Task"])
    if not question or not options:
        compact = re.sub(r"\s+", " ", text).strip()
        return compact[:max_chars]

    context = re.sub(r"\s+", " ", context).strip()
    question = re.sub(r"\s+", " ", question).strip()
    options = re.sub(r"\s+", " ", options).strip()
    context_excerpt = context[: max(0, max_chars - len(question) - len(options) - 320)]
    return (
        "You are answering a GRID task-bank multiple-choice item. "
        "Select all correct options and output only #### followed by a Python list.\n\n"
        f"Context excerpt: {context_excerpt}\n\n"
        f"Question: {question}\n\n"
        f"Options: {options}\n\n"
        'Answer format example: #### ["A"]'
    )


def _parse_options(value: Any) -> list[str]:
    parsed = _parse_json_maybe(value)
    if isinstance(parsed, list):
        return [str(item).strip().upper() for item in parsed if str(item).strip().upper() in {"A", "B", "C", "D"}]
    return []


def _ground_truth_from_row(row: pd.Series) -> Any:
    for key in ("ground_truth", "sft_ground_truth"):
        value = row.get(key)
        if value is not None and str(value).strip() not in {"", "None", "nan"}:
            return value
    reward_model = _parse_json_maybe(row.get("reward_model"))
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth")
    return None


def _build_row(row: pd.Series, index: int, *, max_chars: int) -> dict[str, Any]:
    ground_truth = _ground_truth_from_row(row)
    options = _parse_options(ground_truth)
    gt_text = json.dumps(options, ensure_ascii=False)
    prompt_text = _compact_prompt(_first_user_text(row.get("prompt")), max_chars=max_chars)
    prompt_messages = [{"role": "user", "content": prompt_text}]
    sft_response = "#### " + gt_text
    sft_messages = prompt_messages + [{"role": "assistant", "content": sft_response}]
    source_row_index = row.get("__source_row_index", index)
    if pd.isna(source_row_index):
        source_row_index = index
    return {
        "prompt": prompt_messages,
        "messages": sft_messages,
        "enable_thinking": False,
        "prompt_text": prompt_text,
        "sft_response": sft_response,
        "reward_model": {"ground_truth": gt_text},
        "data_source": "grid_task_bank_docker_smoke",
        "ability": "knowledge_graph_extraction",
        "ground_truth": gt_text,
        "extra_info": {
            "task_type": "grid_task_bank_option_list",
            "expected_options": options,
            "source_row_index": int(source_row_index),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare compact VERL smoke Parquet files from GRID task-bank data.")
    parser.add_argument("--source-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-rows", type=int, default=4)
    parser.add_argument("--val-rows", type=int, default=2)
    parser.add_argument("--max-prompt-chars", type=int, default=1200)
    args = parser.parse_args()

    source = Path(args.source_parquet).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(source)
    rows = []
    for index, row in df.iterrows():
        if _parse_options(_ground_truth_from_row(row)):
            rows.append(_build_row(row, index, max_chars=args.max_prompt_chars))
        if len(rows) >= args.train_rows + args.val_rows:
            break
    if len(rows) < args.train_rows + args.val_rows:
        raise RuntimeError(f"Only found {len(rows)} option-list rows in {source}")

    train_rows = rows[: args.train_rows]
    val_rows = rows[args.train_rows : args.train_rows + args.val_rows]
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    manifest = {
        "source_parquet": str(source),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "max_prompt_chars": args.max_prompt_chars,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
