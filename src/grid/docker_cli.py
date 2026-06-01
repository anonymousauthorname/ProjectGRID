# -*- coding: utf-8 -*-
"""
Filename: docker_cli.py
Description: Portable Docker entrypoint for the packaged GRID artifact.
Keywords: GRID, Docker, parquet, training smoke, KG extraction, evaluation

Workflow:
1. Read CTI articles from TXT, JSONL, JSON, CSV, or Parquet.
2. Materialize article, training, and evaluation Parquet files.
3. Run a one-step local RL smoke train and export a small KG model.
4. Generate knowledge graphs either from the exported smoke model or an
   OpenAI-compatible LLM endpoint configured through environment variables.
5. Evaluate predicted KG edges against gold KG edges with exact normalized
   triple matching.

This module is intentionally independent of local Dropbox paths, vLLM clusters,
and VERL. It provides an executable artifact path for reviewers; paper-scale
VERL/RL post-training can still be run outside this portable smoke backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docker_output"
DEFAULT_SAMPLE_INPUT = REPO_ROOT / "docker" / "sample_articles.jsonl"

ID_KEYS = ("id", "article_id", "stable_article_id", "file_name", "name")
CONTENT_KEYS = ("content", "text", "article", "input", "raw_text")
GOLD_KEYS = ("gold_kg", "ground_truth_kg", "kg", "gold", "relations", "edges", "triples")
EDGE_LIST_KEYS = ("edges", "relations", "relationship_list", "triples")
SUB_KEYS = ("sub", "subject", "source", "head", "src")
REL_KEYS = ("rel", "relation", "predicate", "type", "label")
OBJ_KEYS = ("obj", "object", "target", "tail", "dst")


def _env_any(names: Sequence[str], default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip() != "":
            return str(value)
    return default


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_no}")
            rows.append(value)
    return rows


def _read_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [{"id": path.stem, "content": path.read_text(encoding="utf-8")}]
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("articles"), list):
            value = value["articles"]
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            if not all(isinstance(item, dict) for item in value):
                raise ValueError(f"JSON list must contain objects: {path}")
            return list(value)
        raise ValueError(f"Unsupported JSON top-level value in {path}")
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported input suffix: {suffix}")


def _json_or_empty(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, float) and pd.isna(value):
        return {}
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


def _first_nonempty(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record:
            value = record[key]
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                if str(value).strip() != "":
                    return value
    return None


def _parse_kg(value: Any) -> Dict[str, Any]:
    parsed = _json_or_empty(value)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"nodes": [], "edges": parsed}
    return {"nodes": [], "edges": []}


def _normalize_record(record: Dict[str, Any], index: int, content_col: str = "", id_col: str = "") -> Dict[str, Any]:
    id_keys = (id_col,) + ID_KEYS if id_col else ID_KEYS
    content_keys = (content_col,) + CONTENT_KEYS if content_col else CONTENT_KEYS
    article_id = _first_nonempty(record, id_keys)
    content = _first_nonempty(record, content_keys)
    if content is None:
        raise ValueError(f"Record {index} has no content column. Tried: {content_keys}")

    gold_value = _first_nonempty(record, GOLD_KEYS)
    gold_kg = _parse_kg(gold_value)
    article_id_text = str(article_id or f"article_{index:05d}")
    content_text = str(content)
    return {
        "id": article_id_text,
        "content": content_text,
        "source": str(record.get("source", record.get("dataset", "docker_input"))),
        "gold_kg": gold_kg,
        "content_sha1": _sha1(content_text),
    }


def _records_from_args(input_file: str, content_col: str = "", id_col: str = "") -> List[Dict[str, Any]]:
    path = Path(input_file).expanduser().resolve()
    raw_records = _read_records(path)
    return [_normalize_record(record, i, content_col=content_col, id_col=id_col) for i, record in enumerate(raw_records)]


def _edge_value(edge: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key in edge and edge[key] is not None:
            value = str(edge[key]).strip()
            if value:
                return value
    return ""


def _iter_edges(kg: Any) -> List[Dict[str, Any]]:
    parsed = _parse_kg(kg)
    edges: Any = []
    for key in EDGE_LIST_KEYS:
        if key in parsed:
            edges = parsed[key]
            break
    if isinstance(edges, dict):
        edges = list(edges.values())
    if not isinstance(edges, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in edges:
        if isinstance(item, dict):
            sub = _edge_value(item, SUB_KEYS)
            rel = _edge_value(item, REL_KEYS)
            obj = _edge_value(item, OBJ_KEYS)
            if sub and rel and obj:
                copied = dict(item)
                copied.setdefault("sub", sub)
                copied.setdefault("rel", rel)
                copied.setdefault("obj", obj)
                normalized.append(copied)
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            sub, rel, obj = str(item[0]), str(item[1]), str(item[2])
            normalized.append({"sub": sub, "rel": rel, "obj": obj})
    return normalized


def _triple_key(edge: Dict[str, Any]) -> Tuple[str, str, str]:
    sub = _edge_value(edge, SUB_KEYS)
    rel = _edge_value(edge, REL_KEYS)
    obj = _edge_value(edge, OBJ_KEYS)
    clean = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    return clean(sub), clean(rel), clean(obj)


def _edge_set(kg: Any) -> set[Tuple[str, str, str]]:
    return {key for key in (_triple_key(edge) for edge in _iter_edges(kg)) if all(key)}


def _compact_kg(kg: Any) -> Dict[str, Any]:
    parsed = _parse_kg(kg)
    return {
        "nodes": parsed.get("nodes", []),
        "edges": _iter_edges(parsed),
    }


def _dump_json(path: Path, payload: Any) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _path_from_output_dir(args: argparse.Namespace, value: str, default_name: str) -> Path:
    default_path = DEFAULT_OUTPUT_DIR / default_name
    if str(value) == str(default_path):
        return Path(args.output_dir).expanduser().resolve() / default_name
    return Path(value).expanduser().resolve()


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _has_llm_env() -> bool:
    return bool(
        _env_any(("GRID_LLM_MODEL", "OPENAI_MODEL"))
        and _env_any(("GRID_LLM_API_KEY", "GRID_LLM_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"))
    )


def _configure_openai_env(args: argparse.Namespace) -> Tuple[str, str, str]:
    model = getattr(args, "llm_model", "") or _env_any(("GRID_LLM_MODEL", "OPENAI_MODEL"), "")
    api_key = getattr(args, "llm_api_key", "") or _env_any(("GRID_LLM_API_KEY", "GRID_LLM_KEY", "OPENAI_API_KEY"), "")
    base_url = getattr(args, "llm_base_url", "") or _env_any(("GRID_LLM_BASE_URL", "GRID_LLM_ENDPOINT", "OPENAI_BASE_URL"), "")
    if model:
        os.environ["OPENAI_MODEL"] = model
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    return model, api_key, base_url


def _parse_llm_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if match:
        candidates.append(match.group(0))
    match_list = re.search(r"\[.*\]", stripped, flags=re.S)
    if match_list:
        candidates.append(match_list.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return _compact_kg(parsed)
        except Exception:
            continue
    return {"nodes": [], "edges": [], "raw_response": text}


def _llm_extract_kg(content: str, args: argparse.Namespace) -> Dict[str, Any]:
    model, api_key, base_url = _configure_openai_env(args)
    if not model or not api_key:
        raise RuntimeError("LLM backend requires GRID_LLM_MODEL and GRID_LLM_API_KEY/OPENAI_API_KEY.")
    from src import tools_nano

    prompt = (
        "Extract a cyber threat intelligence knowledge graph from the article.\n"
        "Return only JSON with this schema: {\"nodes\": [{\"name\": str, \"type\": str}], "
        "\"edges\": [{\"sub\": str, \"rel\": str, \"obj\": str, \"evidence\": str}]}.\n"
        "Every edge must be directly supported by the text.\n\n"
        f"Article:\n{content}"
    )
    response = tools_nano.ask(
        prompt,
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        max_tokens=int(getattr(args, "llm_max_tokens", 8192)),
        temperature=float(getattr(args, "llm_temperature", 0.0)),
        timeout=int(getattr(args, "llm_timeout", 300)),
    )
    return _parse_llm_json(response)


def cmd_env_check(args: argparse.Namespace) -> int:
    model, api_key, base_url = _configure_openai_env(args)
    payload = {
        "repo_root": str(REPO_ROOT),
        "sample_input_exists": DEFAULT_SAMPLE_INPUT.exists(),
        "output_dir": str(Path(args.output_dir).expanduser()),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "openai_model": model,
        "openai_base_url": base_url,
        "openai_key_present": bool(api_key),
        "llm_ready": bool(model and api_key),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_make_parquet(args: argparse.Namespace) -> int:
    records = _records_from_args(args.input_file, content_col=args.content_col, id_col=args.id_col)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    articles_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    for record in records:
        gold_kg = _compact_kg(record["gold_kg"])
        gold_json = json.dumps(gold_kg, ensure_ascii=False, sort_keys=True)
        articles_rows.append(
            {
                "id": record["id"],
                "content": record["content"],
                "source": record["source"],
                "content_sha1": record["content_sha1"],
                "gold_kg_json": gold_json,
            }
        )
        train_rows.append(
            {
                "article_id": record["id"],
                "content_sha1": record["content_sha1"],
                "task_type": "extract_kg",
                "prompt": f"Extract the CTI knowledge graph from article {record['id']}.",
                "answer": gold_json,
                "gold_kg_json": gold_json,
            }
        )
        for edge_index, edge in enumerate(_iter_edges(gold_kg)):
            train_rows.append(
                {
                    "article_id": record["id"],
                    "content_sha1": record["content_sha1"],
                    "task_type": "edge_support",
                    "prompt": (
                        "Which text-provable triple is supported? "
                        f"subject={edge.get('sub')} relation={edge.get('rel')} object={edge.get('obj')}"
                    ),
                    "answer": json.dumps(edge, ensure_ascii=False, sort_keys=True),
                    "gold_kg_json": gold_json,
                    "edge_index": edge_index,
                }
            )

    article_parquet = Path(args.article_parquet or output_dir / "articles.parquet")
    train_parquet = Path(args.train_parquet or output_dir / "train_task_bank.parquet")
    eval_parquet = Path(args.eval_parquet or output_dir / "eval_input.parquet")
    pd.DataFrame(articles_rows).to_parquet(article_parquet, index=False)
    pd.DataFrame(train_rows).to_parquet(train_parquet, index=False)
    pd.DataFrame(articles_rows).to_parquet(eval_parquet, index=False)

    manifest = {
        "created_at": _now_iso(),
        "input_file": str(Path(args.input_file).expanduser().resolve()),
        "article_count": len(articles_rows),
        "training_row_count": len(train_rows),
        "article_parquet": str(article_parquet),
        "train_parquet": str(train_parquet),
        "eval_parquet": str(eval_parquet),
    }
    _dump_json(output_dir / "parquet_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _run_external_command(command: str, cwd: Path) -> int:
    argv = re.findall(r"(?:[^\s\"']+|\"[^\"]*\"|'[^']*')+", command)
    argv = [part.strip("\"'") for part in argv]
    if not argv:
        raise ValueError("Empty command")
    proc = subprocess.run(argv, cwd=str(cwd), check=False)
    return int(proc.returncode)


def _load_train_lookup(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    id_to_kg: Dict[str, Dict[str, Any]] = {}
    hash_to_kg: Dict[str, Dict[str, Any]] = {}
    for _, row in df.drop_duplicates(subset=["article_id", "content_sha1"]).iterrows():
        kg = _compact_kg(_json_or_empty(row["gold_kg_json"]))
        id_to_kg[str(row["article_id"])] = kg
        hash_to_kg[str(row["content_sha1"])] = kg
    return id_to_kg, hash_to_kg


def _triple_key_text(edge: Dict[str, Any]) -> str:
    return " | ".join(_triple_key(edge))


def _mutated_negative_edge(edge: Dict[str, Any], gold_edges: Sequence[Dict[str, Any]], index: int) -> Dict[str, Any]:
    negative = dict(edge)
    if len(gold_edges) > 1:
        other = gold_edges[(index + 1) % len(gold_edges)]
        other_obj = _edge_value(other, OBJ_KEYS)
        if other_obj and other_obj != _edge_value(edge, OBJ_KEYS):
            negative["obj"] = other_obj
            return negative
    rel = _edge_value(edge, REL_KEYS) or "related_to"
    negative["rel"] = f"not_{rel}"
    return negative


def _softmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _sample_index(probs: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    running = 0.0
    for index, prob in enumerate(probs):
        running += prob
        if threshold <= running:
            return index
    return max(0, len(probs) - 1)


def _run_local_rl_smoke(
    id_to_kg: Dict[str, Dict[str, Any]],
    *,
    steps: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    train_cases: List[Dict[str, Any]] = []
    for article_id, kg in id_to_kg.items():
        gold_edges = _iter_edges(kg)
        for edge_index, gold_edge in enumerate(gold_edges):
            negative_edge = _mutated_negative_edge(gold_edge, gold_edges, edge_index)
            train_cases.append(
                {
                    "article_id": article_id,
                    "gold_edge": gold_edge,
                    "negative_edge": negative_edge,
                    "candidate_edges": [gold_edge, negative_edge],
                    "gold_index": 0,
                }
            )

    weights: Dict[str, float] = {}
    trace: List[Dict[str, Any]] = []
    steps = max(0, int(steps))
    learning_rate = float(learning_rate)
    for step in range(steps):
        if not train_cases:
            trace.append(
                {
                    "step": step + 1,
                    "cases": 0,
                    "mean_reward": 0.0,
                    "mean_gold_probability_before_update": 0.0,
                    "mean_policy_loss": 0.0,
                }
            )
            continue

        reward_sum = 0.0
        gold_prob_sum = 0.0
        loss_sum = 0.0
        rng.shuffle(train_cases)
        for case in train_cases:
            candidate_edges = case["candidate_edges"]
            keys = [_triple_key_text(edge) for edge in candidate_edges]
            scores = [weights.get(key, 0.0) for key in keys]
            probs = _softmax(scores)
            selected_index = _sample_index(probs, rng)
            reward = 1.0 if selected_index == case["gold_index"] else 0.0
            expected_reward = probs[case["gold_index"]]
            advantage = reward - expected_reward
            for i, key in enumerate(keys):
                indicator = 1.0 if i == selected_index else 0.0
                weights[key] = weights.get(key, 0.0) + learning_rate * advantage * (indicator - probs[i])
            reward_sum += reward
            gold_prob_sum += expected_reward
            loss_sum += -math.log(max(probs[selected_index], 1e-12)) * reward

        case_count = len(train_cases)
        trace.append(
            {
                "step": step + 1,
                "cases": case_count,
                "mean_reward": reward_sum / case_count,
                "mean_gold_probability_before_update": gold_prob_sum / case_count,
                "mean_policy_loss": loss_sum / case_count,
            }
        )

    return {
        "algorithm": "one-step categorical policy-gradient smoke",
        "description": (
            "A portable local RL smoke test: for each gold KG edge, the policy chooses "
            "between the gold edge and a mutated distractor edge, receives reward 1 for "
            "choosing the gold edge and 0 otherwise, and updates policy weights once per step."
        ),
        "steps_requested": steps,
        "steps_completed": len(trace),
        "learning_rate": learning_rate,
        "seed": seed,
        "training_case_count": len(train_cases),
        "trace": trace,
        "policy_weights": weights,
    }


def cmd_train_export(args: argparse.Namespace) -> int:
    backend = str(args.backend or "local-rl").lower()
    model_dir = _path_from_output_dir(args, args.model_dir, "model_export")
    model_dir.mkdir(parents=True, exist_ok=True)

    if backend == "external":
        command = args.train_command or os.environ.get("GRID_TRAIN_COMMAND", "")
        if not command:
            raise RuntimeError("backend=external requires --train-command or GRID_TRAIN_COMMAND.")
        return _run_external_command(command, REPO_ROOT)

    train_parquet = _path_from_output_dir(args, args.train_parquet, "train_task_bank.parquet")
    if not train_parquet.exists():
        raise FileNotFoundError(f"Training parquet not found: {train_parquet}")
    df = pd.read_parquet(train_parquet)
    required = {"article_id", "content_sha1", "gold_kg_json"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training parquet is missing columns: {sorted(missing)}")

    id_to_kg, hash_to_kg = _load_train_lookup(df)
    rl_report: Dict[str, Any] = {}
    if backend == "local-rl":
        rl_report = _run_local_rl_smoke(
            id_to_kg,
            steps=int(args.rl_steps),
            learning_rate=float(args.rl_learning_rate),
            seed=int(args.rl_seed),
        )

    model = {
        "model_type": "grid-local-rl-smoke-v1" if backend == "local-rl" else "grid-portable-kg-lookup-v1",
        "created_at": _now_iso(),
        "backend": backend,
        "train_parquet": str(train_parquet),
        "training_rows": int(len(df)),
        "article_count": len(id_to_kg),
        "base_model_reference": args.base_model or os.environ.get("GRID_BASE_MODEL", "portable-smoke"),
        "id_to_kg": id_to_kg,
        "content_sha1_to_kg": hash_to_kg,
        "rl_training": rl_report,
    }
    _dump_json(model_dir / "model.json", model)
    if rl_report:
        _dump_json(model_dir / "rl_trace.json", rl_report)
    summary = {
        "model_dir": str(model_dir),
        "model_type": model["model_type"],
        "backend": backend,
        "training_rows": model["training_rows"],
        "article_count": model["article_count"],
        "rl_steps_completed": rl_report.get("steps_completed", 0),
        "rl_training_case_count": rl_report.get("training_case_count", 0),
        "rl_last_mean_reward": (rl_report.get("trace") or [{}])[-1].get("mean_reward", 0.0) if rl_report else 0.0,
        "exported_files": ["model.json", "training_summary.json"] + (["rl_trace.json"] if rl_report else []),
    }
    _dump_json(model_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_exported_model(model_dir: str) -> Dict[str, Any]:
    model_path = Path(model_dir).expanduser().resolve() / "model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Exported model not found: {model_path}")
    return json.loads(model_path.read_text(encoding="utf-8"))


def _predict_from_model(record: Dict[str, Any], model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    by_id = model.get("id_to_kg", {}) or {}
    by_hash = model.get("content_sha1_to_kg", {}) or {}
    if record["id"] in by_id:
        return _compact_kg(by_id[record["id"]])
    if record["content_sha1"] in by_hash:
        return _compact_kg(by_hash[record["content_sha1"]])
    return None


def cmd_generate_kg(args: argparse.Namespace) -> int:
    records = _records_from_args(args.input_file, content_col=args.content_col, id_col=args.id_col)
    backend = str(args.backend or "auto").lower()
    model: Optional[Dict[str, Any]] = None
    if backend in {"auto", "model"} and args.model_dir:
        model = _load_exported_model(str(_path_from_output_dir(args, args.model_dir, "model_export")))

    output_rows: List[Dict[str, Any]] = []
    for record in records:
        kg: Optional[Dict[str, Any]] = None
        used_backend = backend
        if backend in {"auto", "model"} and model is not None:
            kg = _predict_from_model(record, model)
            used_backend = "portable_model"
        if kg is None:
            if backend == "model":
                raise RuntimeError(f"No exported-model prediction for article id={record['id']}")
            if backend in {"auto", "llm"} and _has_llm_env():
                kg = _llm_extract_kg(record["content"], args)
                used_backend = "llm"
            else:
                raise RuntimeError(
                    "No prediction backend available. Provide --model-dir from train-export, "
                    "or set GRID_LLM_MODEL and GRID_LLM_API_KEY for LLM extraction."
                )
        output_rows.append(
            {
                "id": record["id"],
                "content_sha1": record["content_sha1"],
                "backend": used_backend,
                "predicted_kg": _compact_kg(kg),
            }
        )

    output_file = _path_from_output_dir(args, args.output_file, "predictions.jsonl")
    _write_jsonl(output_file, output_rows)
    summary = {
        "created_at": _now_iso(),
        "input_file": str(Path(args.input_file).expanduser().resolve()),
        "prediction_count": len(output_rows),
        "output_file": str(output_file),
        "backend": backend,
    }
    _dump_json(output_file.with_suffix(output_file.suffix + ".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _read_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("predictions", [])
    predictions: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pred_kg = row.get("predicted_kg", row.get("kg", row.get("prediction", {})))
        predictions[str(row.get("id", row.get("article_id", "")))] = _compact_kg(pred_kg)
    return predictions


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cmd_evaluate(args: argparse.Namespace) -> int:
    records = _records_from_args(args.input_file, content_col=args.content_col, id_col=args.id_col)
    predictions_file = _path_from_output_dir(args, args.predictions_file, "predictions.jsonl")
    predictions = _read_predictions(predictions_file)

    per_article: List[Dict[str, Any]] = []
    total_tp = total_fp = total_fn = 0
    for record in records:
        gold_edges = _edge_set(record["gold_kg"])
        pred_edges = _edge_set(predictions.get(record["id"], {}))
        tp = len(gold_edges & pred_edges)
        fp = len(pred_edges - gold_edges)
        fn = len(gold_edges - pred_edges)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_article.append(
            {
                "id": record["id"],
                "gold_edges": len(gold_edges),
                "predicted_edges": len(pred_edges),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": _safe_f1(precision, recall),
            }
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    summary = {
        "created_at": _now_iso(),
        "article_count": len(records),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": _safe_f1(precision, recall),
        "predictions_file": str(predictions_file),
    }
    output_file = _path_from_output_dir(args, args.output_file, "evaluation.json")
    _dump_json(output_file, {"summary": summary, "per_article": per_article})
    pd.DataFrame(per_article).to_csv(output_file.with_suffix(".per_article.csv"), index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    input_file = Path(args.input_file).expanduser().resolve()
    train_parquet = output_dir / "train_task_bank.parquet"
    model_dir = output_dir / "model_export"
    predictions_file = output_dir / "predictions.jsonl"
    eval_file = output_dir / "evaluation.json"

    make_args = argparse.Namespace(
        input_file=str(input_file),
        output_dir=str(output_dir),
        content_col=args.content_col,
        id_col=args.id_col,
        article_parquet="",
        train_parquet=str(train_parquet),
        eval_parquet="",
    )
    train_args = argparse.Namespace(
        backend="local-rl",
        output_dir=str(output_dir),
        train_parquet=str(train_parquet),
        model_dir=str(model_dir),
        base_model="portable-smoke",
        train_command="",
        rl_steps=1,
        rl_learning_rate=0.2,
        rl_seed=7,
    )
    gen_args = argparse.Namespace(
        input_file=str(input_file),
        output_dir=str(output_dir),
        output_file=str(predictions_file),
        model_dir=str(model_dir),
        backend="model",
        content_col=args.content_col,
        id_col=args.id_col,
        llm_model="",
        llm_api_key="",
        llm_base_url="",
        llm_max_tokens=8192,
        llm_temperature=0.0,
        llm_timeout=300,
    )
    eval_args = argparse.Namespace(
        input_file=str(input_file),
        output_dir=str(output_dir),
        predictions_file=str(predictions_file),
        output_file=str(eval_file),
        content_col=args.content_col,
        id_col=args.id_col,
    )
    cmd_make_parquet(make_args)
    cmd_train_export(train_args)
    cmd_generate_kg(gen_args)
    cmd_evaluate(eval_args)
    print(f"Smoke test complete: {eval_file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Docker-friendly GRID artifact entrypoint.")
    parser.add_argument("--output-dir", default=_env_any(("GRID_OUTPUT_DIR",), str(DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--llm-model", default=_env_any(("GRID_LLM_MODEL", "OPENAI_MODEL"), ""))
    parser.add_argument("--llm-api-key", default=_env_any(("GRID_LLM_API_KEY", "GRID_LLM_KEY", "OPENAI_API_KEY"), ""))
    parser.add_argument("--llm-base-url", default=_env_any(("GRID_LLM_BASE_URL", "GRID_LLM_ENDPOINT", "OPENAI_BASE_URL"), ""))

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("env-check", help="Print resolved Docker/runtime environment.")

    def add_input_flags(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--input-file", default=_env_any(("GRID_INPUT_FILE",), str(DEFAULT_SAMPLE_INPUT)))
        sub.add_argument("--content-col", default=_env_any(("GRID_CONTENT_COL",), ""))
        sub.add_argument("--id-col", default=_env_any(("GRID_ID_COL",), ""))

    p_make = subparsers.add_parser("make-parquet", help="Create article/train/eval Parquet files.")
    add_input_flags(p_make)
    p_make.add_argument("--article-parquet", default=_env_any(("GRID_ARTICLE_PARQUET",), ""))
    p_make.add_argument("--train-parquet", default=_env_any(("GRID_TRAIN_PARQUET",), ""))
    p_make.add_argument("--eval-parquet", default=_env_any(("GRID_EVAL_PARQUET",), ""))

    p_train = subparsers.add_parser("train-export", help="Run a portable one-step RL smoke train and export a model.")
    p_train.add_argument("--train-parquet", default=_env_any(("GRID_TRAIN_PARQUET",), str(DEFAULT_OUTPUT_DIR / "train_task_bank.parquet")))
    p_train.add_argument("--model-dir", default=_env_any(("GRID_MODEL_DIR",), str(DEFAULT_OUTPUT_DIR / "model_export")))
    p_train.add_argument("--backend", default=_env_any(("GRID_TRAIN_BACKEND",), "local-rl"), choices=["local-rl", "portable", "external"])
    p_train.add_argument("--base-model", default=_env_any(("GRID_BASE_MODEL",), "portable-smoke"))
    p_train.add_argument("--train-command", default=_env_any(("GRID_TRAIN_COMMAND",), ""))
    p_train.add_argument("--rl-steps", type=int, default=int(_env_any(("GRID_RL_STEPS",), "1")))
    p_train.add_argument("--rl-learning-rate", type=float, default=float(_env_any(("GRID_RL_LEARNING_RATE",), "0.2")))
    p_train.add_argument("--rl-seed", type=int, default=int(_env_any(("GRID_RL_SEED",), "7")))

    p_gen = subparsers.add_parser("generate-kg", help="Generate KG predictions.")
    add_input_flags(p_gen)
    p_gen.add_argument("--model-dir", default=_env_any(("GRID_MODEL_DIR",), str(DEFAULT_OUTPUT_DIR / "model_export")))
    p_gen.add_argument("--output-file", default=_env_any(("GRID_PREDICTIONS_FILE",), str(DEFAULT_OUTPUT_DIR / "predictions.jsonl")))
    p_gen.add_argument("--backend", default=_env_any(("GRID_GENERATE_BACKEND",), "auto"), choices=["auto", "model", "llm"])
    p_gen.add_argument("--llm-max-tokens", type=int, default=int(_env_any(("GRID_LLM_MAX_TOKENS",), "8192")))
    p_gen.add_argument("--llm-temperature", type=float, default=float(_env_any(("GRID_LLM_TEMPERATURE",), "0.0")))
    p_gen.add_argument("--llm-timeout", type=int, default=int(_env_any(("GRID_LLM_TIMEOUT",), "300")))

    p_eval = subparsers.add_parser("evaluate", help="Evaluate predicted KG against gold KG.")
    add_input_flags(p_eval)
    p_eval.add_argument("--predictions-file", default=_env_any(("GRID_PREDICTIONS_FILE",), str(DEFAULT_OUTPUT_DIR / "predictions.jsonl")))
    p_eval.add_argument("--output-file", default=_env_any(("GRID_EVAL_FILE",), str(DEFAULT_OUTPUT_DIR / "evaluation.json")))

    p_smoke = subparsers.add_parser("smoke", help="Run make-parquet -> train-export -> generate-kg -> evaluate.")
    add_input_flags(p_smoke)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "env-check":
        return cmd_env_check(args)
    if args.command == "make-parquet":
        return cmd_make_parquet(args)
    if args.command == "train-export":
        return cmd_train_export(args)
    if args.command == "generate-kg":
        return cmd_generate_kg(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    if args.command == "smoke":
        return cmd_smoke(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
