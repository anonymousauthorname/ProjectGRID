# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DATA_ROOT = REPO_ROOT / "train-data"
DEFAULT_SOURCE_PARQUET = TRAIN_DATA_ROOT / "data" / "representative_articles" / "representative_articles_20.parquet"
DEFAULT_OUTPUT_ROOT = TRAIN_DATA_ROOT / "data" / "representative_articles" / "complexity_sampling_outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="📚 从 complex_order 全量数据按复杂度分布做代表性抽样")
    parser.add_argument("--source-parquet", default=str(DEFAULT_SOURCE_PARQUET))
    parser.add_argument("--sample-count", type=int, default=500)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--dedupe-by-stable-article-id",
        action="store_true",
        help="先按 stable_article_id 去重成文章级，再按文章复杂度做代表性抽样",
    )
    parser.add_argument(
        "--output-parquet-name",
        default="sampled_source_articles_500_complexity_distribution_repr_sorted.parquet",
    )
    return parser


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_jsonish(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            return {}
    return {}


def extract_complexity_score(extra_info: Dict[str, Any]) -> float:
    score_candidates = [
        extra_info.get("文章复杂度分数"),
        (extra_info.get("文章复杂度") or {}).get("final_complexity_score"),
        extra_info.get("step6_题目复杂度_排序有效值"),
        extra_info.get("step6_题目复杂度"),
    ]
    for value in score_candidates:
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def enrich_step6_pair_complexity_aliases(extra_info: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(extra_info)
    payload = enriched.get("step6_gold_edge复杂度")
    if not isinstance(payload, dict):
        return enriched
    gold_edges = list(payload.get("gold_edges") or [])
    enriched["step6_参与题目实体对复杂度payload"] = payload
    enriched["step6_参与题目实体对复杂度列表"] = gold_edges
    enriched["step6_参与题目实体对数量"] = int(payload.get("gold_edge_count") or len(gold_edges) or 0)
    enriched["step6_参与题目实体对复杂度均值"] = float(
        payload.get("gold_edge_complexity_mean", payload.get("题目复杂度", 0.0)) or 0.0
    )
    enriched["step6_参与题目实体对复杂度来源"] = str(payload.get("题目复杂度来源") or payload.get("gold_edge_source") or "")
    return enriched


def build_evenly_spaced_indices(total_size: int, sample_count: int) -> List[int]:
    if sample_count <= 0:
        return []
    if total_size <= sample_count:
        return list(range(total_size))
    raw_positions = [
        int(round(i * (total_size - 1) / (sample_count - 1)))
        for i in range(sample_count)
    ]
    deduped = sorted(set(raw_positions))
    if len(deduped) == sample_count:
        return deduped

    used = set(deduped)
    cursor = 0
    while len(deduped) < sample_count and cursor < total_size:
        if cursor not in used:
            deduped.append(cursor)
            used.add(cursor)
        cursor += 1
    return sorted(deduped[:sample_count])


def write_sampling_png(all_scores: Sequence[float], sampled_scores: Sequence[float], png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("⚠️ matplotlib 不可用，跳过 PNG")
        return

    ensure_dir(png_path.parent)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    axes[0].hist(all_scores, bins=50, alpha=0.65, label="all_articles", color="#1f77b4", density=True)
    axes[0].hist(sampled_scores, bins=50, alpha=0.55, label="sampled_articles", color="#ff7f0e", density=True)
    axes[0].set_title("Complexity Distribution")
    axes[0].set_xlabel("complexity score")
    axes[0].set_ylabel("density")
    axes[0].legend()

    sorted_all = sorted(float(x) for x in all_scores)
    sorted_sampled = sorted(float(x) for x in sampled_scores)
    axes[1].plot(
        sorted_all,
        [(idx + 1) / len(sorted_all) for idx in range(len(sorted_all))],
        label="all_articles",
        color="#1f77b4",
    )
    axes[1].plot(
        sorted_sampled,
        [(idx + 1) / len(sorted_sampled) for idx in range(len(sorted_sampled))],
        label="sampled_articles",
        color="#ff7f0e",
    )
    axes[1].set_title("Complexity CDF")
    axes[1].set_xlabel("complexity score")
    axes[1].set_ylabel("cdf")
    axes[1].legend()

    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    xs = sorted(float(v) for v in a)
    ys = sorted(float(v) for v in b)
    i = j = 0
    n = max(len(xs), 1)
    m = max(len(ys), 1)
    ks = 0.0
    while i < len(xs) or j < len(ys):
        if j >= len(ys) or (i < len(xs) and xs[i] <= ys[j]):
            x = xs[i]
        else:
            x = ys[j]
        while i < len(xs) and xs[i] <= x:
            i += 1
        while j < len(ys) and ys[j] <= x:
            j += 1
        ks = max(ks, abs(i / n - j / m))
    return ks


def main() -> int:
    args = build_parser().parse_args()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    run_dir = output_root / f"run_{now_tag()}"
    ensure_dir(run_dir)

    print(f"📦 读取源 parquet: {source_parquet}")
    df = pd.read_parquet(source_parquet)
    score_list: List[float] = []
    stable_ids: List[str] = []
    for row_idx, row in enumerate(df.to_dict("records")):
        extra_info = load_jsonish(row.get("extra_info"))
        score_list.append(extract_complexity_score(extra_info))
        stable_ids.append(str(extra_info.get("stable_article_id") or row.get("stable_article_id") or f"row_{row_idx}"))

    df = df.copy()
    df["__article_complexity_score"] = score_list
    df["__stable_article_id"] = stable_ids
    df["__source_row_index"] = list(range(len(df)))

    source_row_count = int(len(df))
    if bool(args.dedupe_by_stable_article_id):
        before = len(df)
        
        df = df.drop_duplicates(subset=["__stable_article_id"], keep="first").reset_index(drop=True)
        print(f"🧹 文章级去重完成: rows {before} -> articles {len(df)}")

    df = df.sort_values(["__article_complexity_score", "__stable_article_id"]).reset_index(drop=True)

    sample_indices = build_evenly_spaced_indices(len(df), int(args.sample_count))
    sampled_df = df.iloc[sample_indices].copy().reset_index(drop=True)
    sampled_df["__complexity_sort_rank"] = list(range(len(sampled_df)))
    sampled_df["extra_info"] = [
        enrich_step6_pair_complexity_aliases(load_jsonish(value))
        for value in sampled_df["extra_info"].tolist()
    ]

    sampled_scores = sampled_df["__article_complexity_score"].astype(float).tolist()
    if any(sampled_scores[i] > sampled_scores[i + 1] for i in range(len(sampled_scores) - 1)):
        raise RuntimeError("❌ 抽样结果没有保持复杂度升序")

    output_parquet = run_dir / args.output_parquet_name
    sampled_df.to_parquet(output_parquet, engine="pyarrow", index=False)

    all_scores = df["__article_complexity_score"].astype(float).tolist()
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    summary = {
        "source_parquet": str(source_parquet),
        "output_parquet": str(output_parquet),
        "source_row_count": source_row_count,
        "article_count_all": int(len(df)),
        "article_count_sampled": int(len(sampled_df)),
        "dedupe_by_stable_article_id": bool(args.dedupe_by_stable_article_id),
        "sorted_simple_to_complex_verified": True,
        "sample_indices": sample_indices,
        "ks_statistic": float(ks_statistic(all_scores, sampled_scores)),
        "all_quantiles": {str(q): float(df["__article_complexity_score"].quantile(q)) for q in qs},
        "sample_quantiles": {str(q): float(sampled_df["__article_complexity_score"].quantile(q)) for q in qs},
        "min_complexity_all": float(min(all_scores)),
        "max_complexity_all": float(max(all_scores)),
        "min_complexity_sampled": float(min(sampled_scores)),
        "max_complexity_sampled": float(max(sampled_scores)),
        "mean_complexity_all": float(sum(all_scores) / max(len(all_scores), 1)),
        "mean_complexity_sampled": float(sum(sampled_scores) / max(len(sampled_scores), 1)),
    }
    dump_json(run_dir / "sampling_summary.json", summary)
    dump_json(
        run_dir / "sampled_article_manifest.json",
        [
            {
                "sample_order": idx,
                "stable_article_id": str(sampled_df.iloc[idx]["__stable_article_id"]),
                "complexity_score": float(sampled_df.iloc[idx]["__article_complexity_score"]),
            }
            for idx in range(len(sampled_df))
        ],
    )
    dump_text(
        run_dir / "summary.txt",
        "\n".join(
            [
                f"source_parquet: {source_parquet}",
                f"output_parquet: {output_parquet}",
                f"source_row_count: {source_row_count}",
                f"article_count_all: {len(df)}",
                f"article_count_sampled: {len(sampled_df)}",
                f"dedupe_by_stable_article_id: {bool(args.dedupe_by_stable_article_id)}",
                f"sorted_simple_to_complex_verified: True",
                f"min_complexity_sampled: {min(sampled_scores):.6f}",
                f"max_complexity_sampled: {max(sampled_scores):.6f}",
                f"mean_complexity_sampled: {sum(sampled_scores) / max(len(sampled_scores), 1):.6f}",
                f"ks_statistic: {summary['ks_statistic']:.6f}",
            ]
        ),
    )
    write_sampling_png(all_scores, sampled_scores, run_dir / "complexity_distribution_compare.png")

    print(
        f"✅ 抽样完成: all={len(df)} | sampled={len(sampled_df)} | "
        f"range={min(sampled_scores):.6f}-{max(sampled_scores):.6f} | "
        f"ks={summary['ks_statistic']:.6f}"
    )
    print(f"📁 运行目录: {run_dir}")
    print(f"📄 sampled parquet: {output_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
