#!/usr/bin/env python3
"""
06_aggregate_and_analyze.py

Median-of-3 aggregation, C0/C1/C2 invariance analysis, and LLM-grader
reliability (Krippendorff's alpha) over the frozen automated grading data.

NO API CALLS ARE MADE.

Input
-----
- autogrades_parsed_v2.csv

Output
------
- aggregated_scores.csv       (108 rows: one per response, median-of-3 score)
- invariance_summary.csv      (per response model x condition: mean/std,
                                 and delta from C0_base)
- grader_reliability.txt      (Krippendorff's alpha across the 3 graders)

Purpose
-------
This script performs the two analysis steps listed as outstanding in the
project decisions log that do NOT require the pending human-grading data:

  1. Median-of-3 aggregation of the 324 automated judgments down to 108
     per-response scores (3 graders x 108 response cells = 324 rows in).
  2. Core C0 (base) / C1 (character-swap) / C2 (lexical-substitution)
     invariance analysis: how much does the aggregated score move across
     conditions, per response-generation model.
  3. LLM-grader reliability (Krippendorff's alpha) across the 3 graders'
     raw per-judgment scores. This is grader-to-grader reliability, NOT
     the pending human-vs-LLM-grader reliability, which still requires
     the human grading data.

Notes
-----
- If the third-party `krippendorff` package is installed, it is used for
  the alpha calculation (recommended: it is a widely used, tested
  implementation). Otherwise this script falls back to a manual
  implementation of the standard coincidence-matrix algorithm
  (Hayes & Krippendorff, 2007). Given how central this number is to the
  paper, cross-checking the manual fallback's output against the
  `krippendorff` package (or another reference implementation, e.g. R's
  `irr`) before reporting it is strongly recommended.
- The alpha reported here uses the interval difference function
  (delta(v, v') = (v - v')^2), which is the common default for numeric
  1-5 Likert-style scores. If the team decides an ordinal difference
  function is more appropriate for these grades, that would require a
  small change to the `delta` function below -- flagging this as a
  design choice worth confirming with the team rather than assuming.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


# --------------------------------------------------------------------------
# Krippendorff's alpha (interval metric), with optional use of the
# `krippendorff` package if available.
# --------------------------------------------------------------------------

def delta_interval(v1: float, v2: float) -> float:
    return (v1 - v2) ** 2


def krippendorff_alpha_manual(units: list[list[float]]) -> float:
    """
    units: list of per-unit rating lists (each unit = one judged item,
    each inner list = the ratings given to it by different coders).
    Assumes no missing data within a unit (all coders rated all units),
    which holds for this dataset (every response cell has exactly 3
    grader scores).
    """
    coincidence: dict[float, dict[float, float]] = defaultdict(lambda: defaultdict(float))
    n_total = 0.0

    for ratings in units:
        m_u = len(ratings)
        if m_u < 2:
            continue
        weight = 1.0 / (m_u - 1)
        for i in range(m_u):
            for j in range(m_u):
                if i == j:
                    continue
                coincidence[ratings[i]][ratings[j]] += weight
        n_total += m_u

    if n_total == 0:
        raise ValueError("No valid units with >=2 ratings; cannot compute alpha.")

    values = sorted(coincidence.keys() | {v for row in coincidence.values() for v in row})
    marginals = {v: sum(coincidence[v].values()) for v in values}

    Do = 0.0
    for v in values:
        for v2 in values:
            Do += coincidence[v][v2] * delta_interval(v, v2)
    Do /= n_total

    De = 0.0
    for v in values:
        for v2 in values:
            De += marginals[v] * marginals[v2] * delta_interval(v, v2)
    De /= n_total * (n_total - 1)

    if De == 0:
        raise ValueError("Expected disagreement is zero; alpha is undefined (check data).")

    return 1.0 - (Do / De)


def krippendorff_alpha(units: list[list[float]]) -> tuple[float, str]:
    """Returns (alpha, method_used_string)."""
    try:
        import krippendorff  # type: ignore
        import numpy as np  # type: ignore

        max_coders = max(len(u) for u in units)
        matrix = np.full((max_coders, len(units)), np.nan)
        for col, ratings in enumerate(units):
            for row, val in enumerate(ratings):
                matrix[row, col] = val
        alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement="interval")
        return alpha, "krippendorff package (interval)"
    except ImportError:
        return krippendorff_alpha_manual(units), "manual fallback implementation (interval) -- please cross-check"


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def load_rows(input_csv: Path) -> list[dict]:
    with input_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_median_of_3(rows: list[dict]) -> list[dict]:
    """Group the 324 raw judgments by generation_cell_id (108 groups of 3
    grader scores each) and take the median score per group."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["generation_cell_id"]].append(r)

    aggregated = []
    for cell_id, group_rows in groups.items():
        if len(group_rows) != 3:
            print(f"WARNING: {cell_id} has {len(group_rows)} grader judgments, expected 3. Skipping.")
            continue
        scores = [float(r["score"]) for r in group_rows]
        first = group_rows[0]
        aggregated.append({
            "generation_cell_id": cell_id,
            "family_id": first["family_id"],
            "base_id": first["base_id"],
            "condition_code": first["condition_code"],
            "condition_label": first["condition_label"],
            "response_model_key": first["response_model_key"],
            "response_model_display": first["response_model_display"],
            "grader_scores": scores,
            "median_score": statistics.median(scores),
        })
    return aggregated


def write_aggregated_csv(aggregated: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation_cell_id", "family_id", "base_id", "condition_code",
            "condition_label", "response_model_key", "response_model_display",
            "grader_scores", "median_score",
        ])
        for a in aggregated:
            writer.writerow([
                a["generation_cell_id"], a["family_id"], a["base_id"],
                a["condition_code"], a["condition_label"],
                a["response_model_key"], a["response_model_display"],
                ";".join(str(s) for s in a["grader_scores"]),
                a["median_score"],
            ])
    print(f"Wrote {len(aggregated)} aggregated rows -> {out_path}")


def invariance_summary(aggregated: list[dict], out_path: Path) -> None:
    """Per (response_model_key, condition_code): mean/std of median scores,
    plus delta from that model's C0_base mean."""
    by_model_cond: dict[tuple[str, str], list[float]] = defaultdict(list)
    for a in aggregated:
        by_model_cond[(a["response_model_key"], a["condition_code"])].append(a["median_score"])

    models = sorted({k[0] for k in by_model_cond})
    conditions = ["C0_base", "C1_character_swap", "C2_lexical_substitution"]

    rows_out = []
    for model in models:
        base_scores = by_model_cond.get((model, "C0_base"), [])
        base_mean = statistics.mean(base_scores) if base_scores else None
        for cond in conditions:
            scores = by_model_cond.get((model, cond), [])
            if not scores:
                continue
            mean = statistics.mean(scores)
            std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            delta = (mean - base_mean) if base_mean is not None else None
            rows_out.append({
                "response_model_key": model,
                "condition_code": cond,
                "n": len(scores),
                "mean_score": round(mean, 4),
                "std_score": round(std, 4),
                "delta_from_C0_base": round(delta, 4) if delta is not None else "",
            })

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "response_model_key", "condition_code", "n",
            "mean_score", "std_score", "delta_from_C0_base",
        ])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote invariance summary ({len(rows_out)} rows) -> {out_path}")

    print("\n--- Invariance summary (preview) ---")
    for r in rows_out:
        print(f"  {r['response_model_key']:10s} {r['condition_code']:25s} "
              f"n={r['n']:3d}  mean={r['mean_score']:.3f}  "
              f"delta_from_C0={r['delta_from_C0_base']}")


def grader_reliability(rows: list[dict], out_path: Path) -> None:
    """Krippendorff's alpha across the 3 graders, computed over the raw
    per-judgment scores (one unit = one generation_cell_id, one coder =
    one grader)."""
    groups: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        groups[r["generation_cell_id"]][r["grader_key"]] = float(r["score"])

    units = [list(v.values()) for v in groups.values() if len(v) >= 2]
    alpha, method = krippendorff_alpha(units)

    lines = [
        "LLM-grader reliability (Krippendorff's alpha)",
        "=" * 50,
        f"Method: {method}",
        f"Units (response cells): {len(units)}",
        f"Coders (graders) per unit: {len(next(iter(groups.values())))}",
        f"Alpha: {alpha:.4f}",
        "",
        "Note: this is grader-to-grader (LLM vs LLM) reliability, computed",
        "from the 324 automated judgments only. It is NOT the human-vs-LLM",
        "reliability figure, which still requires the pending human grading",
        "data (162 judgments, 3 human raters, DeepSeek-only subset).",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote grader reliability report -> {out_path}")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("autogrades_parsed_v2.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("."))
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    rows = load_rows(args.input)
    print(f"Loaded {len(rows)} raw judgment rows from {args.input}")

    aggregated = aggregate_median_of_3(rows)
    write_aggregated_csv(aggregated, args.outdir / "aggregated_scores.csv")

    invariance_summary(aggregated, args.outdir / "invariance_summary.csv")

    grader_reliability(rows, args.outdir / "grader_reliability.txt")


if __name__ == "__main__":
    main()
