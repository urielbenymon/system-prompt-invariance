#!/usr/bin/env python3
"""
03_reparse_autogrades.py

Post-hoc deterministic parser widening for the frozen autograding outputs.

NO API CALLS ARE MADE.

Input
-----
- autogrades_raw.jsonl
- grading_manifest.json

Output
------
- autogrades_raw_v2.jsonl
- autogrades_parsed_v2.csv
- grading_manifest_v2.json

Purpose
-------
Preserve the original API-returned judgments while widening score parsing to
accept deterministic score formats actually produced by the graders.

Parsing priority
----------------
1. [RESULT] N             -> primary_result
2. [Score: N]             -> fallback_bracket_colon
3. [Score N]              -> fallback_bracket_no_colon
4. terminal/bare Score: N -> fallback_bare_score

N must be an integer from 1 through 5.

Rows already parsed by the original parser are verified, not silently changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_INPUT = Path("autogrades_raw.jsonl")
DEFAULT_OUTPUT = Path("autogrades_raw_v2.jsonl")
DEFAULT_CSV = Path("autogrades_parsed_v2.csv")
DEFAULT_MANIFEST = Path("grading_manifest.json")
DEFAULT_MANIFEST_V2 = Path("grading_manifest_v2.json")


PATTERNS = [
    (
        "primary_result",
        re.compile(r"\[\s*RESULT\s*\]\s*[:\-]?\s*([1-5])\b", re.IGNORECASE),
    ),
    (
        "fallback_bracket_colon",
        re.compile(r"\[\s*Score\s*:\s*([1-5])\s*\]", re.IGNORECASE),
    ),
    (
        "fallback_bracket_no_colon",
        re.compile(r"\[\s*Score\s+([1-5])\s*\]", re.IGNORECASE),
    ),
    (
        "fallback_bare_score",
        re.compile(r"(?:^|\n)\s*Score\s*:\s*([1-5])\b", re.IGNORECASE),
    ),
]


CSV_FIELDS = [
    "judgment_id",
    "generation_cell_id",
    "family_id",
    "base_id",
    "variant_id",
    "condition_code",
    "condition_label",
    "capability",
    "task",
    "response_model_key",
    "response_model_display",
    "response_model_id",
    "grader_key",
    "grader_display",
    "requested_grader_model_id",
    "requested_provider_display",
    "returned_provider",
    "temperature",
    "top_p",
    "grading_status",
    "successful_attempt_number",
    "original_parse_status",
    "original_score",
    "parse_status",
    "parse_method",
    "score",
    "feedback",
    "finish_reason",
    "truncated",
    "grader_prompt_tokens",
    "grader_completion_tokens",
    "grader_reasoning_tokens",
    "grader_total_tokens",
    "grader_reported_cost",
    "openrouter_response_id",
    "completed_at_utc",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_no}: {exc}"
                ) from exc
    return rows


def parse_score(text: Optional[str]) -> Tuple[Optional[int], str]:
    if text is None:
        return None, "unparsed"

    for method, pattern in PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return int(matches[-1].group(1)), method

    return None, "unparsed"


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically widen score parsing for frozen autogrades."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-v2", type=Path, default=DEFAULT_MANIFEST_V2)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if len(rows) != 324:
        raise ValueError(f"Expected 324 judgments; found {len(rows)}.")

    original_parsed = sum(
        r.get("parse_status") == "parsed" and r.get("score") in {1, 2, 3, 4, 5}
        for r in rows
    )
    original_unparsed = len(rows) - original_parsed

    updated: List[Dict[str, Any]] = []
    method_counts: Dict[str, int] = {
        "primary_result": 0,
        "fallback_bracket_colon": 0,
        "fallback_bracket_no_colon": 0,
        "fallback_bare_score": 0,
        "unparsed": 0,
    }

    disagreements = 0
    unresolved_ids: List[str] = []

    for original in rows:
        row = dict(original)

        # Preserve original parser state explicitly.
        row["original_parse_status"] = original.get("parse_status")
        row["original_score"] = original.get("score")

        new_score, method = parse_score(original.get("raw_grader_output"))
        row["score"] = new_score
        row["parse_method"] = method
        row["parse_status"] = "parsed" if new_score is not None else "unparsed"

        method_counts[method] = method_counts.get(method, 0) + 1

        # Critical sanity check: an originally valid score must remain identical.
        if (
            original.get("parse_status") == "parsed"
            and original.get("score") in {1, 2, 3, 4, 5}
        ):
            if new_score != int(original["score"]):
                disagreements += 1

        if new_score is None:
            unresolved_ids.append(original.get("judgment_id", "<missing-id>"))

        updated.append(row)

    widened_parsed = sum(r["parse_status"] == "parsed" for r in updated)

    # Hard fail rather than silently emitting a compromised v2 dataset.
    if disagreements:
        raise RuntimeError(
            f"ABORT: {disagreements} originally parsed scores changed under widened parser."
        )
    if unresolved_ids:
        raise RuntimeError(
            "ABORT: widened parser did not recover every score. "
            f"Unresolved judgments: {unresolved_ids}"
        )

    write_jsonl(args.output, updated)
    write_csv(args.csv, updated)

    manifest = {}
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    manifest_v2 = dict(manifest)
    manifest_v2["post_hoc_parser_widening"] = {
        "reason": (
            "GPT-4 Turbo and Llama 3.1 70B Instruct sometimes returned "
            "'[Score: N]', '[Score N]', or bare 'Score: N' instead of the "
            "instructed '[RESULT] N' format. The raw grader outputs were not "
            "regenerated or altered; only deterministic score extraction was widened."
        ),
        "source_raw_file": str(args.input),
        "source_raw_file_sha256": sha256_file(args.input),
        "original_parsed": original_parsed,
        "original_unparsed": original_unparsed,
        "widened_parsed": widened_parsed,
        "widened_unparsed": len(updated) - widened_parsed,
        "parse_method_counts": method_counts,
        "verified_no_change_to_original_parsed_scores": disagreements == 0,
        "disagreements_with_original_scores": disagreements,
        "api_calls_made": 0,
        "original_raw_file_modified": False,
        "v2_raw_file": str(args.output),
        "v2_raw_file_sha256": sha256_file(args.output),
        "v2_csv_file": str(args.csv),
    }

    args.manifest_v2.write_text(
        json.dumps(manifest_v2, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fallback_total = (
        method_counts["fallback_bracket_colon"]
        + method_counts["fallback_bracket_no_colon"]
        + method_counts["fallback_bare_score"]
    )

    print(f"Original: {original_parsed}/324 parsed, {original_unparsed}/324 unparsed")
    print(
        f"Widened:  {widened_parsed}/324 parsed "
        f"({method_counts['primary_result']} via [RESULT], "
        f"{fallback_total} via fallback), "
        f"{324 - widened_parsed}/324 unparsed"
    )
    print(f"  [Score: N]: {method_counts['fallback_bracket_colon']}")
    print(f"  [Score N]:  {method_counts['fallback_bracket_no_colon']}")
    print(f"  Score: N:    {method_counts['fallback_bare_score']}")
    print(f"Disagreements with original scores: {disagreements}")
    print()
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.csv}")
    print(f"Wrote: {args.manifest_v2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
