#!/usr/bin/env python3
"""
make_rater_packets.py

Builds blinded human-grading packets from responses_official.jsonl.

- Scope: DeepSeek V3 + DeepSeek R1 responses only (full-share design).
- Each of 3 raters receives the SAME 54 items, in an independently
  seeded random order.
- Model identity, condition label, and all provenance/metadata fields
  are stripped before raters ever see a row.
- A private id_mapping.csv links opaque item_id -> real generation_cell_id
  for later rejoining; this file is NEVER given to raters.

No API calls. Pure local transformation of already-frozen data.
"""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

DEEPSEEK_KEYS = {"deepseek_v3", "deepseek_r1"}
NUM_RATERS = 3

RATER_CSV_FIELDS = [
    "item_id",
    "task_prompt",
    "system_instruction",
    "response_to_evaluate",
    "reference_answer",
    "score_rubric",
    "score",
    "comments",
]

ATTESTATION_TEXT = (
    "I confirm that the scores I record below reflect my own independent "
    "judgment, without assistance from any AI/LLM tool. Rater: ____________  "
    "Date: ____________"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def render_rubric(rubric: dict) -> str:
    required = [
        "criteria",
        "score1_description",
        "score2_description",
        "score3_description",
        "score4_description",
        "score5_description",
    ]
    missing = [k for k in required if k not in rubric]
    if missing:
        raise ValueError(f"Rubric missing fields: {missing}")

    lines = [f"Criteria: {rubric['criteria']}", ""]
    for i in range(1, 6):
        lines.append(f"Score {i}: {rubric[f'score{i}_description']}")
    return "\n".join(lines)


def build_item_pool(rows):
    """Filter to DeepSeek-only, assign fixed opaque item_ids (stable order),
    and strip to rater-facing content only."""
    filtered = [r for r in rows if r.get("response_model_key") in DEEPSEEK_KEYS]

    if len(filtered) != 54:
        raise ValueError(
            f"Expected 54 DeepSeek-only rows (9 families x 3 conditions x 2 models); "
            f"found {len(filtered)}."
        )

    # Stable, deterministic base ordering so item_id assignment is reproducible
    # regardless of input file row order.
    filtered.sort(key=lambda r: r["generation_cell_id"])

    pool = []
    mapping = []
    for i, row in enumerate(filtered, start=1):
        item_id = f"item_{i:03}"
        pool.append(
            {
                "item_id": item_id,
                "task_prompt": row["input"],
                "system_instruction": row["system_prompt"],
                "response_to_evaluate": row["response_text"],
                "reference_answer": row["reference_answer"],
                "score_rubric": render_rubric(row["score_rubric"]),
                "score": "",
                "comments": "",
            }
        )
        mapping.append(
            {
                "item_id": item_id,
                "generation_cell_id": row["generation_cell_id"],
                "family_id": row["family_id"],
                "condition_code": row["condition_code"],
                "response_model_key": row["response_model_key"],
            }
        )
    return pool, mapping


def write_rater_csv(path: Path, items: list, seed: int):
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RATER_CSV_FIELDS)
        writer.writeheader()
        for row in shuffled:
            writer.writerow(row)

    return [row["item_id"] for row in shuffled]


def write_mapping_csv(path: Path, mapping: list):
    fields = ["item_id", "generation_cell_id", "family_id", "condition_code", "response_model_key"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in mapping:
            writer.writerow(row)


def write_instructions(path: Path):
    text = f"""# Human Grading Instructions

You will grade {54} model responses. For each row:

1. Read the system instruction and task prompt.
2. Read the response to evaluate.
3. Read the reference answer (an example of a score-5 response).
4. Read the score rubric carefully.
5. Enter a single integer score (1-5) in the `score` column, based strictly
   on the rubric -- not on your general impression of response quality.
6. Optionally, note your reasoning in the `comments` column.

## AI Assistance Policy

Do NOT use any AI/LLM tool (ChatGPT, Claude, Copilot, etc.) to help decide
scores. You may look up unfamiliar words or general facts if needed to
understand a task, but the scoring judgment itself must be entirely your own.

{ATTESTATION_TEXT}

## Notes

- Item order is randomized and differs between raters. This is intentional.
- Some system instructions may contain unusual wording, typos, or phrasing.
  Grade the response on its merits per the rubric regardless of how the
  instruction is phrased.
- Please complete all rows in one or a few sittings if possible, to keep
  your grading standards consistent across items.
"""
    path.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("responses_official.jsonl"))
    ap.add_argument("--outdir", type=Path, default=Path("."))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.input)
    pool, mapping = build_item_pool(rows)

    packet_orders = {}
    for rater_id in range(1, NUM_RATERS + 1):
        packet_path = args.outdir / f"rater_{rater_id}_packet.csv"
        order = write_rater_csv(packet_path, pool, seed=rater_id)
        packet_orders[f"rater_{rater_id}"] = {"seed": rater_id, "item_order": order}
        print(f"Wrote {packet_path} (54 items, seed={rater_id})")

    mapping_path = args.outdir / "id_mapping_PRIVATE.csv"
    write_mapping_csv(mapping_path, mapping)
    print(f"Wrote {mapping_path} (DO NOT share with raters)")

    instructions_path = args.outdir / "rater_instructions.md"
    write_instructions(instructions_path)
    print(f"Wrote {instructions_path}")

    manifest = {
        "source_file": str(args.input),
        "source_file_sha256": sha256_file(args.input),
        "scope": "DeepSeek V3 + DeepSeek R1 only (full-share design)",
        "total_items": 54,
        "num_raters": NUM_RATERS,
        "shuffle_method": "random.Random(seed=rater_id).shuffle, seed = rater number (1,2,3)",
        "item_id_assignment": "Fixed, stable-sorted by generation_cell_id before shuffling; identical across raters",
        "blinded_fields_removed": [
            "generation_cell_id", "family_id", "variant_id", "condition_code",
            "condition_label", "response_model_key", "response_model_display",
            "requested_model_id", "capability", "all token/cost/provider/timestamp fields",
        ],
        "ai_assistance_policy": "Disallowed for scoring judgment; attestation required (see rater_instructions.md)",
        "rater_packet_orders": packet_orders,
    }
    manifest_path = args.outdir / "packet_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
