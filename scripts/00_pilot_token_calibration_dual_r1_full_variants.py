#!/usr/bin/env python3
"""
Pilot token-calibration runner for the BiGGen prompt-invariance experiment.

Purpose
-------
Read all 27 rows in prompts_master_final.jsonl (9 families x 3 conditions) and run
every prompt variant through 5 model/settings configurations:

  1) DeepSeek V3, temperature 0
  2) DeepSeek R1, temperature 0
  3) DeepSeek R1, temperature 0.6
  4) GPT-4o, temperature 0
  5) o3-mini, medium reasoning (temperature parameter omitted)

Total pilot cells: 27 x 5 = 135.

The dual R1 runs serve two purposes:
- choose the R1 temperature for the official experiment using the pre-specified
  qualitative repetition/incoherence rule;
- after that choice, use ONLY the winning R1 temperature's token data, together
  with V3/GPT-4o/o3-mini, to set the final universal token cap.

This pilot is calibration data only. Do not grade it as the official experiment.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"

PILOT_MAX_TOKENS = 16000
TIMEOUT_SECONDS = 300
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = [2, 5]

MODELS = [
    {
        "key": "deepseek_v3_temp0",
        "name": "DeepSeek V3 (temp=0)",
        "id": "deepseek/deepseek-chat",
        "provider": "novita",
        "provider_name": "NovitaAI",
        "temperature": 0.0,
        "reasoning": None,
    },
    {
        "key": "deepseek_r1_temp0",
        "name": "DeepSeek R1 (temp=0)",
        "id": "deepseek/deepseek-r1",
        "provider": "novita",
        "provider_name": "NovitaAI",
        "temperature": 0.0,
        "reasoning": None,
    },
    {
        "key": "deepseek_r1_temp06",
        "name": "DeepSeek R1 (temp=0.6)",
        "id": "deepseek/deepseek-r1",
        "provider": "novita",
        "provider_name": "NovitaAI",
        "temperature": 0.6,
        "reasoning": None,
    },
    {
        "key": "gpt4o_temp0",
        "name": "GPT-4o (temp=0)",
        "id": "openai/gpt-4o",
        "provider": "openai",
        "provider_name": "OpenAI",
        "temperature": 0.0,
        "reasoning": None,
    },
    {
        "key": "o3_mini_medium",
        "name": "o3-mini (medium reasoning)",
        "id": "openai/o3-mini",
        "provider": "openai",
        "provider_name": "OpenAI",
        "temperature": None,
        "reasoning": {"effort": "medium"},
    },
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def validate_master(rows):
    if len(rows) != 27:
        raise ValueError(f"Expected 27 master rows; found {len(rows)}.")

    expected_conditions = {
        "C0_base",
        "C1_character_swap",
        "C2_lexical_substitution",
    }

    if sorted({r["family_id"] for r in rows}) != list(range(9)):
        raise ValueError("Expected family_id values 0..8.")

    for fid in range(9):
        fam = [r for r in rows if r["family_id"] == fid]
        if len(fam) != 3:
            raise ValueError(f"Family {fid} has {len(fam)} rows; expected 3.")
        if {r["condition_code"] for r in fam} != expected_conditions:
            raise ValueError(f"Family {fid} does not contain all 3 conditions.")

        for field in (
            "base_id",
            "input",
            "reference_answer",
            "score_rubric",
            "task",
            "capability",
        ):
            values = [r[field] for r in fam]
            if not (values[0] == values[1] == values[2]):
                raise ValueError(
                    f"Family {fid}: field '{field}' differs across conditions."
                )


def cell_id(row, model):
    return f"{row['variant_id']}::{model['key']}"


def existing_statuses(path):
    statuses = {}
    if Path(path).exists():
        for row in load_jsonl(path):
            statuses[row["pilot_cell_id"]] = row.get("generation_status")
    return statuses


def classify_status(code):
    if code == 408:
        return "timeout_http_408", True
    if code == 429:
        return "rate_limited", True
    if 500 <= code <= 599:
        return "server_error", True
    if 400 <= code <= 499:
        return "client_error", False
    return "http_error", False


def payload_for(row, model):
    payload = {
        "model": model["id"],
        "messages": [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["input"]},
        ],
        "max_tokens": PILOT_MAX_TOKENS,
        "provider": {
            "only": [model["provider"]],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }

    if model["temperature"] is not None:
        payload["temperature"] = model["temperature"]

    if model["reasoning"] is not None:
        payload["reasoning"] = model["reasoning"]

    return payload


def one_attempt(session, headers, payload, attempt_number):
    started = now()
    t0 = time.monotonic()

    log = {
        "attempt_number": attempt_number,
        "started_at_utc": started,
        "ended_at_utc": None,
        "elapsed_seconds": None,
        "outcome": None,
        "http_status": None,
        "retryable": None,
        "error_type": None,
        "error_message": None,
    }

    try:
        response = session.post(
            URL,
            headers=headers,
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )

        log["ended_at_utc"] = now()
        log["elapsed_seconds"] = round(time.monotonic() - t0, 3)
        log["http_status"] = response.status_code

        if response.status_code >= 400:
            outcome, retryable = classify_status(response.status_code)
            log["outcome"] = outcome
            log["retryable"] = retryable
            try:
                log["error_message"] = json.dumps(
                    response.json(), ensure_ascii=False
                )[:4000]
            except Exception:
                log["error_message"] = response.text[:4000]
            return log, None

        try:
            data = response.json()
        except ValueError as exc:
            log["outcome"] = "invalid_json_response"
            log["retryable"] = True
            log["error_type"] = type(exc).__name__
            log["error_message"] = str(exc)
            return log, None

        if not data.get("choices"):
            log["outcome"] = "missing_choices"
            log["retryable"] = True
            log["error_message"] = json.dumps(data, ensure_ascii=False)[:4000]
            return log, None

        log["outcome"] = "success"
        log["retryable"] = False
        return log, data

    except requests.Timeout as exc:
        log.update(
            ended_at_utc=now(),
            elapsed_seconds=round(time.monotonic() - t0, 3),
            outcome="timeout",
            retryable=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return log, None

    except requests.ConnectionError as exc:
        log.update(
            ended_at_utc=now(),
            elapsed_seconds=round(time.monotonic() - t0, 3),
            outcome="connection_error",
            retryable=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return log, None

    except requests.RequestException as exc:
        log.update(
            ended_at_utc=now(),
            elapsed_seconds=round(time.monotonic() - t0, 3),
            outcome="request_error",
            retryable=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return log, None


def usage_from(data):
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}

    reasoning_tokens = details.get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = usage.get("reasoning_tokens") or details.get("reasoning")

    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": usage.get("total_tokens"),
        "reported_cost": usage.get("cost"),
    }


def generate_cell(session, headers, row, model):
    attempts = []
    data = None
    payload = payload_for(row, model)

    for attempt_number in range(1, MAX_ATTEMPTS + 1):
        log, data = one_attempt(
            session,
            headers,
            payload,
            attempt_number,
        )
        attempts.append(log)

        if data is not None or not log["retryable"]:
            break

        if attempt_number < MAX_ATTEMPTS:
            delay = BACKOFF_SECONDS[
                min(attempt_number - 1, len(BACKOFF_SECONDS) - 1)
            ]
            time.sleep(delay)

    result = {
        "pilot_cell_id": cell_id(row, model),
        "family_id": row["family_id"],
        "source_instance_idx": row["source_instance_idx"],
        "base_id": row["base_id"],
        "variant_id": row["variant_id"],
        "condition_code": row["condition_code"],
        "condition_label": row["condition_label"],
        "capability": row["capability"],
        "task": row["task"],

        "response_model_key": model["key"],
        "response_model_display": model["name"],
        "requested_model_id": model["id"],
        "requested_provider_slug": model["provider"],
        "requested_provider_display": model["provider_name"],

        "temperature_requested": model["temperature"],
        "temperature_applied": model["temperature"],
        "temperature_note": (
            None
            if model["temperature"] is not None
            else "o3-mini receives no temperature parameter; reasoning effort is medium."
        ),
        "reasoning_config": model["reasoning"],

        "pilot_max_tokens": PILOT_MAX_TOKENS,
        "request_timeout_seconds": TIMEOUT_SECONDS,
        "max_attempts": MAX_ATTEMPTS,

        "system_prompt": row["system_prompt"],
        "input": row["input"],

        "generation_status": None,
        "successful_attempt_number": None,
        "attempts": attempts,

        "response_text": None,
        "finish_reason": None,
        "truncated": None,

        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "reported_cost": None,

        "returned_model_id": None,
        "openrouter_response_id": None,
        "created": None,
        "completed_at_utc": now(),
    }

    if data is None:
        result["generation_status"] = "unresolved_after_max_attempts"
        return result

    choice = data["choices"][0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")
    usage = usage_from(data)

    result.update(
        {
            "generation_status": "success",
            "successful_attempt_number": next(
                a["attempt_number"]
                for a in attempts
                if a["outcome"] == "success"
            ),
            "response_text": message.get("content"),
            "finish_reason": finish_reason,
            "truncated": finish_reason in {"length", "max_tokens"},
            **usage,
            "returned_model_id": data.get("model"),
            "openrouter_response_id": data.get("id"),
            "created": data.get("created"),
        }
    )

    return result


SUMMARY_FIELDS = [
    "pilot_cell_id",
    "family_id",
    "source_instance_idx",
    "condition_code",
    "condition_label",
    "response_model_key",
    "response_model_display",
    "requested_model_id",
    "requested_provider_display",
    "temperature_applied",
    "generation_status",
    "successful_attempt_number",
    "finish_reason",
    "truncated",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "reported_cost",
]


def make_summary(jsonl_path, csv_path):
    rows = load_jsonl(jsonl_path) if Path(jsonl_path).exists() else []

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in SUMMARY_FIELDS})


def write_manifest(path, master_path):
    manifest = {
        "purpose": (
            "Pilot token-cap calibration and DeepSeek-R1 temperature selection only; "
            "not for official grading or final analysis."
        ),
        "created_at_utc": now(),
        "master_file": str(master_path),
        "master_file_sha256": sha256(master_path),
        "master_rows": 27,
        "pilot_conditions": [
            "C0_base",
            "C1_character_swap",
            "C2_lexical_substitution"
        ],
        "pilot_rows_used": 27,
        "expected_generation_cells": 135,
        "openrouter_endpoint": URL,
        "pilot_max_tokens": PILOT_MAX_TOKENS,
        "request_timeout_seconds": TIMEOUT_SECONDS,
        "max_attempts": MAX_ATTEMPTS,
        "models": MODELS,
        "r1_temperature_selection_rule": (
            "Manually inspect all 27 DeepSeek-R1 temperature-0 responses across C0, C1, and C2 "
            "for visible repetition, looping, or incoherence, and compare them against the "
            "corresponding 27 temperature-0.6 responses. If any temperature-0 response exhibits "
            "the targeted pathology, select temperature 0.6 for the official experiment; "
            "otherwise retain temperature 0 for consistency with the other temperature-controlled "
            "models. Do not choose based on token count alone."
        ),
        "token_cap_rule": (
            "After choosing the official R1 temperature, use only the token data from "
            "DeepSeek V3 temp=0, GPT-4o temp=0, o3-mini medium reasoning, and DeepSeek R1 "
            "at the chosen temperature to calibrate the final universal token cap. "
            "Do not pool both R1 temperatures into the cap calculation."
        ),
    }

    Path(path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the 135-cell full-variant pilot with dual DeepSeek-R1 temperatures."
    )
    parser.add_argument(
        "--master",
        type=Path,
        default=Path("prompts_master_final.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("responses_pilot_dual_r1.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("pilot_token_summary_dual_r1.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("pilot_run_manifest_dual_r1.json"),
    )
    parser.add_argument(
        "--retry-unresolved",
        action="store_true",
        help="Retry previously unresolved cells; successful cells are always skipped.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit(
            "OPENROUTER_API_KEY is not set. "
            "Set it as an environment variable; do not hard-code it."
        )

    rows = load_jsonl(args.master)
    validate_master(rows)

    write_manifest(args.manifest, args.master)
    existing = existing_statuses(args.output)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "BiGGen Prompt Invariance Pilot - Dual R1 Temperature",
    }

    session = requests.Session()
    total_cells = len(rows) * len(MODELS)

    print("Validated 27-row master file.")
    print("Using all 27 prompt variants: 9 families x 3 conditions.")
    print(
        f"Pilot cells: {total_cells} = "
        "27 prompt variants x 5 model/settings configurations."
    )
    print("DeepSeek R1 will run at both temperature 0 and 0.6.")
    print(
        f"Pilot max_tokens={PILOT_MAX_TOKENS}, "
        f"timeout={TIMEOUT_SECONDS}s, "
        f"max_attempts={MAX_ATTEMPTS}"
    )
    print()

    cell_index = 0

    for row in rows:
        for model in MODELS:
            cell_index += 1
            cid = cell_id(row, model)
            prior_status = existing.get(cid)

            if prior_status == "success":
                print(
                    f"[{cell_index:02}/{total_cells}] "
                    f"SKIP successful {cid}"
                )
                continue

            if prior_status is not None and not args.retry_unresolved:
                print(
                    f"[{cell_index:02}/{total_cells}] "
                    f"SKIP prior={prior_status} {cid} "
                    "(use --retry-unresolved to retry)"
                )
                continue

            print(
                f"[{cell_index:02}/{total_cells}] RUN "
                f"family={row['family_id']} | {model['name']}"
            )

            result = generate_cell(
                session,
                headers,
                row,
                model,
            )

            append_jsonl(args.output, result)
            existing[cid] = result["generation_status"]
            make_summary(args.output, args.summary)

            if result["generation_status"] == "success":
                print(
                    "   success | "
                    f"completion_tokens={result['completion_tokens']} | "
                    f"reasoning_tokens={result['reasoning_tokens']} | "
                    f"finish={result['finish_reason']} | "
                    f"attempt={result['successful_attempt_number']}"
                )
            else:
                print(
                    "   unresolved | "
                    f"attempts={len(result['attempts'])}"
                )

    make_summary(args.output, args.summary)

    print()
    print(f"Detailed pilot output: {args.output}")
    print(f"Token summary:         {args.summary}")
    print(f"Run manifest:          {args.manifest}")
    print()
    print(
        "Next step: manually compare the 27 R1 temp=0 outputs against the "
        "27 R1 temp=0.6 outputs using the pre-specified repetition/incoherence rule. "
        "Then exclude the losing R1 setting from the token-cap calculation."
    )


if __name__ == "__main__":
    main()
