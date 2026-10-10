#!/usr/bin/env python3
"""
02_autograde_responses.py

Official LLM-as-a-judge grading runner for the prompt-invariance experiment.

Input
-----
- responses_official.jsonl
- OPENROUTER_API_KEY environment variable

Output
------
- autogrades_raw.jsonl
- autogrades_parsed.csv
- grading_manifest.json

Design
------
Each of the 108 frozen official responses is graded independently by 3 evaluator LMs:
1) GPT-4 Turbo
2) Llama 3.1 70B Instruct
3) Claude Opus 4

Total expected judgments: 108 x 3 = 324.

The grading prompt adapts BiGGen Bench's Prometheus absolute-assessment template.
Because this experiment manipulates the system instruction itself, the evaluator
is shown BOTH:
- the exact condition-specific system instruction; and
- the invariant user input.

The evaluator is also shown:
- the frozen response to evaluate;
- the reference answer;
- the instance-specific 1–5 score rubric.

The expected grader output is:
    Feedback: ... [RESULT] N
where N is an integer from 1 to 5.

Important:
- This script NEVER modifies responses_official.jsonl.
- Successful grader judgments are never regenerated automatically.
- API/provider failures may be retried up to MAX_ATTEMPTS.
- A successful API response with an unparsable score is preserved as-is and
  is NOT silently regenerated. It is marked parse_status="unparsed".
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_INPUT = Path("responses_official.jsonl")
DEFAULT_OUTPUT = Path("autogrades_raw.jsonl")
DEFAULT_PARSED = Path("autogrades_parsed.csv")
DEFAULT_MANIFEST = Path("grading_manifest.json")

GRADER_MAX_TOKENS = 2048
TIMEOUT_SECONDS = 300
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = [2, 5]

# We use temperature=0 for evaluator reproducibility, matching BiGGen's stated
# evaluator temperature. top_p=0.9 is also retained from BiGGen.
# repetition_penalty=1.03 is intentionally omitted because it is not a
# consistently supported cross-provider parameter for this heterogeneous panel.
GRADERS: List[Dict[str, Any]] = [
    {
        "grader_key": "gpt4_turbo",
        "display_name": "GPT-4 Turbo",
        "model_id": "openai/gpt-4-turbo",
        "provider_slug": "openai",
        "provider_display": "OpenAI",
        "temperature": 0.0,
        "top_p": 0.9,
    },
    {
        "grader_key": "llama_3_1_70b_instruct",
        "display_name": "Llama 3.1 70B Instruct",
        "model_id": "meta-llama/llama-3.1-70b-instruct",
        "provider_slug": "deepinfra",
        "provider_display": "DeepInfra (Turbo)",
        "temperature": 0.0,
        "top_p": 0.9,
    },
    {
        "grader_key": "claude_opus_4",
        "display_name": "Claude Opus 4",
        "model_id": "anthropic/claude-opus-4",
        "provider_slug": "google-vertex",
        "provider_display": "Google Vertex",
        "temperature": 0.0,
        "top_p": 0.9,
    },
]

ABSOLUTE_GRADING_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criterion are given.

1. Write detailed feedback that assesses the quality of the response strictly based on the given score rubric, not by evaluating the response in general.
2. After writing the feedback, write a score that is an integer between 1 and 5. Refer to the score rubric when assigning the score.
3. The output format must be exactly:
Feedback: (write feedback for the criterion) [RESULT] (an integer from 1 to 5)
4. Do not generate any other opening, closing, or explanation.

###The instruction to evaluate:
System instruction:
{system_prompt}

User input:
{user_input}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{score_rubric}

###Feedback:
"""

SCORE_RUBRIC_TEMPLATE = """Criteria:
{criteria}

Score 1:
{score1_description}

Score 2:
{score2_description}

Score 3:
{score3_description}

Score 4:
{score4_description}

Score 5:
{score5_description}
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def validate_official_responses(rows: List[Dict[str, Any]]) -> None:
    if len(rows) != 108:
        raise ValueError(f"Expected 108 official responses; found {len(rows)}.")

    ids = [r.get("generation_cell_id") for r in rows]
    if None in ids:
        raise ValueError("At least one response row is missing generation_cell_id.")
    if len(set(ids)) != 108:
        raise ValueError("generation_cell_id values are not unique.")

    expected_conditions = {
        "C0_base",
        "C1_character_swap",
        "C2_lexical_substitution",
    }
    if {r["condition_code"] for r in rows} != expected_conditions:
        raise ValueError("Official response conditions are incomplete or unexpected.")

    required_fields = {
        "generation_cell_id",
        "family_id",
        "base_id",
        "variant_id",
        "condition_code",
        "condition_label",
        "capability",
        "task",
        "system_prompt",
        "input",
        "reference_answer",
        "score_rubric",
        "response_model_key",
        "response_model_display",
        "requested_model_id",
        "generation_status",
        "response_text",
        "finish_reason",
        "truncated",
    }

    for i, row in enumerate(rows):
        missing = required_fields - set(row.keys())
        if missing:
            raise ValueError(
                f"Official response row {i} missing fields: {sorted(missing)}"
            )

        if row["generation_status"] != "success":
            raise ValueError(
                f"Official response row {i} is not successful: "
                f"{row['generation_status']}"
            )

        # Empty responses are legitimate experimental outcomes. Do not reject them
        # here: preserve and grade them as-is rather than blocking the entire run.
        # The generation-stage `empty_response` flag remains available for analysis.


def format_rubric(rubric: Dict[str, Any]) -> str:
    required = {
        "criteria",
        "score1_description",
        "score2_description",
        "score3_description",
        "score4_description",
        "score5_description",
    }
    missing = required - set(rubric.keys())
    if missing:
        raise ValueError(f"Rubric missing fields: {sorted(missing)}")
    return SCORE_RUBRIC_TEMPLATE.format(**rubric)


def build_grading_prompt(row: Dict[str, Any]) -> str:
    return ABSOLUTE_GRADING_TEMPLATE.format(
        system_prompt=row["system_prompt"],
        user_input=row["input"],
        response=(
            row["response_text"]
            if row["response_text"] is not None
            else ""
        ),
        reference_answer=row["reference_answer"],
        score_rubric=format_rubric(row["score_rubric"]),
    )


def judgment_id(row: Dict[str, Any], grader: Dict[str, Any]) -> str:
    return f"{row['generation_cell_id']}::{grader['grader_key']}"


def existing_statuses(path: Path) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    if not path.exists():
        return statuses

    for row in load_jsonl(path):
        jid = row.get("judgment_id")
        if jid:
            statuses[jid] = row.get("grading_status", "unknown")
    return statuses


def classify_http_error(status_code: int) -> Tuple[str, bool]:
    if status_code == 408:
        return "timeout_http_408", True
    if status_code == 429:
        return "rate_limited", True
    if 500 <= status_code <= 599:
        return "server_error", True
    if 400 <= status_code <= 499:
        return "client_error", False
    return "http_error", False


def sanitize_error_message(message: Optional[str], api_key: str) -> Optional[str]:
    if message is None:
        return None

    sanitized = message.replace(api_key, "[REDACTED_API_KEY]") if api_key else message

    # Defensive removal of bearer-like credentials if a library error echoes a header.
    sanitized = re.sub(
        r"Bearer\s+[A-Za-z0-9._\-]+",
        "Bearer [REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized[:4000]


def build_payload(prompt: str, grader: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": grader["model_id"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an evaluator grading another model's response. "
                    "Follow the supplied rubric and output format exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": GRADER_MAX_TOKENS,
        "temperature": grader["temperature"],
        "top_p": grader["top_p"],
        "provider": {
            "only": [grader["provider_slug"]],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
    return payload


def one_attempt(
    session: requests.Session,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    attempt_number: int,
    api_key: str,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    started_at = utc_now_iso()
    t0 = time.monotonic()

    log: Dict[str, Any] = {
        "attempt_number": attempt_number,
        "started_at_utc": started_at,
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
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )

        log["ended_at_utc"] = utc_now_iso()
        log["elapsed_seconds"] = round(time.monotonic() - t0, 3)
        log["http_status"] = response.status_code

        if response.status_code >= 400:
            outcome, retryable = classify_http_error(response.status_code)
            log["outcome"] = outcome
            log["retryable"] = retryable

            try:
                raw_error = json.dumps(response.json(), ensure_ascii=False)
            except Exception:
                raw_error = response.text

            log["error_message"] = sanitize_error_message(raw_error, api_key)
            return log, None

        try:
            data = response.json()
        except ValueError as exc:
            log["outcome"] = "invalid_json_response"
            log["retryable"] = True
            log["error_type"] = type(exc).__name__
            log["error_message"] = sanitize_error_message(str(exc), api_key)
            return log, None

        if not data.get("choices"):
            log["outcome"] = "missing_choices"
            log["retryable"] = True
            log["error_message"] = sanitize_error_message(
                json.dumps(data, ensure_ascii=False), api_key
            )
            return log, None

        log["outcome"] = "success"
        log["retryable"] = False
        return log, data

    except requests.Timeout as exc:
        log.update(
            ended_at_utc=utc_now_iso(),
            elapsed_seconds=round(time.monotonic() - t0, 3),
            outcome="timeout",
            retryable=True,
            error_type=type(exc).__name__,
            error_message=sanitize_error_message(str(exc), api_key),
        )
        return log, None

    except requests.ConnectionError as exc:
        log.update(
            ended_at_utc=utc_now_iso(),
            elapsed_seconds=round(time.monotonic() - t0, 3),
            outcome="connection_error",
            retryable=True,
            error_type=type(exc).__name__,
            error_message=sanitize_error_message(str(exc), api_key),
        )
        return log, None

    except requests.RequestException as exc:
        log.update(
            ended_at_utc=utc_now_iso(),
            elapsed_seconds=round(time.monotonic() - t0, 3),
            outcome="request_error",
            retryable=False,
            error_type=type(exc).__name__,
            error_message=sanitize_error_message(str(exc), api_key),
        )
        return log, None


def extract_usage(data: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}

    reasoning_tokens = details.get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = usage.get("reasoning_tokens") or details.get("reasoning")

    return {
        "grader_prompt_tokens": usage.get("prompt_tokens"),
        "grader_completion_tokens": usage.get("completion_tokens"),
        "grader_reasoning_tokens": reasoning_tokens,
        "grader_total_tokens": usage.get("total_tokens"),
        "grader_reported_cost": usage.get("cost"),
    }


RESULT_PATTERNS = [
    re.compile(r"\[\s*RESULT\s*\]\s*[:\-]?\s*([1-5])\b", re.IGNORECASE),
    re.compile(r"\bRESULT\s*[:\-]\s*([1-5])\b", re.IGNORECASE),
]


def parse_score(text: Optional[str]) -> Tuple[Optional[int], str]:
    if text is None:
        return None, "unparsed"

    for pattern in RESULT_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            # Use the final explicit result if a model mentions RESULT more than once.
            return int(matches[-1]), "parsed"

    return None, "unparsed"


def extract_feedback(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    # Remove the final [RESULT] segment while preserving the actual feedback.
    match = None
    for pattern in RESULT_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            match = matches[-1]
            break

    feedback = text[: match.start()] if match else text
    feedback = feedback.strip()

    if feedback.lower().startswith("feedback:"):
        feedback = feedback[len("feedback:"):].strip()

    return feedback or None


def grade_one(
    session: requests.Session,
    headers: Dict[str, str],
    row: Dict[str, Any],
    grader: Dict[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    prompt = build_grading_prompt(row)
    payload = build_payload(prompt, grader)

    attempts: List[Dict[str, Any]] = []
    successful_data: Optional[Dict[str, Any]] = None

    for attempt_number in range(1, MAX_ATTEMPTS + 1):
        log, data = one_attempt(
            session=session,
            headers=headers,
            payload=payload,
            attempt_number=attempt_number,
            api_key=api_key,
        )
        attempts.append(log)

        if data is not None:
            successful_data = data
            break

        if not log.get("retryable", False):
            break

        if attempt_number < MAX_ATTEMPTS:
            delay = BACKOFF_SECONDS[
                min(attempt_number - 1, len(BACKOFF_SECONDS) - 1)
            ]
            time.sleep(delay)

    result: Dict[str, Any] = {
        "judgment_id": judgment_id(row, grader),

        # Link back to the immutable response
        "generation_cell_id": row["generation_cell_id"],
        "family_id": row["family_id"],
        "base_id": row["base_id"],
        "variant_id": row["variant_id"],
        "condition_code": row["condition_code"],
        "condition_label": row["condition_label"],
        "capability": row["capability"],
        "task": row["task"],
        "response_model_key": row["response_model_key"],
        "response_model_display": row["response_model_display"],
        "response_model_id": row["requested_model_id"],

        # Exact grading materials
        "system_prompt": row["system_prompt"],
        "input": row["input"],
        "reference_answer": row["reference_answer"],
        "score_rubric": row["score_rubric"],
        "response_text": row["response_text"],
        "grader_prompt": prompt,

        # Requested evaluator configuration
        "grader_key": grader["grader_key"],
        "grader_display": grader["display_name"],
        "requested_grader_model_id": grader["model_id"],
        "requested_provider_slug": grader["provider_slug"],
        "requested_provider_display": grader["provider_display"],
        "temperature": grader["temperature"],
        "top_p": grader["top_p"],
        "repetition_penalty": None,
        "grader_max_tokens": GRADER_MAX_TOKENS,
        "request_timeout_seconds": TIMEOUT_SECONDS,
        "max_attempts": MAX_ATTEMPTS,

        # Grading result
        "grading_status": None,
        "successful_attempt_number": None,
        "attempts": attempts,
        "raw_grader_output": None,
        "feedback": None,
        "score": None,
        "parse_status": None,
        "finish_reason": None,
        "truncated": None,

        # Usage / provenance
        "grader_prompt_tokens": None,
        "grader_completion_tokens": None,
        "grader_reasoning_tokens": None,
        "grader_total_tokens": None,
        "grader_reported_cost": None,
        "returned_grader_model_id": None,
        "returned_provider": None,
        "openrouter_response_id": None,
        "created": None,
        "completed_at_utc": utc_now_iso(),
    }

    if successful_data is None:
        result["grading_status"] = "unresolved_after_max_attempts"
        result["parse_status"] = "not_applicable"
        return result

    choice = successful_data["choices"][0]
    message = choice.get("message") or {}
    raw_output = message.get("content")
    finish_reason = choice.get("finish_reason")

    score, parse_status = parse_score(raw_output)
    feedback = extract_feedback(raw_output)
    usage = extract_usage(successful_data)

    result.update(
        {
            "grading_status": "success",
            "successful_attempt_number": next(
                a["attempt_number"]
                for a in attempts
                if a["outcome"] == "success"
            ),
            "raw_grader_output": raw_output,
            "feedback": feedback,
            "score": score,
            "parse_status": parse_status,
            "finish_reason": finish_reason,
            "truncated": finish_reason in {"length", "max_tokens"},
            **usage,
            "returned_grader_model_id": successful_data.get("model"),
            "returned_provider": successful_data.get("provider"),
            "openrouter_response_id": successful_data.get("id"),
            "created": successful_data.get("created"),
        }
    )

    return result


PARSED_FIELDS = [
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
    "parse_status",
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


def rebuild_parsed_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows = load_jsonl(jsonl_path) if jsonl_path.exists() else []

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PARSED_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in PARSED_FIELDS})


def write_manifest(
    manifest_path: Path,
    input_path: Path,
    script_path: Path,
) -> None:
    manifest = {
        "experiment_stage": "official_autograding",
        "purpose": (
            "Grade the frozen 108 official response cells using three independent "
            "evaluator language models."
        ),
        "created_at_utc": utc_now_iso(),

        "input_file": str(input_path),
        "input_file_sha256": sha256_file(input_path),
        "expected_response_cells": 108,
        "graders": 3,
        "expected_judgments": 324,

        "openrouter_endpoint": OPENROUTER_URL,
        "provider_fallbacks_allowed": False,
        "grader_max_tokens": GRADER_MAX_TOKENS,
        "request_timeout_seconds": TIMEOUT_SECONDS,
        "max_attempts": MAX_ATTEMPTS,
        "grader_models": GRADERS,

        "grading_protocol": {
            "framework": "BiGGen/Prometheus-style absolute direct assessment",
            "score_scale": "integer 1-5",
            "expected_output_format": "Feedback: ... [RESULT] N",
            "adaptation": (
                "Unlike BiGGen's released run_response_eval.py, which passes the "
                "instance input as orig_instruction, this experiment includes both "
                "the exact condition-specific system instruction and the user input "
                "because the independent variable is a system-instruction transformation."
            ),
            "reference_answer_provided": True,
            "instance_specific_rubric_provided": True,
            "pairwise_comparison": False,
        },

        "decoding": {
            "temperature": 0.0,
            "top_p": 0.9,
            "repetition_penalty": None,
            "repetition_penalty_note": (
                "BiGGen reports repetition_penalty=1.03, but it is intentionally "
                "omitted here because it is not a consistently supported parameter "
                "across the heterogeneous OpenAI, Meta-hosted, and Anthropic grader endpoints."
            ),
        },

        "retry_policy": {
            "transient_api_failures_retryable": True,
            "successful_api_judgment_regenerated_automatically": False,
            "unparsable_successful_output_regenerated_automatically": False,
        },

        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "requests_version": getattr(requests, "__version__", None),
        },

        "script_file": str(script_path),
        "script_file_sha256": (
            sha256_file(script_path) if script_path.exists() else None
        ),
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def audit_output(path: Path) -> Dict[str, Any]:
    rows = load_jsonl(path) if path.exists() else []

    successful = [r for r in rows if r.get("grading_status") == "success"]
    unresolved = [r for r in rows if r.get("grading_status") != "success"]
    parsed = [r for r in successful if r.get("parse_status") == "parsed"]

    return {
        "rows_written": len(rows),
        "unique_judgment_ids": len({r.get("judgment_id") for r in rows}),
        "successful_rows": len(successful),
        "unresolved_rows": len(unresolved),
        "parsed_scores": len(parsed),
        "unparsed_scores": len(successful) - len(parsed),
        "truncated_rows": sum(bool(r.get("truncated")) for r in successful),
        "retry_success_rows": sum(
            1
            for r in successful
            if (r.get("successful_attempt_number") or 0) > 1
        ),
        "total_reported_cost": sum(
            float(r.get("grader_reported_cost") or 0.0)
            for r in successful
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Autograde 108 frozen official responses with 3 evaluator LMs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Frozen official responses JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Raw autogrades JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--parsed",
        type=Path,
        default=DEFAULT_PARSED,
        help=f"Parsed autogrades CSV (default: {DEFAULT_PARSED})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Grading manifest JSON (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--retry-unresolved",
        action="store_true",
        help=(
            "Retry only previously unresolved API judgments. "
            "Successful judgments are always skipped, including successful "
            "judgments whose score parser failed."
        ),
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 2

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    rows = load_jsonl(args.input)
    validate_official_responses(rows)

    script_path = Path(__file__).resolve()
    write_manifest(args.manifest, args.input, script_path)

    existing = existing_statuses(args.output)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Prompt Invariance Official Autograding",
    }

    session = requests.Session()
    total_judgments = len(rows) * len(GRADERS)

    print("Official autograding configuration validated.")
    print("Frozen response cells: 108.")
    print("Evaluator models: 3.")
    print(f"Expected judgments: {total_judgments}.")
    print("Scoring: BiGGen/Prometheus-style direct assessment, 1-5.")
    print("System instruction + user input are both shown to evaluators.")
    print("Temperature=0, top_p=0.9.")
    print("Provider fallbacks disabled.")
    print()

    judgment_index = 0
    generated_this_run = 0
    skipped_existing = 0

    for row in rows:
        for grader in GRADERS:
            judgment_index += 1
            jid = judgment_id(row, grader)
            prior_status = existing.get(jid)

            if prior_status == "success":
                print(
                    f"[{judgment_index:03}/{total_judgments}] "
                    f"SKIP successful {jid}"
                )
                skipped_existing += 1
                continue

            if prior_status is not None and not args.retry_unresolved:
                print(
                    f"[{judgment_index:03}/{total_judgments}] "
                    f"SKIP prior={prior_status} {jid} "
                    "(use --retry-unresolved to retry)"
                )
                skipped_existing += 1
                continue

            print(
                f"[{judgment_index:03}/{total_judgments}] RUN "
                f"family={row['family_id']} "
                f"condition={row['condition_code']} "
                f"response_model={row['response_model_display']} "
                f"grader={grader['display_name']}"
            )

            result = grade_one(
                session=session,
                headers=headers,
                row=row,
                grader=grader,
                api_key=api_key,
            )

            append_jsonl(args.output, result)
            existing[jid] = result["grading_status"]
            generated_this_run += 1

            # Keep a usable parsed CSV even if interrupted.
            rebuild_parsed_csv(args.output, args.parsed)

            if result["grading_status"] == "success":
                print(
                    "    success | "
                    f"score={result['score']} | "
                    f"parse={result['parse_status']} | "
                    f"finish={result['finish_reason']} | "
                    f"attempt={result['successful_attempt_number']}"
                )
            else:
                last = result["attempts"][-1] if result["attempts"] else {}
                print(
                    "    unresolved | "
                    f"last_outcome={last.get('outcome')} | "
                    f"attempts={len(result['attempts'])}"
                )

    rebuild_parsed_csv(args.output, args.parsed)
    audit = audit_output(args.output)

    print()
    print("Official autograding run complete/resumed.")
    print(f"Generated this run: {generated_this_run}")
    print(f"Skipped existing judgments: {skipped_existing}")
    print(f"Rows written: {audit['rows_written']}")
    print(f"Unique judgment IDs: {audit['unique_judgment_ids']}")
    print(f"Successful API judgments: {audit['successful_rows']}")
    print(f"Unresolved API judgments: {audit['unresolved_rows']}")
    print(f"Parsed scores: {audit['parsed_scores']}")
    print(f"Unparsed successful outputs: {audit['unparsed_scores']}")
    print(f"Truncated grader outputs: {audit['truncated_rows']}")
    print(f"Successes requiring retry: {audit['retry_success_rows']}")
    print(f"Total reported grader cost: ${audit['total_reported_cost']:.6f}")
    print()
    print(f"Raw autogrades: {args.output}")
    print(f"Parsed autogrades: {args.parsed}")
    print(f"Grading manifest: {args.manifest}")

    if (
        audit["rows_written"] == 324
        and audit["unique_judgment_ids"] == 324
        and audit["successful_rows"] == 324
        and audit["parsed_scores"] == 324
    ):
        print()
        print(
            "INTEGRITY CHECK: 324/324 judgments succeeded and produced parsed 1-5 scores."
        )
        print(
            "Freeze autogrades_raw.jsonl before aggregation or human-grading comparison."
        )
    else:
        print()
        print(
            "INTEGRITY CHECK NOT YET COMPLETE: inspect unresolved or unparsed "
            "judgments before aggregate analysis."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
