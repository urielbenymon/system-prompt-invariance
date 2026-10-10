#!/usr/bin/env python3

import json
import statistics
from pathlib import Path

FILES = {
    "Official generation": "responses_official.jsonl",
    "Pilot generation": "responses_pilot_dual_r1.jsonl",
    "Automated grading": "autogrades_raw_v2.jsonl",
}


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"{path}: invalid JSON on line {line_num}: {e}"
                )
    return rows


def audit_file(label, filename):
    path = Path(filename)

    if not path.exists():
        print(f"\n{label}")
        print(f"  ERROR: File not found: {path.resolve()}")
        return None

    rows = load_jsonl(path)

    attempt_times = []
    successful_attempt_times = []
    total_attempts = 0
    rows_with_retries = 0
    failed_attempts = 0
    missing_elapsed = 0

    for row in rows:
        attempts = row.get("attempts", [])

        if len(attempts) > 1:
            rows_with_retries += 1

        total_attempts += len(attempts)

        for attempt in attempts:
            elapsed = attempt.get("elapsed_seconds")

            if isinstance(elapsed, (int, float)):
                elapsed = float(elapsed)
                attempt_times.append(elapsed)

                if attempt.get("ok") is True:
                    successful_attempt_times.append(elapsed)
            else:
                missing_elapsed += 1

            if attempt.get("ok") is not True:
                failed_attempts += 1

    total_time = sum(attempt_times)

    print("\n" + "=" * 68)
    print(label)
    print("=" * 68)
    print(f"File:                         {filename}")
    print(f"Experimental rows/cells:      {len(rows)}")
    print(f"Total API attempts:            {total_attempts}")
    print(f"Rows requiring retries:        {rows_with_retries}")
    print(f"Failed/non-success attempts:   {failed_attempts}")
    print(f"Attempts missing elapsed time: {missing_elapsed}")
    print(f"Cumulative request time:       {total_time:.3f} seconds")
    print(f"Cumulative request time:       {total_time / 60:.3f} minutes")
    print(f"Cumulative request time:       {total_time / 3600:.3f} hours")

    if attempt_times:
        print(f"Mean time per API attempt:     {statistics.mean(attempt_times):.3f} s")
        print(f"Median time per API attempt:   {statistics.median(attempt_times):.3f} s")
        print(f"Minimum attempt time:          {min(attempt_times):.3f} s")
        print(f"Maximum attempt time:          {max(attempt_times):.3f} s")

    if successful_attempt_times:
        print(
            f"Mean successful request time:  "
            f"{statistics.mean(successful_attempt_times):.3f} s"
        )
        print(
            f"Median successful req. time:   "
            f"{statistics.median(successful_attempt_times):.3f} s"
        )

    return {
        "label": label,
        "rows": len(rows),
        "attempts": total_attempts,
        "retries": rows_with_retries,
        "failed_attempts": failed_attempts,
        "time_seconds": total_time,
    }


def main():
    print("Q8 COMPUTE / RUNTIME AUDIT")
    print("This script is read-only; it does not modify the JSONL files.")

    results = []

    for label, filename in FILES.items():
        result = audit_file(label, filename)
        if result is not None:
            results.append(result)

    if results:
        total_seconds = sum(r["time_seconds"] for r in results)
        total_rows = sum(r["rows"] for r in results)
        total_attempts = sum(r["attempts"] for r in results)

        print("\n" + "=" * 68)
        print("OVERALL")
        print("=" * 68)
        print(f"Total experimental rows/cells: {total_rows}")
        print(f"Total API attempts:             {total_attempts}")
        print(f"Total cumulative request time:  {total_seconds:.3f} seconds")
        print(f"Total cumulative request time:  {total_seconds / 60:.3f} minutes")
        print(f"Total cumulative request time:  {total_seconds / 3600:.3f} hours")


if __name__ == "__main__":
    main()