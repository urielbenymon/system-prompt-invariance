"""
PURPOSE:
Audit the character-swap variants for the BiGGen system-prompt invariance experiment.

The script first performs automatic structural checks to verify that each variant
contains exactly one valid adjacent-character transposition and that all non-system
BiGGen fields remain unchanged. It then displays each original/variant pair in a
human-readable format so the researcher can approve, reject, or flag it for revision.

HOW TO USE:
Place this script in the same folder as the original BiGGen JSONL and the
character-swap variant JSONL. Run:

    python .\02_audit_character_variants.py

For each row, inspect the displayed material and enter:
    a = approve
    r = reject
    f = flag for further review
    q = quit

Manual decisions are saved separately in:
    character_swap_audit_results.jsonl
"""

import json
from pathlib import Path


# ---------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------

# Untouched BiGGen source file.
BASE_FILE = Path("biggen_system_prompt_vs_instruction.jsonl")

# Derived character-swap variants.
VARIANT_FILE = Path(
    "biggen_system_prompt_vs_instruction_character_swap.jsonl"
)

# Separate audit log.
AUDIT_FILE = Path("character_swap_audit_results.jsonl")


# ---------------------------------------------------------------------
# FIELDS THAT MUST NEVER CHANGE
# ---------------------------------------------------------------------

FROZEN_FIELDS = [
    "id",
    "capability",
    "task",
    "instance_idx",
    "input",
    "reference_answer",
    "score_rubric",
]


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def load_jsonl(path):
    """Load a JSONL file and return a list of dictionaries."""

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                )

    return rows


def check_adjacent_swap(original_word, transformed_word):
    """
    Verify that transformed_word differs from original_word only by
    transposing one pair of adjacent characters.
    """

    checks = {}

    # Rule: character swap cannot change word length.
    checks["same_word_length"] = (
        len(original_word) == len(transformed_word)
    )

    # Stop detailed testing if lengths differ.
    if not checks["same_word_length"]:
        checks["exactly_two_positions_changed"] = False
        checks["positions_are_adjacent"] = False
        checks["pure_transposition"] = False
        return checks

    # Identify character positions that differ.
    differences = [
        index
        for index, (original_char, transformed_char)
        in enumerate(zip(original_word, transformed_word))
        if original_char != transformed_char
    ]

    # A single adjacent swap should affect exactly two positions.
    checks["exactly_two_positions_changed"] = (
        len(differences) == 2
    )

    if len(differences) != 2:
        checks["positions_are_adjacent"] = False
        checks["pure_transposition"] = False
        return checks

    first_position, second_position = differences

    # The two changed character positions must be neighbors.
    checks["positions_are_adjacent"] = (
        second_position == first_position + 1
    )

    # The two characters must simply exchange positions.
    checks["pure_transposition"] = (
        original_word[first_position]
        == transformed_word[second_position]
        and
        original_word[second_position]
        == transformed_word[first_position]
    )

    return checks


def audit_pair(base_row, variant_row):
    """
    Perform all automatic checks for one baseline/variant pair.
    """

    checks = {}

    original_prompt = base_row["system_prompt"]
    transformed_prompt = variant_row["system_prompt"]

    original_word = variant_row.get("target_original", "")
    transformed_word = variant_row.get("target_transformed", "")

    # Verify that the variant points back to the correct source row.
    checks["base_id_matches"] = (
        variant_row.get("base_id") == base_row.get("id")
    )

    # Verify expected experimental condition metadata.
    checks["condition_correct"] = (
        variant_row.get("condition") == "character_swap"
    )

    checks["transformation_correct"] = (
        variant_row.get("transformation")
        == "adjacent_character_swap"
    )

    # Verify every frozen BiGGen field remains exactly unchanged.
    frozen_results = {}

    for field in FROZEN_FIELDS:
        frozen_results[field] = (
            base_row.get(field) == variant_row.get(field)
        )

    checks["frozen_fields"] = frozen_results

    # Verify exactly one source-word occurrence was selected.
    checks["original_word_occurs_once"] = (
        original_prompt.count(original_word) == 1
    )

    # Reconstruct what the transformed prompt SHOULD be.
    if checks["original_word_occurs_once"]:

        expected_prompt = original_prompt.replace(
            original_word,
            transformed_word,
            1
        )

        checks["only_intended_prompt_change"] = (
            transformed_prompt == expected_prompt
        )

    else:
        checks["only_intended_prompt_change"] = False

    # Rule-sheet preference: ordinary alphabetic word.
    checks["alphabetic_target"] = (
        original_word.isalpha()
        and transformed_word.isalpha()
    )

    # Rule-sheet preference: word length at least five characters.
    checks["target_length_at_least_5"] = (
        len(original_word) >= 5
    )

    # Verify metadata says exactly one manipulation occurred.
    checks["change_count_is_one"] = (
        variant_row.get("change_count") == 1
    )

    # Perform the detailed adjacent-character checks.
    swap_checks = check_adjacent_swap(
        original_word,
        transformed_word
    )

    checks.update(swap_checks)

    # Flatten all Boolean checks, including frozen-field checks.
    top_level_values = [
        value
        for key, value in checks.items()
        if key != "frozen_fields"
    ]

    frozen_values = list(
        checks["frozen_fields"].values()
    )

    automatic_pass = (
        all(top_level_values)
        and all(frozen_values)
    )

    return checks, automatic_pass


def print_boolean(label, value):
    """Print PASS or FAIL for an automatic audit criterion."""

    status = "PASS" if value else "FAIL"
    print(f"[{status}] {label}")


def display_instance(index, base_row, variant_row, checks, automatic_pass):
    """
    Print one complete audit instance in a readable console layout.
    """

    print("\n" + "=" * 110)
    print(f"INSTANCE {index}")
    print("=" * 110)

    print("\nBASE ID:")
    print(base_row.get("id"))

    print("\nORIGINAL SYSTEM PROMPT:")
    print(base_row.get("system_prompt"))

    print("\nCHARACTER-SWAP SYSTEM PROMPT:")
    print(variant_row.get("system_prompt"))

    print("\nRECORDED CHANGE:")
    print(
        f"{variant_row.get('target_original')}"
        f"  ->  "
        f"{variant_row.get('target_transformed')}"
    )

    print("\nAUTOMATIC AUDIT:")
    print_boolean(
        "Base ID matches",
        checks["base_id_matches"]
    )

    print_boolean(
        "Condition metadata is character_swap",
        checks["condition_correct"]
    )

    print_boolean(
        "Transformation metadata is adjacent_character_swap",
        checks["transformation_correct"]
    )

    print_boolean(
        "Original target occurs exactly once",
        checks["original_word_occurs_once"]
    )

    print_boolean(
        "Only intended system-prompt change occurred",
        checks["only_intended_prompt_change"]
    )

    print_boolean(
        "Target contains only alphabetic characters",
        checks["alphabetic_target"]
    )

    print_boolean(
        "Target word length is at least 5",
        checks["target_length_at_least_5"]
    )

    print_boolean(
        "Recorded change count equals 1",
        checks["change_count_is_one"]
    )

    print_boolean(
        "Original/transformed word lengths match",
        checks["same_word_length"]
    )

    print_boolean(
        "Exactly two character positions changed",
        checks["exactly_two_positions_changed"]
    )

    print_boolean(
        "Changed positions are adjacent",
        checks["positions_are_adjacent"]
    )

    print_boolean(
        "Change is a pure adjacent transposition",
        checks["pure_transposition"]
    )

    print("\nFROZEN BIGGEN FIELDS:")

    for field, unchanged in checks["frozen_fields"].items():
        print_boolean(
            f"{field} unchanged",
            unchanged
        )

    print("\nAUTOMATIC RESULT:")
    print(
        "PASS - all mechanical checks succeeded"
        if automatic_pass
        else "FAIL - at least one mechanical check failed"
    )

    print("\n" + "-" * 110)

    print("\nUSER INPUT:")
    print(base_row.get("input"))

    print("\nREFERENCE ANSWER:")
    print(base_row.get("reference_answer"))

    print("\nSCORE RUBRIC:")
    print(
        json.dumps(
            base_row.get("score_rubric"),
            indent=2,
            ensure_ascii=False
        )
    )

    print("\n" + "-" * 110)

    print("\nMANUAL RULE-SHEET REVIEW:")
    print("[ ] Only the system prompt changed.")
    print("[ ] Exactly one transformation occurred.")
    print("[ ] The selected word is an eligible ordinary alphabetic word.")
    print("[ ] No proper noun or protected constraint was altered.")
    print("[ ] The typo remains human-readable / recoverable.")
    print("[ ] The swap did not create a different valid word with another meaning.")
    print("[ ] The instruction's operative meaning remains intact.")
    print("[ ] The original reference answer remains applicable.")
    print("[ ] The original BiGGen score rubric remains applicable verbatim.")


def save_audit_results(results):
    """Write all completed audit decisions to a separate JSONL file."""

    with AUDIT_FILE.open("w", encoding="utf-8") as file:

        for result in results:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                + "\n"
            )


# ---------------------------------------------------------------------
# MAIN AUDIT WORKFLOW
# ---------------------------------------------------------------------

def main():

    print("Loading baseline and character-swap files...")

    base_rows = load_jsonl(BASE_FILE)
    variant_rows = load_jsonl(VARIANT_FILE)

    print(f"Baseline rows: {len(base_rows)}")
    print(f"Character-swap rows: {len(variant_rows)}")

    # Both datasets should contain exactly the same number of instances.
    if len(base_rows) != len(variant_rows):
        raise ValueError(
            "Baseline and character-swap files contain different numbers of rows."
        )

    # Map source rows by BiGGen ID rather than relying only on row order.
    base_by_id = {
        row["id"]: row
        for row in base_rows
    }

    audit_results = []

    for index, variant_row in enumerate(variant_rows):

        base_id = variant_row.get("base_id")

        if base_id not in base_by_id:
            raise ValueError(
                f"Variant row {index} refers to unknown base_id: {base_id}"
            )

        base_row = base_by_id[base_id]

        checks, automatic_pass = audit_pair(
            base_row,
            variant_row
        )

        display_instance(
            index,
            base_row,
            variant_row,
            checks,
            automatic_pass
        )

        while True:

            decision = input(
                "\nDecision "
                "[a = approve, r = reject, f = flag, q = quit]: "
            ).strip().lower()

            if decision in {"a", "r", "f", "q"}:
                break

            print("Please enter a, r, f, or q.")

        if decision == "q":

            print("\nAudit stopped by user.")

            # Preserve decisions already completed.
            save_audit_results(audit_results)

            print(
                f"Completed decisions saved to: {AUDIT_FILE}"
            )

            return

        decision_map = {
            "a": "approved",
            "r": "rejected",
            "f": "flagged_for_review",
        }

        notes = input(
            "Optional audit note "
            "(press Enter for none): "
        ).strip()

        audit_record = {
            "base_id": base_row["id"],
            "variant_id": variant_row.get("variant_id"),
            "instance_idx": base_row.get("instance_idx"),
            "target_original": variant_row.get("target_original"),
            "target_transformed": variant_row.get("target_transformed"),
            "automatic_pass": automatic_pass,
            "automatic_checks": checks,
            "manual_decision": decision_map[decision],
            "manual_note": notes,
        }

        audit_results.append(audit_record)

        # Save after every decision so progress is never lost.
        save_audit_results(audit_results)

    print("\n" + "=" * 110)
    print("AUDIT COMPLETE")
    print("=" * 110)

    approved = sum(
        result["manual_decision"] == "approved"
        for result in audit_results
    )

    rejected = sum(
        result["manual_decision"] == "rejected"
        for result in audit_results
    )

    flagged = sum(
        result["manual_decision"] == "flagged_for_review"
        for result in audit_results
    )

    automatic_failures = sum(
        not result["automatic_pass"]
        for result in audit_results
    )

    print(f"\nApproved: {approved}")
    print(f"Rejected: {rejected}")
    print(f"Flagged: {flagged}")
    print(f"Automatic failures: {automatic_failures}")

    print(
        f"\nAudit results saved to:\n{AUDIT_FILE}"
    )


if __name__ == "__main__":
    main()