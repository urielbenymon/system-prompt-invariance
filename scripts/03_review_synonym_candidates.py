"""
PURPOSE:
Review the ordered lexical candidates for the WordNet-based synonym
transformation condition in the BiGGen system-prompt invariance experiment.

The script displays one original system prompt at a time together with its
eligible candidate words in left-to-right order. The researcher can mark
each candidate as valid for WordNet lookup, skipped/ineligible, or flagged
for further discussion.

HOW TO USE:
Place this script in the same folder as:

    wordnet_synonym_candidates.jsonl

Run:

    python .\03_review_synonym_candidates.py

For each candidate, enter:

    v = valid for WordNet lookup
    s = skip / reject
    f = flag for discussion
    q = quit

The script does NOT modify the candidate inventory. It writes decisions to:

    synonym_candidate_review_results.jsonl
"""

import json
from pathlib import Path


# ---------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------

# Candidate inventory created from the original BiGGen system prompts.
CANDIDATE_FILE = Path("wordnet_synonym_candidates.jsonl")

# Separate review log.
REVIEW_FILE = Path("synonym_candidate_review_results.jsonl")


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def load_jsonl(path):
    """
    Load a JSONL file into a list of dictionaries.
    """

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


def save_review_results(results):
    """
    Save all completed review decisions to a separate JSONL file.
    """

    with REVIEW_FILE.open("w", encoding="utf-8") as file:

        for result in results:

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                + "\n"
            )


def display_prompt_header(row_index, row):
    """
    Print prompt-level information before candidate review.
    """

    print("\n" + "=" * 110)
    print(f"PROMPT {row_index}")
    print("=" * 110)

    print("\nBASE ID:")
    print(row.get("base_id"))

    print("\nINSTANCE INDEX:")
    print(row.get("instance_idx"))

    print("\nORIGINAL SYSTEM PROMPT:")
    print(row.get("system_prompt_original"))

    print("\nSELECTION PROTOCOL:")
    print(row.get("selection_protocol"))

    print("\nEXCLUSION PROTOCOL:")
    print(row.get("exclusion_protocol"))

    print("\n" + "-" * 110)


def display_candidate(candidate):
    """
    Print one lexical candidate in a readable format.
    """

    print(
        f"Candidate #{candidate.get('candidate_rank')}: "
        f"{candidate.get('word')} "
        f"[POS: {candidate.get('pos')}]"
    )


# ---------------------------------------------------------------------
# MAIN REVIEW WORKFLOW
# ---------------------------------------------------------------------

def main():

    print(f"Loading candidate inventory: {CANDIDATE_FILE}")

    rows = load_jsonl(CANDIDATE_FILE)

    print(f"Rows loaded: {len(rows)}")

    review_results = []

    for row_index, row in enumerate(rows):

        display_prompt_header(row_index, row)

        prompt_review = {
            "base_id": row.get("base_id"),
            "instance_idx": row.get("instance_idx"),
            "system_prompt_original": row.get("system_prompt_original"),
            "candidate_reviews": [],
        }

        candidates = row.get("eligible_candidates", [])

        if not candidates:
            print("\nNo candidates found for this prompt.")

            prompt_review["prompt_status"] = "no_candidates"

            review_results.append(prompt_review)

            save_review_results(review_results)

            continue

        for candidate in candidates:

            display_candidate(candidate)

            while True:

                decision = input(
                    "Decision "
                    "[v = valid, s = skip, f = flag, q = quit]: "
                ).strip().lower()

                if decision in {"v", "s", "f", "q"}:
                    break

                print("Please enter v, s, f, or q.")

            if decision == "q":

                print("\nReview stopped by user.")

                review_results.append(prompt_review)

                save_review_results(review_results)

                print(
                    f"Completed review decisions saved to:\n{REVIEW_FILE}"
                )

                return

            decision_map = {
                "v": "valid_for_wordnet_lookup",
                "s": "skipped_or_ineligible",
                "f": "flagged_for_discussion",
            }

            notes = input(
                "Optional note "
                "(press Enter for none): "
            ).strip()

            candidate_review = {
                "candidate_rank": candidate.get("candidate_rank"),
                "word": candidate.get("word"),
                "pos": candidate.get("pos"),
                "decision": decision_map[decision],
                "notes": notes,
            }

            prompt_review["candidate_reviews"].append(
                candidate_review
            )

            save_review_results(
                review_results + [prompt_review]
            )

        # Mark prompt complete after reviewing all candidates.
        prompt_review["prompt_status"] = "candidate_review_complete"

        review_results.append(prompt_review)

        save_review_results(review_results)

    print("\n" + "=" * 110)
    print("CANDIDATE REVIEW COMPLETE")
    print("=" * 110)

    total_candidates = sum(
        len(row.get("candidate_reviews", []))
        for row in review_results
    )

    valid_candidates = sum(
        review.get("decision") == "valid_for_wordnet_lookup"
        for row in review_results
        for review in row.get("candidate_reviews", [])
    )

    skipped_candidates = sum(
        review.get("decision") == "skipped_or_ineligible"
        for row in review_results
        for review in row.get("candidate_reviews", [])
    )

    flagged_candidates = sum(
        review.get("decision") == "flagged_for_discussion"
        for row in review_results
        for review in row.get("candidate_reviews", [])
    )

    print(f"\nTotal candidates reviewed: {total_candidates}")
    print(f"Valid for WordNet lookup: {valid_candidates}")
    print(f"Skipped / ineligible: {skipped_candidates}")
    print(f"Flagged: {flagged_candidates}")

    print(
        f"\nReview results saved to:\n{REVIEW_FILE}"
    )


if __name__ == "__main__":
    main()