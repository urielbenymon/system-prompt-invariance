"""
PURPOSE:
Query WordNet for synonym evidence for the lexical-substitution candidates
that were approved during manual candidate review.

This script does NOT automatically choose the final synonym. Instead, it
retrieves WordNet synsets, definitions, and eligible one-word lemma candidates
for every approved target word while preserving the original left-to-right
candidate order.

HOW TO USE:
Place this script in the same folder as:

    synonym_candidate_review_results.jsonl

Run:

    python .\04_wordnet_synonym_lookup.py

The script creates:

    wordnet_lookup_results.jsonl

The resulting file will be used during the next manual contextual-equivalence
review, where the first candidate with an acceptable WordNet-supported synonym
will be selected.

DEPENDENCIES:
If NLTK is not already installed, run:

    pip install nltk

The script will attempt to download the WordNet data automatically the first
time it is needed.
"""

# Import JSON support for reading and writing JSONL files.
import json

# Import Path for clean file-path handling.
from pathlib import Path

# Import NLTK.
import nltk

# Import the WordNet corpus interface.
from nltk.corpus import wordnet as wn


# ---------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------

# Manual review results from the candidate-audit stage.
REVIEW_FILE = Path("synonym_candidate_review_results.jsonl")

# Output file containing all WordNet lookup evidence.
OUTPUT_FILE = Path("wordnet_lookup_results.jsonl")


# ---------------------------------------------------------------------
# WORDNET SETUP
# ---------------------------------------------------------------------

def ensure_wordnet():
    """
    Make sure the WordNet corpora needed by NLTK are installed.

    If they are missing, download them automatically.
    """

    try:

        # Attempt a simple WordNet lookup.
        wn.synsets("test")

    except LookupError:

        print("WordNet data not found.")
        print("Downloading WordNet...")

        # Download the core WordNet lexical database.
        nltk.download("wordnet")

        # Download multilingual WordNet metadata that NLTK may use.
        nltk.download("omw-1.4")

        print("WordNet download complete.")


# ---------------------------------------------------------------------
# JSONL HELPERS
# ---------------------------------------------------------------------

def load_jsonl(path):
    """
    Load a JSONL file into a list of dictionaries.
    """

    rows = []

    # Open the file using UTF-8 encoding.
    with path.open("r", encoding="utf-8") as file:

        # Read one JSON object per line.
        for line_number, line in enumerate(file, start=1):

            # Ignore empty lines.
            if not line.strip():
                continue

            try:

                # Parse the JSON row.
                rows.append(json.loads(line))

            except json.JSONDecodeError as error:

                # Stop with a useful error if a row is malformed.
                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"of {path}: {error}"
                )

    return rows


def save_jsonl(path, rows):
    """
    Save a list of dictionaries as JSONL.
    """

    with path.open("w", encoding="utf-8") as file:

        for row in rows:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


# ---------------------------------------------------------------------
# PART-OF-SPEECH MAPPING
# ---------------------------------------------------------------------

def convert_pos(pos_label):
    """
    Convert our candidate POS labels into WordNet POS constants.

    Returns None for unsupported labels.
    """

    mapping = {
        "noun": wn.NOUN,
        "verb": wn.VERB,
        "adjective": wn.ADJ,
        "adverb": wn.ADV,
    }

    return mapping.get(
        str(pos_label).lower()
    )


# ---------------------------------------------------------------------
# SYNONYM FILTERING
# ---------------------------------------------------------------------

def is_single_word_candidate(lemma_name):
    """
    Determine whether a WordNet lemma is eligible for our experiment.

    We currently exclude:
    - multiword expressions
    - underscore-separated expressions
    - hyphenated expressions
    - tokens containing non-alphabetic characters

    This implements the additional constraint adopted after candidate review:
    the C2 transformation should remain a one-word -> one-word substitution.
    """

    # WordNet represents spaces inside multiword expressions with underscores.
    if "_" in lemma_name:
        return False

    # Exclude hyphenated compounds.
    if "-" in lemma_name:
        return False

    # Require an ordinary alphabetic word.
    if not lemma_name.isalpha():
        return False

    return True


def normalize_for_comparison(word):
    """
    Normalize lexical forms for simple case-insensitive comparison.
    """

    return word.strip().lower()


# ---------------------------------------------------------------------
# WORDNET LOOKUP
# ---------------------------------------------------------------------

def lookup_candidate(word, pos_label):
    """
    Retrieve WordNet evidence for one approved candidate word.

    The function returns:
    - all matching synsets for the specified POS
    - each synset's definition
    - all lemma names
    - filtered one-word synonym candidates

    No synonym is automatically selected.
    """

    # Convert our human-readable POS label to a WordNet POS value.
    wn_pos = convert_pos(pos_label)

    # If the POS is unsupported, return an explicit error status.
    if wn_pos is None:

        return {
            "lookup_status": "unsupported_pos",
            "synset_count": 0,
            "synsets": [],
            "eligible_synonym_candidates": [],
        }

    # Ask WordNet for synsets matching this word and POS.
    synsets = wn.synsets(
        word,
        pos=wn_pos
    )

    # Store structured information about every synset.
    synset_records = []

    # Store distinct eligible synonym forms across all synsets.
    eligible_synonyms = []

    # Normalize the source word for comparison.
    normalized_source = normalize_for_comparison(word)

    # Process every WordNet sense separately.
    for synset in synsets:

        # Retrieve every lemma name in this synset.
        lemma_names = [
            lemma.name()
            for lemma in synset.lemmas()
        ]

        # Store eligible alternatives from this specific sense.
        eligible_from_synset = []

        for lemma_name in lemma_names:

            # Reject multiword, hyphenated, or non-alphabetic forms.
            if not is_single_word_candidate(lemma_name):
                continue

            # Do not treat the original word itself as a synonym replacement.
            if (
                normalize_for_comparison(lemma_name)
                == normalized_source
            ):
                continue

            # Add the candidate to this synset's eligible list.
            eligible_from_synset.append(
                lemma_name
            )

            # Add it to the overall list only once.
            if lemma_name not in eligible_synonyms:
                eligible_synonyms.append(
                    lemma_name
                )

        # Record all useful WordNet information for auditing.
        synset_records.append({
            "synset": synset.name(),
            "definition": synset.definition(),
            "examples": synset.examples(),
            "all_lemmas": lemma_names,
            "eligible_one_word_lemmas": eligible_from_synset,
        })

    # Assign a descriptive lookup status.
    if not synsets:

        lookup_status = "no_wordnet_synsets"

    elif not eligible_synonyms:

        lookup_status = "synsets_found_no_eligible_synonyms"

    else:

        lookup_status = "eligible_synonyms_found"

    # Return the complete lookup record.
    return {
        "lookup_status": lookup_status,
        "synset_count": len(synsets),
        "synsets": synset_records,
        "eligible_synonym_candidates": eligible_synonyms,
    }


# ---------------------------------------------------------------------
# DISPLAY HELPERS
# ---------------------------------------------------------------------

def print_lookup_summary(
    prompt_index,
    candidate,
    lookup_result
):
    """
    Print a concise human-readable WordNet lookup summary.
    """

    print("\n" + "-" * 100)

    print(
        f"PROMPT {prompt_index} | "
        f"CANDIDATE #{candidate.get('candidate_rank')}"
    )

    print(
        f"Word: {candidate.get('word')}"
    )

    print(
        f"POS: {candidate.get('pos')}"
    )

    print(
        f"WordNet status: "
        f"{lookup_result['lookup_status']}"
    )

    print(
        f"Synsets found: "
        f"{lookup_result['synset_count']}"
    )

    synonyms = (
        lookup_result[
            "eligible_synonym_candidates"
        ]
    )

    if synonyms:

        print(
            "Eligible one-word synonyms: "
            + ", ".join(synonyms)
        )

    else:

        print(
            "Eligible one-word synonyms: NONE"
        )


# ---------------------------------------------------------------------
# MAIN WORKFLOW
# ---------------------------------------------------------------------

def main():

    # Ensure the lexical resource is available.
    ensure_wordnet()

    # Confirm that the review file exists.
    if not REVIEW_FILE.exists():

        raise FileNotFoundError(
            f"Could not find {REVIEW_FILE}.\n"
            "Place this script in the same folder as "
            "synonym_candidate_review_results.jsonl."
        )

    # Load the manually reviewed candidate inventory.
    review_rows = load_jsonl(
        REVIEW_FILE
    )

    print(
        f"Loaded {len(review_rows)} reviewed prompts."
    )

    # Create the final list of lookup records.
    output_rows = []

    # Process prompts in original dataset order.
    for prompt_index, row in enumerate(review_rows):

        # Retrieve the original system prompt.
        original_prompt = row.get(
            "system_prompt_original"
        )

        # Retrieve manual candidate decisions.
        candidate_reviews = row.get(
            "candidate_reviews",
            []
        )

        # Create the prompt-level WordNet record.
        prompt_record = {
            "base_id": row.get("base_id"),
            "instance_idx": row.get("instance_idx"),
            "system_prompt_original": original_prompt,
            "candidate_lookups": [],
            "selection_rule": (
                "Evaluate candidates in original left-to-right rank order. "
                "Select the first candidate for which at least one "
                "WordNet-supported one-word synonym also passes manual "
                "contextual-equivalence and rubric-preservation review."
            ),
            "additional_constraint": (
                "Hyphenated and multiword lexical forms are excluded "
                "from both target and replacement eligibility."
            ),
            "selected_candidate_rank": None,
            "selected_word": None,
            "selected_synonym": None,
            "selected_synset": None,
            "selection_status": (
                "pending_contextual_equivalence_review"
            ),
        }

        # Show the original prompt once before its candidate lookups.
        print("\n" + "=" * 100)
        print(f"PROMPT {prompt_index}")
        print("=" * 100)

        print("\nORIGINAL SYSTEM PROMPT:")
        print(original_prompt)

        # Process each manually reviewed candidate in rank order.
        for candidate in candidate_reviews:

            # Skip anything you rejected or flagged as ineligible.
            if (
                candidate.get("decision")
                != "valid_for_wordnet_lookup"
            ):
                continue

            # Retrieve the candidate word.
            word = candidate.get("word")

            # Retrieve its manually assigned POS.
            pos_label = candidate.get("pos")

            # Query WordNet.
            lookup_result = lookup_candidate(
                word,
                pos_label
            )

            # Combine audit metadata with lookup evidence.
            candidate_record = {
                "candidate_rank": (
                    candidate.get(
                        "candidate_rank"
                    )
                ),
                "word": word,
                "pos": pos_label,
                "manual_candidate_decision": (
                    candidate.get("decision")
                ),
                "candidate_review_notes": (
                    candidate.get("notes", "")
                ),
                **lookup_result,
                "contextual_equivalence_status": (
                    "pending_manual_review"
                ),
                "contextual_review_notes": "",
                "final_candidate_status": "pending",
            }

            # Add this candidate to the prompt record.
            prompt_record[
                "candidate_lookups"
            ].append(
                candidate_record
            )

            # Print a concise console summary.
            print_lookup_summary(
                prompt_index,
                candidate,
                lookup_result
            )

        # Preserve original candidate ordering explicitly.
        prompt_record[
            "candidate_lookups"
        ].sort(
            key=lambda candidate:
            candidate.get(
                "candidate_rank",
                999999
            )
        )

        # Add this prompt to the output dataset.
        output_rows.append(
            prompt_record
        )

    # Save every lookup result.
    save_jsonl(
        OUTPUT_FILE,
        output_rows
    )

    # Print completion summary.
    print("\n" + "=" * 100)
    print("WORDNET LOOKUP COMPLETE")
    print("=" * 100)

    print(
        f"\nPrompts processed: "
        f"{len(output_rows)}"
    )

    total_candidates = sum(
        len(
            row["candidate_lookups"]
        )
        for row in output_rows
    )

    candidates_with_synonyms = sum(
        candidate[
            "lookup_status"
        ]
        == "eligible_synonyms_found"
        for row in output_rows
        for candidate
        in row["candidate_lookups"]
    )

    print(
        f"Approved candidates queried: "
        f"{total_candidates}"
    )

    print(
        f"Candidates with at least one "
        f"eligible WordNet synonym: "
        f"{candidates_with_synonyms}"
    )

    print(
        f"\nSaved WordNet evidence to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nNEXT STEP:"
        "\nRun a separate contextual-equivalence review."
        "\nFor each prompt, inspect candidates from lowest rank upward."
        "\nStop at the first candidate with a synonym that preserves "
        "the exact system-prompt meaning and original BiGGen rubric."
    )


# Execute the pipeline when the script is run directly.
if __name__ == "__main__":
    main()