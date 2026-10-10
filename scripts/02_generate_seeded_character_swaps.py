"""
PURPOSE:
Generate reproducible adjacent-character swap variants for the 10 BiGGen
system_prompt_vs_instruction prompts.

The target word for each prompt is fixed in advance. For each selected word,
the script enumerates eligible internal adjacent-character pairs, then chooses
one pair reproducibly using a row-specific pseudorandom seed derived from a
single global seed and the prompt's BiGGen ID.

HOW TO USE:
Place this script in the same folder as:

    biggen_system_prompt_vs_instruction.jsonl

Then run:

    python .\02_generate_seeded_character_swaps.py

The script writes:

    biggen_system_prompt_vs_instruction_character_swap.jsonl

Running the script again with the same source file, target-word mapping,
and GLOBAL_SEED will produce the same swaps.
"""

# Import JSON support for reading and writing JSONL files.
import json

# Import Python's pseudorandom number generator.
import random

# Import SHA-256 hashing so each prompt gets a stable row-specific seed.
import hashlib

# Import Path for clean file-path handling.
from pathlib import Path


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

# Set the untouched BiGGen source file.
BASE_FILE = Path("biggen_system_prompt_vs_instruction.jsonl")

# Set the derived character-swap output file.
OUTPUT_FILE = Path(
    "biggen_system_prompt_vs_instruction_character_swap.jsonl"
)

# Fix one experiment-wide seed before generating any results.
GLOBAL_SEED = 2026


# Fix the target word for every BiGGen instance in advance.
TARGET_WORDS = {
    0: "expert",
    1: "expert",
    2: "chatbot",
    3: "Translate",
    4: "Reverse",
    5: "Evaluate",
    6: "Analyze",
    7: "Assess",
    8: "Critically",
    9: "Examine",
}


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

# Load every non-empty line from a JSONL file.
def load_jsonl(path):

    # Create an empty list for parsed rows.
    rows = []

    # Open the JSONL file using UTF-8 encoding.
    with path.open("r", encoding="utf-8") as file:

        # Read each line in order.
        for line in file:

            # Ignore blank lines.
            if not line.strip():
                continue

            # Parse the JSON object and append it to the row list.
            rows.append(json.loads(line))

    # Return the parsed dataset.
    return rows


# Create a deterministic row-specific random generator.
def make_row_rng(base_id):

    # Combine the global seed and prompt ID into one stable string.
    key = f"{GLOBAL_SEED}:{base_id}".encode("utf-8")

    # Hash the string using SHA-256.
    digest = hashlib.sha256(key).hexdigest()

    # Convert part of the hash into a reproducible integer seed.
    row_seed = int(digest[:16], 16)

    # Return both the RNG object and its reproducible seed.
    return random.Random(row_seed), row_seed


# Enumerate internal adjacent-character pairs that are eligible to swap.
def get_eligible_pairs(word):

    # Create an empty list for eligible index pairs.
    eligible_pairs = []

    # Require a reasonably long alphabetic word.
    if len(word) < 5 or not word.isalpha():
        return eligible_pairs

    # Scan internal adjacent positions only.
    #
    # Example:
    # word = "expert"
    # indices = 0 1 2 3 4 5
    #
    # Internal starts are 1, 2, 3.
    # This avoids swapping the first or last character boundary.
    for first_index in range(1, len(word) - 2):

        # The second character is immediately adjacent.
        second_index = first_index + 1

        # Skip identical characters because swapping them changes nothing.
        if word[first_index] == word[second_index]:
            continue

        # Store the admissible adjacent pair.
        eligible_pairs.append(
            [first_index, second_index]
        )

    # Return every eligible internal adjacent pair.
    return eligible_pairs


# Apply one adjacent-character transposition.
def swap_characters(word, first_index, second_index):

    # Convert the word to a mutable list of characters.
    characters = list(word)

    # Swap the two selected characters.
    characters[first_index], characters[second_index] = (
        characters[second_index],
        characters[first_index],
    )

    # Rejoin the list into a transformed word.
    return "".join(characters)


# ---------------------------------------------------------------------
# MAIN WORKFLOW
# ---------------------------------------------------------------------

# Run the complete generation pipeline.
def main():

    # Tell the user what file is being loaded.
    print(f"Loading source file: {BASE_FILE}")

    # Load the original BiGGen rows.
    rows = load_jsonl(BASE_FILE)

    # Verify that the expected 10 rows are present.
    if len(rows) != 10:
        raise ValueError(
            f"Expected 10 BiGGen rows, found {len(rows)}."
        )

    # Create a list for transformed rows.
    output_rows = []

    # Process every BiGGen prompt in order.
    for row_index, row in enumerate(rows):

        # Retrieve the pre-selected target word for this row.
        target_word = TARGET_WORDS[row_index]

        # Read the original system prompt.
        original_prompt = row["system_prompt"]

        # Confirm that the selected target occurs exactly once.
        if original_prompt.count(target_word) != 1:
            raise ValueError(
                f"Row {row_index}: target word {target_word!r} "
                "does not occur exactly once."
            )

        # Enumerate all eligible internal adjacent-character pairs.
        eligible_pairs = get_eligible_pairs(target_word)

        # Stop if no eligible pair exists.
        if not eligible_pairs:
            raise ValueError(
                f"Row {row_index}: no eligible adjacent pairs "
                f"for target word {target_word!r}."
            )

        # Create a reproducible row-specific RNG.
        rng, row_seed = make_row_rng(row["id"])

        # Select one eligible adjacent pair uniformly at random.
        selected_pair = rng.choice(eligible_pairs)

        # Separate the two selected character indices.
        first_index, second_index = selected_pair

        # Create the transformed word.
        transformed_word = swap_characters(
            target_word,
            first_index,
            second_index
        )

        # Replace the target word once inside the system prompt.
        transformed_prompt = original_prompt.replace(
            target_word,
            transformed_word,
            1
        )

        # Copy the complete original BiGGen row.
        derived_row = dict(row)

        # Add experiment-specific metadata.
        derived_row.update({

            # Preserve a pointer back to the original BiGGen instance.
            "base_id": row["id"],

            # Create a unique ID for this transformed condition.
            "variant_id": (
                f"{row['id']}__character_swap"
            ),

            # Record the experimental condition.
            "condition": "character_swap",

            # Record the exact transformation family.
            "transformation": "adjacent_character_swap",

            # Preserve the untouched system prompt for auditing.
            "system_prompt_original": original_prompt,

            # Replace system_prompt with the transformed version.
            "system_prompt": transformed_prompt,

            # Record the fixed target word.
            "target_original": target_word,

            # Record the transformed target word.
            "target_transformed": transformed_word,

            # Record every pair that was eligible for random selection.
            "eligible_swap_pairs_word_0indexed": eligible_pairs,

            # Record the pair actually selected.
            "selected_swap_pair_word_0indexed": selected_pair,

            # Record the global experiment seed.
            "global_seed": GLOBAL_SEED,

            # Record the deterministic row-specific seed.
            "row_seed": row_seed,

            # Record that this prompt received exactly one transformation.
            "change_count": 1,

            # Mark the row as waiting for human audit.
            "audit_status": "pending_manual_review",
        })

        # Append the transformed row to the output dataset.
        output_rows.append(derived_row)

        # Print a concise audit preview.
        print(
            f"Row {row_index}: "
            f"{target_word} -> {transformed_word} "
            f"| eligible={eligible_pairs} "
            f"| selected={selected_pair}"
        )

    # Open the output JSONL file.
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:

        # Write one JSON object per line.
        for row in output_rows:

            # Serialize the row without escaping normal Unicode text.
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    # Confirm completion.
    print("\nGeneration complete.")

    # Show the output filename.
    print(f"Saved to: {OUTPUT_FILE}")

    # Show the experiment-wide seed.
    print(f"Global seed: {GLOBAL_SEED}")


# Run main only when this file is executed directly.
if __name__ == "__main__":
    main()