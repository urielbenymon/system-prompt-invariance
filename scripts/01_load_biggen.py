"""
PURPOSE:
This script loads the BiGGen-Bench test dataset from Hugging Face and isolates
the system_prompt_vs_instruction and system_prompt_vs_demo tasks for our experiment.
It displays the original prompts, inputs, reference answers, and scoring rubrics and
saves each task as a separate local JSONL file without modifying the benchmark content.

HOW TO USE:
Run this script from the project virtual environment with `python .\01_load_biggen.py`.
The script downloads BiGGen-Bench, filters for each target task, prints every selected
instance for inspection, and saves each task as its own JSONL file.
"""

# Import Hugging Face's function for downloading and loading datasets.
from datasets import load_dataset


# Store the Hugging Face identifier for the BiGGen-Bench dataset.
DATASET_NAME = "prometheus-eval/BiGGen-Bench"


# Specify the BiGGen-Bench tasks that we want to retrieve.
TARGET_TASKS = [
    "system_prompt_vs_instruction",
    "system_prompt_vs_demo",
]


# Tell the user that the dataset-loading process is starting.
print("Loading BIGGEN-Bench...")


# Download and load the test split of BiGGen-Bench from Hugging Face.
dataset = load_dataset(DATASET_NAME, split="test")


# Print the total number of instances contained in the test dataset.
print(f"Total rows: {len(dataset)}")


# Print the available dataset columns so we can verify its structure.
print("Columns:", dataset.column_names)


# Loop through each target task separately.
for target_task in TARGET_TASKS:

    # Filter the complete dataset to retain only the current target task.
    subset = dataset.filter(
        # Keep a row only when its task matches the current target task.
        lambda row: row["task"] == target_task
    )


    # Print the number of instances found for the current task.
    print(f"\nRows for {target_task}: {len(subset)}")


    # Loop through every selected BiGGen-Bench instance and track its position.
    for i, row in enumerate(subset):

        # Print a separator to make individual instances easier to distinguish.
        print("\n" + "=" * 100)

        # Print the current row number within this filtered subset.
        print(f"ROW {i}")

        # Print the unique BiGGen-Bench identifier associated with the instance.
        print("ID:", row.get("id"))

        # Print the broader BiGGen-Bench capability to which this instance belongs.
        print("CAPABILITY:", row.get("capability"))

        # Print the specific task name for verification.
        print("TASK:", row.get("task"))


        # Print a heading identifying the system-prompt field.
        print("\nSYSTEM PROMPT:")

        # Print the original system prompt supplied by BiGGen-Bench.
        print(row.get("system_prompt"))


        # Print a heading identifying the user-input field.
        print("\nINPUT:")

        # Print the original user input associated with the system prompt.
        print(row.get("input"))


        # Print a heading identifying the reference-answer field.
        print("\nREFERENCE ANSWER:")

        # Print BiGGen-Bench's reference answer for the current instance.
        print(row.get("reference_answer"))


        # Print a heading identifying the scoring-rubric field.
        print("\nSCORE RUBRIC:")

        # Print BiGGen-Bench's original rubric used to evaluate model responses.
        print(row.get("score_rubric"))


    # Build a separate output filename for the current task.
    output_filename = f"biggen_{target_task}.jsonl"


    # Save the filtered rows as a JSONL file while leaving the source dataset unchanged.
    subset.to_json(output_filename)


    # Confirm that the current task subset was successfully saved locally.
    print(f"\nSaved subset to {output_filename}")