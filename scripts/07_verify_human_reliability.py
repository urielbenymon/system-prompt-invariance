#!/usr/bin/env python3
"""
07_verify_human_reliability.py

Cross-checks the human-only and combined human+LLM Krippendorff's alpha
using the third-party `krippendorff` package, as an independent check
against the manual implementation in 06_aggregate_and_analyze.py.

NO API CALLS ARE MADE.

Requires
--------
    pip install openpyxl krippendorff numpy

Input files (edit the paths below if your filenames differ)
--------------------------------------------------------------
    - rater_1_sheet_formatted_human_grading.xlsx
    - rater_2_sheet_formatted_human_grading.xlsx
    - rater_3_sheet_formatted_human_grading.xlsx
    - human_grading_files/id_mapping_PRIVATE.csv   (from the project zip)
    - aggregated_scores.csv                         (from the project zip)

Output
------
Prints:
    - Human-only Krippendorff's alpha (3 raters)
    - Combined alpha (3 human raters + LLM median as a 4th coder)
    - Pearson r (human median vs LLM median)
    - Exact agreement rate
"""

import csv
import statistics
import openpyxl
import numpy as np
import krippendorff

# ---- Edit these paths if needed ----
RATER_FILES = {
    1: "human_grading_files/rater_1_sheet_formatted_human_grading.xlsx",
    2: "human_grading_files/rater_2_sheet_formatted_human_grading.xlsx",
    3: "human_grading_files/rater_3_sheet_formatted_human_grading.xlsx",
}
ID_MAPPING_PATH = "human_grading_files/id_mapping_PRIVATE.csv"
AGGREGATED_SCORES_PATH = "aggregated_scores.csv"
# -------------------------------------


def load_id_mapping(path):
    mapping = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["item_id"]] = row["generation_cell_id"]
    return mapping


def load_human_scores(rater_files, id_map):
    scores = {}  # cell_id -> {rater_label: score}
    for rater_num, path in rater_files.items():
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            item_id, score = row[0], row[6]
            if item_id is None or score is None:
                continue
            cell_id = id_map[item_id]
            scores.setdefault(cell_id, {})[f"rater_{rater_num}"] = float(score)
    return scores


def load_llm_scores(path):
    scores = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[row["generation_cell_id"]] = float(row["median_score"])
    return scores


def krippendorff_alpha_pkg(units):
    max_coders = max(len(u) for u in units)
    matrix = np.full((max_coders, len(units)), np.nan)
    for col, ratings in enumerate(units):
        for row, val in enumerate(ratings):
            matrix[row, col] = val
    return krippendorff.alpha(reliability_data=matrix, level_of_measurement="interval")


def main():
    id_map = load_id_mapping(ID_MAPPING_PATH)
    human = load_human_scores(RATER_FILES, id_map)
    llm = load_llm_scores(AGGREGATED_SCORES_PATH)

    print(f"Cells with human scores: {len(human)}")
    complete = {c: v for c, v in human.items() if len(v) == 3}
    print(f"Cells with all 3 raters: {len(complete)}")
    missing_llm = [c for c in complete if c not in llm]
    print(f"Cells missing an LLM score: {len(missing_llm)}")
    print()

    # 1. Human-only alpha
    human_units = [list(v.values()) for v in complete.values()]
    alpha_human = krippendorff_alpha_pkg(human_units)
    print(f"Human-only Krippendorff's alpha (3 raters): {alpha_human:.4f}")

    # 2. Human median vs LLM median
    human_medians = {c: statistics.median(v.values()) for c, v in complete.items()}
    diffs = [human_medians[c] - llm[c] for c in complete if c in llm]
    print(f"\nMean(human_median - llm_median): {statistics.mean(diffs):.4f}")
    print(f"Mean absolute difference: {statistics.mean(abs(d) for d in diffs):.4f}")

    xs = [human_medians[c] for c in complete if c in llm]
    ys = [llm[c] for c in complete if c in llm]
    r = np.corrcoef(xs, ys)[0, 1]
    print(f"Pearson r (human median vs LLM median): {r:.4f}")

    # 3. Combined alpha: 3 human raters + LLM median as a 4th "coder"
    combined_units = [list(v.values()) + [llm[c]] for c, v in complete.items() if c in llm]
    alpha_combined = krippendorff_alpha_pkg(combined_units)
    print(f"\nCombined alpha (3 human raters + LLM-median-as-4th-coder): {alpha_combined:.4f}")

    exact = sum(1 for c in complete if c in llm and human_medians[c] == llm[c])
    total = sum(1 for c in complete if c in llm)
    print(f"\nExact agreement (human median == LLM median): {exact}/{total} = {100*exact/total:.1f}%")


if __name__ == "__main__":
    main()
