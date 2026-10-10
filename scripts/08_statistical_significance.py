#!/usr/bin/env python3
"""
08_statistical_significance.py

Statistical significance testing for the two core results tables:

  1. Core invariance (Table 3): paired Wilcoxon signed-rank tests,
     C0 vs. C1 and C0 vs. C2, run separately per response-generation model
     (4 models x 2 conditions = 8 tests). Paired because each of the 9
     families contributes one C0 score and one C1/C2 score per model.

  2. Grader format compliance (Table 5): Cochran's Q test (omnibus) across
     the three graders, followed by pairwise McNemar exact tests (3 pairs)
     if Cochran's Q is significant. Repeated-measures design because all
     three graders scored the same 108 response items -- NOT independent
     samples, so a chi-square test of independence would be the wrong
     tool here.

Both families of tests (8 Wilcoxon + 3 McNemar = 11 total) are corrected
for multiple comparisons using the Holm step-down method, applied
separately within each family.

NO API CALLS ARE MADE.

Input
-----
    - aggregated_scores.csv       (for Wilcoxon, Table 3)
    - autogrades_parsed_v2.csv    (for Cochran's Q / McNemar, Table 5)

Output
------
    - significance_invariance.csv    (8 Wilcoxon results, Holm-corrected)
    - significance_compliance.csv    (Cochran's Q + 3 McNemar results, Holm-corrected)
    Console summary of both.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from itertools import combinations

from scipy.stats import wilcoxon, chi2, binomtest, PermutationMethod


# --------------------------------------------------------------------------
# Holm step-down correction
# --------------------------------------------------------------------------

def holm_correction(pvalues: list[float]) -> list[float]:
    """Returns Holm-adjusted p-values, same order as input."""
    m = len(pvalues)
    indexed = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(indexed):
        adj = (m - rank) * pvalues[i]
        running_max = max(running_max, adj)
        adjusted[i] = min(running_max, 1.0)
    return adjusted


# --------------------------------------------------------------------------
# Table 3: Wilcoxon signed-rank, C0 vs C1 and C0 vs C2, per model
# --------------------------------------------------------------------------

def load_aggregated_scores(path: str) -> dict:
    """Returns {(model, condition): {family_id: median_score}}."""
    data = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["response_model_key"], row["condition_code"])
            data[key][row["family_id"]] = float(row["median_score"])
    return data


def run_wilcoxon_tests(agg_path: str, out_path: str) -> None:
    data = load_aggregated_scores(agg_path)
    models = sorted({k[0] for k in data})
    conditions = ["C1_character_swap", "C2_lexical_substitution"]

    results = []
    for model in models:
        base = data[(model, "C0_base")]
        for cond in conditions:
            perturbed = data[(model, cond)]
            families = sorted(set(base) & set(perturbed))
            x = [base[f] for f in families]
            y = [perturbed[f] for f in families]
            diffs = [a - b for a, b in zip(x, y)]

            n_effective = sum(1 for d in diffs if d != 0)

            if n_effective == 0:
                stat, p = float("nan"), 1.0
                note = "all differences zero (n_eff=0)"
            else:
                # NOTE: method="exact" is NOT actually exact once zeros/ties
                # are present (confirmed in SciPy's own documentation and by
                # independent manual brute-force verification against this
                # exact dataset). Every contrast here has zeros, and two
                # (o3-mini) also have tied nonzero |diff| values, so we use
                # an exhaustive PermutationMethod instead, which SciPy's docs
                # explicitly recommend for this situation and which was
                # cross-validated to match a from-scratch manual calculation.
                pm = PermutationMethod(n_resamples=100000, random_state=0)
                res = wilcoxon(x, y, method=pm)
                stat, p = res.statistic, res.pvalue
                note = f"exact permutation (n_eff={n_effective})"

            results.append({
                "model": model,
                "condition": cond,
                "n": len(families),
                "n_effective": n_effective,
                "statistic": stat,
                "p_raw": p,
                "note": note,
            })

    pvals = [r["p_raw"] for r in results]
    adjusted = holm_correction(pvals)
    for r, p_adj in zip(results, adjusted):
        r["p_holm"] = p_adj

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "condition", "n", "n_effective", "statistic", "p_raw", "p_holm", "note",
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} Wilcoxon results -> {out_path}")
    print("\n--- Wilcoxon signed-rank (Table 3), exact permutation, Holm-corrected ---")
    for r in results:
        sig = "*" if r["p_holm"] < 0.05 else " "
        print(f"  {r['model']:12s} {r['condition']:25s} n={r['n']:2d} (n_eff={r['n_effective']:2d})  "
              f"p_raw={r['p_raw']:.4f}  p_holm={r['p_holm']:.4f} {sig}  ({r['note']})")


# --------------------------------------------------------------------------
# Table 5: Cochran's Q (omnibus) + McNemar (pairwise), graders on same items
# --------------------------------------------------------------------------

def load_compliance_matrix(path: str) -> dict:
    """Returns {generation_cell_id: {grader_key: is_compliant (0/1)}}."""
    matrix = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            compliant = 1 if row["parse_method"] == "primary_result" else 0
            matrix[row["generation_cell_id"]][row["grader_key"]] = compliant
    return matrix


def cochrans_q(binary_matrix: list[list[int]]) -> tuple[float, float]:
    """binary_matrix: rows = subjects (items), columns = treatments (graders)."""
    n = len(binary_matrix)
    k = len(binary_matrix[0])
    col_sums = [sum(row[j] for row in binary_matrix) for j in range(k)]
    row_sums = [sum(row) for row in binary_matrix]

    numerator = (k - 1) * (k * sum(c ** 2 for c in col_sums) - sum(col_sums) ** 2)
    denominator = k * sum(row_sums) - sum(r ** 2 for r in row_sums)

    if denominator == 0:
        return float("nan"), 1.0

    Q = numerator / denominator
    p = chi2.sf(Q, df=k - 1)
    return Q, p


def mcnemar_exact(a_scores: list[int], b_scores: list[int]) -> tuple[int, int, float]:
    """Exact McNemar test on paired binary outcomes.
    Returns (n_discordant_b_only, n_discordant_c_only, p_value)."""
    b = sum(1 for a, b_ in zip(a_scores, b_scores) if a == 1 and b_ == 0)
    c = sum(1 for a, b_ in zip(a_scores, b_scores) if a == 0 and b_ == 1)
    n = b + c
    if n == 0:
        return b, c, 1.0
    result = binomtest(min(b, c), n, p=0.5, alternative="two-sided")
    return b, c, result.pvalue


def run_grader_tests(parsed_path: str, out_path: str) -> None:
    matrix = load_compliance_matrix(parsed_path)
    graders = sorted({g for v in matrix.values() for g in v})
    items = sorted(c for c, v in matrix.items() if len(v) == len(graders))

    binary_matrix = [[matrix[item][g] for g in graders] for item in items]

    Q, p_omnibus = cochrans_q(binary_matrix)
    print(f"\n--- Cochran's Q (Table 5 omnibus, {len(items)} items x {len(graders)} graders) ---")
    print(f"  Q = {Q:.4f}, df = {len(graders)-1}, p = {p_omnibus:.6f}")

    pairwise_results = []
    for g1, g2 in combinations(graders, 2):
        a_scores = [matrix[item][g1] for item in items]
        b_scores = [matrix[item][g2] for item in items]
        b, c, p = mcnemar_exact(a_scores, b_scores)
        pairwise_results.append({
            "grader_1": g1, "grader_2": g2,
            "discordant_1_only": b, "discordant_2_only": c,
            "p_raw": p,
        })

    pvals = [r["p_raw"] for r in pairwise_results]
    adjusted = holm_correction(pvals)
    for r, p_adj in zip(pairwise_results, adjusted):
        r["p_holm"] = p_adj

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "grader_1", "grader_2", "statistic_or_Q",
                          "df", "discordant_1_only", "discordant_2_only",
                          "p_raw", "p_holm"])
        writer.writerow(["cochrans_q", "", "", f"{Q:.4f}", len(graders) - 1,
                          "", "", f"{p_omnibus:.6f}", ""])
        for r in pairwise_results:
            writer.writerow(["mcnemar_exact", r["grader_1"], r["grader_2"], "",
                              "", r["discordant_1_only"], r["discordant_2_only"],
                              f"{r['p_raw']:.6f}", f"{r['p_holm']:.6f}"])

    print(f"\nWrote Cochran's Q + McNemar results -> {out_path}")
    print("\n--- McNemar pairwise (post-hoc), Holm-corrected ---")
    for r in pairwise_results:
        sig = "*" if r["p_holm"] < 0.05 else " "
        print(f"  {r['grader_1']:25s} vs {r['grader_2']:25s}  "
              f"discordant=({r['discordant_1_only']},{r['discordant_2_only']})  "
              f"p_raw={r['p_raw']:.4f}  p_holm={r['p_holm']:.4f} {sig}")


def main() -> None:
    run_wilcoxon_tests("aggregated_scores.csv", "significance_invariance.csv")
    run_grader_tests("autogrades_parsed_v2.csv", "significance_compliance.csv")


if __name__ == "__main__":
    main()
