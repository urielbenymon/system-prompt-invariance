# System-Prompt Invariance for LLM-as-Judge Evaluation

Code, prompts, model outputs, and grading data for:

> **System-Prompt Invariance for LLM-as-Judge Evaluation: A Linguistic Robustness Study**
> Uriel Benymon, Albert Ming Wei, Zhicheng Guo, Xinbei Qiu, Clara Resnick
> *Workshop on Linguistic Principles for Foundation Models (LP4FM), NeurIPS 2026*

We test whether task-defining system prompts are behaviorally stable under small, task-preserving perturbations. Nine `system_prompt_vs_instruction` families from [BiGGen Bench](https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench) (grounding capability) are each rendered in three matched conditions:

| Code | Condition | Transformation |
|---|---|---|
| `C0_base` | Unperturbed | Original BiGGen system prompt |
| `C1_character_swap` | Character-level | One seeded adjacent-character transposition in a pre-selected word |
| `C2_lexical_substitution` | Lexical | One WordNet-supported synonym substitution, manually verified for contextual equivalence |

Four response-generation models (DeepSeek V3, DeepSeek R1, GPT-4o, o3-mini) produce **108 responses** (9 × 3 × 4). Each response is scored by three LLM graders (GPT-4 Turbo, Llama 3.1 70B Instruct, Claude Opus 4), giving **324 judgments**, aggregated by median-of-3. A blinded human validation study covers the 54 DeepSeek responses.

## Repository layout

```
scripts/                     Pipeline scripts (run from inside data/; see below)
data/                        All inputs, intermediate files, and frozen outputs
data/human_grading_files/    Blinded rater packets, completed rater sheets, ID mapping, instructions
```

Every script reads and writes files **relative to the current working directory**, so run them from `data/`:

```bash
cd data
python ../scripts/06_aggregate_and_analyze.py
```

## Reproducing the reported results (no API calls)

The analysis stages run entirely offline on the frozen data in this repository:

```bash
pip install -r requirements.txt
cd data
python ../scripts/03_reparse_autogrades.py        # autogrades_raw.jsonl -> autogrades_parsed_v2.csv
python ../scripts/06_aggregate_and_analyze.py     # median-of-3 scores, invariance summary, LLM-grader alpha
python ../scripts/08_statistical_significance.py  # Wilcoxon (invariance) and Cochran's Q / McNemar (compliance), Holm-corrected
python ../scripts/07_verify_human_reliability.py  # human alpha, human-vs-LLM agreement
```

Expected key outputs: human inter-rater Krippendorff's α = 0.9264; Pearson r (human vs. LLM median) = 0.9946; exact agreement 53/54 (98.1%).

## Full pipeline

Stages that call models require an [OpenRouter](https://openrouter.ai) key, read from the environment:

```bash
export OPENROUTER_API_KEY=...   # PowerShell: $env:OPENROUTER_API_KEY="..."
```

Re-running the API stages will **not** reproduce the outputs byte-for-byte. Hosted models change behind floating identifiers, and provider-side sampling isn't fully deterministic. The frozen outputs in `data/` are the ones analyzed in the paper.

| Stage | Script | Input → Output | API |
|---|---|---|---|
| 1. Load benchmark | `01_load_biggen.py` | Hugging Face → `biggen_system_prompt_vs_instruction.jsonl` | – |
| 2. Character swaps | `02_generate_seeded_character_swaps.py` | → `biggen_system_prompt_vs_instruction_character_swap.jsonl` | – |
| 3. Swap audit (interactive) | `02_audit_character_variants.py` | → `character_swap_audit_results.jsonl` | – |
| 4. Synonym candidate review (interactive) | `03_review_synonym_candidates.py` | `wordnet_synonym_candidates.jsonl` → `synonym_candidate_review_results.jsonl` | – |
| 5. WordNet lookup | `04_wordnet_synonym_lookup.py` | → `wordnet_lookup_results.jsonl` | – |
| 6. Contextual review (interactive) | `05_review_wordnet_synonyms.py` | → `wordnet_contextual_review_results.jsonl` | – |
| 7. Manual overrides | `apply_manual_wordnet_overrides.py` | `wordnet_contextual_review_results_render.jsonl` → `wordnet_contextual_review_results_final.jsonl` | – |
| 8. Pilot calibration | `00_pilot_token_calibration_dual_r1_full_variants.py` | `prompts_master_final.jsonl` → `responses_pilot_dual_r1.jsonl`, `pilot_token_summary_dual_r1.csv` | ✓ |
| 9. Official generation | `01_generate_official_responses.py` | `prompts_master_final.jsonl` → `responses_official.jsonl`, `generation_summary.csv`, `generation_manifest.json` | ✓ |
| 10. Automated grading | `02_autograde_responses_adjusted.py` | `responses_official.jsonl` → `autogrades_raw.jsonl`, `autogrades_parsed.csv`, `grading_manifest.json` | ✓ |
| 11. Parser widening | `03_reparse_autogrades.py` | → `autogrades_raw_v2.jsonl`, `autogrades_parsed_v2.csv`, `grading_manifest_v2.json` | – |
| 12. Aggregation & analysis | `06_aggregate_and_analyze.py` | → `aggregated_scores.csv`, `invariance_summary.csv`, `grader_reliability.txt` | – |
| 13. Rater packets | `make_rater_packets.py` | `responses_official.jsonl` → `human_grading_files/` | – |
| 14. Human reliability | `07_verify_human_reliability.py` | rater sheets + `aggregated_scores.csv` → stdout | – |
| 15. Significance tests | `08_statistical_significance.py` | → `significance_invariance.csv`, `significance_compliance.csv` | – |
| 16. Runtime audit | `q8_runtime_audit.py` | generation and grading logs → stdout (compute appendix) | – |

The final per-condition prompt sets are `prompts_base_final.jsonl`, `prompts_character_swap_final_audited.jsonl`, and `prompts_lexical_substitution_final.jsonl`, combined in `prompts_master_final.jsonl` (27 rows = 9 families × 3 conditions). Script numbering reflects the order in which the scripts were written, not strictly the order in which they run; the table above gives the run order. Manifests record SHA-256 hashes of their inputs and of the generating scripts, and the scripts here are byte-identical to the versions that produced them.

## Human validation data

- `rater_{1,2,3}_packet.csv`: the blinded packets given to raters. Model identity, condition, and all provenance fields are stripped, and item order is shuffled independently for each rater.
- `rater_{1,2,3}_sheet_formatted_human_grading.xlsx`: the completed ratings.
- `id_mapping_PRIVATE.csv`: maps the opaque `item_id`s back to generation cells. It was withheld from raters during grading to preserve blinding and is released here so the analysis can be reproduced.
- `rater_instructions.md`: participation and consent information plus grading instructions, as reproduced in Appendix B of the paper.

Raters are identified only by number. This repository doesn't link any rater number to a person.

## License

- **Code** (`scripts/`): [MIT](LICENSE).
- **Data** (`data/`): [CC BY-SA 4.0](LICENSE-DATA). These files are derived from BiGGen Bench (CC BY-SA 4.0), so the derived data carries the same license. Model outputs are additionally subject to the respective providers' terms of use.

## Citation

```bibtex
@inproceedings{benymon2026systemprompt,
  title     = {System-Prompt Invariance for {LLM}-as-Judge Evaluation: A Linguistic Robustness Study},
  author    = {Benymon, Uriel and Wei, Albert Ming and Guo, Zhicheng and Qiu, Xinbei and Resnick, Clara},
  booktitle = {NeurIPS 2026 Workshop on Linguistic Principles for Foundation Models (LP4FM)},
  year      = {2026},
  url       = {https://github.com/urielbenymon/system-prompt-invariance}
}
```
