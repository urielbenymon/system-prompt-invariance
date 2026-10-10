import json
from pathlib import Path

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path("wordnet_contextual_review_results_render.jsonl")
OUTPUT_FILE = Path("wordnet_contextual_review_results_final.jsonl")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_prompt(rows, exact_original_prompt):
    for row in rows:
        if row.get("system_prompt_original") == exact_original_prompt:
            return row

    raise ValueError(
        f"Could not find prompt in JSONL:\n{exact_original_prompt}"
    )


# ---------------------------------------------------------------------
# Manual audit override 1:
# Translate -> Interpret rejected; Translate -> Render approved
# ---------------------------------------------------------------------

TRANSLATE_PROMPT = "Translate the German to English."


def update_translate_prompt(row):
    """
    Preserve the manually reviewed decision:
      Interpret -> rejected
      Render    -> approved
    """

    row["selected_candidate_rank"] = 1
    row["selected_word"] = "Translate"
    row["selected_occurrence_number"] = 1
    row["selected_synonym"] = "render"
    row["selected_synset"] = "translate.v.01"
    row["system_prompt_transformed"] = "Render the German to English."
    row["selection_status"] = "approved"

    row["manual_reaudit"] = {
        "reason": "Interpretive-invariance review",
        "reviewed_candidates": [
            {
                "word": "Translate",
                "replacement": "interpret",
                "decision": "rejected",
                "synset": "translate.v.01",
                "justification": (
                    "'Interpret' permits a reasonable alternative reading in which "
                    "the model explains or construes the German rather than strictly "
                    "translating it. This introduces interpretive openness that could "
                    "alter the requested operation."
                ),
            },
            {
                "word": "Translate",
                "replacement": "render",
                "decision": "approved",
                "synset": "translate.v.01",
                "justification": (
                    "In the explicit German-to-English construction, 'render' preserves "
                    "the requested translation operation and does not introduce a "
                    "reasonable alternative interpretation that would materially alter "
                    "the expected response or scoring rubric."
                ),
            },
        ],
    }


# ---------------------------------------------------------------------
# Manual audit override 2:
# query -> question rejected
# expert -> no usable candidate
# generation -> genesis rejected
# make -> create approved
# ---------------------------------------------------------------------

QUERY_PROMPT = (
    "You are an expert of query generation. "
    "I want make more user queries to make a better retrieval. "
    "Given the popular dataset, make more user queries."
)


def update_query_prompt(row):
    """
    Replace the earlier query -> question choice with the manually reviewed
    make -> create substitution.
    """

    row["selected_candidate_rank"] = None
    row["selected_word"] = "make"
    row["selected_occurrence_number"] = 2
    row["selected_synonym"] = "create"
    row["selected_synset"] = "01620211-v"

    row["system_prompt_transformed"] = (
        "You are an expert of query generation. "
        "I want make more user queries to make a better retrieval. "
        "Given the popular dataset, create more user queries."
    )

    row["selection_status"] = "approved"

    row["manual_reaudit"] = {
        "reason": "Expanded contextual and interpretive-invariance review",
        "reviewed_candidates": [
            {
                "word": "expert",
                "replacement": None,
                "decision": "no_usable_candidate",
                "synset": None,
                "justification": (
                    "No usable WordNet-supported replacement was identified for "
                    "'expert' in this context."
                ),
            },
            {
                "word": "query",
                "replacement": "question",
                "decision": "rejected",
                "synset": "07208256-n",
                "justification": (
                    "In 'query generation,' 'query' is used in the technical "
                    "information-retrieval sense of a retrieval request/input. "
                    "'Question generation' can instead refer to generating "
                    "natural-language interrogative questions, introducing a "
                    "reasonable alternative task interpretation."
                ),
            },
            {
                "word": "generation",
                "replacement": "genesis",
                "decision": "rejected",
                "synset": "07338980-n",
                "justification": (
                    "Although WordNet groups 'generation' and 'genesis' under "
                    "'a coming into being,' 'query genesis' can denote the origin "
                    "or emergence of queries rather than the process of producing "
                    "them. The substitution therefore changes the contextual "
                    "interpretation."
                ),
            },
            {
                "word": "make",
                "replacement": "create",
                "decision": "approved",
                "synset": "01620211-v",
                "justification": (
                    "WordNet places 'make' and 'create' in the same creation sense "
                    "('make or cause to be or to become'). In 'make more user queries,' "
                    "'create' preserves the intended operation of producing additional "
                    "queries without changing task scope, interpretation, or rubric "
                    "applicability."
                ),
            },
        ],
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    rows = load_jsonl(INPUT_FILE)

    translate_row = find_prompt(rows, TRANSLATE_PROMPT)
    update_translate_prompt(translate_row)

    query_row = find_prompt(rows, QUERY_PROMPT)
    update_query_prompt(query_row)

    save_jsonl(rows, OUTPUT_FILE)

    print(f"Loaded:  {INPUT_FILE}")
    print(f"Updated: {TRANSLATE_PROMPT}")
    print(f"Updated: {QUERY_PROMPT}")
    print(f"Wrote:   {OUTPUT_FILE}")


if __name__ == "__main__":
    main()