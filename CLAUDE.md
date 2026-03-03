# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Epilepsy drug prediction pipeline for ~332 Ugandan patients across 3 visits (0, 6, 12 months). Goal: predict which antiepileptic drugs a clinician should prescribe at each visit, given clinical history and prior medication record.

**Conda env:** `global_llm`
**Scripts dir:** `scripts/`
**Data dir:** `data/combined_dataset.csv` (semicolon-delimited, 332 patients)
**API keys:** stored in `scripts/.env` — `TOGETHER_API_KEY`, `OPENAI_API_KEY`, `CLAUDE_API_KEY`

---

## Two Pipelines

### Pipeline A — Original (`scripts/`)
4 stages, run in order:

1. **`extract_phenos.py`** → `csv_feats_gpt_oss.json`
   Extracts 18 features per patient per visit (8 clinical + 10 drugs) from `combined_dataset.csv`. Drug fields store `Exposure_before` + `Action_taken` across all visits cumulatively.

2. **`fix_drug_backfill.py`** → `csv_backfilled_gpt_oss.json`
   Backfills V1 drug entries inferred from V2/V3 observations. Marks them `Reasoning: "Backfilled from..."`.

3. **`reconc_all.py`** → `csv_reconciled_gpt_oss.json`
   Reconciles drug timelines across visits. **This is the main input to prediction.**

4. **`predict_drugs.py`** → `drug/all_drugs/{model}_v{N}_ext_options.json`
   Predicts 3 drug options per patient per visit. Key function: `build_extracted_content()` — passes non-drug features from current visit + **previous visit's drug status only** (no current-visit plan leakage). Configuration at top of file: `model`, `visit_num`, `input_mode`, `prompt_file`.

**Output format (`ext_options`):** `option_1/2/3` each with `drugs: [{drug, action}]` + `label` + `rationale` + `think` (reasoning trace).

---

### Pipeline B — Leak-Free (`scripts/*_clean.py`)
Separates observation from prescription at raw text level before any extraction.

1. **`split_input_output.py`** → `split_results.json`
2. **`build_ground_truth.py`** → `ground_truth.json` (prescribed/stopped drugs per visit)
3. **`extract_from_input.py`** → `extracted_features.json`
4. **`predict_drugs_clean.py`** → `drug/clean/...`

---

## PDF Pipeline (`pdf_scripts/`)
375 patient folders in `all_patient_pdfs/`. Each folder has `*_merged.txt` with `DOCUMENT 1/2/3` sections = Visit 1/2/3.

**`pdf_scripts/pdf_extract.py`** — reads `*_merged.txt`, extracts 15 features (no per-visit split yet), saves to `pdf_scripts/extracted_pdf_features.json`.

> **Note:** PDF data requires cleaning before pipeline runs. The per-visit split from DOCUMENT sections is not yet implemented for the full Pipeline A flow.

---

## Grading

**Script:** `grade_preds.py`
- Hardcoded to read `openai_gptoss120b_v{1,2,3}_ext_options.json` and `csv_reconciled_gpt_oss.json`
- Match rule: GT active set == pred active set (drug names only, actions stripped)
- GT active = drugs where action ∈ {continue, start}; `stop` only counted when `Exposure_before == on_arrival`
- mono = gt_size ≤ 1, poly = gt_size ≥ 2

**Results (332 patients, gptoss120b):**
| Visit | Overall | Mono | Poly |
|-------|---------|------|------|
| V1 | 61.1% | 72.4% | 22.7% |
| V2 | 71.1% | 89.5% | 30.8% |
| V3 | 66.9% | 93.0% | 27.3% |
| All | 66.4% | | |

**Output:** `drug/all_drugs/grading_results.csv`
Columns: `patient_id, visit, mono_poly, match, matched_option, gt_active, gt_size, option_1_active, option_2_active, option_3_active`

---

## Key Data Files

| File | Description |
|------|-------------|
| `data/combined_dataset.csv` | Raw patient CSV, semicolon-delimited |
| `scripts/csv_reconciled_gpt_oss.json` | Main reconciled features (332 pats) — primary pipeline input |
| `scripts/drug/all_drugs/openai_gptoss120b_v{1,2,3}_ext_options.json` | gptoss predictions |
| `scripts/drug/all_drugs/grading_results.csv` | Grading results (996 rows) |
| `scripts/drug/all_drugs/active_drugs_per_visit.csv` | GT active drugs: pid × v1/v2/v3 |
| `scripts/drug/all_drugs/pred_active_drugs_per_visit.csv` | Pred active drugs, pipe-delimited options |
| `scripts/ground_truth.json` | Clean pipeline GT |
| `scripts/extracted_features.json` | Clean pipeline extracted features |

---

## 10 Drugs Tracked
`clobazam, clonazepam, valproate, ethosuximide, levetiracetam, lamotrigine, phenobarbital, phenytoin, topiramate, carbamazepine`

---

## Model / Provider Routing
`predict_drugs.py` auto-detects provider from model name:
- `claude-*` → Anthropic API (key: `CLAUDE_API_KEY`)
- `gpt-*`, `o1/o3/o4` → OpenAI Responses API (key: `OPENAI_API_KEY`)
- anything else → Together AI (key: `TOGETHER_API_KEY`)

Thinking/reasoning traces are captured for all providers. Output filenames are auto-derived from model name + visit + mode.

---

## Viewer / Feedback

- **`viewer/app.py`** — Local Streamlit viewer for all 332 patients
- **`viewer_feedback/app.py`** — Hosted feedback viewer (20 sampled patients)
  - Name-based login; per-reviewer CSV saved to `viewer_feedback/evaluations/`
  - 20 patients sampled: 10 poly_wrong, 5 mono_wrong, 3 poly_correct, 2 mono_correct
  - Deployed repo: https://github.com/skartik04/drug_feedback

---

## Notebook
**`scripts/inspect_outputs.ipynb`** — Grading visualizations. Reads `grading_results.csv`. Shows accuracy by visit, mono vs poly, accuracy by GT size, which option matched, sample cases.

---

## Run Commands
```bash
# Full Pipeline A
conda run -n global_llm python3 scripts/extract_phenos.py
conda run -n global_llm python3 scripts/fix_drug_backfill.py
conda run -n global_llm python3 scripts/reconc_all.py
conda run -n global_llm python3 scripts/predict_drugs.py   # edit model/visit_num at top first

# Grade
conda run -n global_llm python3 scripts/grade_preds.py

# Execute notebook
conda run -n global_llm jupyter nbconvert --to notebook --execute scripts/inspect_outputs.ipynb --output scripts/inspect_outputs.ipynb

# Viewers
conda run -n global_llm streamlit run viewer/app.py
conda run -n global_llm streamlit run viewer_feedback/app.py

# PDF extraction
conda run -n global_llm python3 pdf_scripts/pdf_extract.py
```
