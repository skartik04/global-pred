# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What We're Building
A **multi-agent LLM diagnostic swarm** for epilepsy patients in Uganda. The goal is a paper (targeting NeurIPS) showing that a dynamic team of LLM agents — simulating a full clinical diagnostic team — can reason through complex seizure cases the way a real specialist team would.

**This is NOT a standard prediction pipeline.** Accuracy against ground truth is not the deliverable. The contribution is the **agentic framework itself**: how the system decides which specialist agents to recruit, when, and why — and the quality of clinical reasoning that emerges.

---

## Execution Architecture

**3-phase execution with orchestrator as final decision-maker.**

### Phase 1 — Parallel Assessments
Each specialist reads the full patient history independently and produces an assessment. The orchestrator decides which specialists to fire based on the patient presentation.

| Specialist | Role | Triggered when |
|---|---|---|
| **Seizure Diagnostician** | Classifies seizure type/syndrome (focal vs generalized, West, LGS, Landau-Kleffner, Rolandic). Reads EEG findings. Flags when semiology ≠ etiology. | Always |
| **Pediatrician** | Developmental context (CP, global delay, HIE), weight-based dosing, age-specific safety, growth trajectory. | Children, developmental delay, CP, birth injury |
| **Infectious Disease / Tropical Medicine** | Differentiates epilepsy from infectious etiologies (cerebral malaria, NCC, HIV CNS). Assesses acute infection impact on seizure control. | Fever, infection signs, new-onset seizures, tropical cues |
| **Clinical Setting & Formulary Specialist** | Determines resource level from context (LMIC vs high-resource). Adjusts for drug availability, cost, national formulary. Generalized — not hardcoded to Uganda. | Always |

### Phase 2 — Serial Prescribing + Safety Review
Each agent sees full patient history + all Phase 1 outputs.

| Specialist | Role |
|---|---|
| **Prescribing Epileptologist** | Takes diagnosis + assessments → makes treatment plan. Handles treatment response (continue/escalate/switch), DRE detection, longitudinal reasoning. |
| **Clinical Pharmacologist** | Reviews the prescription for drug-drug interactions, drug-seizure contraindications, dosing errors. **Advisor, not gatekeeper** — reports flags, does not veto. |

### Phase 3 — Orchestrator Final Decision
The orchestrator sees EVERYTHING: full patient history + all 6 agent outputs. It synthesizes and makes the final integrated treatment decision. No single specialist can hijack the outcome.

### Key Design Principles
- **Orchestrator is the attending at the team meeting** — it heard everyone, it decides
- **Every agent sees full patient history** — not just the previous agent's summary
- **Serial agents also see prior agent outputs** — the epileptologist sees the diagnostician's classification, the pharmacologist sees the prescription
- **Pharmacologist is an advisor, not a gate** — it flags concerns, orchestrator weighs them
- **Dynamic team composition** — not every patient gets all 6 specialists (simple adult case: 4 agents; complex pediatric case with infection: all 6)

### Typical Routing
- Simple adult: Diagnostician → Setting → Epileptologist → Pharmacologist (4 agents)
- Pediatric: Diagnostician → Pediatrician → Setting → Epileptologist → Pharmacologist (5 agents)
- Complex pediatric + infection: All 6 → Orchestrator (full team)

---

## Why These 6 Specialists (Evidence from Doctor Feedback)

Derived from analysis of 60 visit-level feedback entries across 20 patients from two reviewing physicians (JP and Raj). See `feedback analysis/` for raw CSVs.

1. **Seizure Diagnostician** — #1 LLM failure was misclassifying seizure type. West Syndrome missed (64_Owinyi), focal-vs-generalized confusion (8_Tisma, 260_Mukisa), rare syndromes missed (100_Otim Landau-Kleffner, 262_Mutyaba LGS).

2. **Prescribing Epileptologist** — Longitudinal treatment response errors: de-escalating despite continued seizures (64_Owinyi), changing working regimens unnecessarily (131_Muwanguzi, 227_Muluwaya), missing DRE signals (58_Sekyeru, 90_Odongo).

3. **Clinical Pharmacologist** — Drug-drug interactions missed (PHB+CLB: 320_Nakibirango), drug-seizure contraindications (CBZ worsening myoclonic: 67_Ssenyonjo, 317_Nantume), weight-based dosing gaps (85_Mirembe, 90_Odongo).

4. **Pediatrician** — Most patients are children. Weight gain → dose recalculation needed (90_Odongo), developmental context affects drug choice (CP patients), formulation knowledge (syrup-to-mg: 85_Mirembe).

5. **Infectious Disease** — Uganda-specific: cerebral malaria, NCC, HIV CNS. Acute infection exacerbating seizures (320_Nakibirango V3). Distinguishing epilepsy from infectious seizure etiologies.

6. **Setting Specialist** — Drug availability varies by setting (90_Odongo: "VPA more available than LEV in Uganda"). Generalized so the framework works for any clinical setting.

**Not included as agents (handled in preprocessing):**
- Medication reconciliation / timeline tracking → already solved by `clean_output.json` pipeline
- Clinical data quality review (stacked notes, missing data) → handled upstream before agents see the input

---

## Data Available (this folder)

| File | Description |
|------|-------------|
| `split_results.json` | 279 patients × 3 visits. `input_text` = clinical observations. `output_text` = prescription. Clean input/output split — no leakage. |
| `clean_output.json` | 279 patients × 3 visits. Single clean prescription text per visit, built from `output_text` + `output_columns`. Includes V2-backfilled V1 for 123 not-on-med patients. |
| `drug_gt.json` | 279 patients × 3 visits. `{prescribed: [...], stopped: [...]}` — extracted drug decisions. Reference only, not oracle. |
| `predict_prompt.txt` | 7-stage reasoning pipeline prompt (used by single-model baseline `predict_drugs_clean.py`). |
| `inspect_pred_input.ipynb` | Notebook to inspect the exact input string that goes to the model for any patient × visit. |

**Primary input to the agent system:** Built on the fly from `split_results.json` (clinical notes) + `clean_output.json` (prior prescriptions) + `data/combined_dataset.csv` (demographics). See `build_pred_input.py` or `predict_drugs_clean.py:build_input()` for the exact construction.

**GT caveat:** `drug_gt.json` is a noisy reference extracted by LLM from free text. Do not treat match rate against it as the success metric.

---

## Clean Output Pipeline (Preprocessing)

`build_clean_output.py` produces `clean_output.json` — a single clean prescription string per patient per visit.

- **Standard visits:** Merges `output_text` + `output_columns` from `split_results.json`.
- **Not-on-med V1 (123 patients):** Backfills V1 prescription from V2 data. Uses V2 `input_text` + V2 `output_text` + V2 `output_columns` with semantic inclusion rules (drug already active at V2 → must be from V1, since no visits exist between V1 and V2). Conservative: ambiguous = exclude. Does NOT copy V2 doses.
- **`rerun_clean_output.py`**: Reruns any empty visits. 1 genuinely empty: `102_Mulungi Rodney Visit_3`.

---

## Clinical Context
- ~279 Ugandan epilepsy patients, 3 visits (0, 6, 12 months)
- Seizure differential in Uganda is broader than Western settings: epilepsy, cerebral malaria, neurocysticercosis, HIV-related CNS, limited formulary
- 10 tracked ASMs: `carbamazepine, clobazam, clonazepam, ethosuximide, lamotrigine, levetiracetam, phenobarbital, phenytoin, topiramate, valproate`
- Neurologist collaborators on the team for ground truth validation

---

## Run Commands
```bash
# Conda env
conda run -n global_llm python3 agentic/<script>.py

# API keys in agentic/.env: TOGETHER_API_KEY, OPENAI_API_KEY, CLAUDE_API_KEY
```

---

## What Already Exists (preprocessing, not the main task)
- `split_input_output.py` → `split_results.json` (LLM splits free text into observations vs plan)
- `build_clean_output.py` → `clean_output.json` (LLM merges output fields into single prescription per visit)
- `build_drug_gt.py` → `drug_gt.json` (LLM extracts drug decisions from prescription side)
- `build_pred_input.py` → `pred_input.json` (deterministic input construction, no LLM)
- `predict_drugs_clean.py` → `drug/{model}_v{N}_options.json` (single-model baseline predictions)
- `count_visits.py` / `rerun_failed.py` → `visit_counts.json`

These are upstream data prep. The agentic system is built on top of `split_results.json` + `clean_output.json`.





ALWAYS READ THESE FILES TOO:

1. raw_text/CLAUDE.md — Full architecture with 6 specialists, 3-phase execution, routing logic, evidence from
   doctor feedback, data files, preprocessing pipeline docs
2. memory/agentic_workspace.md — Updated with finalized specialist list, execution phases, evidence base
3. memory/feedback_report_before_act.md — New: report before editing
4. memory/feedback_agents_medical.md — New: agents = real doctors, not ML tasks
5. memory/MEMORY.md — Index updated with all new entries + clean output pipeline state