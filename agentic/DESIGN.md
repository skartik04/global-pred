# Agentic Epilepsy Drug Prediction — Design Document

## Overview

A multi-agent LLM diagnostic swarm that simulates a clinical team meeting to predict antiepileptic drug prescriptions for Ugandan epilepsy patients. The system dynamically recruits specialist agents based on each patient's clinical presentation, producing not just a drug recommendation but a full reasoning trace showing how each specialist contributed to the decision.

**Target venue:** NeurIPS

**Core contribution:** The agentic framework itself — dynamic team composition, clinically motivated execution phases, and the quality of reasoning that emerges. Not accuracy against ground truth.

---

## Part 1: Data Cleaning Pipeline

### Raw Data
- **Source:** `data/combined_dataset.csv` — 279 Ugandan epilepsy patients, semicolon-delimited, 3 visits each (0, 6, 12 months)
- Each patient row contains free-text clinical notes and structured medication columns across all visits

### Step 1: Input/Output Split (`split_input_output.py`)

Splits raw clinical text into observation (input) vs prescription (output) per visit using an LLM.

- **Input text:** What the clinician observed — HPI, exam findings, seizure history, EEG results, current medications on arrival
- **Output text:** What the clinician decided — drug prescriptions, dose changes, follow-up plans
- **Output columns:** Structured medication fields from the CSV (e.g. "Medication dosage and if there was a change")

This split is the foundation of the leak-free pipeline: the prediction model sees only observations (input), never the prescription (output) for the current visit.

**Output:** `split_results.json` — 279 patients × 3 visits, each with `input_text`, `output_text`, `output_columns`

### Step 2: Clean Prescription Output (`build_clean_output.py`)

Merges `output_text` + `output_columns` into a single clean prescription string per visit.

**Two prompt types:**

1. **Standard prompt** (all visits for patients already on medication): Merge the free-text plan and structured medication column into one unified prescription. Remove observations, keep only treatment decisions.

2. **Backfill prompt** (Visit 1 for 123 not-on-medication patients): These patients had no prior medication at V1 — V1 is their first prescription. V1 output may be incomplete or empty. To recover missing V1 drug info, the LLM receives:
   - V1 output_text + V1 output_columns (what we have from V1)
   - V2 input_text (clinical notes at 6-month follow-up — mentions what patient has been taking)
   - V2 output_text + V2 output_columns (V2 prescription decisions)

   **Key justification:** There are NO visits between V1 and V2. Any drug that V2 describes as already active (patient was already taking it when they arrived at V2) must have been started at V1.

   **Semantic inclusion rules (conservative):**
   - INCLUDE: drugs V2 describes as already active — "continue", "maintain", "same as before", "still on", "has been on", dose adjustments to an existing drug (the drug itself is from V1)
   - EXCLUDE: drugs V2 introduces as new — "start", "add", "initiate", "switch to", "trial of"
   - EXCLUDE: ambiguous cases with no clear signal — omission is safer than incorrect inclusion
   - Do NOT copy V2 doses onto V1 — V2 dose may reflect a V2 change, not what was prescribed at V1

**Edge cases:**
- 1 genuinely empty visit: `102_Mulungi Rodney Visit_3` (no clinical data exists)
- `52_Ssemwogerere Farid Visit_1`: only case where V2 output column was the sole source of V1 drug info (manually verified)

**Output:** `clean_output.json` — 279 patients × 3 visits, single clean prescription string each

### Step 3: Ground Truth Extraction (`build_drug_gt.py`)

Extracts structured drug decisions from the prescription side: `{prescribed: [...], stopped: [...]}` per visit.

**Output:** `drug_gt.json` — noisy reference extracted by LLM from free text. May not always be correct. Not treated as oracle.

### Step 4: Prediction Input Construction (`build_pred_input.py` / `predict_drugs_clean.py:build_input()`)

Builds the input string that goes to the prediction model. Deterministic — no LLM involved.

**Structure for patient P at visit V:**
```
Age: X | Sex: M/F | Diagnosis: Y
Seizure onset: Z | Seizure duration: W

[Visit 1 (0 months) - Clinical Notes]
<input_text from split_results>

[Visit 1 (0 months) - Prescription]
<clean_output>

[Visit 2 (6 months) - Clinical Notes]        ← only if predicting V2 or V3
<input_text from split_results>

[Visit 2 (6 months) - Prescription]           ← only if predicting V3
<clean_output>

[Current Visit - Clinical Notes]              ← current visit: notes only
<input_text from split_results>
                                              ← NO prescription — this is what we predict
```

**Key properties:**
- Prior visits include both clinical notes AND prescription (the model sees what was decided before)
- Current visit includes only clinical notes (no leakage of the current prescription)
- No patient identifiers in the prompt — patient ID is only used as a JSON key for storage
- Demographics from `data/combined_dataset.csv` (age, sex, diagnosis, seizure onset, duration)

---

## Part 2: Agentic Architecture

### Why Agents?

Analysis of 60 visit-level feedback entries across 20 patients from two reviewing neurologists (JP and Raj) revealed that a single LLM consistently fails at tasks requiring distinct medical specializations:

| Failure pattern | Frequency | Required expertise |
|---|---|---|
| Seizure type/syndrome misclassification | Most common | Epileptology, neurophysiology |
| Weight-based dosing errors in children | Common | Pediatrics, pharmacology |
| Drug-seizure type contraindications missed | Common | Pharmacology |
| De-escalating working regimens | Common | Treatment response analysis |
| Missing infectious etiologies | Occasional | Tropical medicine |
| Ignoring drug availability constraints | Occasional | Health systems knowledge |

A single model trying to be all these specialists at once performs poorly. A team of focused agents, each bringing their domain expertise, produces better clinical reasoning.

### 6 Specialist Agents

#### 1. Seizure Diagnostician

**Real-world analogue:** Diagnostic neurologist / neurophysiologist

**Expertise:** Seizure type classification (focal vs generalized vs unknown), epilepsy syndrome recognition (West syndrome, Lennox-Gastaut, Landau-Kleffner, Rolandic epilepsy, Dravet), EEG interpretation from clinical notes, distinguishing semiology (what the seizure looks like) from etiology (where it originates).

**Why needed:** The #1 LLM failure. JP on 64_Owinyi Golden: "Sounds like West Syndrome based on semiology and EEG, wouldn't assume focal epilepsy." JP on 8_Tisma Natabi: "LLM should be picking up focal versus generalized rather than the semiology, since the EEG showed focal epilepsy." JP on 100_Otim Fortunate: "If patient does have Landau-Kleffner needs to be on clobazam or a benzodiazepine." Misclassification cascades into wrong drug selection across all subsequent visits.

**Output:** Seizure classification, syndrome identification (if applicable), confidence level, and reasoning. Flags when clinical features and EEG findings disagree.

#### 2. Prescribing Epileptologist

**Real-world analogue:** Attending epileptologist

**Expertise:** Treatment planning given seizure type + medication history + treatment response. Longitudinal reasoning across visits. Drug-resistant epilepsy (DRE) detection (failed 2+ optimized ASMs). Escalation logic (monotherapy → adjunctive → polytherapy). Knowing when NOT to change a working regimen.

**Why needed:** JP on 64_Owinyi Golden V3: "Wouldn't de-escalate given continued seizures. Would keep topiramate since that one seems to work best." Raj on 131_Muwanguzi: "If pt is seizure free on 2 drugs, that needs to be continued." JP on 90_Odongo Moses: "Since patient has DRE, decision might have been made that increased or optimized dose is futile." The LLM's default tendency is to always suggest alternatives, even when the current regimen is working.

**Output:** Concrete treatment plan — which drugs to continue/start/stop/adjust, with reasoning tied to the diagnostic classification and treatment history.

#### 3. Clinical Pharmacologist

**Real-world analogue:** Clinical pharmacist / pharmacologist

**Expertise:** Drug-drug interactions, drug-seizure type contraindications, safety profiles, weight-based dose checking, formulation knowledge (syrup vs tablet conversions), side effect assessment.

**Why needed:** JP on 320_Nakibirango Calvin: "We need to tell LLM to never mix PHB and clobazam." JP on 67_Ssenyonjo Waswa / 317_Nantume Shabila: CBZ worsening drop attacks / myoclonic seizures (drug-seizure type contraindication). JP on 85_Mirembe Mercy: "Usually 2ml depakote is 80mg" (formulation knowledge). JP on 90_Odongo Moses: "Would double check that both meds are at the weight based max dosing since children tend to gain weight every 6 months."

**Role:** Advisor, not gatekeeper. Reviews the epileptologist's proposed prescription and flags concerns. Does NOT rewrite the prescription or veto — reports flags for the orchestrator to weigh. This prevents the pharmacologist from becoming authoritarian and overriding well-reasoned treatment plans.

**Output:** Safety review — flagged interactions, contraindications, dosing concerns, with severity and reasoning. No alternative prescriptions.

#### 4. Pediatrician

**Real-world analogue:** General pediatrician

**Expertise:** Developmental context (cerebral palsy, global developmental delay, HIE, birth injury), weight-based dosing and growth trajectory, age-specific drug safety, comorbidity management in children (sickle cell, ADHD, behavioral issues).

**Why needed:** Most patients in this dataset are children. JP on 10_Muduku Matthew: "Probably need weight dosing adjustment. LLM didn't mention that as consideration for why there were more breakthroughs." JP on 90_Odongo Moses: "Children tend to gain weight every 6 months" — doses that were appropriate 6 months ago may now be subtherapeutic. JP on 85_Mirembe Mercy: "2 notes stacked together, first age is 4 months, second is 1 year" — developmental trajectory matters.

**Why separate from pharmacologist:** The pharmacologist knows drugs. The pediatrician knows the child. A 2-year-old with spastic quadriplegia, developmental delay, and weighing 8kg has different management considerations than a 12-year-old with focal seizures — that's pediatric expertise, not pharmacology.

**Output:** Developmental assessment, weight/age-appropriate dosing flags, comorbidity considerations, growth-related dose recalculation needs.

#### 5. Infectious Disease / Tropical Medicine Specialist

**Real-world analogue:** ID / tropical medicine physician

**Expertise:** Differentiating epilepsy from infectious seizure etiologies (cerebral malaria, neurocysticercosis, HIV-related CNS disease). Assessing acute infection impact on seizure control. Drug interactions between ASMs and antimalarials/antiretrovirals.

**Why needed:** The seizure differential in Uganda is fundamentally broader than in Western settings. A patient presenting with new-onset seizures + fever could be NCC, cerebral malaria, or HIV encephalopathy — management is completely different from epilepsy. JP on 320_Nakibirango Calvin V3: "LLM should comment on patient being acutely ill with infection in terms of consideration. Suspect increased seizures in the setting of acute infection."

**Output:** Differential assessment — is this truly epilepsy or could it be infectious? If infection is present, how does it affect seizure management? Flags when antiparasitic/antimicrobial treatment might change the ASM plan.

#### 6. Clinical Setting & Formulary Specialist

**Real-world analogue:** Health systems / public health physician

**Expertise:** Drug availability by setting, national formulary constraints, cost-effectiveness, supply chain realities. Generalized to any clinical setting — reads context clues from the notes to determine if LMIC or high-resource, then adjusts accordingly.

**Why needed:** Raj on 90_Odongo Moses: "VPA is more commonly available than LEV and provided by the national formulary, and as such more frequently prescribed." The best drug on paper is useless if it's not in stock. In a high-resource setting, this agent has minimal influence; in LMIC, it's critical.

**Why generalized:** The same framework should work on patients from any setting. The agent reads the clinical context and adapts — not hardcoded to Uganda.

**Output:** Setting assessment, drug availability/cost flags, formulary-appropriate alternatives when a recommended drug is likely unavailable.

### Execution Flow

```
Patient Input (clinical notes + prior prescriptions + demographics)
                    │
                    ▼
            ┌─────────────┐
            │ ORCHESTRATOR │ ── reads patient, decides which Phase 1 agents to recruit
            └──────┬──────┘
                   │
    ╔══════════════╧═══════════════╗
    ║      PHASE 1 (Parallel)      ║
    ║                              ║
    ║  ┌──────────────────────┐    ║
    ║  │ Seizure Diagnostician │   ║  ← always
    ║  └──────────────────────┘    ║
    ║  ┌──────────────────────┐    ║
    ║  │ Pediatrician          │   ║  ← if child / developmental
    ║  └──────────────────────┘    ║
    ║  ┌──────────────────────┐    ║
    ║  │ Infectious Disease    │   ║  ← if infection cues
    ║  └──────────────────────┘    ║
    ║  ┌──────────────────────┐    ║
    ║  │ Setting Specialist    │   ║  ← always
    ║  └──────────────────────┘    ║
    ║                              ║
    ║  Each sees: full patient     ║
    ║  history only                ║
    ╚══════════════╤═══════════════╝
                   │
    ╔══════════════╧═══════════════╗
    ║      PHASE 2 (Serial)        ║
    ║                              ║
    ║  ┌────────────────────────┐  ║
    ║  │ Prescribing             │  ║
    ║  │ Epileptologist          │  ║  ← sees full history + Phase 1 outputs
    ║  └───────────┬────────────┘  ║
    ║              │               ║
    ║  ┌───────────▼────────────┐  ║
    ║  │ Clinical                │  ║
    ║  │ Pharmacologist          │  ║  ← sees full history + Phase 1 + prescription
    ║  └────────────────────────┘  ║
    ║                              ║
    ╚══════════════╤═══════════════╝
                   │
            ┌──────▼──────┐
            │ ORCHESTRATOR │ ── sees EVERYTHING → final treatment decision
            └─────────────┘
```

### What Each Component Receives

| Component | Input |
|---|---|
| **Orchestrator (triage)** | Full patient input string |
| **Phase 1 agents** | Full patient input string (each independently) |
| **Prescribing Epileptologist** | Full patient input string + all Phase 1 agent outputs |
| **Clinical Pharmacologist** | Full patient input string + all Phase 1 agent outputs + epileptologist's treatment plan |
| **Orchestrator (final)** | Full patient input string + all 6 agent outputs |

### Design Principles

1. **Agents are real doctors, not ML tasks.** Each agent represents a distinct medical specialist you'd find at a clinical team meeting. Preprocessing tasks (medication reconciliation, data quality) are handled upstream in the cleaning pipeline, not by agents.

2. **Orchestrator is the final decision-maker.** Like the attending at a team meeting — hears everyone, weighs all inputs, makes the call. No single specialist can hijack the outcome.

3. **Pharmacologist advises, doesn't veto.** Reports safety flags for the orchestrator to consider. If the pharmacologist disagrees with the epileptologist, the orchestrator decides — not the pharmacologist.

4. **Every agent sees full patient history.** No telephone game. Phase 2 agents additionally see Phase 1 outputs. The orchestrator sees everything.

5. **Dynamic team composition.** Not every patient needs all 6 specialists. The orchestrator decides based on the clinical presentation. Simple adult focal epilepsy case: 4 agents. Complex pediatric case with infection: all 6.

6. **Reasoning traces are the output.** The final drug list matters, but the real deliverable is the reasoning: which agents were recruited, what each one found, how the orchestrator weighed their inputs, and why the final decision was made.

---

## Part 3: Data Files

| File | Description |
|---|---|
| `split_results.json` | 279 patients × 3 visits. `input_text` (clinical observations) + `output_text` (prescription). The leak-free foundation. |
| `clean_output.json` | 279 patients × 3 visits. Single clean prescription string per visit. Built from output_text + output_columns, with V2 backfill for not-on-med V1 patients. |
| `drug_gt.json` | 279 patients × 3 visits. `{prescribed: [...], stopped: [...]}`. Noisy LLM-extracted reference, not oracle. |
| `data/combined_dataset.csv` | Raw patient CSV, semicolon-delimited. Demographics + clinical notes + medication columns. |
| `feedbacks/feedback_JP.csv` | 60 visit-level feedback entries from neurologist JP (20 patients × 3 visits). 4-way judgment + comments. |
| `feedbacks/feedback_Raj.csv` | 60 visit-level feedback entries from neurologist Raj (20 patients × 3 visits). Yes/No judgment + comments. |
| `predict_prompt.txt` | 7-stage reasoning prompt for single-model baseline. |
| `inspect_pred_input.ipynb` | Notebook to view exact model input for any patient × visit. |
| `.env.example` | Template for API keys: `TOGETHER_API_KEY`, `OPENAI_API_KEY`, `CLAUDE_API_KEY`. |

## Clinical Context

- ~279 Ugandan epilepsy patients, 3 visits (0, 6, 12 months)
- Predominantly pediatric population
- Seizure differential broader than Western settings: epilepsy, cerebral malaria, neurocysticercosis, HIV-related CNS
- Limited formulary — 10 tracked ASMs: carbamazepine, clobazam, clonazepam, ethosuximide, lamotrigine, levetiracetam, phenobarbital, phenytoin, topiramate, valproate
- Neurologist collaborators provide ground truth validation and feedback

## Run Commands

```bash
# Conda env for all scripts
conda run -n global_llm python3 agentic/<script>.py

# API keys: copy .env.example to .env and fill in
cp agentic/.env.example agentic/.env

# Inspect prediction input for any patient
conda run -n global_llm jupyter notebook agentic/inspect_pred_input.ipynb
```
