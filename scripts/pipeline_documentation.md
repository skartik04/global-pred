# Leak-Free Drug Prediction Pipeline

## Problem Statement

Clinical notes in this epilepsy cohort (~333 patients, 3 visits at 0/6/12 months) mix **observational data** with the **doctor's prescription plan** in the same text fields. Approximately 93% of HPI (History of Presenting Illness) fields contain an embedded "Plan" section; the "Current drug regimen" column mixes prior medications with new prescriptions. This means any extraction model sees the drug decisions that the prediction model is supposed to predict — a **data leakage** problem.

This pipeline cleanly separates **input** (observations) from **output** (plan) before any extraction, ensuring the prediction model never sees the current visit's answer.

---

## Pipeline Architecture

```
                         ┌──────────────────────────┐
                         │    Raw CSV (333 patients, │
                         │    semicolon-delimited)   │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                     │
        ┌───────────────────────┐                        │
        │  STEP 1: SPLIT        │                        │
        │  (split_input_output  │                        │
        │   .py)                │                        │
        │                       │                        │
        │  LLM separates each   │                        │
        │  visit into:          │                        │
        │  • INPUT  (obs)       │                        │
        │  • OUTPUT (plan)      │                        │
        └───┬───────────┬───────┘                        │
            │           │                                │
            │           ▼                                │
            │   ┌───────────────────────┐                │
            │   │  STEP 1b: GROUND      │                │
            │   │  TRUTH                │                │
            │   │  (build_ground_truth  │                │
            │   │   .py)                │                │
            │   │                       │                │
            │   │  Extract drug labels  │                │
            │   │  from OUTPUT side +   │                │
            │   │  reconcile undocu-    │                │
            │   │  mented Rx            │                │
            │   └───────────┬───────────┘                │
            │               │                            │
            ▼               │                            │
    ┌───────────────────┐   │                            │
    │  STEP 2: EXTRACT  │   │                            │
    │  (extract_from_   │   │                            │
    │   input.py)       │   │                            │
    │                   │   │                            │
    │  18-feature       │   │                            │
    │  extraction from  │   │                            │
    │  INPUT-only text  │   │                            │
    └───────┬───────────┘   │                            │
            │               │                            │
            ▼               ▼                            │
    ┌─────────────────────────────────┐                  │
    │  STEP 3: PREDICT                │                  │
    │  (predict_drugs_clean.py)       │                  │
    │                                 │                  │
    │  Drug recommendation from:      │                  │
    │  • Extracted features (Step 2)  │                  │
    │  • Prior plans as med history   │                  │
    │    (Step 1b, V1..V(N-1))        │                  │
    │  • Prior drug observations      │                  │
    │                                 │                  │
    │  Evaluated against ground truth │◄─────────────────┘
    │  from Step 1b                   │
    └─────────────────────────────────┘
```

### Key Design Decisions

1. **No drug backfill to inputs** — backfilling Visit 1 input with drugs from Visit 2 observations would leak ground truth (those drugs came from Visit 1's plan, which is what we're predicting).

2. **Carry forward prior plans** — for Visit N prediction, include plans from V1..V(N-1) as medication history. This is not leakage: the doctor at Visit 2 knows what was prescribed at Visit 1.

3. **Three-source medication history** — the prediction model sees:
   - **(a)** Current visit's drug observations (what patient is on now)
   - **(b)** Prior visits' documented plans (from ground truth)
   - **(c)** Prior visits' drug observations not covered by (a) or (b)

4. **Undocumented prescriptions** — if a patient arrives at Visit 2 on a drug not in any prior documented plan, attribute it to the most recent prior visit (documentation gap, not obtained elsewhere).

5. **Step 1 is fully LLM-based** — clinical notes are nuanced enough to warrant LLM splitting for all patients, not just edge cases.

---

## Step 1: Split Input/Output

**Script:** `split_input_output.py`
**Model:** `openai/gpt-oss-120b` (via Together AI)
**Output:** `split_results.json`

### Column Pre-Classification

Before sending text to the LLM, columns are pre-classified:

| Classification | Columns |
|---|---|
| **Always INPUT** | Age, Sex, Seizure Diagnosis, risk factor checkboxes, developmental status, weight, seizure frequency, medication status |
| **Always OUTPUT** | `Current dose`, `Medication dosage and if there was a change in medication (6/12 months)` |
| **Needs LLM splitting** | `History of Presenting Illness`, `Second Entry(6 months)`, `Third Entry(12 months)`, `Current drug regimen` |

### Split Prompt (abbreviated)

> You are a clinical note splitter. Given text from an epilepsy clinic note, separate it into:
>
> 1. **INPUT (observations):** Clinical history, examination findings, what the patient is currently taking (observed state), seizure descriptions, developmental information, past medical history.
>
> 2. **OUTPUT (treatment plan):** Prescriptions, dose changes, new medications ordered, investigations ordered, referrals, follow-up plans.
>
> Key distinctions:
> - "currently on sodium valproate" → INPUT (observation)
> - "Tabs Carbamazepine 200 mg bd x 1/12" → OUTPUT (prescription)
> - "episodes reduced with carbamazepine" → INPUT (treatment response)
> - "Start phenytoin 5 mls bd" → OUTPUT (new prescription)

### Worked Example: Mwania Sheldon

**Raw HPI text (Visit 1):**
> 1-year 10/12 month old with no known chronic illnesses, was well till 4/12 ago when he developed convulsions that started with the right upper and lower limbs that later progressed to involve the whole body. The child has had convulsions 4 times since August 2022 (last one was yesterday) and has not been on anticonvulsant medication till yesterday when he was started on phenytoin syrup 5 mls bd. Mother reports that she started ANC at 3 months and attended monthly until she delivered. Pregnancy was uneventful. Delivered by SVD at term BWT 4.3 kg, neonatal period was uneventful. No history of any febrile illnesses. There is history of seizure disorder in the family (maternal grandfather). O/E: FGC, very irritable with excessive crying, no pallor, no jaundice, no dehydration, no edema. CNS: Very irritable, fully conscious, normal tone in all limbs, no FNDs...

**After Split:**

| Field | Content |
|---|---|
| **input_text** | *(full HPI above — all observational)* |
| **output_text** | *(empty — no plan found in HPI; Visit 1 had "Not on Medication" status and no plan column)* |
| **input_columns** | Age: 1 5/6, Sex: M, Seizure Diagnosis: Epilepsy, Medication status: Not on Medication, Risk Factors: Unknown (Checked) |
| **output_columns** | *(empty)* |

**Visit 2 split:**

| Field | Content |
|---|---|
| **input_text** | "2 1/2 year old male with Epilepsy on sodium valproate seizure onset August 2022. Medication began January 2023, since medication began he had one episode of seizure attack (duration: 20 min)" |
| **output_text** | *(empty)* |
| **input_columns** | Date: 26/06/2023, Sex: M, Weight: 15.5 kg, Seizure frequency: Reduced |
| **output_columns** | "1. Sodium Valproate 150 mg bd x 3/12" |

**Visit 3 split:**

| Field | Content |
|---|---|
| **input_text** | "Reviewed a 3-year-old male with epilepsy on Sodium Valproate at 18 mg/kg/day. Mother reports seizures are frequent and not well controlled. O/E: FGC, Afebrile, Not pale, No Jaundice. CNS: Fully alert, Normal tone and No FNDs" |
| **output_text** | *(empty)* |
| **input_columns** | Date: 25/01/2024, Sex: M, Weight: 16.2 kg, Seizure frequency: Increased |
| **output_columns** | "1. Increase Na Valproate to 25 mg/kg/day. Tabs Sodium Valproate 200mg BD x 2/12 2. Tabs Levetiracetam 250 mg B.D x 2/12" |

> **Note:** For Sheldon's visits, the HPI text was entirely observational (no embedded plan), so `output_text` was empty. The prescriptions lived in the always-OUTPUT dose columns, which were automatically classified.

---

## Step 1b: Ground Truth Construction

**Script:** `build_ground_truth.py`
**Output:** `ground_truth.json`

### How it works

1. Parse OUTPUT text + output_columns to identify which of the 10 tracked drugs are prescribed, stopped, or not mentioned.
2. Parse INPUT text to identify which drugs are observed (on_arrival, past_tried).
3. **Reconciliation:** if Visit N's INPUT shows the patient on drug X, but no prior visit's plan prescribed drug X → attribute drug X to the most recent prior visit's plan.

### Worked Example: Mwania Sheldon

**Visit 1 ground truth:**

The OUTPUT side is empty (no plan documented). However, at Visit 2 the patient arrives "on sodium valproate" and "medication began January 2023" — the same date as Visit 1. Since no prior plan prescribed valproate, the reconciliation logic attributes it to Visit 1:

```
Visit_1:
  prescribed_drugs: [valproate]          ← reconciled from V2 observation
  stopped_drugs:    []
  observed_on_arrival: [phenytoin]       ← V1 input says "started on phenytoin"
  source: "reconciled_from_later_observation"
```

**Visit 2 ground truth:**

The output_columns contain "Sodium Valproate 150 mg bd x 3/12":

```
Visit_2:
  prescribed_drugs: [valproate]
  stopped_drugs:    []
  observed_on_arrival: [valproate]       ← V2 input says "on sodium valproate"
  source: "documented_plan"
```

**Visit 3 ground truth:**

The output_columns contain both valproate and levetiracetam:

```
Visit_3:
  prescribed_drugs: [levetiracetam, valproate]
  stopped_drugs:    []
  observed_on_arrival: [valproate]
  source: "documented_plan"
```

### Reconciliation Summary (5-patient trial)

| Patient | Drug | Attributed to | Reason |
|---|---|---|---|
| Mwania Sheldon | valproate | Visit 1 | V2 says "on sodium valproate", V1 had no plan |
| Kazibwe Ryan | carbamazepine | Visit 1 | V2 says "episodes reduced with carbamazepine", V1 had no plan |
| Asio Esther Jane | valproate | Visit 2 | V3 says "on Sodium Valproate", V2 plan only documented ethosuximide |

---

## Step 2: Structured Extraction (INPUT Only)

**Script:** `extract_from_input.py`
**Prompt:** `extract_input_prompt.txt`
**Output:** `extracted_features.json`

### Key changes from old pipeline

| Old Pipeline | New Pipeline |
|---|---|
| Extracts from raw CSV columns (HPI includes plan) | Extracts from `input_text` only (plan removed) |
| Drug schema: `Use_status: <current\|previous\|unclear>; Temporal_language: <ongoing\|initiation\|dose_change\|stop\|unclear>` | Drug schema: `Status: <on_arrival\|past_tried\|not_mentioned>` |
| `Action_taken` field present (leaks treatment decisions) | No `Action_taken` field |

### Extraction Prompt (key sections)

> You are an expert clinician and medical NLP assistant reviewing a single epilepsy clinic visit note.
>
> **IMPORTANT:** The text you receive contains ONLY observational clinical information. No treatment plans, prescriptions, or medication orders are included.
>
> For each of 18 features, extract:
> - `value`: short clean statement or "Not mentioned."
> - `supporting_text`: exact quoted text from the note
> - `reasoning`: how you decided
> - `confidence`: high / med / low
>
> **Drug features (9-18):** For each drug, value must be either "Not mentioned." or:
> ```
> "Status: <on_arrival|past_tried|not_mentioned>"
> ```
> - `on_arrival`: patient is currently taking this drug (observed)
> - `past_tried`: patient previously used this drug but no longer on it
> - `not_mentioned`: drug not mentioned in the text

### Worked Example: Mwania Sheldon

**Visit 1 — Extracted Features (selected):**

| Feature | Value | Supporting Text |
|---|---|---|
| Age_Years | 1.8 | "1-year 10/12 month old" |
| Sex_Female | No | "Patient Gender: M" |
| OnsetTimingYears | 0.3 | "Age of Onset of Seizure: 1 year 6/12 months" |
| SeizureFreq | Four seizures since August 2022, most recent yesterday | "convulsions 4 times since August 2022 (last one was yesterday)" |
| StatusOrProlonged | Not mentioned. | |
| CognitivePriority | Not mentioned. | |
| Risk_Factors | family_history_epilepsy | "history of seizure disorder in the family (maternal grandfather)" |
| SeizureType | Focal onset convulsive seizures progressing to generalized tonic-clonic | "convulsions that started with the right upper and lower limbs that later progressed to involve the whole body" |
| drug_phenytoin | **Status: on_arrival** | "started on phenytoin syrup 5 mls bd" |
| drug_valproate | Not mentioned. | |
| *(all other drugs)* | Not mentioned. | |

**Visit 2 — Extracted Features (selected):**

| Feature | Value | Supporting Text |
|---|---|---|
| Age_Years | 2.5 | "2 1/2 year old male" |
| OnsetTimingYears | 0.8 | "seizure onset August 2022" (visit June 2023) |
| SeizureFreq | Reduced; one seizure since medication began | "since medication began he had one episode of seizure attack" |
| StatusOrProlonged | One seizure lasting 20 minutes | "duration: 20 min" |
| drug_valproate | **Status: on_arrival** | "on sodium valproate" |
| drug_phenytoin | Not mentioned. | *(phenytoin not in V2 input text)* |
| *(all other drugs)* | Not mentioned. | |

**Visit 3 — Extracted Features (selected):**

| Feature | Value | Supporting Text |
|---|---|---|
| Age_Years | 3.0 | "3-year-old male" |
| SeizureFreq | Frequent seizures, increased | "seizures are frequent and not well controlled" |
| drug_valproate | **Status: on_arrival** | "on Sodium Valproate at 18 mg/kg/day" |
| *(all other drugs)* | Not mentioned. | |

---

## Step 3: Drug Prediction

**Script:** `predict_drugs_clean.py`
**Prompt:** `predict_prompt_clean.txt`
**Output:** `drug/clean/<model>_v<visit>_clean_top3.{json,csv}`

### How the prediction input is constructed

For Visit N, the model receives:

1. **Extracted clinical features** (from Step 2) — demographics, seizure characteristics, risk factors
2. **Supporting clinical text** (from Step 1 input_text) — safe, no leakage
3. **Medication history from three sources:**
   - **(a)** Current visit's extracted drug observations (what patient is on now)
   - **(b)** Prior visits' documented plans from ground truth (V1..V(N-1))
   - **(c)** Prior visits' drug observations not already covered by (a) or (b)

### Prediction Prompt (key sections)

> **ROLE:** You are an expert epileptologist. Recommend the top 3 most appropriate anti-seizure medications (ASMs) using structured clinical reasoning.
>
> **SETTING:** Uganda, resource-limited, cost-sensitive prescribing. ONLY consider these 10 ASMs: carbamazepine, clobazam, clonazepam, ethosuximide, lamotrigine, levetiracetam, phenobarbital, phenytoin, topiramate, valproate.
>
> **HIERARCHICAL REASONING PIPELINE:**
> - Stage 0: Input sanity and uncertainty
> - Stage 1: Seizure-type compatibility shortlist
> - Stage 2: Hard safety and special-population modifiers
> - Stage 3: Practicality, Cost, and LMIC Baseline (anchor to VPA/CBZ/PB)
> - Stage 4: Side-effect alignment and comorbidity tie-breakers
> - Stage 5: Monotherapy first, polytherapy logic if needed
>
> **OUTPUT FORMAT:**
> ```
> ---SECTION 1: CLINICAL REASONING---
> [reasoning through the pipeline]
>
> ---SECTION 2: TOP-3 DRUG RECOMMENDATIONS---
> Rank 1: <drug> | reason: <1-2 sentences>
> Rank 2: <drug> | reason: <1-2 sentences>
> Rank 3: <drug> | reason: <1-2 sentences>
> ```

### Worked Example: Mwania Sheldon

#### Visit 1 Prediction

**What the model sees:**

```
=== CLINICAL FEATURES ===
  Age_Years: 1.8
  Sex_Female: No
  OnsetTimingYears: 0.3
  SeizureFreq: Four seizures since August 2022, most recent yesterday
  StatusOrProlonged: Not mentioned.
  CognitivePriority: Not mentioned.
  Risk_Factors: Possible risk factors are: family_history_epilepsy.
  SeizureType: Focal onset convulsive seizures progressing to generalized tonic-clonic.

=== MEDICATION HISTORY ===
Current drug observations (this visit):
  - phenytoin: currently taking (on_arrival)

No prior documented plans available (this is Visit 1).
```

**Model output:**

| Rank | Drug | Reasoning |
|---|---|---|
| 1 | **carbamazepine** | First-line, inexpensive sodium-channel blocker ideal for focal seizures with secondary generalization; aligns with Ugandan practice and offers a better safety profile than phenytoin in a toddler. |
| 2 | **levetiracetam** | Broad-spectrum, inexpensive relative to newer agents, rapid titration, and good tolerability; useful if carbamazepine is ineffective or not tolerated. |
| 3 | **phenytoin** | Currently being taken; retains utility as a fallback option, but carries higher risk of protein-binding variability and long-term adverse effects in young children. |

**Ground truth:** valproate (reconciled from V2 observation)
**Match:** No top-1 match; no top-3 match

---

#### Visit 2 Prediction

**What the model sees:**

```
=== CLINICAL FEATURES ===
  Age_Years: 2.5
  Sex_Female: No
  OnsetTimingYears: 0.8
  SeizureFreq: Reduced; one seizure episode since medication began
  StatusOrProlonged: One seizure lasting 20 minutes reported
  CognitivePriority: Not mentioned.
  Risk_Factors: Possible risk factors are: none mentioned.
  SeizureType: Seizure type not clearly described in this note.

=== MEDICATION HISTORY ===
Current drug observations (this visit):
  - valproate: currently taking (on_arrival)

Prior documented plans:
  - Visit 1 plan: prescribed valproate

Prior drug exposure (observed at earlier visits, not in any plan):
  - phenytoin: was taking at Visit 1
```

> **Note the phenytoin line.** This comes from source (c) — Visit 1's extraction saw phenytoin as `on_arrival`, but no plan ever prescribed it. Without source (c), the model at Visit 2 would have no knowledge of the patient's phenytoin history.

**Model output:**

| Rank | Drug | Reasoning |
|---|---|---|
| 1 | **valproate** | Already providing good control (only one seizure since initiation) and is the first-line broad-spectrum agent for unknown seizure types in this age group; continued use is preferred over switching. |
| 2 | **levetiracetam** | Broad-spectrum, inexpensive, rapid onset without need for slow titration; an effective alternative if VPA tolerance issues arise. |
| 3 | **phenobarbital** | Very low cost, effective across seizure types, and remains a reliable backup in resource-limited settings despite its sedating profile. |

**Ground truth:** valproate
**Match:** Top-1 match

---

#### Visit 3 Prediction

**What the model sees:**

```
=== CLINICAL FEATURES ===
  Age_Years: 3.0
  Sex_Female: No
  OnsetTimingYears: Not mentioned.
  SeizureFreq: Frequent seizures, increased
  StatusOrProlonged: Not mentioned.
  CognitivePriority: Not mentioned.
  Risk_Factors: Possible risk factors are: none mentioned.
  SeizureType: Not mentioned.

=== MEDICATION HISTORY ===
Current drug observations (this visit):
  - valproate: currently taking (on_arrival)

Prior documented plans:
  - Visit 1 plan: prescribed valproate
  - Visit 2 plan: prescribed valproate

Prior drug exposure (observed at earlier visits, not in any plan):
  - phenytoin: was taking at Visit 1
```

**Model output:**

| Rank | Drug | Reasoning |
|---|---|---|
| 1 | **valproate** | Already effective, broad-spectrum coverage for unknown seizure types, inexpensive and aligns with Ugandan first-line practice; continue as monotherapy with routine liver function monitoring. |
| 2 | **levetiracetam** | Broad-spectrum, rapid titration, minimal interactions, and well-tolerated in toddlers; ideal adjunct or alternative if valproate efficacy wanes or side-effects emerge. |
| 3 | **phenobarbital** | Cheap, widely available, effective for many seizure phenotypes; useful as a third-line add-on when valproate + levetiracetam do not achieve adequate control, accepting increased sedation. |

**Ground truth:** valproate + levetiracetam
**Match:** Top-1 partial match (valproate), Top-2 captures levetiracetam

---

## Summary: Information Flow for Mwania Sheldon

```
VISIT 1                         VISIT 2                         VISIT 3
────────                        ────────                        ────────

Raw CSV                         Raw CSV                         Raw CSV
  │                               │                               │
  ▼                               ▼                               ▼
STEP 1: Split                   STEP 1: Split                   STEP 1: Split
  INPUT: HPI (observational)      INPUT: "on sodium valproate"    INPUT: "on VPA at 18mg/kg/day,
         phenytoin on_arrival            seizure onset Aug 2022           seizures frequent"
  OUTPUT: (empty)                 OUTPUT: "VPA 150mg bd x3/12"    OUTPUT: "Increase VPA to 25mg/
                                                                          kg/day + LEV 250mg BD"
  │          │                    │          │                    │          │
  ▼          ▼                    ▼          ▼                    ▼          ▼
STEP 2    STEP 1b              STEP 2    STEP 1b              STEP 2    STEP 1b
Extract   Ground Truth         Extract   Ground Truth         Extract   Ground Truth
features  prescribed:          features  prescribed:          features  prescribed:
from      [valproate]          from      [valproate]          from      [VPA, LEV]
INPUT     (reconciled)         INPUT                          INPUT
  │          │                    │          │                    │          │
  └────┬─────┘                    └────┬─────┘                    └────┬─────┘
       ▼                               ▼                               ▼
    STEP 3: Predict V1              STEP 3: Predict V2              STEP 3: Predict V3
    Features + no prior plans       Features + V1 plan              Features + V1,V2 plans
    + phenytoin on_arrival          + VPA on_arrival                + VPA on_arrival
                                    + phenytoin from V1 obs         + phenytoin from V1 obs
       │                               │                               │
       ▼                               ▼                               ▼
    Rank 1: CBZ                     Rank 1: VPA  ✓                  Rank 1: VPA  ✓
    Rank 2: LEV                     Rank 2: LEV                     Rank 2: LEV  ✓
    Rank 3: PHT                     Rank 3: PB                      Rank 3: PB
```

---

## Files

| File | Purpose |
|---|---|
| `split_input_output.py` | Step 1: LLM-based text splitting for all visits |
| `build_ground_truth.py` | Step 1b: Ground truth extraction + reconciliation |
| `extract_from_input.py` | Step 2: Feature extraction from INPUT only |
| `extract_input_prompt.txt` | Step 2: Extraction prompt (18 features, simplified drug schema) |
| `predict_drugs_clean.py` | Step 3: Drug prediction with prior plans as medication history |
| `predict_prompt_clean.txt` | Step 3: Prediction prompt (hierarchical clinical reasoning) |
| `eval_clean.ipynb` | Evaluation notebook |

### Intermediate Outputs

| File | Contents |
|---|---|
| `split_results.json` | Per-patient, per-visit INPUT/OUTPUT split |
| `ground_truth.json` | Per-patient, per-visit prescribed/stopped drugs + reconciliation |
| `extracted_features.json` | Per-patient, per-visit 18-feature extraction |
| `drug/clean/*_v{1,2,3}_clean_top3.json` | Full prediction results with reasoning |
| `drug/clean/*_v{1,2,3}_clean_top3.csv` | Summary CSV (patient_id, rank_1/2/3, reasons) |

---

## Running the Pipeline

```bash
cd /home/shreyas/NLP/global-pred
source .venv/bin/activate

# Step 1: Split input/output (15 LLM calls for 5 patients)
python scripts/split_input_output.py

# Step 1b: Build ground truth
python scripts/build_ground_truth.py

# Step 2: Extract features from INPUT only
python scripts/extract_from_input.py

# Step 3: Predict drugs (edit visit_num in script, or run loop)
python scripts/predict_drugs_clean.py
```

To run on all patients, set `limit_patients = None` in each script.

---

## Trial Run Results (5 patients, 15 predictions)

| Metric | Value |
|---|---|
| Top-1 Accuracy | 8/15 (53%) |
| Top-3 Accuracy | 11/15 (73%) |
| Reconciliations needed | 3 (Sheldon VPA→V1, Ryan CBZ→V1, Asio VPA→V2) |
