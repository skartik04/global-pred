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

### Split Prompt (full)

```
You are a clinical note splitter. Given text from an epilepsy clinic note, separate it into:

1. INPUT (observations): Clinical history, examination findings, what the patient is currently
   taking (observed state), seizure descriptions, developmental information, past medical history,
   and any other factual observations.

2. OUTPUT (treatment plan): Prescriptions, dose changes, new medications ordered, investigations
   ordered, referrals, follow-up plans. Look for markers like "Plan:", numbered prescription lists
   (e.g., "1. Tabs Carbamazepine 200mg bd"), medication orders with dosages and durations
   (e.g., "x 1/12", "x 3/12"), "Continue [drug] [dose]", "Start [drug]", "Increase [drug]",
   "Switch to [drug]".

Key distinctions:
- "currently on sodium valproate" or "on carbamazepine" -> INPUT (observation of current state)
- "Tabs Carbamazepine 200 mg bd x 1/12" -> OUTPUT (prescription order)
- "episodes reduced with carbamazepine" -> INPUT (treatment response observation)
- "Start phenytoin 5 mls bd" -> OUTPUT (new prescription)
- "Continue Na Valproate 100 mg morning x 1/52" -> OUTPUT (prescription plan)
- "Do EEG" or "Review with results" -> OUTPUT (investigation/follow-up ordered)
- "was started on phenytoin yesterday" -> INPUT (observation of recent event)
- "Previously on phenobarbital" -> INPUT (medication history)
- "Sodium Valproate, Phenobarbital, Tegretol" (drug list without dosing instructions)
  -> INPUT (observation of what patient is on)
- "1. Sodium Valproate 150 mg BD x 3/12, 2. Folic Acid 5 mg OD x 3/12"
  -> OUTPUT (prescription list with doses and durations)

Return a JSON object with exactly two keys:
{
  "input_text": "...(all observational content, preserving original wording)...",
  "output_text": "...(all plan/prescription content, preserving original wording)..."
}

Preserve the original wording as much as possible. If no plan/prescription is found,
set output_text to "". If no observations are found, set input_text to "".
Do not output anything except the JSON object.
```

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

### Extraction Prompt (full)

<details>
<summary>Click to expand full extraction prompt (extract_input_prompt.txt)</summary>

```
You are an expert clinician and medical NLP assistant reviewing a single epilepsy clinic visit
note for one patient.

IMPORTANT: The text you receive contains ONLY observational clinical information. No treatment
plans, prescriptions, or medication orders are included. Your job is to extract what was observed,
not what was decided.

The note may be noisy, incomplete, or formatted inconsistently. This note corresponds to one visit
only. Other visits are not provided, so some clinical information may be missing.

The note will come from a CSV extraction; formatting may be fragmented, line-broken, or out of
order. Ignore formatting artifacts and extract meaning only from actual clinical text.

Your task is to extract a fixed set of features from this one note and return them as a single
JSON object.

GLOBAL EXTRACTION PRINCIPLE (IMPORTANT):
- Do not rely only on exact keywords. Use clinical reading comprehension to map the meaning of the
  note into the requested features.
- You may use direct clinical implication within THIS note (for example, a clearly stated
  neurodevelopmental disorder implies developmental impairment; a clearly stated convulsive seizure
  description implies seizure semiology), but you must not invent facts that are not supported by
  the text.
- Do not use any information from other visits or background knowledge.

For each feature:
- Look only at what is written in this note.
- Do not assume or infer information from other visits or background knowledge.
- If the information is present, summarize it cleanly in your own words.
- If the information is not present, set the value to exactly "Not mentioned." (including the period).
- You never carry information across visits. You never guess. You never assume continuation of
  drugs or diagnoses. You only report what is in this one note.

Confidence:
- Use "high" if the information is clearly present or clearly not mentioned in this note.
- Use "med" or "low" only when the wording is ambiguous or hard to interpret.

Supporting text:
- "supporting_text" must contain exact quoted text snippets copied from the note (verbatim).
- Keep supporting_text as short as possible while still proving the extraction.
- If multiple snippets are needed, separate them with " | ".
- If not mentioned, supporting_text must be "" (empty string).

OUTPUT FORMAT

You must output a single JSON object with the following keys, in this exact spelling and order:

{
  "Age_Years": { ... },
  "Sex_Female": { ... },
  "OnsetTimingYears": { ... },
  "SeizureFreq": { ... },
  "StatusOrProlonged": { ... },
  "CognitivePriority": { ... },
  "Risk_Factors": { ... },
  "SeizureType": { ... },
  "drug_clobazam": { ... },
  "drug_clonazepam": { ... },
  "drug_valproate": { ... },
  "drug_ethosuximide": { ... },
  "drug_levetiracetam": { ... },
  "drug_lamotrigine": { ... },
  "drug_phenobarbital": { ... },
  "drug_phenytoin": { ... },
  "drug_topiramate": { ... },
  "drug_carbamazepine": { ... }
}

Each feature value must itself be a JSON object with exactly four fields:
- "value": a short, clean statement or label for that feature (or "Not mentioned.").
- "supporting_text": exact quoted text from the note (or "" if not mentioned).
- "reasoning": a short explanation of how you decided the value, based only on this note.
- "confidence": "high", "med", or "low".

FEATURE DEFINITIONS AND RULES

1. Age_Years — Age of the patient at the time of this visit, in years.
2. Sex_Female — Whether the patient is female ("Yes"/"No"/"Not mentioned.").
3. OnsetTimingYears — Duration of epilepsy at the time of this visit, in years since first seizure.
4. SeizureFreq — Short phrase describing how often seizures are occurring.
5. StatusOrProlonged — Any mention of status epilepticus or prolonged seizures.
6. CognitivePriority — Developmental delay, psychiatric problems, or behavioral problems.
7. Risk_Factors — List from: febrile_seizures, childhood_meningitis, trauma_to_head,
   perinatal_complication, family_history_epilepsy, cerebral_malaria, cerebrovascular_disease.
8. SeizureType — Dominant seizure type or semiology described in this note.

9-18. Drug features (drug_clobazam through drug_carbamazepine)

For each drug, "value" must be either "Not mentioned." or:

"Status: <on_arrival|past_tried|not_mentioned>"

Definitions:
- on_arrival: patient is currently taking this drug at this visit (observed to be on it)
- past_tried: patient previously used this drug but is no longer on it
- not_mentioned: drug is not mentioned in the observational text

Drug name matching (synonyms):
 9. drug_clobazam: clobazam, Frisium
10. drug_clonazepam: clonazepam, Rivotril, Klonopin
11. drug_valproate: valproate, sodium valproate, valproic acid, Depakote, Epilim
12. drug_ethosuximide: ethosuximide, Zarontin
13. drug_levetiracetam: levetiracetam, Keppra
14. drug_lamotrigine: lamotrigine, Lamictal
15. drug_phenobarbital: phenobarbital, phenobarbitone, Gardenal
16. drug_phenytoin: phenytoin, Dilantin, Epanutin
17. drug_topiramate: topiramate, Topamax
18. drug_carbamazepine: carbamazepine, Tegretol, Carbatrol, CBZ

Produce only the JSON object described above, with all feature keys present, and no additional text.
```

</details>

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

### Prediction Prompt (full)

<details>
<summary>Click to expand full prediction prompt (predict_prompt_clean.txt)</summary>

```
ROLE
You are an expert epileptologist. Your job is to recommend the top 3 most appropriate anti-seizure
medications (ASMs) using a structured, clinically grounded reasoning process. This is decision
support only.

SETTING AND CONSTRAINTS
- Setting: Uganda, resource-limited, cost-sensitive prescribing.
- You MUST ONLY consider and recommend from these 10 ASMs. Never recommend any other drug.
  1) carbamazepine (CBZ)    6) levetiracetam (LEV)
  2) clobazam (CLB)         7) phenobarbital (PB)
  3) clonazepam (CZP)       8) phenytoin (PHT)
  4) ethosuximide (ESM)     9) topiramate (TPM)
  5) lamotrigine (LTG)     10) valproate (VPA)

INPUT
- Demographics (age, sex)
- Seizure characteristics (type, frequency, onset timing)
- History of status epilepticus or prolonged seizures
- Cognitive/developmental status
- Risk factors / etiology
- Medication history from two sources:
  (a) Current observations: what the patient is currently taking or was previously taking
      (observed at this visit). Status key: on_arrival = currently taking; past_tried = previously
      used.
  (b) Prior documented plans: what was prescribed at previous visits (carried forward as historical
      context). This tells you what treatments have been tried and their implicit outcomes.
  Continuation of a working regimen is strongly preferred over switching.

HIERARCHICAL REASONING PIPELINE (FOLLOW IN ORDER)

Stage 0: Input sanity and uncertainty
- Determine seizure type and whether classification is confident vs mixed/uncertain.
- If seizure type is uncertain or mixed, prefer broad-spectrum options compatible with multiple
  seizure types and be cautious about drugs known to worsen generalized seizure types.

Stage 1: Seizure-type compatibility shortlist (select AND reject)
- Actively select the most relevant candidate drugs for the seizure type.
- Explicitly reject or strongly deprioritize drugs that are mismatched or known to worsen that
  seizure type.

Stage 2: Hard safety and special-population modifiers (strong penalties)
- Pregnancy or female of childbearing potential: strongly penalize VPA; minimize polytherapy.
- Hepatic impairment or hepatic failure risk: strongly penalize VPA.
- Low albumin states: be cautious with highly protein-bound drugs (PHT, VPA).

Stage 3: Practicality, Cost, and The LMIC Baseline
- In this resource-limited Uganda setting, older/cheaper generation drugs (VPA, CBZ, PB) are the
  foundational baseline of care.
- You MUST anchor your initial choices to VPA (for generalized/mixed) or CBZ (for focal) as the
  default first-line agents.
- You should ONLY pivot to newer, more expensive agents (LEV, LTG, TPM) if there is a strict,
  documented contraindication to the older drugs.
- Note: PB is highly accessible and used often here despite its sedation. Do not penalize it to
  0.0 unless strictly contraindicated.

Stage 4: Side-effect alignment and comorbidity tie-breakers (soft optimization)
- Use cognition vulnerability, mood/behavior risk, weight concerns, and other risks to choose
  among remaining reasonable options.
- LEV: behavioral disturbance risk.
- TPM: cognitive/language issues; weight loss; metabolic acidosis; nephrolithiasis.
- VPA: weight gain; hyperammonemia risk; pancreatitis risk.
- PB: sedating.

Stage 5: Monotherapy first, polytherapy logic if needed
- Prefer one strong option if reasonable.
- If adding an adjunct, prefer a different mechanism where possible.
- Avoid combining multiple sodium-channel dominant drugs (CBZ, PHT, LTG).

SEIZURE-TYPE COMPATIBILITY GUIDANCE FOR THESE 10 DRUGS

Focal seizures: Common first options: CBZ, LTG. Alternatives: LEV, VPA.
GTCS: Strong option: VPA. If VPA unsuitable: LTG, LEV, TPM.
Absence seizures: Strong options: ESM or VPA. Alternative: LTG.
  Strongly avoid CBZ and PHT (seizure worsening risk).
Myoclonic seizures: Strong option: VPA. Alternatives: LEV, TPM.
  Avoid CBZ and PHT (worsen myoclonic seizures). Caution: LTG can aggravate myoclonus.
Tonic or atonic seizures: Tonic: PHT and LTG effective. Atonic: VPA drug of choice.

SCORING INSTRUCTIONS
- Select exactly 3 drugs from the 10 ASMs, ranked from most to least appropriate.
- Rank 1 = single best choice. Rank 2 = next-best alternative. Rank 3 = reasonable third option.
- Strongly favor continuing a current medication that is appropriate for the seizure type.

EXAMPLES (for calibration)

Example 1 — Monotherapy (generalized seizures, child)
Top-3: valproate, phenobarbital, carbamazepine

Example 2 — Monotherapy (focal seizures, adult male)
Top-3: carbamazepine, valproate, phenytoin

Example 3 — Dual therapy (refractory generalized, child)
Top-3: valproate, clobazam, levetiracetam

Example 4 — Absence seizures (child)
Top-3: ethosuximide, valproate, lamotrigine

OUTPUT FORMAT (STRICT — FOLLOW EXACTLY)

Produce your response in exactly two sections:

---SECTION 1: CLINICAL REASONING---
Reason through the clinical picture using the pipeline above as a guide.

---SECTION 2: TOP-3 DRUG RECOMMENDATIONS---
Rank 1: <drug_name> | reason: <1-2 sentences>
Rank 2: <drug_name> | reason: <1-2 sentences>
Rank 3: <drug_name> | reason: <1-2 sentences>

Drug names must be from: carbamazepine, clobazam, clonazepam, ethosuximide, lamotrigine,
levetiracetam, phenobarbital, phenytoin, topiramate, valproate
```

</details>

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

**Model thinking trace (internal reasoning before output):**

<details>
<summary>Click to expand thinking trace</summary>

> We need to produce the answer: clinical reasoning and top 3 drug recommendations. Patient: 1.8-year-old male (young toddler). Seizure type: focal onset convulsive seizures progressing to generalized tonic-clonic (i.e., focal seizures with secondary generalization). Currently on phenytoin (on arrival). No prior other plans.
>
> We must follow hierarchy.
>
> Stage 0: Input sanity. Seizure type appears clear: focal onset convulsive with secondary generalization, i.e., focal seizure evolving to GTCS. Good confidence.
>
> Stage 1: Seizure-type compatibility shortlist. For focal seizures: first options CBZ, LTG; alternatives: LEV, VPA. Adjuncts: CLB. PHT can be considered but weigh adverse effects. Also PB.
>
> But current med: phenytoin is already being taken. PHT is a sodium channel blocker. For focal seizures, PHT is acceptable but not first line in many guidelines (especially for a toddler?). PHT is enzyme inducer; but may be used. However, phenytoin's use in infants/toddlers? Usually not first-line due to side effects, bone issues, and risk of neurodevelopmental impact. In resource-limited settings, could be used.
>
> We must recommend top 3 ASMs, rank order, considering continuity: continuation of a working regimen is strongly preferred. Must evaluate if phenytoin is appropriate: patient is 1.8 years old; phenytoin is okay for focal seizures but there are concerns about neurodevelopment, sedation etc. However, it's current. Should we keep phenytoin as Rank 1? The guidelines: "Continuation of a working regimen is strongly preferred over switching." Since we have no info on efficacy: we have had 4 seizures since August 2022 (which is 4 months). Not sure about frequency baseline, but still having seizures while on phenytoin? The note says "currently taking phenytoin observed". It does not state seizure control; 4 seizures since Aug (maybe that's the total while on phenytoin). If still having seizures, maybe not fully controlled. But we don't have efficacy data. Could consider switching to better first-line CBZ, LTG, but we also consider cost and LMIC baseline: CBZ is cheaper and standard for focal seizures. So recommend switching from phenytoin to carbamazepine perhaps.
>
> Thus ranking: 1) carbamazepine (first-line focal, cheap). 2) levetiracetam (broad-spectrum, easy titration, safe). 3) phenytoin (if continue or as fallback).
>
> Thus final top 3: carbamazepine, levetiracetam, phenytoin.

</details>

**Model clinical reasoning (output Section 1):**

> **Stage 0 – Input sanity:** The patient is a 1.8-year-old male with a clear seizure phenotype: focal onset convulsive seizures that secondarily generalize to a tonic-clonic seizure. The classification is confident; no mixed or ambiguous seizure types are reported.
>
> **Stage 1 – Seizure-type compatibility shortlist:** For focal seizures the guideline-preferred agents are carbamazepine (CBZ) and lamotrigine (LTG). Alternatives that are also effective include levetiracetam (LEV) and valproate (VPA). Phenytoin (PHT) is a historical option for focal seizures but is generally lower-ranked because of its side-effect profile, especially in very young children.
>
> **Stage 2 – Safety / special-population modifiers:** Age ~2y: No pregnancy concerns. Protein binding: PHT is highly protein-bound; in a toddler with rapidly changing albumin, free levels can be unpredictable, adding a safety penalty. Cognitive impact: PHT and high-dose CBZ can affect development, but CBZ's risk is lower than PHT's in this age group.
>
> **Stage 3 – Practicality, cost, and LMIC baseline:** Uganda's resource-limited setting makes older, inexpensive drugs the default. CBZ is cheap, widely available, and the standard first-line for focal seizures. PHT is already being taken, but given its less favorable safety profile in toddlers and the availability of a cheaper, more seizure-specific drug (CBZ), it should be considered only as a backup.
>
> **Stage 4 – Side-effect alignment:** CBZ: Possible hyponatremia or rash, but monitorable. LEV: Generally well-tolerated; behavioral upset is rare in toddlers. PHT: Long-term bone health concerns, potential for gingival hyperplasia, and higher interaction burden; thus a lower rank.
>
> **Stage 5 – Monotherapy first:** Monotherapy is the goal. Given a clear focal-onset seizure type, the cheap, guideline-aligned first-line agent (CBZ) is the top recommendation.

**Model output (Section 2):**

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

**Model thinking trace (internal reasoning before output):**

<details>
<summary>Click to expand thinking trace</summary>

> We need to produce reasoning and top 3 recommendations.
>
> Patient: 1.8-year-old male (sex female: No => male). So infant/toddler. Seizure type unknown, not clearly described. Currently on valproate (started at visit 1 and continued). Prior exposure phenytoin at visit 1 (maybe trial). Seizure frequency reduced, only one seizure since medication started. Had a prolonged seizure 20 minutes (possible status? Not full status epilepticus, but prolonged). Unknown cognitive status. No risk factors. Need to decide top 3 ASMs.
>
> Given seizure type uncertain, Stage 0: uncertain, broad-spectrum recommended. Broad-spectrum among list: levetiracetam, lamotrigine, topiramate, valproate, clobazam, clonazepam. Valproate already used and appears effective (only one seizure). Continue valproate unless contraindicated. In infants, VPA is commonly used for generalized seizures, but concerns: hepatic toxicity in children under 2? Valproate is used but careful monitoring. In resource-limited setting, VPA is cheap and accessible, also first-line for generalized.
>
> We should produce top 3: 1) Valproate (continue). 2) Levetiracetam (broad-spectrum, well tolerated, no need for titration, accessible). 3) Phenobarbital (older, cheap, works for many seizure types, sedation but acceptable). Could also consider lamotrigine but needs slow titration; not ideal for urgent control.
>
> Thus rank: Valproate, Levetiracetam, Phenobarbital.

</details>

**Model clinical reasoning (output Section 1):**

> **Stage 0 – Seizure-type certainty:** The chart does not specify semiology; seizure type is therefore uncertain. In such cases a broad-spectrum ASM that is effective for both focal and generalized events is preferred, while avoiding agents known to aggravate particular syndromes (e.g., carbamazepine or phenytoin for myoclonic/absence patterns).
>
> **Stage 1 – Compatibility shortlist:** Broad-spectrum candidates: valproate (VPA), levetiracetam (LEV), lamotrigine (LTG), topiramate (TPM), clobazam (CLB) and clonazepam (CZP). Narrow-spectrum/focal-leaning drugs that could worsen an unknown seizure type (especially myoclonic or absence) are carbamazepine (CBZ) and phenytoin (PHT); both are therefore de-prioritized. Phenobarbital (PB) retains utility across seizure types and is inexpensive, so it remains on the shortlist.
>
> **Stage 2 – Safety / special-population modifiers:** 1.8-year-old male, so pregnancy-related penalties do not apply. No reported hepatic disease; VPA is acceptable but warrants routine liver monitoring in young children. Prior exposure to phenytoin (ineffective/discontinued) reinforces avoidance of that drug.
>
> **Stage 3 – Practicality, cost, LMIC baseline:** Uganda's essential-medicine formulary readily supplies VPA, phenobarbital, and phenytoin. VPA is the default for generalized-type seizures and is already providing good seizure control (only one breakthrough seizure). LEV, while newer, is increasingly available at modest cost and does not require slow titration.
>
> **Stage 4 – Side-effect alignment & comorbidities:** No cognitive or behavioral concerns reported; behavioral risk of LEV is acceptable. Sedation from phenobarbital is tolerable in a toddler. VPA's weight-gain and metabolic side-effects are less concerning in a young infant with good growth.
>
> **Stage 5 – Monotherapy vs polytherapy:** Current monotherapy with VPA appears largely effective (marked reduction in seizure frequency). The goal is to maintain seizure control while preserving a simple regimen.

**Model output (Section 2):**

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

**Model thinking trace (internal reasoning before output):**

<details>
<summary>Click to expand thinking trace</summary>

> We need to recommend top 3 ASMs for this patient.
>
> We have limited info: 1.8-year-old male (sex female: No; male). Very young toddler. Seizure type unknown, unspecified, frequent, increased. Currently on valproate (VPA) ongoing from prior visits. Also had phenytoin previously (maybe tried but not now). No other data.
>
> We must follow pipeline.
>
> Stage 0: Uncertainty. Seizure type not specified -> uncertain/mixed. Therefore need broad-spectrum options that are safe for unknown type. Avoid drugs that worsen generalized seizures like CBZ, PHT for possible generalized myoclonic/absence etc. Also need to consider age: under 2 years. Certain drugs have age restrictions: Valproate is generally used in children for generalized seizures. Lamotrigine is okay but requires slow titration. Levetiracetam is broad-spectrum and used in infants. Phenobarbital is also used in infants. Ethosuximide is for absence only, not appropriate without known absence. Topiramate may be used but caution with growth, metabolic effects. Clobazam clonazepam are benzodiazepines used as adjunct, but sedation.
>
> Current med is VPA. It's continued. Should we keep?
>
> Contraindications: No pregnancy. No hepatic disease known. So VPA acceptable. For toddlers, VPA is common.
>
> But VPA has known risk of hepatotoxicity in children under 2, especially those with metabolic disorders, but still used as first-line for generalized seizures. However, risk is higher in children <2 years; alternative like levetiracetam is often preferred due to safety. But guidelines vary. In resource-limited Uganda, VPA is cheap and first-line for generalized seizures. However, due to age <2, VPA caution. But we have no info of metabolic disease.
>
> Given continuation already on VPA, we should likely keep as Rank 1, unless strong reason to switch. Since seizure type unknown, VPA covers many generalized types. Also may treat focal.
>
> Thus ranking: 1) Valproate (continue). 2) Levetiracetam (broad-spectrum, safe). 3) Phenobarbital (cheap, effective for many types, but sedation).

</details>

**Model clinical reasoning (output Section 1):**

> **Stage 0 – Input sanity and uncertainty:** The patient is a 1.8-year-old male with frequent seizures of unspecified semiology; seizure type is thus uncertain/mixed. Because the phenotype is unknown we must select a broad-spectrum ASM that will not aggravate any major generalized seizure subtype (e.g., absence, myoclonic).
>
> **Stage 1 – Seizure-type compatibility shortlist:** Broad-spectrum agents: levetiracetam, lamotrigine, topiramate, valproate, clobazam, clonazepam. Narrow-spectrum/focal-leaning drugs that can worsen generalized patterns (carbamazepine, phenytoin, phenobarbital to a lesser extent) are deprioritized unless needed for focal control.
>
> **Stage 2 – Hard safety / special-population modifiers:** No pregnancy or child-bearing concerns. No reported hepatic failure or severe malnutrition. Age < 2 years raises a modest concern for valproate-related hepatotoxicity, but this risk is acceptable when the drug is already providing seizure control. Phenytoin is highly protein-bound; low albumin in a toddler could alter free levels, so it is not preferred.
>
> **Stage 3 – Practicality, cost, LMIC baseline:** In Uganda, valproate is inexpensive and forms the cornerstone for generalized or mixed seizures. Continuation of an already-established regimen is strongly favored. Newer agents are only introduced if there is a clear contraindication or inadequate response.
>
> **Stage 4 – Side-effect alignment:** Valproate: Good efficacy across generalized patterns; hepatic monitoring is advisable. Levetiracetam: Very favorable side-effect profile, minimal drug-drug interactions, rapid titration—ideal backup. Phenobarbital: Widely available but sedating; still acceptable as a third-line cheap option.
>
> **Stage 5 – Monotherapy first:** Monotherapy with valproate remains appropriate. If seizure control remains suboptimal, adding levetiracetam (different mechanism) would be the preferred adjunct.

**Model output (Section 2):**

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
