"""
Step 1b: Build ground truth from OUTPUT side of split results.

Extracts which drugs the doctor actually prescribed at each visit, then
reconciles undocumented prescriptions (if a patient arrives at Visit N on a drug
that no prior plan documented, attribute it to the earliest undocumented visit).

Input:  split_results.json (from Step 1)
Output: ground_truth.json
"""

import os
import json
import asyncio
import random
import dotenv
from tqdm import tqdm
from together import AsyncTogether

_HERE = os.path.dirname(os.path.abspath(__file__))
dotenv.load_dotenv(os.path.join(_HERE, '.env'))

# ============ CONFIGURATION ============
model = 'openai/gpt-oss-120b'
split_results_path = os.path.join(_HERE, 'split_results.json')
max_concurrency = 8
limit_patients = 5  # Set to int for testing, None for all
output_file = os.path.join(_HERE, 'ground_truth.json')
# ========================================

DRUG_LIST = [
    "carbamazepine", "clobazam", "clonazepam", "ethosuximide",
    "lamotrigine", "levetiracetam", "phenobarbital", "phenytoin",
    "topiramate", "valproate",
]

PLAN_EXTRACT_PROMPT = """You are a clinical pharmacist reviewing a treatment plan from an epilepsy clinic visit.

Given the plan/prescription text below, identify which of these 10 anti-seizure medications (ASMs) are mentioned:
carbamazepine, clobazam, clonazepam, ethosuximide, lamotrigine, levetiracetam, phenobarbital, phenytoin, topiramate, valproate

Also consider common synonyms:
- carbamazepine: Tegretol, Carbatrol, CBZ
- clobazam: Frisium
- clonazepam: Rivotril, Klonopin
- ethosuximide: Zarontin, Ethoxusamide, Ethoxusimide
- lamotrigine: Lamictal
- levetiracetam: Keppra
- phenobarbital: phenobarbitone, Gardenal
- phenytoin: Dilantin, Epanutin
- topiramate: Topamax
- valproate: sodium valproate, valproic acid, Depakote, Epilim, Na Valproate

For each drug, classify as:
- "prescribed": drug appears in the plan with dosage/instructions (actively prescribed, continued, dose changed)
- "stopped": drug is explicitly discontinued or stopped in this plan
- "not_mentioned": drug not mentioned in the plan text

Return a JSON object with exactly these 10 keys:
{
  "carbamazepine": "prescribed|stopped|not_mentioned",
  "clobazam": "prescribed|stopped|not_mentioned",
  "clonazepam": "prescribed|stopped|not_mentioned",
  "ethosuximide": "prescribed|stopped|not_mentioned",
  "lamotrigine": "prescribed|stopped|not_mentioned",
  "levetiracetam": "prescribed|stopped|not_mentioned",
  "phenobarbital": "prescribed|stopped|not_mentioned",
  "phenytoin": "prescribed|stopped|not_mentioned",
  "topiramate": "prescribed|stopped|not_mentioned",
  "valproate": "prescribed|stopped|not_mentioned"
}

Do not output anything except the JSON object."""

OBSERVATION_EXTRACT_PROMPT = """You are a clinical pharmacist reviewing observational notes from an epilepsy clinic visit.

Given the observational text below, identify which of these 10 anti-seizure medications (ASMs) the patient is described as currently taking or having previously taken:
carbamazepine, clobazam, clonazepam, ethosuximide, lamotrigine, levetiracetam, phenobarbital, phenytoin, topiramate, valproate

Also consider common synonyms:
- carbamazepine: Tegretol, Carbatrol, CBZ
- clobazam: Frisium
- clonazepam: Rivotril, Klonopin
- ethosuximide: Zarontin, Ethoxusamide, Ethoxusimide
- lamotrigine: Lamictal
- levetiracetam: Keppra
- phenobarbital: phenobarbitone, Gardenal
- phenytoin: Dilantin, Epanutin
- topiramate: Topamax
- valproate: sodium valproate, valproic acid, Depakote, Epilim, Na Valproate

For each drug, classify as:
- "on_arrival": patient is currently taking this drug (observed to be on it)
- "past_tried": patient was previously on this drug but no longer
- "not_mentioned": drug not mentioned in the observational text

Return a JSON object with exactly these 10 keys:
{
  "carbamazepine": "on_arrival|past_tried|not_mentioned",
  "clobazam": "on_arrival|past_tried|not_mentioned",
  "clonazepam": "on_arrival|past_tried|not_mentioned",
  "ethosuximide": "on_arrival|past_tried|not_mentioned",
  "lamotrigine": "on_arrival|past_tried|not_mentioned",
  "levetiracetam": "on_arrival|past_tried|not_mentioned",
  "phenobarbital": "on_arrival|past_tried|not_mentioned",
  "phenytoin": "on_arrival|past_tried|not_mentioned",
  "topiramate": "on_arrival|past_tried|not_mentioned",
  "valproate": "on_arrival|past_tried|not_mentioned"
}

Do not output anything except the JSON object."""


# --- Helpers ---

def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
        t = t.strip()
    return t


def robust_json_load(text: str) -> dict:
    raw = strip_code_fences(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def combine_output_text(visit_data: dict) -> str:
    """Combine output_text and output_columns into a single text for parsing."""
    parts = []
    output_text = visit_data.get("output_text", "")
    if output_text:
        parts.append(output_text)
    for col, val in visit_data.get("output_columns", {}).items():
        if val:
            parts.append(f"{col}: {val}")
    return "\n".join(parts)


def combine_input_text(visit_data: dict) -> str:
    """Combine input_text and input_columns into a single text for observation parsing."""
    parts = []
    input_text = visit_data.get("input_text", "")
    if input_text:
        parts.append(input_text)
    for col, val in visit_data.get("input_columns", {}).items():
        if val and "Risk Factors" not in col:  # Skip checkbox columns
            parts.append(f"{col}: {val}")
    return "\n".join(parts)


# --- Async LLM ---

async def call_llm_with_retries(
    client: AsyncTogether,
    system_prompt: str,
    user_content: str,
    model: str = "openai/gpt-oss-120b",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    semaphore: asyncio.Semaphore | None = None,
    max_retries: int = 6,
):
    async def _one_call():
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    for attempt in range(max_retries):
        try:
            if semaphore is None:
                return await _one_call()
            async with semaphore:
                return await _one_call()
        except Exception as e:
            msg = str(e).lower()
            retryable = any(s in msg for s in [
                "rate", "429", "timeout", "temporar", "overload", "503", "500", "connection"
            ])
            if not retryable or attempt == max_retries - 1:
                print(f"[ERROR] API call failed: {e}")
                return ""
            base = 0.8 * (2 ** attempt)
            await asyncio.sleep(base + random.random() * 0.3)
    return ""


# --- Processing ---

async def extract_plan_drugs(client, semaphore, output_text: str) -> dict:
    """Extract prescribed/stopped drugs from plan text."""
    if not output_text.strip():
        return {drug: "not_mentioned" for drug in DRUG_LIST}

    response = await call_llm_with_retries(
        client, PLAN_EXTRACT_PROMPT, output_text,
        model=model, semaphore=semaphore,
    )
    parsed = robust_json_load(response) if response else {}

    result = {}
    for drug in DRUG_LIST:
        val = parsed.get(drug, "not_mentioned")
        if val not in ("prescribed", "stopped", "not_mentioned"):
            val = "not_mentioned"
        result[drug] = val
    return result


async def extract_observed_drugs(client, semaphore, input_text: str) -> dict:
    """Extract observed drug status from input text."""
    if not input_text.strip():
        return {drug: "not_mentioned" for drug in DRUG_LIST}

    response = await call_llm_with_retries(
        client, OBSERVATION_EXTRACT_PROMPT, input_text,
        model=model, semaphore=semaphore,
    )
    parsed = robust_json_load(response) if response else {}

    result = {}
    for drug in DRUG_LIST:
        val = parsed.get(drug, "not_mentioned")
        if val not in ("on_arrival", "past_tried", "not_mentioned"):
            val = "not_mentioned"
        result[drug] = val
    return result


async def process_patient(client, semaphore, patient_id, visits):
    """Process one patient: extract plan drugs + observed drugs for all visits."""
    result = {}

    for visit_name in ["Visit_1", "Visit_2", "Visit_3"]:
        visit_data = visits.get(visit_name, {})
        output_text = combine_output_text(visit_data)
        input_text = combine_input_text(visit_data)

        plan_drugs = await extract_plan_drugs(client, semaphore, output_text)
        observed_drugs = await extract_observed_drugs(client, semaphore, input_text)

        prescribed = [d for d in DRUG_LIST if plan_drugs[d] == "prescribed"]
        stopped = [d for d in DRUG_LIST if plan_drugs[d] == "stopped"]

        result[visit_name] = {
            "prescribed_drugs": prescribed,
            "stopped_drugs": stopped,
            "observed_on_arrival": [d for d in DRUG_LIST if observed_drugs[d] == "on_arrival"],
            "observed_past_tried": [d for d in DRUG_LIST if observed_drugs[d] == "past_tried"],
            "source": "documented_plan",
            "plan_drugs_raw": plan_drugs,
            "observed_drugs_raw": observed_drugs,
        }

    return patient_id, result


def reconcile_undocumented(ground_truth: dict):
    """Reconcile undocumented prescriptions across visits.

    If a patient arrives at Visit N on drug X, but no prior visit's plan
    prescribed drug X, attribute it to the earliest visit without a documented
    plan for that drug (documentation gap assumption).
    """
    reconciliation_log = []

    for pid, visits in ground_truth.items():
        for visit_num in [2, 3]:
            visit_name = f"Visit_{visit_num}"
            visit_data = visits.get(visit_name, {})
            on_arrival = set(visit_data.get("observed_on_arrival", []))

            if not on_arrival:
                continue

            # Check which drugs are accounted for in prior plans
            accounted = set()
            for prev_num in range(1, visit_num):
                prev_name = f"Visit_{prev_num}"
                prev_data = visits.get(prev_name, {})
                accounted.update(prev_data.get("prescribed_drugs", []))

            # Drugs on arrival but not in any prior plan
            undocumented = on_arrival - accounted

            for drug in undocumented:
                # Attribute to the most recent prior visit
                target_visit = f"Visit_{visit_num - 1}"
                target_data = visits.get(target_visit, {})

                if drug not in target_data.get("prescribed_drugs", []):
                    target_data.setdefault("prescribed_drugs", []).append(drug)
                    if target_data.get("source") == "documented_plan":
                        target_data["source"] = "reconciled_from_later_observation"
                    reconciliation_log.append(
                        f"{pid}: {drug} observed at {visit_name}, "
                        f"attributed to {target_visit} plan (undocumented)"
                    )

    return reconciliation_log


async def main_async():
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        print("Error: TOGETHER_API_KEY not found")
        return

    print(f"\n{'='*60}")
    print(f"STEP 1b: BUILD GROUND TRUTH")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Input: {split_results_path}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    with open(split_results_path, 'r', encoding='utf-8') as f:
        split_results = json.load(f)

    patient_ids = list(split_results.keys())
    if limit_patients:
        patient_ids = patient_ids[:limit_patients]
        print(f"Limited to {limit_patients} patients")

    print(f"Total patients: {len(patient_ids)}")

    client = AsyncTogether(api_key=together_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    ground_truth = {}
    pbar = tqdm(total=len(patient_ids), desc="Building ground truth")

    tasks = [
        process_patient(client, semaphore, pid, split_results[pid])
        for pid in patient_ids
    ]

    for coro in asyncio.as_completed(tasks):
        pid, result = await coro
        ground_truth[pid] = result
        pbar.update(1)

    pbar.close()

    # Reconcile undocumented prescriptions
    print("\nReconciling undocumented prescriptions...")
    log = reconcile_undocumented(ground_truth)
    if log:
        print(f"  {len(log)} reconciliations made:")
        for entry in log[:20]:
            print(f"    {entry}")
        if len(log) > 20:
            print(f"    ... and {len(log) - 20} more")
    else:
        print("  No reconciliations needed.")

    # Reorder by original patient order
    ordered = {pid: ground_truth[pid] for pid in patient_ids if pid in ground_truth}

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)

    # Summary stats
    total_prescribed = 0
    total_reconciled = 0
    for pid, visits in ordered.items():
        for vname, vdata in visits.items():
            total_prescribed += len(vdata.get("prescribed_drugs", []))
            if vdata.get("source") == "reconciled_from_later_observation":
                total_reconciled += 1

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"{'='*60}")
    print(f"Saved {len(ordered)} patients to {output_file}")
    print(f"Total prescriptions across all visits: {total_prescribed}")
    print(f"Visits with reconciled prescriptions: {total_reconciled}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main_async())
