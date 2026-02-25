"""
Step 1: LLM-based text splitting for all visits.

Separates each visit's raw text into INPUT (observations) and OUTPUT (plan/prescriptions).
Pre-classifies unambiguous columns; sends ambiguous text to LLM for splitting.

Output: split_results.json
"""

import os
import json
import asyncio
import random
import dotenv
import pandas as pd
from tqdm import tqdm
from together import AsyncTogether

_HERE = os.path.dirname(os.path.abspath(__file__))
dotenv.load_dotenv(os.path.join(_HERE, '.env'))

# ============ CONFIGURATION ============
model = 'openai/gpt-oss-120b'
csv_path = os.path.join(os.path.dirname(_HERE), 'data', 'combined_dataset.csv')
max_concurrency = 8
limit_patients = 5  # Set to int for testing, None for all
output_file = os.path.join(_HERE, 'split_results.json')
# ========================================

SPLIT_SYSTEM_PROMPT = """You are a clinical note splitter. Given text from an epilepsy clinic note, separate it into:

1. INPUT (observations): Clinical history, examination findings, what the patient is currently taking (observed state), seizure descriptions, developmental information, past medical history, and any other factual observations.

2. OUTPUT (treatment plan): Prescriptions, dose changes, new medications ordered, investigations ordered, referrals, follow-up plans. Look for markers like "Plan:", numbered prescription lists (e.g., "1. Tabs Carbamazepine 200mg bd"), medication orders with dosages and durations (e.g., "x 1/12", "x 3/12"), "Continue [drug] [dose]", "Start [drug]", "Increase [drug]", "Switch to [drug]".

Key distinctions:
- "currently on sodium valproate" or "on carbamazepine" -> INPUT (observation of current state)
- "Tabs Carbamazepine 200 mg bd x 1/12" -> OUTPUT (prescription order)
- "episodes reduced with carbamazepine" -> INPUT (treatment response observation)
- "Start phenytoin 5 mls bd" -> OUTPUT (new prescription)
- "Continue Na Valproate 100 mg morning x 1/52" -> OUTPUT (prescription plan)
- "Do EEG" or "Review with results" -> OUTPUT (investigation/follow-up ordered)
- "was started on phenytoin yesterday" -> INPUT (observation of recent event)
- "Previously on phenobarbital" -> INPUT (medication history)
- "Sodium Valproate, Phenobarbital, Tegretol" (drug list without dosing instructions) -> INPUT (observation of what patient is on)
- "1. Sodium Valproate 150 mg BD x 3/12, 2. Folic Acid 5 mg OD x 3/12" -> OUTPUT (prescription list with doses and durations)

Return a JSON object with exactly two keys:
{
  "input_text": "...(all observational content, preserving original wording)...",
  "output_text": "...(all plan/prescription content, preserving original wording)..."
}

Preserve the original wording as much as possible. If no plan/prescription is found, set output_text to "". If no observations are found, set input_text to "".
Do not output anything except the JSON object."""

# --- Column classifications ---

# Visit 1: Always INPUT
ALWAYS_INPUT_V1 = [
    'Date of visit(0 months)', 'Age', 'Date of Birth:', 'Sex:',
    'Seizure Diagnosis',
    'Detailed description of seizure history',
    'Age of onset of seizure', 'Duration of Seizure',
    'Pre-ictal description', 'Ictal description', 'Post-ictal description',
    'Developmental Delay ', 'Duration with Developmental delay',
    'If delayed, specify',
    'Developmental Regression', 'Duration with Developmental Regression',
    'If delayed, specify.1',
    'Behavioural Problems', 'Psychiatric ',
    'Medication status',
    'Main Risk Factors (choice=Family history of epilepsy)',
    'Main Risk Factors (choice=Perinatal asphyxia)',
    'Main Risk Factors (choice=Prematurity/ Low birth weight)',
    'Main Risk Factors (choice=Intrauterine infections (TORCHES))',
    'Main Risk Factors (choice=Neonatal infections)',
    'Main Risk Factors (choice=Neonatal Jaundice)',
    'Main Risk Factors (choice=Childhood cerebral malaria)',
    'Main Risk Factors (choice=Tetanus)',
    'Main Risk Factors (choice=Childhood meningitis.)',
    'Main Risk Factors (choice=Genetic disorders)',
    'Main Risk Factors (choice=Familial History of Psychomotor abnormalities)',
    'Main Risk Factors (choice=Trauma to the head)',
    'Main Risk Factors (choice=Perinatal injuries)',
    'Main Risk Factors (choice=Febrile Seizures)',
    'Main Risk Factors (choice=Unknown)',
    'Main Risk Factors (choice=Other)',
]

# Visit 1: Always OUTPUT
ALWAYS_OUTPUT_V1 = ['Current dose']

# Visit 1: Needs LLM splitting
SPLIT_V1 = ['History of Presenting Illness', 'Current drug regimen']

# Visit 2
ALWAYS_INPUT_V2 = [
    'Date of visit(6 months)', 'Sex:',
    'Patients weight', 'Seizure frequency (6 months)', 'Specify ',
]
ALWAYS_OUTPUT_V2 = ['Medication dosage and if there was a change in medication(6 months)']
SPLIT_V2 = ['Second Entry(6 months)']

# Visit 3
ALWAYS_INPUT_V3 = [
    'Date of visit(12 months)', 'Sex:',
    'Patients weight.1', 'Seizure frequency (12 months)',
]
ALWAYS_OUTPUT_V3 = ['Medication dosage and if there was a change in medication(12 months)']
SPLIT_V3 = ['Third Entry(12 months)']


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
        # Try to find JSON object in the text
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {"input_text": "", "output_text": ""}


def safe_get(entry, col, columns):
    if col in columns:
        val = entry[col]
        return str(val).strip() if pd.notna(val) else ""
    return ""


def collect_columns(entry, col_list, columns):
    """Collect column values into a dict, skipping empty values."""
    result = {}
    for col in col_list:
        val = safe_get(entry, col, columns)
        if val:
            result[col] = val
    return result


def build_split_text(entry, split_cols, columns):
    """Build labeled text from columns that need LLM splitting."""
    parts = []
    for col in split_cols:
        val = safe_get(entry, col, columns)
        if val:
            parts.append(f"=== FIELD: {col} ===\n{val}")
    return "\n\n".join(parts)


# --- Async LLM ---

async def call_llm_with_retries(
    client: AsyncTogether,
    system_prompt: str,
    user_content: str,
    model: str = "openai/gpt-oss-120b",
    temperature: float = 0.0,
    max_tokens: int = 4096,
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


# --- Main processing ---

async def process_all(df, client, max_concurrency):
    semaphore = asyncio.Semaphore(max_concurrency)
    columns = df.columns

    # Build all tasks: (patient_id, visit_num, split_text)
    visit_configs = [
        (1, ALWAYS_INPUT_V1, ALWAYS_OUTPUT_V1, SPLIT_V1),
        (2, ALWAYS_INPUT_V2, ALWAYS_OUTPUT_V2, SPLIT_V2),
        (3, ALWAYS_INPUT_V3, ALWAYS_OUTPUT_V3, SPLIT_V3),
    ]

    patient_order = []
    tasks = []

    for idx, entry in df.iterrows():
        record_id = safe_get(entry, 'Record ID', columns)
        name = safe_get(entry, 'Name: ', columns)
        if record_id and name:
            patient_id = f"{record_id}_{name}"
        elif record_id:
            patient_id = record_id
        elif name:
            patient_id = name
        else:
            patient_id = f"patient_{idx}"

        patient_order.append(patient_id)

        for visit_num, input_cols, output_cols, split_cols in visit_configs:
            input_columns = collect_columns(entry, input_cols, columns)
            output_columns = collect_columns(entry, output_cols, columns)
            split_text = build_split_text(entry, split_cols, columns)

            tasks.append((patient_id, visit_num, input_columns, output_columns, split_text))

    # Run LLM calls for tasks with non-empty split text
    results = {}
    pbar = tqdm(total=len(tasks), desc="Splitting visits", unit="call")

    async def _run_one(task):
        patient_id, visit_num, input_columns, output_columns, split_text = task

        if split_text.strip():
            response = await call_llm_with_retries(
                client, SPLIT_SYSTEM_PROMPT, split_text,
                model=model, semaphore=semaphore,
            )
            parsed = robust_json_load(response) if response else {}
        else:
            parsed = {}

        return (patient_id, visit_num, {
            "input_text": parsed.get("input_text", ""),
            "output_text": parsed.get("output_text", ""),
            "input_columns": input_columns,
            "output_columns": output_columns,
        })

    for coro in asyncio.as_completed([_run_one(t) for t in tasks]):
        patient_id, visit_num, visit_result = await coro
        results.setdefault(patient_id, {})[f"Visit_{visit_num}"] = visit_result
        pbar.update(1)

    pbar.close()

    # Order results by patient order and visit order
    ordered = {}
    for pid in patient_order:
        per = results.get(pid, {})
        ordered[pid] = {
            "Visit_1": per.get("Visit_1", {}),
            "Visit_2": per.get("Visit_2", {}),
            "Visit_3": per.get("Visit_3", {}),
        }

    return ordered


async def main_async():
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        print("Error: TOGETHER_API_KEY not found")
        return

    print(f"\n{'='*60}")
    print(f"STEP 1: SPLIT INPUT/OUTPUT")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"CSV: {csv_path}")
    print(f"Output: {output_file}")
    print(f"Concurrency: {max_concurrency}")
    print(f"{'='*60}\n")

    df = pd.read_csv(
        csv_path, sep=';', engine='python',
        quotechar='"', doublequote=True, escapechar='\\',
    )
    dedup_cols = ['Name: ', 'Date of visit(0 months)',
                  'Date of visit(6 months)', 'Date of visit(12 months)']
    df = df.drop_duplicates(subset=dedup_cols)

    if limit_patients:
        df = df.head(limit_patients)
        print(f"Limited to {limit_patients} patients")

    print(f"Total patients: {len(df)}")

    client = AsyncTogether(api_key=together_key)
    results = await process_all(df, client, max_concurrency)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"DONE! Saved {len(results)} patients to {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main_async())
