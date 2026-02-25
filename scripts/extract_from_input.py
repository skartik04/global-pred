"""
Step 2: Structured feature extraction from INPUT-only text.

Reads split_results.json from Step 1, constructs per-visit user content from
input_text + input_columns (no plan/prescription text), and runs 18-feature
extraction via LLM.

Drug features use simplified schema: "Status: <on_arrival|past_tried|not_mentioned>"
(no Action_taken / Temporal_language).

Input:  split_results.json
Output: extracted_features.json
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
prompt_file = os.path.join(_HERE, 'extract_input_prompt.txt')
max_concurrency = 8
limit_patients = 5  # Set to int for testing, None for all
output_file = os.path.join(_HERE, 'extracted_features.json')
# ========================================

FEATURE_NAMES = [
    "Age_Years",
    "Sex_Female",
    "OnsetTimingYears",
    "SeizureFreq",
    "StatusOrProlonged",
    "CognitivePriority",
    "Risk_Factors",
    "SeizureType",
    "drug_clobazam",
    "drug_clonazepam",
    "drug_valproate",
    "drug_ethosuximide",
    "drug_levetiracetam",
    "drug_lamotrigine",
    "drug_phenobarbital",
    "drug_phenytoin",
    "drug_topiramate",
    "drug_carbamazepine",
]

# Risk factor column names (for formatting)
RISK_FACTOR_COLS = [
    ('Main Risk Factors (choice=Family history of epilepsy)', 'Family history of epilepsy'),
    ('Main Risk Factors (choice=Perinatal asphyxia)', 'Perinatal asphyxia'),
    ('Main Risk Factors (choice=Prematurity/ Low birth weight)', 'Prematurity/Low birth weight'),
    ('Main Risk Factors (choice=Intrauterine infections (TORCHES))', 'Intrauterine infections'),
    ('Main Risk Factors (choice=Neonatal infections)', 'Neonatal infections'),
    ('Main Risk Factors (choice=Neonatal Jaundice)', 'Neonatal Jaundice'),
    ('Main Risk Factors (choice=Childhood cerebral malaria)', 'Childhood cerebral malaria'),
    ('Main Risk Factors (choice=Tetanus)', 'Tetanus'),
    ('Main Risk Factors (choice=Childhood meningitis.)', 'Childhood meningitis'),
    ('Main Risk Factors (choice=Genetic disorders)', 'Genetic disorders'),
    ('Main Risk Factors (choice=Familial History of Psychomotor abnormalities)', 'Familial psychomotor abnormalities'),
    ('Main Risk Factors (choice=Trauma to the head)', 'Trauma to the head'),
    ('Main Risk Factors (choice=Perinatal injuries)', 'Perinatal injuries'),
    ('Main Risk Factors (choice=Febrile Seizures)', 'Febrile Seizures'),
    ('Main Risk Factors (choice=Unknown)', 'Unknown'),
    ('Main Risk Factors (choice=Other)', 'Other'),
]


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


def parse_extraction_response(response_text: str, identifier: str) -> dict:
    """Parse LLM extraction response into structured features dict."""
    parsed_data = {}

    if not response_text:
        return {fn: {"Answer": "", "Reasoning": "", "Supporting_Text": "", "Confidence": ""}
                for fn in FEATURE_NAMES}

    raw = strip_code_fences(response_text)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON object
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                print(f"[WARN] JSON parse failed for {identifier}")
                return {fn: {"Answer": "", "Reasoning": "", "Supporting_Text": "", "Confidence": ""}
                        for fn in FEATURE_NAMES}
        else:
            print(f"[WARN] No JSON found for {identifier}")
            return {fn: {"Answer": "", "Reasoning": "", "Supporting_Text": "", "Confidence": ""}
                    for fn in FEATURE_NAMES}

    for feature_name in FEATURE_NAMES:
        feat_obj = data.get(feature_name, {})

        if isinstance(feat_obj, str):
            value, reasoning, supporting_text, confidence = feat_obj, "", "", ""
        elif isinstance(feat_obj, dict):
            value = feat_obj.get("value", "")
            reasoning = feat_obj.get("reasoning", "")
            supporting_text = feat_obj.get("supporting_text", "")
            confidence = feat_obj.get("confidence", "")
        else:
            value, reasoning, supporting_text, confidence = "", "", "", ""

        parsed_data[feature_name] = {
            "Answer": value,
            "Reasoning": reasoning,
            "Supporting_Text": supporting_text,
            "Confidence": confidence,
        }

    return parsed_data


def format_risk_factors(input_columns: dict) -> str:
    """Format risk factor checkboxes into a readable string."""
    checked = []
    for col_name, display_name in RISK_FACTOR_COLS:
        val = input_columns.get(col_name, "")
        if val and val.lower() == "checked":
            checked.append(display_name)
    if checked:
        return "Checked risk factors: " + ", ".join(checked)
    return "No risk factors checked"


def build_user_content(visit_num: int, visit_data: dict) -> str:
    """Build user content for extraction from split results."""
    input_text = visit_data.get("input_text", "")
    input_columns = visit_data.get("input_columns", {})

    if visit_num == 1:
        lines = [
            f"Visit 1 (Initial Visit - 0 months):",
            f"Initial Visit Date: {input_columns.get('Date of visit(0 months)', '')}",
            f"Patient Age: {input_columns.get('Age', '')}",
            f"Date of Birth: {input_columns.get('Date of Birth:', '')}",
            f"Patient Gender: {input_columns.get('Sex:', '')}",
            f"Seizure Diagnosis: {input_columns.get('Seizure Diagnosis', '')}",
            f"Detailed Description of Seizure History: {input_columns.get('Detailed description of seizure history', '')}",
            f"Age of Onset of Seizure: {input_columns.get('Age of onset of seizure', '')}",
            f"Duration of Seizure: {input_columns.get('Duration of Seizure', '')}",
            f"Pre-ictal description: {input_columns.get('Pre-ictal description', '')}",
            f"Ictal description: {input_columns.get('Ictal description', '')}",
            f"Post-ictal description: {input_columns.get('Post-ictal description', '')}",
            f"Developmental Delay: {input_columns.get('Developmental Delay ', '')}",
            f"Duration with Developmental delay: {input_columns.get('Duration with Developmental delay', '')}",
            f"If delayed, specify: {input_columns.get('If delayed, specify', '')}",
            f"Developmental Regression: {input_columns.get('Developmental Regression', '')}",
            f"Behavioural Problems: {input_columns.get('Behavioural Problems', '')}",
            f"Psychiatric: {input_columns.get('Psychiatric ', '')}",
            f"Medication status: {input_columns.get('Medication status', '')}",
            f"{format_risk_factors(input_columns)}",
        ]
        if input_text:
            lines.append("")
            lines.append("Clinical Notes (observations only, no treatment plan):")
            lines.append(input_text)

    elif visit_num == 2:
        lines = [
            f"Visit 2 (6 months follow-up):",
            f"6-Month Visit Date: {input_columns.get('Date of visit(6 months)', '')}",
            f"Patient Gender: {input_columns.get('Sex:', '')}",
            f"Patient Weight: {input_columns.get('Patients weight', '')}",
            f"Seizure Frequency: {input_columns.get('Seizure frequency (6 months)', '')}",
            f"Additional Specifications: {input_columns.get('Specify ', '')}",
        ]
        if input_text:
            lines.append("")
            lines.append("Clinical Notes (observations only, no treatment plan):")
            lines.append(input_text)
        lines.append("")
        lines.append("Note:")
        lines.append("- Baseline age was recorded at the initial visit.")
        lines.append("- This is approximately 6 months after baseline.")
        lines.append("- Follow-up documentation may omit unchanged baseline information.")

    elif visit_num == 3:
        lines = [
            f"Visit 3 (12 months follow-up):",
            f"12-Month Visit Date: {input_columns.get('Date of visit(12 months)', '')}",
            f"Patient Gender: {input_columns.get('Sex:', '')}",
            f"Patient Weight: {input_columns.get('Patients weight.1', '')}",
            f"Seizure Frequency: {input_columns.get('Seizure frequency (12 months)', '')}",
        ]
        if input_text:
            lines.append("")
            lines.append("Clinical Notes (observations only, no treatment plan):")
            lines.append(input_text)
        lines.append("")
        lines.append("Note:")
        lines.append("- Baseline age was recorded at the initial visit.")
        lines.append("- This is approximately 12 months after baseline.")
        lines.append("- Follow-up documentation may omit unchanged baseline information.")
    else:
        return ""

    return "\n".join(lines)


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


# --- Main ---

async def main_async():
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        print("Error: TOGETHER_API_KEY not found")
        return

    print(f"\n{'='*60}")
    print(f"STEP 2: EXTRACT FEATURES (INPUT ONLY)")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Input: {split_results_path}")
    print(f"Prompt: {prompt_file}")
    print(f"Output: {output_file}")
    print(f"Concurrency: {max_concurrency}")
    print(f"{'='*60}\n")

    # Load split results
    with open(split_results_path, 'r', encoding='utf-8') as f:
        split_results = json.load(f)

    # Load system prompt
    with open(prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    patient_ids = list(split_results.keys())
    if limit_patients:
        patient_ids = patient_ids[:limit_patients]
        print(f"Limited to {limit_patients} patients")

    print(f"Total patients: {len(patient_ids)}")

    # Build all tasks: (patient_id, visit_num, user_content)
    tasks = []
    for pid in patient_ids:
        visits = split_results[pid]
        for visit_num in [1, 2, 3]:
            visit_name = f"Visit_{visit_num}"
            visit_data = visits.get(visit_name, {})
            user_content = build_user_content(visit_num, visit_data)
            if user_content.strip():
                tasks.append((pid, visit_name, user_content))

    print(f"Total extraction calls: {len(tasks)}")

    client = AsyncTogether(api_key=together_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    results = {}
    pbar = tqdm(total=len(tasks), desc="Extracting features", unit="call")

    async def _run_one(task):
        pid, visit_name, content = task
        response = await call_llm_with_retries(
            client, system_prompt, content,
            model=model, semaphore=semaphore,
        )
        return pid, visit_name, response

    for coro in asyncio.as_completed([_run_one(t) for t in tasks]):
        pid, visit_name, response = await coro
        features = parse_extraction_response(response, f"{pid}_{visit_name}")
        results.setdefault(pid, {})[visit_name] = features
        pbar.update(1)

    pbar.close()

    # Order results
    ordered = {}
    visit_order = ["Visit_1", "Visit_2", "Visit_3"]
    for pid in patient_ids:
        per = results.get(pid, {})
        ordered[pid] = {v: per.get(v, {}) for v in visit_order}

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)

    # Summary
    n_empty = sum(1 for pid in ordered for v in visit_order
                  if not ordered[pid].get(v))
    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"{'='*60}")
    print(f"Saved {len(ordered)} patients to {output_file}")
    if n_empty:
        print(f"  Warning: {n_empty} visit extractions returned empty")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main_async())
