import os
import json
import asyncio
import random
import dotenv
import pandas as pd
from tqdm import tqdm
from together import AsyncTogether

# ---- Global Constants ----
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

# ---- Helper Functions ----

def parse_reasoning_output(response_text, identifier):
    def strip_code_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            first_newline = t.find("\n")
            if first_newline != -1:
                t = t[first_newline + 1 :]
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
            t = t.strip()
        return t

    parsed_data = {"id": identifier, "features": {}}

    if not response_text:
        return parsed_data

    raw = strip_code_fences(response_text)

    try:
        data = json.loads(raw)
    except Exception as e:
        print(f"[WARN] JSON parse failed for {identifier}: {e}")
        return parsed_data

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

        parsed_data["features"][feature_name] = {
            "Answer": value,
            "Reasoning": reasoning,
            "Supporting_Text": supporting_text,
            "Confidence": confidence,
        }

    return parsed_data


def save_results(patient_responses, output_filename_base):
    json_data = {}
    print(f"Parsing responses for {len(patient_responses)} patients...")

    for patient_id, visits in patient_responses.items():
        patient_data = {}
        for visit_name, response_text in visits.items():
            parsed_entry = parse_reasoning_output(response_text, f"{patient_id}_{visit_name}")
            patient_data[visit_name] = parsed_entry["features"]
        json_data[patient_id] = patient_data

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, f"{output_filename_base}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(json_data)} patients to {json_path}")


# ---- Async Together AI Call Wrapper ----

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
            retryable = any(s in msg for s in ["rate", "429", "timeout", "temporar", "overload", "503", "500", "connection"])
            if not retryable or attempt == max_retries - 1:
                return ""
            base = 0.8 * (2 ** attempt)
            await asyncio.sleep(base + random.random() * 0.3)


# ---- Async Inference: CSV ----

async def run_inference_csv_async(path, num_entries_to_process, client, system_prompt, max_concurrency=8):
    print(f"Processing CSV from: {path}")

    try:
        raw_data = pd.read_csv(
            path,
            sep=";",
            engine="python",
            quotechar='"',
            doublequote=True,
            escapechar="\\",
        )
        dedup_cols = ['Name: ', 'Date of visit(0 months)', 'Date of visit(6 months)', 'Date of visit(12 months)']
        raw_data = raw_data.drop_duplicates(subset=dedup_cols)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}

    limit = min(len(raw_data), num_entries_to_process)
    use_patient_id = "Patient ID" in raw_data.columns

    print(f"Starting async API calls for {limit} patients (3 calls each)...")

    visit_order = ["Visit_1", "Visit_2", "Visit_3"]
    patient_order = []
    tasks = []

    for index, entry in raw_data.head(limit).iterrows():
        def safe_get(col_name):
            if col_name in raw_data.columns:
                val = entry[col_name]
                return val if pd.notna(val) else ""
            return ""

        record_id = ""
        if "Record ID" in raw_data.columns and pd.notna(entry.get("Record ID")):
            record_id = str(entry["Record ID"]).strip()
        
        if record_id and "Name: " in raw_data.columns and pd.notna(entry.get("Name: ")):
            patient_id = f"{record_id}_{str(entry['Name: ']).strip()}"
        elif record_id and use_patient_id and pd.notna(entry.get("Patient ID")):
            patient_id = f"{record_id}_{str(entry['Patient ID']).strip()}"
        elif record_id:
            patient_id = record_id
        elif "Name: " in raw_data.columns and pd.notna(entry.get("Name: ")):
            patient_id = str(entry["Name: "]).strip()
        elif use_patient_id and pd.notna(entry.get("Patient ID")):
            patient_id = str(entry["Patient ID"]).strip()
        else:
            patient_id = f"patient_{index}"

        patient_order.append(patient_id)

        visit_1_desc = f"""Visit 1 (Initial Visit - 0 months):
Initial Visit Date: {safe_get('Date of visit(0 months)')}
Patient Age: {safe_get('Age')}
Patient Gender: {safe_get('Sex:')}
Seizure Diagnosis: {safe_get('Seizure Diagnosis')}
Medical History / History of Presenting Illness: {safe_get('History of Presenting Illness')}
Detailed Description of Seizure History: {safe_get('Detailed description of seizure history')}
Age of Onset of Seizure: {safe_get('Age of onset of seizure')}
Duration of Seizure: {safe_get('Duration of Seizure')}
Prior Medications / Drug Regimen (prior drug list; may also include a new prescription plan, sometimes marked with a date): {safe_get('Current drug regimen')}
Current Dose: {safe_get('Current dose')}
"""

        visit_2_desc = f"""Visit 2 (6 months follow-up):
6-Month Visit Date: {safe_get('Date of visit(6 months)')}
Patient Gender: {safe_get('Sex:')}
Medication Notes (6-month visit; may describe continuation or change): {safe_get('Second Entry(6 months)')}
Medication Prescription (6-month visit; may describe dosage change): {safe_get('Medication dosage and if there was a change in medication(6 months)')}
Additional Specifications: {safe_get('Specify')}

Note:
- Baseline age was recorded at the initial visit.
- This is approximately 6 months after baseline.
- Follow-up documentation may omit unchanged baseline information.
"""

        visit_3_desc = f"""Visit 3 (12 months follow-up):
12-Month Visit Date: {safe_get('Date of visit(12 months)')}
Patient Gender: {safe_get('Sex:')}
Medication Notes (12-month visit; may describe continuation or change): {safe_get('Third Entry(12 months)')}
Medication Prescription (12-month visit; may describe dosage change): {safe_get('Medication dosage and if there was a change in medication(12 months)')}

Note:
- Baseline age was recorded at the initial visit.
- This is approximately 12 months after baseline.
- Follow-up documentation may omit unchanged baseline information.
"""

        tasks.append((patient_id, "Visit_1", visit_1_desc))
        tasks.append((patient_id, "Visit_2", visit_2_desc))
        tasks.append((patient_id, "Visit_3", visit_3_desc))

    semaphore = asyncio.Semaphore(max_concurrency)
    results = {}

    pbar = tqdm(total=len(tasks), desc="Visits", unit="call")

    async def _run_one(task):
        patient_id, visit_name, content = task
        out = await call_llm_with_retries(
            client=client,
            system_prompt=system_prompt,
            user_content=content,
            semaphore=semaphore,
        )
        return patient_id, visit_name, out

    for coro in asyncio.as_completed([_run_one(t) for t in tasks]):
        patient_id, visit_name, response_text = await coro
        results.setdefault(patient_id, {})[visit_name] = response_text
        pbar.update(1)

    pbar.close()
    
    ordered_results = {}
    for pid in patient_order:
        per = results.get(pid, {})
        ordered_results[pid] = {v: per.get(v, "") for v in visit_order}
    
    return ordered_results


# ---- Main ----

async def main_async():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv.load_dotenv(os.path.join(script_dir, ".env"))
    together_key = os.getenv("TOGETHER_API_KEY")

    if not together_key:
        print("Error: TOGETHER_API_KEY not found in .env file")
        return

    csv_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'global', 'data', 'combined_dataset.csv')
    num_entries_to_process = 6  # FULL RUN
    max_concurrency = 8  # parallel API calls
    prompt_path = os.path.join(script_dir, "extract_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_path}")
        return

    client = AsyncTogether(api_key=together_key)

    patient_responses = await run_inference_csv_async(
        csv_path, num_entries_to_process, client, system_prompt, max_concurrency=max_concurrency
    )

    if patient_responses:
        save_results(patient_responses, "csv_feats_gpt_oss_trial")


if __name__ == "__main__":
    asyncio.run(main_async())
