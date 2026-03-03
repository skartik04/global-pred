"""
Extract clinical features from individual patient visit txt files (PDF OCR).
Reads _1.txt / _2.txt / _3.txt from each folder in all_patient_pdfs/.
Output: pdf_scripts/pdf_feats.json
"""
import os
import json
import asyncio
import random
import dotenv
from tqdm import tqdm
from together import AsyncTogether

# ---- Configuration ----
MODEL = "openai/gpt-oss-120b"
MAX_CONCURRENCY = 8
MAX_RETRIES = 6
LIMIT_PATIENTS = None   # Set to int to limit, None for all patients

# ---- Constants ----
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

MAX_VISITS = 10  # Maximum visit number to search for

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PDF_DIR = os.path.join(_ROOT, "all_patient_pdfs")


# ---- File Discovery ----

def find_visit_files(folder: str) -> dict:
    """Return {visit_name: path} for all visits found in folder (Visit_1 .. Visit_N)."""
    files = os.listdir(folder)
    visits = {}
    for n in range(1, MAX_VISITS + 1):
        suffixes = [f"_{n}.txt", f" {n}.txt"]
        for fname in files:
            if any(fname.endswith(s) for s in suffixes):
                visits[f"Visit_{n}"] = os.path.join(folder, fname)
                break
    return visits


def find_all_patients() -> list:
    """Return [(patient_name, {visit_name: path}), ...] sorted by name."""
    patients = []
    for entry in sorted(os.listdir(PDF_DIR)):
        folder = os.path.join(PDF_DIR, entry)
        if not os.path.isdir(folder):
            continue
        visits = find_visit_files(folder)
        if visits:
            patients.append((entry, visits))
    return patients


# ---- Parsing ----

def parse_reasoning_output(response_text, identifier):
    def strip_code_fences(text):
        t = text.strip()
        if t.startswith("```"):
            first_newline = t.find("\n")
            if first_newline != -1:
                t = t[first_newline + 1:]
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


def save_results(patient_responses, output_path):
    json_data = {}
    print(f"Parsing responses for {len(patient_responses)} patients...")

    for patient_id, visits in patient_responses.items():
        patient_data = {}
        for visit_name, response_text in visits.items():
            parsed_entry = parse_reasoning_output(response_text, f"{patient_id}_{visit_name}")
            patient_data[visit_name] = parsed_entry["features"]
        json_data[patient_id] = patient_data

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(json_data)} patients to {output_path}")


# ---- Async API Call ----

async def call_llm_with_retries(
    client,
    system_prompt,
    user_content,
    model=MODEL,
    temperature=0.0,
    max_tokens=8192,
    semaphore=None,
    max_retries=MAX_RETRIES,
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
    return ""


# ---- Async Inference ----

async def run_inference_pdf_async(patients, system_prompt, client, max_concurrency=MAX_CONCURRENCY):
    """
    patients: [(patient_name, {visit_name: txt_path}), ...]
    Returns {patient_name: {visit_name: response_text}}
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    # Build flat task list: (patient_name, visit_name, txt_path)
    tasks = []
    patient_visit_order = {}  # patient_name -> [visit_name, ...]
    for patient_name, visits in patients:
        patient_visit_order[patient_name] = list(visits.keys())
        for visit_name, txt_path in visits.items():
            tasks.append((patient_name, visit_name, txt_path))

    results = {}
    pbar = tqdm(total=len(tasks), desc="Visits", unit="call")

    async def _run_one(task):
        patient_name, visit_name, txt_path = task
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            visit_text = f.read()
        out = await call_llm_with_retries(
            client=client,
            system_prompt=system_prompt,
            user_content=visit_text,
            semaphore=semaphore,
        )
        return patient_name, visit_name, out

    for coro in asyncio.as_completed([_run_one(t) for t in tasks]):
        patient_name, visit_name, response_text = await coro
        results.setdefault(patient_name, {})[visit_name] = response_text
        pbar.update(1)

    pbar.close()

    # Restore order
    ordered = {}
    for patient_name, visits in patients:
        per = results.get(patient_name, {})
        ordered[patient_name] = {v: per.get(v, "") for v in visits.keys()}

    return ordered


# ---- Main ----

async def main_async():
    dotenv.load_dotenv(os.path.join(_ROOT, "scripts", ".env"))
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        print("Error: TOGETHER_API_KEY not found in scripts/.env")
        return

    prompt_path = os.path.join(_HERE, "pdf_extract_pervisit_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    all_patients = find_all_patients()
    if LIMIT_PATIENTS is not None:
        all_patients = all_patients[:LIMIT_PATIENTS]

    print(f"Model:    {MODEL}")
    print(f"Patients: {len(all_patients)}")
    print(f"Concurrency: {MAX_CONCURRENCY}\n")
    for name, visits in all_patients:
        print(f"  {name}: {list(visits.keys())}")
    print()

    client = AsyncTogether(api_key=together_key)
    patient_responses = await run_inference_pdf_async(all_patients, system_prompt, client)

    output_path = os.path.join(_HERE, "pdf_feats.json")
    save_results(patient_responses, output_path)


if __name__ == "__main__":
    asyncio.run(main_async())
