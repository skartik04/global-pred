"""
Patient-level epilepsy relevance filter.
Reads _merged.txt for each patient, asks LLM if this is an epilepsy/seizure patient.
Output: pdf_scripts/pdf_filter_results.json  {patient_id: "YES"/"NO"}
"""
import os
import json
import asyncio
import random
import dotenv
from tqdm import tqdm
from together import AsyncTogether

MODEL = "openai/gpt-oss-120b"
MAX_CONCURRENCY = 8
MAX_RETRIES = 6
LIMIT_PATIENTS = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PDF_DIR = os.path.join(_ROOT, "all_patient_pdfs")

SYSTEM_PROMPT = """You are a medical record classifier. Output exactly one token: YES or NO.

YES only if the primary purpose of any of these clinical notes is seizure or epilepsy evaluation or management. Evidence must appear in Diagnosis/Impression/Assessment or History of Presenting Complaint or Plan, such as:
- a seizure or epilepsy diagnosis (epilepsy, seizure disorder, convulsions, febrile seizures, status epilepticus, ICD G40 etc)
- active seizure discussion (events, frequency, last seizure, seizure semiology)
- a seizure-specific plan (start/change/stop an antiseizure medication because of seizures, EEG for seizures, seizure precautions, neurology/epilepsy clinic follow-up)

NO if seizures or antiseizure medications are mentioned only as: past history, problem list, "no history of seizures", family history, allergy list, or a background medication list while the visit is clearly for something else (ENT, cough, tonsils, skin, dental, etc). Medication name alone is NOT enough because many antiseizure medications are used for other indications.

Output only YES or NO. Nothing else."""


def find_merged_txt(folder: str):
    for fname in os.listdir(folder):
        if fname.endswith("_merged.txt"):
            return os.path.join(folder, fname)
    return None


def find_all_patients():
    patients = []
    for entry in sorted(os.listdir(PDF_DIR)):
        folder = os.path.join(PDF_DIR, entry)
        if not os.path.isdir(folder):
            continue
        merged = find_merged_txt(folder)
        if merged:
            patients.append((entry, merged))
    return patients


async def call_llm(client, user_content, semaphore):
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_content},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
            return (resp.choices[0].message.content or "").strip().upper()
        except Exception as e:
            msg = str(e).lower()
            retryable = any(s in msg for s in ["rate", "429", "timeout", "temporar", "overload", "503", "500", "connection"])
            if not retryable or attempt == MAX_RETRIES - 1:
                print(f"[ERROR] {e}")
                return "ERROR"
            await asyncio.sleep(0.8 * (2 ** attempt) + random.random() * 0.3)
    return "ERROR"


async def main_async():
    dotenv.load_dotenv(os.path.join(_ROOT, "scripts", ".env"))
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        print("Error: TOGETHER_API_KEY not found")
        return

    patients = find_all_patients()
    if LIMIT_PATIENTS:
        patients = patients[:LIMIT_PATIENTS]

    print(f"Model:    {MODEL}")
    print(f"Patients: {len(patients)}\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    client = AsyncTogether(api_key=together_key)
    results = {}

    pbar = tqdm(total=len(patients), desc="Filtering", unit="patient")

    async def _run_one(patient_name, merged_path):
        with open(merged_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        answer = await call_llm(client, text, semaphore)
        # Normalise — startswith check to handle "YES." / "NO." / "YES\n" etc
        if answer.startswith("YES"):
            answer = "YES"
        elif answer.startswith("NO"):
            answer = "NO"
        else:
            answer = "ERROR"
        return patient_name, answer

    for coro in asyncio.as_completed([_run_one(n, p) for n, p in patients]):
        patient_name, answer = await coro
        results[patient_name] = answer
        pbar.update(1)

    pbar.close()

    out_path = os.path.join(_HERE, "pdf_filter_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    yes   = sum(1 for v in results.values() if v == "YES")
    no    = sum(1 for v in results.values() if v == "NO")
    error = sum(1 for v in results.values() if v == "ERROR")
    print(f"\nDone. YES={yes}  NO={no}  ERROR={error}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main_async())
