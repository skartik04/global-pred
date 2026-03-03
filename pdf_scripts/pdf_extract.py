"""
Extract clinical features from merged patient PDF text files.
Reads *_merged.txt from each patient folder, passes to gptoss, saves JSON.

Output: pdf_scripts/extracted_pdf_features.json
"""
import os
import re
import json
import asyncio
import random
import dotenv
from tqdm import tqdm
from together import AsyncTogether

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL = "openai/gpt-oss-120b"
MAX_CONCURRENCY = 8
MAX_RETRIES = 6
limit_patients = 400   # Set to int to process only N patients; None for all
# ───────────────────────────────────────────────────────────────────────────────

_HERE    = os.path.dirname(os.path.abspath(__file__))          # pdf_scripts/
_ROOT    = os.path.dirname(_HERE)                               # project root
PDF_DIR  = os.path.join(_ROOT, "all_patient_pdfs")             # where patient folders live

FEATURES = [
    "Age",
    "Sex_Female",
    "SeizureFreqTrajectory",
    "RiskFactors",
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

def find_merged_txts() -> list[tuple[str, str]]:
    """Return [(patient_name, merged_txt_path), ...] for every patient folder."""
    tasks = []
    for entry in sorted(os.listdir(PDF_DIR)):
        if not os.path.isdir(os.path.join(PDF_DIR, entry)):
            continue
        folder = os.path.join(PDF_DIR, entry)
        for fname in os.listdir(folder):
            if fname.endswith("_merged.txt"):
                tasks.append((entry, os.path.join(folder, fname)))
                break
    return tasks


def parse_json_response(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


async def call_model(
    client: AsyncTogether,
    system_prompt: str,
    user_content: str,
    semaphore: asyncio.Semaphore,
) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_content},
                    ],
                    max_tokens=4096,
                    temperature=0,
                )
            return resp.choices[0].message.content or ""
        except Exception as e:
            msg = str(e).lower()
            retryable = any(s in msg for s in [
                "rate", "429", "timeout", "temporar", "overload",
                "503", "502", "500", "bad gateway", "connection",
            ])
            if not retryable or attempt == MAX_RETRIES - 1:
                print(f"[ERROR] {e}")
                return ""
            await asyncio.sleep(0.8 * (2 ** attempt) + random.random() * 0.3)
    return ""


async def run_all(tasks, system_prompt, client) -> dict:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    results   = {}
    pbar      = tqdm(total=len(tasks), desc="Extracting")

    async def _one(patient_name, txt_path):
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            merged_text = f.read()
        user_content = f"PATIENT: {patient_name}\n\n{merged_text}"
        raw    = await call_model(client, system_prompt, user_content, semaphore)
        parsed = parse_json_response(raw)
        return patient_name, parsed, raw

    for coro in asyncio.as_completed([_one(n, p) for n, p in tasks]):
        patient_name, parsed, raw = await coro
        results[patient_name] = parsed
        pbar.update(1)

    pbar.close()
    return results


def build_clean_output(results: dict) -> dict:
    out = {}
    for patient, parsed in results.items():
        patient_out = {}
        for feat in FEATURES:
            feat_data = parsed.get(feat, {})
            if isinstance(feat_data, dict):
                patient_out[feat] = {
                    "value":           feat_data.get("value", "Not mentioned."),
                    "supporting_text": feat_data.get("supporting_text", ""),
                    "reasoning":       feat_data.get("reasoning", ""),
                    "confidence":      feat_data.get("confidence", ""),
                }
            else:
                patient_out[feat] = {
                    "value": str(feat_data) if feat_data else "Not mentioned.",
                    "supporting_text": "", "reasoning": "", "confidence": "",
                }
        out[patient] = patient_out
    return out


def main():
    dotenv.load_dotenv(os.path.join(_ROOT, "scripts", ".env"))
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY not set in scripts/.env")
        return

    with open(os.path.join(_HERE, "pdf_extract_prompt.txt"), "r", encoding="utf-8") as f:
        system_prompt = f.read()

    tasks = find_merged_txts()
    if limit_patients:
        tasks = tasks[:limit_patients]

    print(f"Model:    {MODEL}")
    print(f"Patients: {len(tasks)}")
    print(f"Concurrency: {MAX_CONCURRENCY}\n")

    client  = AsyncTogether(api_key=api_key)
    results = asyncio.run(run_all(tasks, system_prompt, client))

    out_path = os.path.join(_HERE, "extracted_pdf_features.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(build_clean_output(results), f, indent=2, ensure_ascii=False)

    n_failed = sum(1 for p in results.values() if not p)
    print(f"\nDone. {len(results)} patients processed.")
    print(f"  Output: {out_path}")
    if n_failed:
        print(f"  WARNING: {n_failed} patients had unparseable responses")


if __name__ == "__main__":
    main()
