"""
Repair script: re-runs failed/missing extractions and updates pdf_feats.json in place.
Handles:
  1. Patients with non-standard (descriptive) filenames missed by the main script
  2. Empty visit entries (parse failed / API returned empty)
"""
import os
import json
import asyncio
import random
import dotenv
from together import AsyncTogether

MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 6

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PDF_DIR = os.path.join(_ROOT, "all_patient_pdfs")

FEATURE_NAMES = [
    "Age_Years", "Sex_Female", "OnsetTimingYears", "SeizureFreq",
    "StatusOrProlonged", "CognitivePriority", "Risk_Factors", "SeizureType",
    "drug_clobazam", "drug_clonazepam", "drug_valproate", "drug_ethosuximide",
    "drug_levetiracetam", "drug_lamotrigine", "drug_phenobarbital",
    "drug_phenytoin", "drug_topiramate", "drug_carbamazepine",
]

# Patients whose txt files have descriptive names instead of _1/_2/_3
SPECIAL_MAPPINGS = {
    "AHAISIBWE_ISABELLA": {
        "Visit_1": "isabella_ihaisibwe.txt",
        "Visit_2": "isabella;_6_months.txt",
        "Visit_3": "isabella-_1yr.txt",
    },
    "MAYRA_ATIM": {
        "Visit_1": "mayra.txt",
        "Visit_2": "mayra_6_months.txt",
        "Visit_3": "mayra_1yr.txt",
    },
    "SSENKUNDA_JORAM": {
        "Visit_1": "SSENKUNDA_JORAM.txt",
    },
}


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
            value          = feat_obj.get("value", "")
            reasoning      = feat_obj.get("reasoning", "")
            supporting_text = feat_obj.get("supporting_text", "")
            confidence     = feat_obj.get("confidence", "")
        else:
            value, reasoning, supporting_text, confidence = "", "", "", ""

        parsed_data["features"][feature_name] = {
            "Answer": value,
            "Reasoning": reasoning,
            "Supporting_Text": supporting_text,
            "Confidence": confidence,
        }
    return parsed_data


async def call_llm_with_retries(client, system_prompt, user_content, semaphore):
    async def _one_call():
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.0,
            max_tokens=8192,
        )
        return (resp.choices[0].message.content or "").strip()

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                return await _one_call()
        except Exception as e:
            msg = str(e).lower()
            retryable = any(s in msg for s in ["rate", "429", "timeout", "temporar", "overload", "503", "500", "connection"])
            if not retryable or attempt == MAX_RETRIES - 1:
                print(f"[ERROR] {e}")
                return ""
            await asyncio.sleep(0.8 * (2 ** attempt) + random.random() * 0.3)
    return ""


async def main_async():
    dotenv.load_dotenv(os.path.join(_ROOT, "scripts", ".env"))
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        print("Error: TOGETHER_API_KEY not found")
        return

    prompt_path = os.path.join(_HERE, "pdf_extract_pervisit_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    output_path = os.path.join(_HERE, "pdf_feats.json")
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build task list: (patient_id, visit_name, txt_path)
    tasks = []

    # 1. Missing patients with special filename mappings
    for patient_id, visit_map in SPECIAL_MAPPINGS.items():
        folder = os.path.join(PDF_DIR, patient_id)
        for visit_name, fname in visit_map.items():
            txt_path = os.path.join(folder, fname)
            if os.path.exists(txt_path):
                tasks.append((patient_id, visit_name, txt_path))
            else:
                print(f"[WARN] File not found: {txt_path}")

    # 2. Empty visits in existing data
    for pid, visits in data.items():
        for vname, feats in visits.items():
            if not feats:
                # Find the txt file for this visit
                folder = os.path.join(PDF_DIR, pid)
                n = int(vname.split("_")[1])
                found = None
                for fname in os.listdir(folder):
                    if fname.endswith(f"_{n}.txt") or fname.endswith(f" {n}.txt"):
                        found = os.path.join(folder, fname)
                        break
                if found:
                    tasks.append((pid, vname, found))
                else:
                    print(f"[WARN] Could not find txt for {pid}/{vname}")

    print(f"Tasks to repair: {len(tasks)}")
    for pid, vname, path in tasks:
        print(f"  {pid} / {vname}  →  {os.path.basename(path)}")
    print()

    semaphore = asyncio.Semaphore(8)
    client = AsyncTogether(api_key=together_key)

    async def _run_one(task):
        pid, visit_name, txt_path = task
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            visit_text = f.read()
        raw = await call_llm_with_retries(client, system_prompt, visit_text, semaphore)
        parsed = parse_reasoning_output(raw, f"{pid}_{visit_name}")
        return pid, visit_name, parsed["features"]

    results = await asyncio.gather(*[_run_one(t) for t in tasks])

    # Update data in place
    updated = 0
    for pid, visit_name, features in results:
        if features:
            if pid not in data:
                data[pid] = {}
            data[pid][visit_name] = features
            print(f"  [OK] {pid} / {visit_name}")
            updated += 1
        else:
            print(f"  [FAIL] {pid} / {visit_name} — still empty after retry")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {updated}/{len(tasks)} repaired. Saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main_async())
