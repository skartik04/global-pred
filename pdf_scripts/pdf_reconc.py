"""
Reconcile per-visit PDF extractions into cumulative longitudinal feature vectors.
Input:  pdf_scripts/pdf_feats.json
Output: pdf_scripts/pdf_reconc.json
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
TEMPERATURE = 0.0
MAX_CONCURRENCY = 8
MAX_RETRIES = 6
LIMIT_PATIENTS = None   # Set to int to limit, None for all

INPUT_FILENAME  = "pdf_feats.json"
OUTPUT_FILENAME = "pdf_reconc.json"
PROMPT_FILENAME = "pdf_reconc_prompt.txt"

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

MAX_VISITS = 10  # Maximum visit number to look for

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


# ---- JSON Helpers ----

def strip_code_fences(text):
    t = (text or "").strip()
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
        t = t.strip()
    return t


def robust_json_load(text):
    raw = strip_code_fences(text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start: end + 1])
            except Exception:
                return None
        return None


def normalize_feature_obj(obj):
    if isinstance(obj, str):
        return {"Answer": obj, "Reasoning": "", "Supporting_Text": "", "Confidence": ""}

    if isinstance(obj, dict):
        if all(k in obj for k in ["Answer", "Reasoning", "Supporting_Text", "Confidence"]):
            return {
                "Answer": obj.get("Answer", ""),
                "Reasoning": obj.get("Reasoning", ""),
                "Supporting_Text": obj.get("Supporting_Text", ""),
                "Confidence": obj.get("Confidence", ""),
            }
        if any(k in obj for k in ["value", "reasoning", "supporting_text", "confidence"]):
            return {
                "Answer": obj.get("value", ""),
                "Reasoning": obj.get("reasoning", ""),
                "Supporting_Text": obj.get("supporting_text", ""),
                "Confidence": obj.get("confidence", ""),
            }
        return {"Answer": "", "Reasoning": "", "Supporting_Text": "", "Confidence": ""}

    return {"Answer": "", "Reasoning": "", "Supporting_Text": "", "Confidence": ""}


def parse_cumulative_output(response_text, identifier):
    data = robust_json_load(response_text)
    if not isinstance(data, dict):
        print(f"[WARN] JSON parse failed for {identifier}")
        return {}
    out = {}
    for feat in FEATURE_NAMES:
        out[feat] = normalize_feature_obj(data.get(feat, {}))
    return out


# ---- Async API Call ----

async def call_llm_with_retries(
    client,
    system_prompt,
    user_content,
    model=MODEL,
    temperature=TEMPERATURE,
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


# ---- Reconciliation Input Builder ----

def build_reconcile_user_content(visits_in_order):
    """visits_in_order: list of visit feature dicts in chronological order."""
    parts = []
    for i, v in enumerate(visits_in_order, 1):
        parts.append(f"VISIT_{i}_JSON:\n{json.dumps(v, ensure_ascii=False)}")
    return "\n\n".join(parts)


# ---- Async Reconciliation Runner ----

async def run_reconciliation_async(input_json_path, output_json_path, system_prompt, client):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    patient_ids = list(data.keys())
    if LIMIT_PATIENTS is not None:
        patient_ids = patient_ids[:LIMIT_PATIENTS]

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    out_data = {}
    tasks = []

    for pid in patient_ids:
        visits = data.get(pid, {})
        # Collect all visits that exist, in order
        visit_keys = [f"Visit_{n}" for n in range(1, MAX_VISITS + 1)
                      if visits.get(f"Visit_{n}")]
        visit_data = [visits[vk] for vk in visit_keys]

        # Always copy Visit_1 directly (no API call needed)
        out_data[pid] = {vk: {} for vk in visit_keys}
        if visit_keys:
            out_data[pid][visit_keys[0]] = visit_data[0]

        # Schedule cumulative reconcile for Visit_2 .. Visit_N
        for i in range(1, len(visit_keys)):
            target_visit = visit_keys[i]
            cumulative = visit_data[:i + 1]  # V1 .. Vi
            tasks.append((pid, target_visit, build_reconcile_user_content(cumulative)))

    pbar = tqdm(total=len(tasks), desc="Reconcile calls", unit="call")

    async def _run_one(task):
        pid, target_visit, user_content = task
        resp_text = await call_llm_with_retries(
            client=client,
            system_prompt=system_prompt,
            user_content=user_content,
            semaphore=semaphore,
        )
        merged = parse_cumulative_output(resp_text, f"{pid}_{target_visit}")
        return pid, target_visit, merged, resp_text

    for coro in asyncio.as_completed([_run_one(t) for t in tasks]):
        pid, target_visit, merged_features, raw_text = await coro
        if merged_features:
            out_data[pid][target_visit] = merged_features
        else:
            out_data[pid][target_visit] = {}
            out_data[pid][f"{target_visit}_RAW_MODEL_OUTPUT"] = raw_text
        pbar.update(1)

    pbar.close()

    print(f"\nSaving results to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"Saved reconciled data for {len(out_data)} patients to {output_json_path}")


# ---- Main ----

async def main_async():
    dotenv.load_dotenv(os.path.join(_ROOT, "scripts", ".env"))
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        print("Error: TOGETHER_API_KEY not found in scripts/.env")
        return

    input_path  = os.path.join(_HERE, INPUT_FILENAME)
    output_path = os.path.join(_HERE, OUTPUT_FILENAME)
    prompt_path = os.path.join(_HERE, PROMPT_FILENAME)

    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    print("=" * 60)
    print("RECONCILIATION CONFIGURATION")
    print("=" * 60)
    print(f"Input:  {INPUT_FILENAME}")
    print(f"Output: {OUTPUT_FILENAME}")
    print(f"Model:  {MODEL}")
    print(f"Limit:  {LIMIT_PATIENTS if LIMIT_PATIENTS else 'all patients'}")
    print("=" * 60 + "\n")

    client = AsyncTogether(api_key=together_key)
    await run_reconciliation_async(input_path, output_path, system_prompt, client)


if __name__ == "__main__":
    asyncio.run(main_async())
