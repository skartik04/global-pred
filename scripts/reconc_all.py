import os
import json
import asyncio
import random
import dotenv
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

VISIT_ORDER = ["Visit_1", "Visit_2", "Visit_3"]


# ---- JSON Helpers ----
def strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
        t = t.strip()
    return t


def robust_json_load(text: str):
    raw = strip_code_fences(text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        # try to salvage first {...} block
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                return None
        return None


def normalize_feature_obj(obj):
    """
    Normalize model output to the expected internal format:
    {
      "Answer": ...,
      "Reasoning": ...,
      "Supporting_Text": ...,
      "Confidence": ...
    }
    Accepts:
      - already-normalized dict
      - dict with value/reasoning/supporting_text/confidence
      - string
    """
    if isinstance(obj, str):
        return {
            "Answer": obj,
            "Reasoning": "",
            "Supporting_Text": "",
            "Confidence": "",
        }

    if isinstance(obj, dict):
        # Preferred format for reconciliation prompt
        if all(k in obj for k in ["Answer", "Reasoning", "Supporting_Text", "Confidence"]):
            return {
                "Answer": obj.get("Answer", ""),
                "Reasoning": obj.get("Reasoning", ""),
                "Supporting_Text": obj.get("Supporting_Text", ""),
                "Confidence": obj.get("Confidence", ""),
            }

        # Fallback if model outputs the extraction-style schema
        if any(k in obj for k in ["value", "reasoning", "supporting_text", "confidence"]):
            return {
                "Answer": obj.get("value", ""),
                "Reasoning": obj.get("reasoning", ""),
                "Supporting_Text": obj.get("supporting_text", ""),
                "Confidence": obj.get("confidence", ""),
            }

        # Unknown dict shape
        return {
            "Answer": "",
            "Reasoning": "",
            "Supporting_Text": "",
            "Confidence": "",
        }

    return {
        "Answer": "",
        "Reasoning": "",
        "Supporting_Text": "",
        "Confidence": "",
    }


def parse_cumulative_output(response_text: str, identifier: str):
    """
    Parses the reconciled cumulative JSON output (schema with Answer/Reasoning/Supporting_Text/Confidence).
    Returns dict of features keyed by FEATURE_NAMES. Empty dict on failure.
    """
    data = robust_json_load(response_text)
    if not isinstance(data, dict):
        print(f"[WARN] CUM JSON parse failed for {identifier}")
        return {}

    out = {}
    for feat in FEATURE_NAMES:
        out[feat] = normalize_feature_obj(data.get(feat, {}))

    return out


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
            retryable = any(
                s in msg
                for s in ["rate", "429", "timeout", "temporar", "overload", "503", "500", "connection"]
            )
            if (not retryable) or (attempt == max_retries - 1):
                return ""
            base = 0.8 * (2 ** attempt)
            await asyncio.sleep(base + random.random() * 0.3)
    return ""


# ---- Reconciliation Prompt Input Builder ----
def build_reconcile_user_content(visits_in_order):
    """
    visits_in_order: list[dict] where each dict is one visit feature-vector
    Returns user message that includes 2 or 3 JSON objects (as text) in order.
    """
    parts = []
    for i, v in enumerate(visits_in_order, 1):
        parts.append(f"VISIT_{i}_JSON:\n{json.dumps(v, ensure_ascii=False)}")
    return "\n\n".join(parts)


# ---- Async Reconciliation Runner ----
async def run_reconciliation_async(
    input_json_path: str,
    output_json_path: str,
    reconcile_system_prompt: str,
    model: str,
    temperature: float,
    max_concurrency: int,
    limit_patients: int | None,
    client: AsyncTogether,
):
    """
    Runs reconciliation on all patients asynchronously.
    Results are accumulated in memory and saved ONCE at the end.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # preserves order in Python 3.7+

    patient_ids = list(data.keys())
    if limit_patients is not None:
        patient_ids = patient_ids[:limit_patients]

    semaphore = asyncio.Semaphore(max_concurrency)

    # Output container (ordered)
    out_data = {}

    # Build reconciliation tasks for Visit_2 and Visit_3 only
    tasks = []
    for pid in patient_ids:
        visits = data.get(pid, {})
        v1 = visits.get("Visit_1", {}) or {}
        v2 = visits.get("Visit_2", {}) or {}
        v3 = visits.get("Visit_3", {}) or {}

        # Always copy Visit_1 directly
        out_data[pid] = {"Visit_1": v1, "Visit_2": {}, "Visit_3": {}}

        # If Visit_2 exists (non-empty), schedule reconcile for cumulative(1+2)
        if isinstance(v1, dict) and isinstance(v2, dict) and v1 and v2:
            user_content_12 = build_reconcile_user_content([v1, v2])
            tasks.append((pid, "Visit_2", user_content_12))

        # If Visit_3 exists (non-empty), schedule reconcile for cumulative(1+2+3)
        if isinstance(v1, dict) and isinstance(v2, dict) and isinstance(v3, dict) and v1 and v2 and v3:
            user_content_123 = build_reconcile_user_content([v1, v2, v3])
            tasks.append((pid, "Visit_3", user_content_123))

    pbar = tqdm(total=len(tasks), desc="Reconcile calls", unit="call")

    async def _run_one(task):
        pid, target_visit, user_content = task
        resp_text = await call_llm_with_retries(
            client=client,
            system_prompt=reconcile_system_prompt,
            user_content=user_content,
            model=model,
            temperature=temperature,
            semaphore=semaphore,
        )
        merged = parse_cumulative_output(resp_text, f"{pid}_{target_visit}")
        return pid, target_visit, merged, resp_text

    # Run concurrently
    for coro in asyncio.as_completed([_run_one(t) for t in tasks]):
        pid, target_visit, merged_features, raw_text = await coro

        if merged_features:
            # Save reconciled cumulative features
            out_data[pid][target_visit] = merged_features
        else:
            # If parse failed, keep empty and save raw for debugging
            out_data[pid][target_visit] = {}
            out_data[pid][f"{target_visit}_RAW_MODEL_OUTPUT"] = raw_text

        pbar.update(1)

    pbar.close()

    # Write output JSON (saved once at the end after all API calls complete)
    print(f"\nSaving results to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved reconciled data for {len(out_data)} patients to {output_json_path}")


# ---- Main (Async) ----

async def main_async():
    # ============ CONFIGURATION ============
    
    # API Key
    script_dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(script_dir, ".env")

    # Input/Output Files (in only_open folder)
    input_filename = "csv_backfilled_gpt_oss_trial.json"    # Input: backfilled extraction results
    output_filename = "csv_reconciled_gpt_oss_trial.json"   # Output: reconciled cumulative results
    prompt_filename = "reconc_prompt.txt"             # Reconciliation prompt
    
    # Processing Settings
    model = "openai/gpt-oss-120b"
    temperature = 0.0
    max_concurrency = 8
    limit_patients = 400  # Set to a number to limit, or None for all patients
    
    # ========================================
    
    dotenv.load_dotenv(key_path)
    together_key = os.getenv("TOGETHER_API_KEY")
    
    if not together_key:
        print("Error: TOGETHER_API_KEY not found in .env file")
        return
    
    # Build full paths
    input_json_path = os.path.join(script_dir, input_filename)
    output_json_path = os.path.join(script_dir, output_filename)
    reconcile_prompt_path = os.path.join(script_dir, prompt_filename)
    
    # Load reconciliation prompt
    try:
        with open(reconcile_prompt_path, "r", encoding="utf-8") as f:
            reconcile_system_prompt = f.read()
    except FileNotFoundError:
        print(f"Error: Reconciliation prompt file not found at {reconcile_prompt_path}")
        return
    
    client = AsyncTogether(api_key=together_key)
    
    print("\n" + "="*60)
    print("RECONCILIATION CONFIGURATION")
    print("="*60)
    print(f"Input:  {input_filename}")
    print(f"Output: {output_filename}")
    print(f"Prompt: {prompt_filename}")
    print(f"Model:  {model}")
    print(f"Limit:  {limit_patients if limit_patients else 'all patients'}")
    print(f"Saving: Once at the end (after all API calls complete)")
    print("="*60 + "\n")
    
    await run_reconciliation_async(
        input_json_path=input_json_path,
        output_json_path=output_json_path,
        reconcile_system_prompt=reconcile_system_prompt,
        model=model,
        temperature=temperature,
        max_concurrency=max_concurrency,
        limit_patients=limit_patients,
        client=client,
    )


if __name__ == "__main__":
    asyncio.run(main_async())
