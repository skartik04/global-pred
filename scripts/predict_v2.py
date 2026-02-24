import os
import re
import json
import asyncio
import random
import aiohttp
import pandas as pd
from tqdm import tqdm
import dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
dotenv.load_dotenv(os.path.join(_HERE, '.env'))

# ============ CONFIGURATION ============

# Model name (Together AI)
model = 'openai/gpt-oss-120b'

# Which visit to process (1, 2, or 3)
visit_num = 3

# Input mode: 'raw' (CSV) or 'extracted' (reconciled JSON)
input_mode = 'extracted'

# Input paths
csv_path = os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'global', 'data', 'combined_dataset.csv')
reconciled_json_path = os.path.join(_HERE, 'csv_reconciled_gpt_oss.json')

# Prompt file
prompt_file = 'all_drugs_v2.txt'

# Processing settings
thinking_budget = 6000
max_tokens = 4096
max_concurrency = 8
limit_patients = None  # Set to int to limit, None for all

# ========================================

DRUG_COLUMNS = [
    "carbamazepine", "clobazam", "clonazepam", "ethosuximide",
    "lamotrigine", "levetiracetam", "phenobarbital", "phenytoin",
    "topiramate", "valproate",
]

DRUG_FEATURE_NAMES = [
    "drug_clobazam", "drug_clonazepam", "drug_valproate", "drug_ethosuximide",
    "drug_levetiracetam", "drug_lamotrigine", "drug_phenobarbital",
    "drug_phenytoin", "drug_topiramate", "drug_carbamazepine",
]

# Feature names for extracted mode (non-drug features only)
FEATURE_NAMES = [
    "Age_Years",
    "Sex_Female",
    "OnsetTimingYears",
    "SeizureFreq",
    "StatusOrProlonged",
    "CognitivePriority",
    "Risk_Factors",
    "SeizureType",
]


def get_short_model_name(model_name: str) -> str:
    """Extract a short name from the model for filenames."""
    name = model_name.replace("/", "_").replace("-", "").replace(".", "")
    if len(name) > 20:
        name = name[:20]
    return name


def _extract_status(answer: str) -> str:
    """Parse drug Answer field into a concise status string."""
    if not answer or answer.strip().lower() in ("not mentioned.", "not mentioned"):
        return None
    # For reconciled multi-visit format like "Visit 1: current (ongoing). Visit 2: ..."
    # Take the last visit's status
    parts = answer.split(".")
    last_meaningful = ""
    for part in reversed(parts):
        part = part.strip()
        if part and any(kw in part.lower() for kw in [
            "current", "previous", "unclear", "ongoing",
            "stop", "initiation", "dose_change"
        ]):
            last_meaningful = part
            break
    if not last_meaningful:
        last_meaningful = answer.strip().rstrip(".")
    # Clean up "Visit N:" prefix if present
    last_meaningful = re.sub(r'^Visit\s*\d+\s*:\s*', '', last_meaningful).strip()
    return last_meaningful


def _format_status(status: str) -> str:
    """Convert raw extracted status into a clean human-readable string.
    Handles both extraction format ('Use_status: current; Temporal_language: ongoing')
    and reconciled format ('current (ongoing)').
    """
    s = status.lower()
    # Determine use
    if "current" in s:
        use = "current"
    elif "previous" in s:
        use = "previous"
    else:
        return "unclear"
    # Determine temporal
    if "ongoing" in s or "backfilled" in s:
        temp = "ongoing"
    elif "dose_change" in s:
        temp = "dose changed"
    elif "initiation" in s:
        temp = "newly started"
    elif "stop" in s:
        temp = "stopped"
    else:
        temp = None
    if temp:
        return f"{use} ({temp})"
    return use


_MED_LEGEND = (
    "Status key: current (ongoing) = still being taken at prior visit; "
    "current (newly started) = just initiated at prior visit; "
    "current (dose changed) = dose was adjusted at prior visit; "
    "previous (stopped) = discontinued before prior visit."
)


def build_current_medications(visit_name: str, all_visits: dict) -> str:
    """Build medication history section based on visit logic."""
    visit_num = int(visit_name.split("_")[1])
    found_any = False

    if visit_num == 1:
        # Visit 1: show ALL drugs with any prior history, EXCEPT those first started
        # at this very visit (initiation). Strip temporal language — just list drug names.
        # The task is to decide continue/stop/add; leaking the outcome here would be cheating.
        lines = ["MEDICATION HISTORY (prior drug exposure — no outcome known):"]
        visit_feats = all_visits.get("Visit_1", {})
        for drug_feat in DRUG_FEATURE_NAMES:
            answer = visit_feats.get(drug_feat, {}).get("Answer", "")
            if not answer or "not mentioned" in answer.lower():
                continue
            status = _extract_status(answer)
            if not status or "initiation" in status.lower():
                continue
            drug_name = drug_feat.replace("drug_", "")
            lines.append(f"- {drug_name}: previously on")
            found_any = True
    else:
        # Visit 2/3: show all drugs from the previous visit with full status + legend
        lines = ["MEDICATION HISTORY (status as of prior visit):", _MED_LEGEND]
        prev_visit = f"Visit_{visit_num - 1}"
        prev_feats = all_visits.get(prev_visit, {})
        for drug_feat in DRUG_FEATURE_NAMES:
            answer = prev_feats.get(drug_feat, {}).get("Answer", "")
            if not answer or "not mentioned" in answer.lower():
                continue
            status = _extract_status(answer)
            if status:
                drug_name = drug_feat.replace("drug_", "")
                lines.append(f"- {drug_name}: {_format_status(status)}")
                found_any = True

    if not found_any:
        lines.append("No prior medication information available.")

    return "\n".join(lines)


def parse_top3(response: str) -> list:
    """Parse top-3 drug recommendations from response.
    Returns list of (rank, drug_name, reason) tuples.
    Handles markdown bold markers and various formatting variants.
    """
    # Strip markdown bold markers for easier matching
    cleaned = response.replace("**", "")
    results = []
    for rank in [1, 2, 3]:
        # Try strict format first: Rank 1: drug | reason: ...
        pattern = rf'Rank\s*{rank}\s*[:\.]\s*(\w+)\s*\|\s*[Rr]eason\s*:\s*(.*?)(?=Rank\s*\d|$)'
        match = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if not match:
            # Fallback: Rank 1: drug — reason or Rank 1: drug - reason
            pattern = rf'Rank\s*{rank}\s*[:\.]\s*(\w+)\s*[\-–—]\s*(.*?)(?=Rank\s*\d|$)'
            match = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if not match:
            # Fallback: Rank 1: drug\nreason (newline separated)
            pattern = rf'Rank\s*{rank}\s*[:\.]\s*(\w+)\s*\n(.*?)(?=Rank\s*\d|$)'
            match = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if not match:
            # Fallback: numbered list like "1. drug | reason:" or "1. drug: reason"
            pattern = rf'(?:^|\n)\s*{rank}\.\s*(\w+)\s*[\|:]\s*(.*?)(?=\n\s*\d\.|$)'
            match = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if match:
            drug = match.group(1).strip().lower()
            reason = match.group(2).strip()
            # Validate drug name
            if drug in DRUG_COLUMNS:
                results.append((rank, drug, reason))
            else:
                # Fuzzy: try prefix match
                candidates = [d for d in DRUG_COLUMNS if d.startswith(drug[:4])]
                if candidates:
                    results.append((rank, candidates[0], reason))
    return results


def parse_structured(thinking: str, content: str) -> dict:
    """Build structured dict from the separate thinking/content returned by Together."""
    clinical_reasoning = ""
    sec1_match = re.search(
        r'---\s*SECTION\s*1.*?---\s*(.*?)(?=---\s*SECTION\s*2|$)',
        content, re.DOTALL | re.IGNORECASE,
    )
    if sec1_match:
        clinical_reasoning = sec1_match.group(1).strip()

    top3 = parse_top3(content)
    drugs_ranked = {f"rank_{r}": d for r, d, _ in top3}
    reasons = {f"rank_{r}_reason": reason for r, _, reason in top3}

    return {"think": thinking, "reasoning": clinical_reasoning, "raw_content": content, **drugs_ranked, **reasons}


def build_raw_content(entry, visit_num: int, columns) -> str:
    """Build content from raw CSV (excluding drug/medication columns)."""
    def safe_get(col):
        if col in columns:
            val = entry[col]
            return val if pd.notna(val) else ""
        return ""

    if visit_num == 1:
        content = f"""Visit 1 (Initial Visit - 0 months):
Initial Visit Date: {safe_get('Date of visit(0 months)')}
Patient Age: {safe_get('Age')}
Patient Gender: {safe_get('Sex:')}
Seizure Diagnosis: {safe_get('Seizure Diagnosis')}
Medical History / History of Presenting Illness: {safe_get('History of Presenting Illness')}
Detailed Description of Seizure History: {safe_get('Detailed description of seizure history')}
Age of Onset of Seizure: {safe_get('Age of onset of seizure')}
Duration of Seizure: {safe_get('Duration of Seizure')}"""

    elif visit_num == 2:
        content = f"""Visit 1 (Initial Visit - 0 months):
Initial Visit Date: {safe_get('Date of visit(0 months)')}
Patient Age: {safe_get('Age')}
Patient Gender: {safe_get('Sex:')}
Seizure Diagnosis: {safe_get('Seizure Diagnosis')}
Medical History / History of Presenting Illness: {safe_get('History of Presenting Illness')}
Detailed Description of Seizure History: {safe_get('Detailed description of seizure history')}
Age of Onset of Seizure: {safe_get('Age of onset of seizure')}
Duration of Seizure: {safe_get('Duration of Seizure')}

Visit 2 (6 months follow-up):
6-Month Visit Date: {safe_get('Date of visit(6 months)')}
Additional Specifications: {safe_get('Specify')}

Note: Follow-up documentation may omit unchanged baseline information."""
    else:
        content = ""

    return content


def build_extracted_content(patient_id: str, visit_name: str, visit_features: dict, all_visits: dict) -> str:
    """Build content from extracted/reconciled JSON (non-drug features + medications)."""
    lines = [f"Patient ID: {patient_id}", f"Visit: {visit_name}", ""]

    visit_1_features = all_visits.get("Visit_1", {})

    lines.append("--- CLINICAL HISTORY ---")
    for feat in FEATURE_NAMES:
        if feat == "Age_Years":
            source_features = visit_1_features
        else:
            source_features = visit_features

        answer = source_features.get(feat, {}).get('Answer', 'Not available')
        reasoning = source_features.get(feat, {}).get('Reasoning', '')
        lines.append(f"{feat}: {answer} (Additional reasoning: {reasoning})")

    lines.append("")
    lines.append(build_current_medications(visit_name, all_visits))

    return "\n".join(lines)


def load_tasks_raw(csv_path: str, visit_num: int):
    """Load tasks from raw CSV."""
    df = pd.read_csv(
        csv_path,
        sep=';',
        engine='python',
        quotechar='"',
        doublequote=True,
        escapechar='\\'
    )

    dedup_cols = ['Name: ', 'Date of visit(0 months)', 'Date of visit(6 months)', 'Date of visit(12 months)']
    df = df.drop_duplicates(subset=dedup_cols)

    tasks = []
    for idx, row in df.iterrows():
        record_id = str(row.get("Record ID", "")).strip() if pd.notna(row.get("Record ID")) else ""
        name = str(row.get("Name: ", "")).strip() if pd.notna(row.get("Name: ")) else ""

        if record_id and name:
            patient_id = f"{record_id}_{name}"
        elif record_id:
            patient_id = record_id
        elif name:
            patient_id = name
        else:
            patient_id = f"patient_{idx}"

        content = build_raw_content(row, visit_num, df.columns)
        if content.strip():
            tasks.append((patient_id, content))

    return tasks


def load_tasks_extracted(json_path: str, visit_num: int):
    """Load tasks from reconciled JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    visit_name = f"Visit_{visit_num}"
    tasks = []

    for pid in data:
        visits = data.get(pid, {})
        visit_features = visits.get(visit_name, {})

        if not visit_features:
            continue

        content = build_extracted_content(pid, visit_name, visit_features, visits)
        tasks.append((pid, content))

    return tasks


# ---- Async Together AI Call — Raw HTTP with thinking tokens ----

async def call_together_with_retries(
    session: aiohttp.ClientSession,
    together_key: str,
    system_prompt: str,
    user_content: str,
    model: str,
    thinking_budget: int,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = 6,
) -> tuple[str, str]:
    """Async raw HTTP call with concurrency cap + exponential backoff.
    Returns (reasoning, content) tuple.
    """
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {together_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
    }

    async def _one_call():
        async with session.post(url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
        msg = data["choices"][0]["message"]
        reasoning = (msg.get("reasoning") or "").strip()
        content = (msg.get("content") or "").strip()
        return reasoning, content

    for attempt in range(max_retries):
        try:
            async with semaphore:
                return await _one_call()
        except Exception as e:
            msg = str(e).lower()
            retryable = any(s in msg for s in ["rate", "429", "timeout", "temporar", "overload", "503", "502", "500", "bad gateway", "connection"])
            if not retryable or attempt == max_retries - 1:
                print(f"[ERROR] API call failed: {e}")
                return "", ""
            base = 0.8 * (2 ** attempt)
            await asyncio.sleep(base + random.random() * 0.3)
    return "", ""


async def run_inference(tasks, system_prompt: str, together_key: str, model: str, thinking_budget: int, max_tokens: int, max_concurrency: int):
    """Run inference on all tasks. Returns dict {pid: (reasoning, content)}."""
    semaphore = asyncio.Semaphore(max_concurrency)
    results = {}

    pbar = tqdm(total=len(tasks), desc="Processing")

    async def _run_one(task, session):
        pid, user_content = task
        reasoning, content = await call_together_with_retries(
            session, together_key, system_prompt, user_content,
            model, thinking_budget, max_tokens, semaphore
        )
        return pid, reasoning, content

    async with aiohttp.ClientSession() as session:
        for coro in asyncio.as_completed([_run_one(t, session) for t in tasks]):
            pid, reasoning, content = await coro
            results[pid] = (reasoning, content)
            pbar.update(1)

    pbar.close()
    return results


async def main_async():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get Together API key
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        print("Error: TOGETHER_API_KEY not found in .env file")
        return

    # Output files
    short_name = get_short_model_name(model)
    mode_str = "raw" if input_mode == "raw" else "ext"
    output_file = os.path.join(script_dir, f"drug/all_drugs/{short_name}_v{visit_num}_{mode_str}_top3.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load system prompt
    prompt_path = os.path.join(script_dir, prompt_file)
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    print(f"\n{'='*60}")
    print(f"TOGETHER AI TOP-3 DRUG PREDICTION v2 (with thinking)")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Visit: {visit_num}")
    print(f"Mode:  {input_mode}")
    print(f"Thinking budget: {thinking_budget} tokens")
    print(f"Concurrency: {max_concurrency}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    # Load tasks based on mode
    print("Loading tasks...")
    if input_mode == "raw":
        print(f"  CSV: {csv_path}")
        tasks = load_tasks_raw(csv_path, visit_num)
    else:
        print(f"  JSON: {reconciled_json_path}")
        tasks = load_tasks_extracted(reconciled_json_path, visit_num)

    if limit_patients:
        tasks = tasks[:limit_patients]
        print(f"  Limited to {limit_patients} patients")

    print(f"Total patients: {len(tasks)}\n")

    # Run inference
    results = await run_inference(tasks, system_prompt, together_key, model, thinking_budget, max_tokens, max_concurrency)

    # Build structured JSON and save (maintain original task order)
    ordered_structured = {}
    csv_file = output_file.replace('.json', '.csv')
    parsed_data = []
    n_failed = 0
    for pid, _ in tasks:
        thinking, content = results.get(pid, ("", ""))
        ordered_structured[pid] = parse_structured(thinking, content)

        top3 = parse_top3(content)
        row = {'patient_id': pid}
        for rank in [1, 2, 3]:
            matches = [t for t in top3 if t[0] == rank]
            if matches:
                _, drug, reason = matches[0]
                row[f'rank_{rank}'] = drug
                row[f'rank_{rank}_reason'] = reason
            else:
                row[f'rank_{rank}'] = None
                row[f'rank_{rank}_reason'] = None
        parsed_data.append(row)
        if not top3:
            n_failed += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ordered_structured, f, indent=2, ensure_ascii=False)

    csv_cols = ['patient_id', 'rank_1', 'rank_2', 'rank_3',
                'rank_1_reason', 'rank_2_reason', 'rank_3_reason']
    df_out = pd.DataFrame(parsed_data, columns=csv_cols)
    df_out.to_csv(csv_file, index=False)

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"{'='*60}")
    print(f"Saved {len(results)} results:")
    print(f"  JSON (with thinking): {output_file}")
    print(f"  CSV (top-3 drugs): {csv_file}")
    if n_failed > 0:
        print(f"  Warning: {n_failed} patients had no drugs parsed")
    print(f"{'='*60}\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()