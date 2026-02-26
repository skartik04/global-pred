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

# Set to True to use the trial (smaller) reconciled JSON, False for full
use_trial_json = False

# Input paths
csv_path = os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'global', 'data', 'combined_dataset.csv')
reconciled_json_path = os.path.join(
    _HERE,
    'csv_reconciled_gpt_oss_trial.json' if use_trial_json else 'csv_reconciled_gpt_oss.json'
)

# Prompt file
prompt_file = 'all_drugs.txt'

# Processing settings
thinking_budget = 6000
max_tokens = 4096
max_concurrency = 8
limit_patients = 400   # Set to int to limit, None for all

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

ALLOWED_ACTIONS = {"continue", "start", "stop"}

_STATUS_LABELS = {
    'current_ongoing':      'current (on arrival)',
    'current_new':          'current (on arrival)',
    'current_dose_changed': 'current (on arrival)',
    'previous_stopped':     'previously tried',
}

_MED_LEGEND = (
    "Status key: current (ongoing) = still being taken at prior visit; "
    "current (newly started) = just initiated at prior visit; "
    "current (dose changed) = dose was adjusted at prior visit; "
    "previous (stopped) = discontinued before prior visit."
)


def get_short_model_name(model_name: str) -> str:
    """Extract a short name from the model for filenames."""
    name = model_name.replace("/", "_").replace("-", "").replace(".", "")
    if len(name) > 20:
        name = name[:20]
    return name


def _last_visit_portion(answer: str) -> str:
    """If answer has multi-visit 'Visit N: ...' format, return only the last segment."""
    parts = re.split(r'(?=\bVisit\s*\d+\s*:)', answer.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        last = re.sub(r'^Visit\s*\d+\s*:\s*', '', parts[-1]).strip().rstrip('.')
        return last
    return answer.strip().rstrip('.')


def _specific_visit_portion(answer: str, visit_n: int) -> str:
    """Extract the Visit N portion from a multi-visit answer string. Returns '' if not found."""
    parts = re.split(r'(?=\bVisit\s*\d+\s*:)', answer.strip())
    for part in parts:
        part = part.strip()
        m = re.match(r'Visit\s*(\d+)\s*:\s*(.*)', part, re.DOTALL)
        if m and int(m.group(1)) == visit_n:
            return m.group(2).strip().rstrip('.')
    return ''


def _parse_drug_answer(answer: str) -> tuple:
    """
    Parse a drug answer string into a semantic status category.

    Returns (status, is_first_time_start) where:
      status            : 'current_ongoing', 'current_new', 'current_dose_changed',
                          'previous_stopped', or None (not mentioned / not relevant)
      is_first_time_start: True if brand-new prescription with no prior exposure
    """
    if not answer or answer.strip().lower() in ('not mentioned.', 'not mentioned', ''):
        return None, False

    portion = _last_visit_portion(answer)
    s = portion.lower()

    # Clearly irrelevant / no-data values
    if s in ('no new information',) or re.match(r'^no_prior_exposure,?\s*no_action$', s):
        return None, False

    # ── Structured format: Exposure_before: X; Action_taken: Y ──
    if 'exposure_before:' in s:
        exp_m = re.search(r'exposure_before:\s*(\w+)', s)
        act_m = re.search(r'action_taken:\s*(\w+)', s)
        exposure = exp_m.group(1) if exp_m else ''
        action   = act_m.group(1) if act_m else ''

        if exposure == 'no_prior_exposure':
            if action == 'start':
                return 'current_new', True   # brand-new prescription
            return None, False               # no prior, no action → irrelevant

        if action == 'stop':
            return 'previous_stopped', False
        if exposure == 'past_tried' and action == 'no_action':
            return 'previous_stopped', False
        if action == 'start':                # past_tried + start = re-initiated
            return 'current_new', False
        if action in ('continue', 'no_action'):
            if 'dose increased' in s or 'dose reduced' in s or 'dose_change' in s:
                return 'current_dose_changed', False
            return 'current_ongoing', False
        return 'current_ongoing', False      # structured fallback

    # ── Simple shorthand: "on_arrival, continue" / "no_prior_exposure, start" ──
    shorthand = re.match(
        r'^(on_arrival|past_tried|no_prior_exposure),?\s*(continue|start|stop|no_action)',
        s
    )
    if shorthand:
        is_no_prior = shorthand.group(1) == 'no_prior_exposure'
        action = shorthand.group(2)
        if action == 'stop' or (shorthand.group(1) == 'past_tried' and action == 'no_action'):
            return 'previous_stopped', False
        if action == 'start':
            return 'current_new', is_no_prior
        if action == 'continue':
            if 'dose increased' in s or 'dose reduced' in s:
                return 'current_dose_changed', False
            return 'current_ongoing', False
        return 'current_ongoing', False

    # ── Free-text patterns ──
    if any(kw in s for kw in ['remains stopped', 'no current use', 'past trial of']):
        return 'previous_stopped', False
    if 'no new information' in s and 'past_tried' in s:
        return 'previous_stopped', False

    has_stop     = bool(re.search(r'\bstop\b', s))
    has_start    = any(kw in s for kw in ['started', 'start', 'initiated', 'restarted'])
    has_continue = any(kw in s for kw in ['continued', 'continue'])
    has_dose_chg = 'dose increased' in s or 'dose reduced' in s or 'dose increase' in s
    is_no_prior  = 'no prior exposure' in s or 'no_prior_exposure' in s

    if has_stop and not has_continue:
        return 'previous_stopped', False
    if has_start and not has_continue:
        return 'current_new', is_no_prior
    if has_dose_chg:
        return 'current_dose_changed', False
    if has_continue or 'on arrival' in s or 'on_arrival' in s:
        return 'current_ongoing', False

    return 'current_ongoing', False          # default fallback


def build_current_medications(visit_name: str, all_visits: dict) -> str:
    """Build medication history section based on visit logic."""
    vnum = int(visit_name.split("_")[1])
    found_any = False

    if vnum == 1:
        # Visit 1: show drugs the patient had prior history with.
        # Exclude brand-new prescriptions being started for the first time here.
        # Distinguish: currently on (on_arrival) vs previously tried (past_tried, stopped).
        lines = ["MEDICATION HISTORY (prior drug exposure — no outcome known):"]
        visit_feats = all_visits.get("Visit_1", {})
        for drug_feat in DRUG_FEATURE_NAMES:
            answer = visit_feats.get(drug_feat, {}).get("Answer", "")
            if not answer or "not mentioned" in answer.lower():
                continue
            status, is_first_time = _parse_drug_answer(answer)
            if status is None or is_first_time:
                continue
            drug_name = drug_feat.replace("drug_", "")
            if status in ('current_ongoing', 'current_dose_changed'):
                label = "current (on arrival)"
            else:
                label = "previously tried"
            lines.append(f"- {drug_name}: {label}")
            found_any = True
    else:
        # Visit 2/3: show all drugs from the previous visit with full status + legend
        lines = [f"MEDICATION HISTORY (as of Visit {vnum - 1}):"]
        prev_visit = f"Visit_{vnum - 1}"
        prev_feats = all_visits.get(prev_visit, {})
        curr_feats = all_visits.get(visit_name, {})
        for drug_feat in DRUG_FEATURE_NAMES:
            answer = prev_feats.get(drug_feat, {}).get("Answer", "")
            status = None
            if answer and "not mentioned" not in answer.lower():
                status, _ = _parse_drug_answer(answer)
            # Fallback: if prev visit had no useful data, check current visit's
            # cumulative answer for the Visit_{N-1} portion
            if status is None:
                curr_answer = curr_feats.get(drug_feat, {}).get("Answer", "")
                if curr_answer:
                    portion = _specific_visit_portion(curr_answer, vnum - 1)
                    if portion and "not mentioned" not in portion.lower():
                        status, _ = _parse_drug_answer(portion)
            if status is None:
                continue
            drug_name = drug_feat.replace("drug_", "")
            lines.append(f"- {drug_name}: {_STATUS_LABELS[status]}")
            found_any = True

    if not found_any:
        lines.append("No prior medication information available.")
    else:
        lines.insert(1, "Status key: current (on arrival) = patient is actively taking this drug coming into this visit; previously tried = drug was used before but is NOT currently active (already stopped).")

    return "\n".join(lines)


def parse_options(content: str) -> dict:
    """Parse 3 regimen options from response."""
    sec2_match = re.search(
        r'---\s*SECTION\s*2.*?---\s*(.*?)$',
        content, re.DOTALL | re.IGNORECASE,
    )
    section_text = sec2_match.group(1).strip() if sec2_match else content

    options = {}
    block_pattern = re.compile(
        r'Option\s+(\d)\s*:\s*(.+?)(?=Option\s+\d\s*:|$)',
        re.DOTALL | re.IGNORECASE,
    )

    for m in block_pattern.finditer(section_text):
        num = int(m.group(1))
        if num not in (1, 2, 3):
            continue

        block_lines = m.group(2).strip().split('\n')
        label = block_lines[0].strip() if block_lines else ""
        drugs = []
        rationale_parts = []
        in_rationale = False

        for line in block_lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            rat_match = re.match(r'Rationale\s*:\s*(.*)', stripped, re.IGNORECASE)
            if rat_match:
                in_rationale = True
                rationale_parts.append(rat_match.group(1).strip())
                continue
            if in_rationale:
                rationale_parts.append(stripped)
                continue
            drug_match = re.match(r'-\s*(\w+)\s*:\s*(\w+)', stripped)
            if drug_match:
                drug_name = drug_match.group(1).lower()
                action = drug_match.group(2).lower()
                if drug_name in DRUG_COLUMNS and action in ALLOWED_ACTIONS:
                    drugs.append({"drug": drug_name, "action": action})

        options[f"option_{num}"] = {
            "label": label,
            "drugs": drugs,
            "rationale": " ".join(rationale_parts),
        }

    return options


def parse_structured(thinking: str, content: str) -> dict:
    """Build structured dict from the separate thinking/content returned by Together."""
    clinical_reasoning = ""
    sec1_match = re.search(
        r'---\s*SECTION\s*1.*?---\s*(.*?)(?=---\s*SECTION\s*2|$)',
        content, re.DOTALL | re.IGNORECASE,
    )
    if sec1_match:
        clinical_reasoning = sec1_match.group(1).strip()

    options = parse_options(content)
    return {"think": thinking, "reasoning": clinical_reasoning, "raw_content": content, **options}


def format_drugs_str(drugs: list) -> str:
    """Format list of drug dicts as semicolon-joined string: drug:action; ..."""
    return "; ".join(f"{d['drug']}:{d['action']}" for d in drugs)


def build_raw_content(entry, vnum: int, columns) -> str:
    """Build content from raw CSV (excluding drug/medication columns)."""
    def safe_get(col):
        if col in columns:
            val = entry[col]
            return val if pd.notna(val) else ""
        return ""

    if vnum == 1:
        content = f"""Visit 1 (Initial Visit - 0 months):
Initial Visit Date: {safe_get('Date of visit(0 months)')}
Patient Age: {safe_get('Age')}
Patient Gender: {safe_get('Sex:')}
Seizure Diagnosis: {safe_get('Seizure Diagnosis')}
Medical History / History of Presenting Illness: {safe_get('History of Presenting Illness')}
Detailed Description of Seizure History: {safe_get('Detailed description of seizure history')}
Age of Onset of Seizure: {safe_get('Age of onset of seizure')}
Duration of Seizure: {safe_get('Duration of Seizure')}"""

    elif vnum == 2:
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
        source_features = visit_1_features if feat == "Age_Years" else visit_features
        answer = source_features.get(feat, {}).get('Answer', 'Not available')
        lines.append(f"{feat}: {answer}")

    lines.append("")
    lines.append(build_current_medications(visit_name, all_visits))

    return "\n".join(lines)


def load_tasks_raw(csv_path: str, vnum: int):
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

        content = build_raw_content(row, vnum, df.columns)
        if content.strip():
            tasks.append((patient_id, content))

    return tasks


def load_tasks_extracted(json_path: str, vnum: int):
    """Load tasks from reconciled JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    visit_name = f"Visit_{vnum}"
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
) -> tuple:
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
    output_file = os.path.join(script_dir, f"drug/all_drugs/{short_name}_v{visit_num}_{mode_str}_options.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load system prompt
    prompt_path = os.path.join(script_dir, prompt_file)
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    print(f"\n{'='*60}")
    print(f"ALL-DRUG REGIMEN OPTIONS PREDICTION (with thinking)")
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
        structured = parse_structured(thinking, content)
        ordered_structured[pid] = structured

        row = {'patient_id': pid}
        for opt_num in [1, 2, 3]:
            key = f"option_{opt_num}"
            opt = structured.get(key, {})
            row[f"{key}_label"] = opt.get("label", "")
            row[f"{key}_drugs"] = format_drugs_str(opt.get("drugs", []))
            row[f"{key}_rationale"] = opt.get("rationale", "")
        parsed_data.append(row)

        if not any(structured.get(f"option_{n}") for n in [1, 2, 3]):
            n_failed += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ordered_structured, f, indent=2, ensure_ascii=False)

    csv_cols = [
        'patient_id',
        'option_1_label', 'option_1_drugs', 'option_1_rationale',
        'option_2_label', 'option_2_drugs', 'option_2_rationale',
        'option_3_label', 'option_3_drugs', 'option_3_rationale',
    ]
    df_out = pd.DataFrame(parsed_data, columns=csv_cols)
    df_out.to_csv(csv_file, index=False)

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"{'='*60}")
    print(f"Saved {len(results)} results:")
    print(f"  JSON (with thinking): {output_file}")
    print(f"  CSV (regimen options): {csv_file}")
    if n_failed > 0:
        print(f"  Warning: {n_failed} patients had no options parsed")
    print(f"{'='*60}\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
