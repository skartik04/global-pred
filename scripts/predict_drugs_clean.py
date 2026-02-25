"""
Step 3: Leak-free drug prediction with prior plans as medication history.

For Visit N prediction, the model sees:
  - Visit N's extracted features (clinical picture from Step 2)
  - Visit N's extracted drug observations (what patient is currently on)
  - Plans from V1..V(N-1) carried forward as medication history (from ground_truth.json)

No data leakage: the model never sees the current visit's plan.

Input:  extracted_features.json (Step 2), ground_truth.json (Step 1b)
Output: drug/clean/<model>_v<visit>_clean_top3.{json,csv}
"""

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
model = 'openai/gpt-oss-120b'
visit_num = 3

extracted_features_path = os.path.join(_HERE, 'extracted_features.json')
ground_truth_path = os.path.join(_HERE, 'ground_truth.json')
prompt_file = os.path.join(_HERE, 'predict_prompt_clean.txt')

thinking_budget = 6000
max_tokens = 4096
max_concurrency = 8
limit_patients = 5  # Set to int for testing, None for all
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

FEATURE_NAMES = [
    "Age_Years", "Sex_Female", "OnsetTimingYears", "SeizureFreq",
    "StatusOrProlonged", "CognitivePriority", "Risk_Factors", "SeizureType",
]


def get_short_model_name(model_name: str) -> str:
    name = model_name.replace("/", "_").replace("-", "").replace(".", "")
    if len(name) > 20:
        name = name[:20]
    return name


def _parse_drug_status(answer: str) -> str | None:
    """Parse drug Answer field from new simplified format: 'Status: on_arrival|past_tried'."""
    if not answer or answer.strip().lower() in ("not mentioned.", "not mentioned", ""):
        return None
    s = answer.lower()
    if "on_arrival" in s:
        return "on_arrival"
    elif "past_tried" in s:
        return "past_tried"
    # Fallback: try old format keywords
    elif "current" in s:
        return "on_arrival"
    elif "previous" in s:
        return "past_tried"
    return None


def build_medication_history(
    visit_num: int,
    visit_features: dict,
    all_visits_features: dict,
    ground_truth_patient: dict,
) -> str:
    """Build medication history from three sources:
    (a) Current visit's extracted drug observations (what patient is on)
    (b) Prior visits' documented plans (from ground truth)
    (c) Prior visits' extracted drug observations (what patient was on at earlier visits)
    """
    lines = ["MEDICATION HISTORY:"]

    # Source (a): Current observations from extracted features
    obs_lines = []
    for drug_feat in DRUG_FEATURE_NAMES:
        answer = visit_features.get(drug_feat, {}).get("Answer", "")
        status = _parse_drug_status(answer)
        if status:
            drug_name = drug_feat.replace("drug_", "")
            if status == "on_arrival":
                obs_lines.append(f"  - {drug_name}: currently taking (observed)")
            elif status == "past_tried":
                obs_lines.append(f"  - {drug_name}: previously tried (observed)")

    if obs_lines:
        lines.append("Current observations (what patient is on/was on at this visit):")
        lines.extend(obs_lines)
    else:
        lines.append("Current observations: No medications observed at this visit.")

    # Source (b): Prior documented plans
    if visit_num >= 2:
        plan_lines = []
        for prev_num in range(1, visit_num):
            prev_name = f"Visit_{prev_num}"
            prev_gt = ground_truth_patient.get(prev_name, {})
            prescribed = prev_gt.get("prescribed_drugs", [])
            stopped = prev_gt.get("stopped_drugs", [])

            if prescribed or stopped:
                parts = []
                if prescribed:
                    parts.append(f"prescribed: {', '.join(prescribed)}")
                if stopped:
                    parts.append(f"stopped: {', '.join(stopped)}")
                plan_lines.append(f"  - Visit {prev_num} plan: {'; '.join(parts)}")

        if plan_lines:
            lines.append("")
            lines.append("Prior documented plans (what was prescribed at previous visits):")
            lines.extend(plan_lines)
        else:
            lines.append("")
            lines.append("Prior documented plans: No prior plans documented.")

        # Source (c): Prior visits' drug observations (drugs seen at earlier visits
        # that aren't already covered by current observations or prior plans)
        current_drugs = set()
        for drug_feat in DRUG_FEATURE_NAMES:
            answer = visit_features.get(drug_feat, {}).get("Answer", "")
            if _parse_drug_status(answer):
                current_drugs.add(drug_feat.replace("drug_", ""))

        plan_drugs = set()
        for prev_num in range(1, visit_num):
            prev_gt = ground_truth_patient.get(f"Visit_{prev_num}", {})
            plan_drugs.update(prev_gt.get("prescribed_drugs", []))
            plan_drugs.update(prev_gt.get("stopped_drugs", []))

        prior_obs_lines = []
        for prev_num in range(1, visit_num):
            prev_name = f"Visit_{prev_num}"
            prev_feats = all_visits_features.get(prev_name, {})
            for drug_feat in DRUG_FEATURE_NAMES:
                answer = prev_feats.get(drug_feat, {}).get("Answer", "")
                status = _parse_drug_status(answer)
                if status:
                    drug_name = drug_feat.replace("drug_", "")
                    if drug_name not in current_drugs and drug_name not in plan_drugs:
                        label = "was taking" if status == "on_arrival" else "had previously tried"
                        prior_obs_lines.append(
                            f"  - {drug_name}: {label} at Visit {prev_num}")
                        # Add to seen sets so we don't repeat
                        plan_drugs.add(drug_name)

        if prior_obs_lines:
            lines.append("")
            lines.append("Prior drug exposure (observed at earlier visits, not in any plan):")
            lines.extend(prior_obs_lines)
    else:
        # Visit 1: no prior plans
        lines.append("")
        lines.append("Prior documented plans: None (this is the initial visit).")

    return "\n".join(lines)


def build_prediction_content(
    patient_id: str,
    visit_name: str,
    visit_features: dict,
    all_visits_features: dict,
    ground_truth_patient: dict,
) -> str:
    """Build user content for drug prediction."""
    visit_num_int = int(visit_name.split("_")[1])
    lines = [f"Patient ID: {patient_id}", f"Visit: {visit_name}", ""]

    # Clinical features (use Visit 1 for Age, current visit for rest)
    visit_1_features = all_visits_features.get("Visit_1", {})

    lines.append("--- CLINICAL HISTORY ---")
    for feat in FEATURE_NAMES:
        if feat == "Age_Years":
            source = visit_1_features
        else:
            source = visit_features

        answer = source.get(feat, {}).get('Answer', 'Not available')
        reasoning = source.get(feat, {}).get('Reasoning', '')
        lines.append(f"{feat}: {answer} (Additional reasoning: {reasoning})")

    # Medication history
    lines.append("")
    lines.append(build_medication_history(
        visit_num_int, visit_features, all_visits_features, ground_truth_patient
    ))

    return "\n".join(lines)


def parse_top3(response: str) -> list:
    """Parse top-3 drug recommendations from response."""
    cleaned = response.replace("**", "")
    results = []
    for rank in [1, 2, 3]:
        pattern = rf'Rank\s*{rank}\s*[:\.]\s*(\w+)\s*\|\s*[Rr]eason\s*:\s*(.*?)(?=Rank\s*\d|$)'
        match = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if not match:
            pattern = rf'Rank\s*{rank}\s*[:\.]\s*(\w+)\s*[\-–—]\s*(.*?)(?=Rank\s*\d|$)'
            match = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if not match:
            pattern = rf'Rank\s*{rank}\s*[:\.]\s*(\w+)\s*\n(.*?)(?=Rank\s*\d|$)'
            match = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if not match:
            pattern = rf'(?:^|\n)\s*{rank}\.\s*(\w+)\s*[\|:]\s*(.*?)(?=\n\s*\d\.|$)'
            match = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if match:
            drug = match.group(1).strip().lower()
            reason = match.group(2).strip()
            if drug in DRUG_COLUMNS:
                results.append((rank, drug, reason))
            else:
                candidates = [d for d in DRUG_COLUMNS if d.startswith(drug[:4])]
                if candidates:
                    results.append((rank, candidates[0], reason))
    return results


def parse_structured(thinking: str, content: str) -> dict:
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

    return {"think": thinking, "reasoning": clinical_reasoning,
            "raw_content": content, **drugs_ranked, **reasons}


def load_tasks(
    extracted_features: dict,
    ground_truth: dict,
    visit_num: int,
) -> list:
    """Build prediction tasks for the given visit."""
    visit_name = f"Visit_{visit_num}"
    tasks = []

    for pid in extracted_features:
        all_visits = extracted_features[pid]
        visit_features = all_visits.get(visit_name, {})

        if not visit_features:
            continue

        gt_patient = ground_truth.get(pid, {})

        content = build_prediction_content(
            pid, visit_name, visit_features, all_visits, gt_patient
        )
        tasks.append((pid, content))

    return tasks


# --- Async Together AI Call (raw HTTP with thinking tokens) ---

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
            retryable = any(s in msg for s in [
                "rate", "429", "timeout", "temporar", "overload",
                "503", "502", "500", "bad gateway", "connection"
            ])
            if not retryable or attempt == max_retries - 1:
                print(f"[ERROR] API call failed: {e}")
                return "", ""
            base = 0.8 * (2 ** attempt)
            await asyncio.sleep(base + random.random() * 0.3)
    return "", ""


async def run_inference(tasks, system_prompt, together_key, model,
                        thinking_budget, max_tokens, max_concurrency):
    semaphore = asyncio.Semaphore(max_concurrency)
    results = {}

    pbar = tqdm(total=len(tasks), desc="Predicting")

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
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        print("Error: TOGETHER_API_KEY not found")
        return

    short_name = get_short_model_name(model)
    output_dir = os.path.join(_HERE, "drug", "clean")
    os.makedirs(output_dir, exist_ok=True)
    json_output = os.path.join(output_dir, f"{short_name}_v{visit_num}_clean_top3.json")
    csv_output = json_output.replace('.json', '.csv')

    # Load system prompt
    with open(prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    print(f"\n{'='*60}")
    print(f"STEP 3: LEAK-FREE DRUG PREDICTION (with thinking)")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Visit: {visit_num}")
    print(f"Thinking budget: {thinking_budget} tokens")
    print(f"Concurrency: {max_concurrency}")
    print(f"Output: {json_output}")
    print(f"{'='*60}\n")

    # Load inputs
    print("Loading inputs...")
    with open(extracted_features_path, 'r', encoding='utf-8') as f:
        extracted_features = json.load(f)
    print(f"  Extracted features: {len(extracted_features)} patients")

    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    print(f"  Ground truth: {len(ground_truth)} patients")

    # Build tasks
    tasks = load_tasks(extracted_features, ground_truth, visit_num)
    if limit_patients:
        tasks = tasks[:limit_patients]
        print(f"  Limited to {limit_patients} patients")
    print(f"Total prediction tasks: {len(tasks)}\n")

    # Run inference
    results = await run_inference(
        tasks, system_prompt, together_key, model,
        thinking_budget, max_tokens, max_concurrency
    )

    # Build structured output (maintain task order)
    ordered_structured = {}
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

    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(ordered_structured, f, indent=2, ensure_ascii=False)

    csv_cols = ['patient_id', 'rank_1', 'rank_2', 'rank_3',
                'rank_1_reason', 'rank_2_reason', 'rank_3_reason']
    df_out = pd.DataFrame(parsed_data, columns=csv_cols)
    df_out.to_csv(csv_output, index=False)

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"{'='*60}")
    print(f"Saved {len(results)} results:")
    print(f"  JSON (with thinking): {json_output}")
    print(f"  CSV (top-3 drugs): {csv_output}")
    if n_failed > 0:
        print(f"  Warning: {n_failed} patients had no drugs parsed")
    print(f"{'='*60}\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
