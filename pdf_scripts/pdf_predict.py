"""
Drug prediction for PDF pipeline.
Reads pdf_reconc.json, runs all visits for all patients, saves to pdf_scripts/drug/.
"""
import os
import re
import json
import asyncio
import random
import aiohttp
from tqdm import tqdm
import dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
dotenv.load_dotenv(os.path.join(_ROOT, "scripts", ".env"))

# ============ CONFIGURATION ============

model = "openai/gpt-oss-120b"

prompt_file = os.path.join(_ROOT, "scripts", "all_drugs.txt")

thinking_budget = 6000
max_tokens      = 16000
max_concurrency = 8
limit_patients  = None   # Set to int to limit, None for all

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

ALLOWED_ACTIONS = {"continue", "start", "stop"}

_STATUS_LABELS = {
    "current_ongoing":      "current (on arrival)",
    "current_new":          "current (on arrival)",
    "current_dose_changed": "current (on arrival)",
    "previous_stopped":     "previously tried",
}


def get_provider(model_name: str) -> str:
    m = model_name.lower()
    if m.startswith("claude"):
        return "anthropic"
    if any(m.startswith(p) for p in ("gpt-", "o1", "o3", "o4", "text-")):
        return "openai"
    return "together"


def get_short_model_name(model_name: str) -> str:
    name = model_name.replace("/", "_").replace("-", "").replace(".", "")
    return name[:20] if len(name) > 20 else name


def _last_visit_portion(answer: str) -> str:
    parts = re.split(r"(?=\bVisit\s*\d+\s*:)", answer.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        last = re.sub(r"^Visit\s*\d+\s*:\s*", "", parts[-1]).strip().rstrip(".")
        return last
    return answer.strip().rstrip(".")


def _specific_visit_portion(answer: str, visit_n: int) -> str:
    parts = re.split(r"(?=\bVisit\s*\d+\s*:)", answer.strip())
    for part in parts:
        part = part.strip()
        m = re.match(r"Visit\s*(\d+)\s*:\s*(.*)", part, re.DOTALL)
        if m and int(m.group(1)) == visit_n:
            return m.group(2).strip().rstrip(".")
    return ""


def _parse_drug_answer(answer: str) -> tuple:
    if not answer or answer.strip().lower() in ("not mentioned.", "not mentioned", ""):
        return None, False

    portion = _last_visit_portion(answer)
    s = portion.lower()

    if s in ("no new information",) or re.match(r"^no_prior_exposure,?\s*no_action$", s):
        return None, False

    if "exposure_before:" in s:
        exp_m = re.search(r"exposure_before:\s*(\w+)", s)
        act_m = re.search(r"action_taken:\s*(\w+)", s)
        exposure = exp_m.group(1) if exp_m else ""
        action   = act_m.group(1) if act_m else ""

        if exposure == "no_prior_exposure":
            if action == "start":
                return "current_new", True
            return None, False

        if action == "stop":
            return "previous_stopped", False
        if exposure == "past_tried" and action == "no_action":
            return "previous_stopped", False
        if action == "start":
            return "current_new", False
        if action in ("continue", "no_action"):
            if "dose increased" in s or "dose reduced" in s or "dose_change" in s:
                return "current_dose_changed", False
            return "current_ongoing", False
        return "current_ongoing", False

    shorthand = re.match(
        r"^(on_arrival|past_tried|no_prior_exposure),?\s*(continue|start|stop|no_action)",
        s,
    )
    if shorthand:
        is_no_prior = shorthand.group(1) == "no_prior_exposure"
        action = shorthand.group(2)
        if action == "stop" or (shorthand.group(1) == "past_tried" and action == "no_action"):
            return "previous_stopped", False
        if action == "start":
            return "current_new", is_no_prior
        if action == "continue":
            if "dose increased" in s or "dose reduced" in s:
                return "current_dose_changed", False
            return "current_ongoing", False
        return "current_ongoing", False

    if any(kw in s for kw in ["remains stopped", "no current use", "past trial of"]):
        return "previous_stopped", False
    if "no new information" in s and "past_tried" in s:
        return "previous_stopped", False

    has_stop     = bool(re.search(r"\bstop\b", s))
    has_start    = any(kw in s for kw in ["started", "start", "initiated", "restarted"])
    has_continue = any(kw in s for kw in ["continued", "continue"])
    has_dose_chg = "dose increased" in s or "dose reduced" in s or "dose increase" in s
    is_no_prior  = "no prior exposure" in s or "no_prior_exposure" in s

    if has_stop and not has_continue:
        return "previous_stopped", False
    if has_start and not has_continue:
        return "current_new", is_no_prior
    if has_dose_chg:
        return "current_dose_changed", False
    if has_continue or "on arrival" in s or "on_arrival" in s:
        return "current_ongoing", False

    return "current_ongoing", False


def build_current_medications(visit_name: str, all_visits: dict) -> str:
    vnum = int(visit_name.split("_")[1])
    found_any = False

    if vnum == 1:
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
            label = "current (on arrival)" if status in ("current_ongoing", "current_dose_changed") else "previously tried"
            lines.append(f"- {drug_name}: {label}")
            found_any = True
    else:
        lines = [f"MEDICATION HISTORY (as of Visit {vnum - 1}):"]
        prev_visit = f"Visit_{vnum - 1}"
        prev_feats = all_visits.get(prev_visit, {})
        curr_feats = all_visits.get(visit_name, {})
        for drug_feat in DRUG_FEATURE_NAMES:
            answer = prev_feats.get(drug_feat, {}).get("Answer", "")
            status = None
            if answer and "not mentioned" not in answer.lower():
                status, _ = _parse_drug_answer(answer)
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


def build_extracted_content(patient_id: str, visit_name: str, visit_features: dict, all_visits: dict) -> str:
    lines = [f"Patient ID: {patient_id}", f"Visit: {visit_name}", ""]
    visit_1_features = all_visits.get("Visit_1", {})

    lines.append("--- CLINICAL HISTORY ---")
    for feat in FEATURE_NAMES:
        source_features = visit_1_features if feat == "Age_Years" else visit_features
        answer = source_features.get(feat, {}).get("Answer", "Not available")
        lines.append(f"{feat}: {answer}")

    lines.append("")
    lines.append(build_current_medications(visit_name, all_visits))
    return "\n".join(lines)


def load_all_tasks(json_path: str):
    """Load all (patient, visit) tasks from pdf_reconc.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    for pid in data:
        visits = data[pid]
        # Collect all Visit_N keys in order
        visit_keys = sorted(
            [k for k in visits if re.match(r"^Visit_\d+$", k)],
            key=lambda k: int(k.split("_")[1]),
        )
        for vk in visit_keys:
            vfeats = visits.get(vk, {})
            if not vfeats:
                continue
            content = build_extracted_content(pid, vk, vfeats, visits)
            tasks.append((pid, vk, content))

    return tasks, data


def parse_options(content: str) -> dict:
    sec2_match = re.search(
        r"---\s*SECTION\s*2.*?---\s*(.*?)$",
        content, re.DOTALL | re.IGNORECASE,
    )
    section_text = sec2_match.group(1).strip() if sec2_match else content

    options = {}
    block_pattern = re.compile(
        r"Option\s+(\d)\s*:\s*(.+?)(?=Option\s+\d\s*:|$)",
        re.DOTALL | re.IGNORECASE,
    )
    for m in block_pattern.finditer(section_text):
        num = int(m.group(1))
        if num not in (1, 2, 3):
            continue
        block_lines = m.group(2).strip().split("\n")
        label = block_lines[0].strip() if block_lines else ""
        drugs, rationale_parts = [], []
        in_rationale = False
        for line in block_lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            rat_match = re.match(r"Rationale\s*:\s*(.*)", stripped, re.IGNORECASE)
            if rat_match:
                in_rationale = True
                rationale_parts.append(rat_match.group(1).strip())
                continue
            if in_rationale:
                rationale_parts.append(stripped)
                continue
            drug_match = re.match(r"-\s*(\w+)\s*:\s*(\w+)", stripped)
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
    clinical_reasoning = ""
    sec1_match = re.search(
        r"---\s*SECTION\s*1.*?---\s*(.*?)(?=---\s*SECTION\s*2|$)",
        content, re.DOTALL | re.IGNORECASE,
    )
    if sec1_match:
        clinical_reasoning = sec1_match.group(1).strip()
    options = parse_options(content)
    return {"think": thinking, "reasoning": clinical_reasoning, "raw_content": content, **options}


def format_drugs_str(drugs: list) -> str:
    return "; ".join(f"{d['drug']}:{d['action']}" for d in drugs)


async def call_api(session, api_key, system_prompt, user_content, model, thinking_budget, max_tokens, semaphore, max_retries=6):
    provider = get_provider(model)

    async def _one_call():
        if provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}],
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
            }
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"{resp.status}: {await resp.text()}")
                data = await resp.json()
            reasoning, content = "", ""
            for block in data.get("content", []):
                if block.get("type") == "thinking":
                    reasoning = block.get("thinking", "")
                elif block.get("type") == "text":
                    content = block.get("text", "")
            return reasoning.strip(), content.strip()

        elif provider == "openai":
            url = "https://api.openai.com/v1/responses"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_output_tokens": max_tokens,
            }
            is_thinking = any(model.lower().startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))
            if is_thinking:
                payload["reasoning"] = {"effort": "none", "summary": "auto"}
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"{resp.status}: {await resp.text()}")
                data = await resp.json()
            reasoning, content = "", ""
            for item in data.get("output", []):
                if item.get("type") == "reasoning":
                    reasoning = " ".join(s.get("text", "") for s in item.get("summary", [])).strip()
                elif item.get("type") == "message":
                    for block in item.get("content", []):
                        if block.get("type") == "output_text":
                            content = block.get("text", "").strip()
            return reasoning, content

        else:  # together
            url = "https://api.together.xyz/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": max_tokens,
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
            }
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"{resp.status}: {await resp.text()}")
                data = await resp.json()
            msg = data["choices"][0]["message"]
            return (msg.get("reasoning") or "").strip(), (msg.get("content") or "").strip()

    for attempt in range(max_retries):
        try:
            async with semaphore:
                return await _one_call()
        except Exception as e:
            msg = str(e).lower()
            retryable = any(s in msg for s in ["rate", "429", "timeout", "temporar", "overload", "503", "502", "500", "bad gateway", "connection"])
            if not retryable or attempt == max_retries - 1:
                print(f"[ERROR] {e}")
                return "", ""
            await asyncio.sleep(0.8 * (2 ** attempt) + random.random() * 0.3)
    return "", ""


async def main_async():
    provider = get_provider(model)
    key_map = {
        "together":  "TOGETHER_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "anthropic": "CLAUDE_API_KEY",
    }
    api_key = os.getenv(key_map[provider])
    if not api_key:
        print(f"Error: {key_map[provider]} not found"); return

    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    reconc_path = os.path.join(_HERE, "pdf_reconc.json")
    tasks, _ = load_all_tasks(reconc_path)

    if limit_patients:
        # Get unique pids in order, keep first limit_patients
        seen, filtered = set(), []
        for t in tasks:
            if t[0] not in seen:
                seen.add(t[0])
            if len(seen) <= limit_patients:
                filtered.append(t)
        tasks = filtered

    short_name = get_short_model_name(model)
    out_dir = os.path.join(_HERE, "drug")
    os.makedirs(out_dir, exist_ok=True)
    output_json = os.path.join(out_dir, f"{short_name}_all_visits_ext_options.json")
    output_csv  = os.path.join(out_dir, f"{short_name}_all_visits_ext_options.csv")

    print("=" * 60)
    print(f"PDF DRUG PREDICTION")
    print("=" * 60)
    print(f"Model:       {model}")
    print(f"Provider:    {provider}")
    print(f"Total tasks: {len(tasks)} (patient × visit pairs)")
    print(f"Concurrency: {max_concurrency}")
    print(f"Output:      {output_json}")
    print("=" * 60 + "\n")

    semaphore = asyncio.Semaphore(max_concurrency)
    results = {}  # (pid, visit_name) -> (thinking, content)

    pbar = tqdm(total=len(tasks), desc="Predicting")

    async def _run_one(pid, vk, user_content, session):
        thinking, content = await call_api(
            session, api_key, system_prompt, user_content,
            model, thinking_budget, max_tokens, semaphore,
        )
        return pid, vk, thinking, content

    async with aiohttp.ClientSession() as session:
        for coro in asyncio.as_completed([_run_one(pid, vk, uc, session) for pid, vk, uc in tasks]):
            pid, vk, thinking, content = await coro
            results[(pid, vk)] = (thinking, content)
            pbar.update(1)

    pbar.close()

    # Build nested output: {pid: {visit_name: structured}}
    out_json = {}
    csv_rows = []
    n_failed = 0

    for pid, vk, _ in tasks:
        thinking, content = results.get((pid, vk), ("", ""))
        structured = parse_structured(thinking, content)
        out_json.setdefault(pid, {})[vk] = structured

        row = {"patient_id": pid, "visit": vk}
        for opt_num in [1, 2, 3]:
            opt = structured.get(f"option_{opt_num}", {})
            row[f"option_{opt_num}_label"]    = opt.get("label", "")
            row[f"option_{opt_num}_drugs"]    = format_drugs_str(opt.get("drugs", []))
            row[f"option_{opt_num}_rationale"] = opt.get("rationale", "")
        csv_rows.append(row)

        if not any(structured.get(f"option_{n}") for n in [1, 2, 3]):
            n_failed += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    import csv
    csv_cols = [
        "patient_id", "visit",
        "option_1_label", "option_1_drugs", "option_1_rationale",
        "option_2_label", "option_2_drugs", "option_2_rationale",
        "option_3_label", "option_3_drugs", "option_3_rationale",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{'='*60}")
    print(f"DONE! {len(results)} patient×visit predictions saved.")
    print(f"  JSON: {output_json}")
    print(f"  CSV:  {output_csv}")
    if n_failed:
        print(f"  WARNING: {n_failed} tasks had no options parsed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main_async())
