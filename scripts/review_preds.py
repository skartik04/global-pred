"""
review_preds.py — Multi-model LLM reviewer for drug prediction reasoning.

Supports:
  Anthropic  — claude-sonnet-4-6, claude-opus-4-6, etc.
  OpenAI     — gpt-4o, o3, o4-mini, gpt-5, etc.

For each patient:
  - Raw medical record from source CSV
  - AI's extracted clinical summary (what gpt-oss-120b saw)
  - AI's full output: thinking trace + staged reasoning + 3 drug options
  → Sends all to the reviewer model with web search enabled
  → Saves structured critique + score per patient

Outputs:
  drug/all_drugs/review_{short_model}_v{N}.json
  drug/all_drugs/review_{short_model}_v{N}.csv
"""

import os
import re
import json
import asyncio
import random
import pandas as pd
import dotenv
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
dotenv.load_dotenv(os.path.join(_HERE, ".env"))

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — change these to switch model / visit
# ══════════════════════════════════════════════════════════════════════════════

visit_num      = 1
reviewer_model = "claude-sonnet-4-6"
# Other options:
#   "claude-opus-4-6"
#   "gpt-4o"
#   "o3"
#   "gpt-5"          (whenever available)
#   "o4-mini"

max_tokens      = 1000
limit_patients  = 2   # int to cap, None = all
max_concurrency = 1

# ══════════════════════════════════════════════════════════════════════════════

DRUG_FEATURE_NAMES = [
    "drug_clobazam", "drug_clonazepam", "drug_valproate", "drug_ethosuximide",
    "drug_levetiracetam", "drug_lamotrigine", "drug_phenobarbital",
    "drug_phenytoin", "drug_topiramate", "drug_carbamazepine",
]
FEATURE_NAMES = [
    "Age_Years", "Sex_Female", "OnsetTimingYears", "SeizureFreq",
    "StatusOrProlonged", "CognitivePriority", "Risk_Factors", "SeizureType",
]


# ── Provider detection ────────────────────────────────────────────────────────

def get_provider(model: str) -> str:
    """Return 'anthropic' or 'openai' based on model name."""
    if model.startswith("claude"):
        return "anthropic"
    return "openai"


def get_short_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", model)[:20]


# ── File paths ────────────────────────────────────────────────────────────────

def make_paths(visit_num: int, model: str) -> dict:
    short = get_short_model_name(model)
    base  = os.path.join(_HERE, "drug", "all_drugs")
    return {
        "pred":   os.path.join(base, f"openai_gptoss120b_v{visit_num}_ext_options.json"),
        "reconc": os.path.join(_HERE, "csv_reconciled_gpt_oss.json"),
        "csv":    os.path.join(os.path.dirname(_HERE), "data", "combined_dataset.csv"),
        "prompt": os.path.join(_HERE, "review_prompt.txt"),
        "out_json": os.path.join(base, f"review_{short}_v{visit_num}.json"),
        "out_csv":  os.path.join(base, f"review_{short}_v{visit_num}.csv"),
    }


# ── Raw record builder ────────────────────────────────────────────────────────

def build_raw_record(row) -> str:
    def sg(col):
        if col in row.index:
            v = row[col]
            s = str(v).strip()
            return s if pd.notna(v) and s not in ("", "nan") else "Not recorded"
        return "Not recorded"

    return f"""=== RAW MEDICAL RECORD (Visit 1 — Initial Presentation) ===
Date: {sg('Date of visit(0 months)')}
Age: {sg('Age')}
Sex: {sg('Sex:')}
Seizure Diagnosis: {sg('Seizure Diagnosis')}

History of Presenting Illness:
{sg('History of Presenting Illness')}

Detailed Seizure History:
{sg('Detailed description of seizure history')}

Age of Onset: {sg('Age of onset of seizure')}
Duration: {sg('Duration of Seizure')}
Pre-ictal: {sg('Pre-ictal description')}
Ictal: {sg('Ictal description')}
Post-ictal: {sg('Post-ictal description')}

Developmental Delay: {sg('Developmental Delay ')}
Behavioural Problems: {sg('Behavioural Problems')}
Psychiatric: {sg('Psychiatric')}

Medication Status at Presentation: {sg('Medication status')}
Current Drug Regimen: {sg('Current drug regimen')}
Date Commenced: {sg('Date of commencement of medication')}
Current Dose: {sg('Current dose')}"""


# ── AI input summary builder ──────────────────────────────────────────────────

def build_ai_input_summary(pid: str, visits: dict) -> str:
    v1 = visits.get("Visit_1", {})
    lines = [f"=== AI MODEL INPUT (Extracted Summary — {pid}) ===",
             "\n-- Clinical Features --"]
    for feat in FEATURE_NAMES:
        ans = v1.get(feat, {}).get("Answer", "Not extracted")
        lines.append(f"{feat}: {ans}")

    lines.append("\n-- Drug History at Visit 1 --")
    found_any = False
    for feat in DRUG_FEATURE_NAMES:
        obj  = v1.get(feat, {})
        ans  = obj.get("Answer", "")
        reas = obj.get("Reasoning", "")
        if ans and "not mentioned" not in ans.lower():
            drug = feat.replace("drug_", "")
            lines.append(f"{drug}: {ans}")
            if reas:
                lines.append(f"  (reasoning: {reas[:120]})")
            found_any = True
    if not found_any:
        lines.append("No prior drug exposure recorded.")

    return "\n".join(lines)


# ── AI output formatter ───────────────────────────────────────────────────────

def build_ai_output(pred: dict) -> str:
    lines = ["=== AI MODEL OUTPUT ==="]

    # Thinking scratchpad omitted — too long, not needed for reviewer
    reasoning = pred.get("reasoning", "").strip()
    if reasoning:
        if len(reasoning) > 1500:
            reasoning = reasoning[:1500] + "\n... [truncated]"
        lines.append("\n-- Stage-by-Stage Clinical Reasoning --")
        lines.append(reasoning)

    lines.append("\n-- Drug Recommendations --")
    for n in [1, 2, 3]:
        opt = pred.get(f"option_{n}", {})
        if not opt:
            continue
        label = opt.get("label", "")
        drugs = opt.get("drugs", [])
        rat   = opt.get("rationale", "")
        drug_str = "; ".join(
            f"{d['drug']}: {d['action']}"
            for d in drugs
            if d.get("action") in {"continue", "start", "stop"}
        )
        lines.append(f"\nOption {n}: {label}")
        lines.append(f"  Drugs: {drug_str or 'none listed'}")
        lines.append(f"  Rationale: {rat}")

    return "\n".join(lines)


# ── Anthropic caller ──────────────────────────────────────────────────────────

async def call_anthropic(
    model: str,
    api_key: str,
    system_prompt: str,
    user_content: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 4,
) -> str:
    import anthropic as ant

    client = ant.Anthropic(api_key=api_key)

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await asyncio.to_thread(
                    client.messages.create,
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    messages=[{"role": "user", "content": user_content}],
                )
            return "\n".join(
                block.text for block in response.content
                if hasattr(block, "text")
            ).strip()
        except Exception as e:
            msg = str(e).lower()
            # Web search tool not available — retry without it
            if "web_search" in msg or "unknown tool" in msg or "tool" in msg:
                try:
                    async with semaphore:
                        response = await asyncio.to_thread(
                            client.messages.create,
                            model=model,
                            max_tokens=max_tokens,
                            system=system_prompt,
                            messages=[{"role": "user", "content": user_content}],
                        )
                    return "\n".join(
                        block.text for block in response.content
                        if hasattr(block, "text")
                    ).strip()
                except Exception as e2:
                    print(f"[ERROR] Anthropic (no-tool fallback) failed: {e2}")
                    return ""
            retryable = any(s in msg for s in ["rate", "429", "overload", "timeout", "503", "500"])
            if not retryable or attempt == max_retries - 1:
                print(f"[ERROR] Anthropic failed: {e}")
                return ""
            await asyncio.sleep(2 ** attempt + random.random())
    return ""


# ── OpenAI caller ─────────────────────────────────────────────────────────────

async def call_openai(
    model: str,
    api_key: str,
    system_prompt: str,
    user_content: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 4,
) -> str:
    import openai

    client = openai.OpenAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            async with semaphore:
                # Try Responses API with web search (works for gpt-4o, o3, gpt-5 etc.)
                try:
                    response = await asyncio.to_thread(
                        client.responses.create,
                        model=model,
                        tools=[{"type": "web_search_preview"}],
                        instructions=system_prompt,
                        input=user_content,
                    )
                    return response.output_text.strip()
                except (AttributeError, Exception) as resp_err:
                    # Responses API not available or model doesn't support it
                    # Fall back to Chat Completions API
                    resp_msg = str(resp_err).lower()
                    if "web_search" in resp_msg or "tool" in resp_msg:
                        # Try chat completions without web search
                        response = await asyncio.to_thread(
                            client.chat.completions.create,
                            model=model,
                            max_tokens=max_tokens,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user",   "content": user_content},
                            ],
                        )
                        return response.choices[0].message.content.strip()
                    raise  # re-raise if it's a different error

        except Exception as e:
            msg = str(e).lower()
            retryable = any(s in msg for s in ["rate", "429", "overload", "timeout", "503", "500"])
            if not retryable or attempt == max_retries - 1:
                print(f"[ERROR] OpenAI failed: {e}")
                return ""
            await asyncio.sleep(2 ** attempt + random.random())
    return ""


# ── Unified caller ────────────────────────────────────────────────────────────

async def call_reviewer(
    model: str,
    api_keys: dict,
    system_prompt: str,
    user_content: str,
    semaphore: asyncio.Semaphore,
) -> str:
    provider = get_provider(model)
    if provider == "anthropic":
        return await call_anthropic(
            model, api_keys["anthropic"], system_prompt, user_content, semaphore
        )
    else:
        return await call_openai(
            model, api_keys["openai"], system_prompt, user_content, semaphore
        )


# ── Parse review text ─────────────────────────────────────────────────────────

def extract_score(text: str):
    m = re.search(r'SCORE:\s*(\d)/5', text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def extract_section(text: str, header: str) -> str:
    m = re.search(
        rf'##\s*{re.escape(header)}\s*\n+(.*?)(?=\n##|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    return m.group(1).strip()[:500] if m else ""


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    api_keys = {
        "anthropic": os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        "openai":    os.getenv("OPENAI_API_KEY"),
    }

    provider = get_provider(reviewer_model)
    if provider == "anthropic" and not api_keys["anthropic"]:
        print("Error: CLAUDE_API_KEY not found in .env"); return
    if provider == "openai" and not api_keys["openai"]:
        print("Error: OPENAI_API_KEY not found in .env"); return

    paths = make_paths(visit_num, reviewer_model)
    os.makedirs(os.path.dirname(paths["out_json"]), exist_ok=True)

    with open(paths["prompt"],  "r", encoding="utf-8") as f:
        system_prompt = f.read()
    with open(paths["pred"],    "r", encoding="utf-8") as f:
        pred_data = json.load(f)
    with open(paths["reconc"],  "r", encoding="utf-8") as f:
        reconc_data = json.load(f)

    df_csv = pd.read_csv(
        paths["csv"], sep=";", engine="python",
        quotechar='"', doublequote=True, escapechar="\\"
    )
    df_csv = df_csv.drop_duplicates(subset=[
        "Name: ", "Date of visit(0 months)",
        "Date of visit(6 months)", "Date of visit(12 months)"
    ])

    # Build task list
    patients = []
    for pid in pred_data:
        if pid not in reconc_data:
            continue
        pred = pred_data[pid]
        if not pred.get("raw_content", "").strip():
            continue

        name_part = "_".join(pid.split("_")[1:]) if "_" in pid else pid
        matches   = df_csv[df_csv["Name: "].str.contains(name_part, na=False)]
        raw_rec   = build_raw_record(matches.iloc[0]) if not matches.empty else "Raw record not found."

        ai_input  = build_ai_input_summary(pid, reconc_data[pid])
        ai_output = build_ai_output(pred)
        content   = f"{raw_rec}\n\n{ai_input}\n\n{ai_output}"
        patients.append((pid, content))

    if limit_patients:
        patients = patients[:limit_patients]

    print(f"\n{'='*60}")
    print(f"LLM REVIEWER")
    print(f"  Reviewer model : {reviewer_model}  ({provider})")
    print(f"  Visit          : {visit_num}")
    print(f"  Patients       : {len(patients)}")
    print(f"  Concurrency    : {max_concurrency}")
    print(f"  Output         : {paths['out_json']}")
    print(f"{'='*60}\n")

    semaphore = asyncio.Semaphore(max_concurrency)
    results   = {}

    async def review_one(pid, content, delay=0):
        if delay:
            await asyncio.sleep(delay)
        review = await call_reviewer(reviewer_model, api_keys, system_prompt, content, semaphore)
        return pid, review

    pbar  = tqdm(total=len(patients), desc="Reviewing")
    tasks = [review_one(pid, c, delay=i*35) for i, (pid, c) in enumerate(patients)]
    for coro in asyncio.as_completed(tasks):
        pid, review = await coro
        results[pid] = review
        pbar.update(1)
    pbar.close()

    # Build output
    out_json = {}
    rows     = []
    for pid, _ in patients:
        review = results.get(pid, "")
        score  = extract_score(review)
        errors = extract_section(review, "CRITICAL ERRORS")
        rec    = extract_section(review, "WHAT YOU WOULD RECOMMEND")
        out_json[pid] = {
            "model":           reviewer_model,
            "score":           score,
            "critical_errors": errors,
            "recommendation":  rec,
            "full_review":     review,
        }
        rows.append({
            "patient_id":            pid,
            "reviewer_model":        reviewer_model,
            "score":                 score,
            "critical_errors":       errors,
            "reviewer_recommendation": rec,
        })

    with open(paths["out_json"], "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(paths["out_csv"], index=False)

    # Summary
    scores = [r["score"] for r in rows if r["score"] is not None]
    print(f"\n{'='*60}")
    print(f"DONE — {len(patients)} patients reviewed")
    if scores:
        from collections import Counter
        print(f"Average score: {sum(scores)/len(scores):.2f}/5  (n={len(scores)})")
        for s in sorted(Counter(scores)):
            print(f"  {s}/5 : {Counter(scores)[s]} patients")
    print(f"\nSaved → {paths['out_json']}")
    print(f"Saved → {paths['out_csv']}")


if __name__ == "__main__":
    asyncio.run(main())
