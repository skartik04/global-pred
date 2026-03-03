"""
Repair failed reconciliation entries in pdf_reconc.json.
Re-runs visits that are empty or have unparseable RAW_MODEL_OUTPUT.
Uses higher max_tokens to handle patients with many visits.
"""
import os, json, asyncio, random, re
import dotenv
from tqdm import tqdm
from together import AsyncTogether

MODEL          = "openai/gpt-oss-120b"
MAX_TOKENS     = 32000   # increased for large multi-visit reconciliations
MAX_CONCURRENCY = 4
MAX_RETRIES    = 6

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

FEATURE_NAMES = [
    "Age_Years","Sex_Female","OnsetTimingYears","SeizureFreq","StatusOrProlonged",
    "CognitivePriority","Risk_Factors","SeizureType","drug_clobazam","drug_clonazepam",
    "drug_valproate","drug_ethosuximide","drug_levetiracetam","drug_lamotrigine",
    "drug_phenobarbital","drug_phenytoin","drug_topiramate","drug_carbamazepine",
]


def strip_code_fences(text):
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r'^```(?:json)?\s*', '', t)
        t = re.sub(r'\s*```$', '', t.strip())
    return t.strip()


def robust_json_load(text):
    raw = strip_code_fences(text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{"); end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end+1])
            except Exception:
                return None
    return None


def normalize_feature_obj(obj):
    if isinstance(obj, str):
        return {"Answer": obj, "Reasoning": "", "Supporting_Text": "", "Confidence": ""}
    if isinstance(obj, dict):
        if all(k in obj for k in ["Answer","Reasoning","Supporting_Text","Confidence"]):
            return {k: obj.get(k,"") for k in ["Answer","Reasoning","Supporting_Text","Confidence"]}
        if any(k in obj for k in ["value","reasoning","supporting_text","confidence"]):
            return {"Answer": obj.get("value",""), "Reasoning": obj.get("reasoning",""),
                    "Supporting_Text": obj.get("supporting_text",""), "Confidence": obj.get("confidence","")}
    return {"Answer":"","Reasoning":"","Supporting_Text":"","Confidence":""}


def parse_cumulative_output(text, identifier):
    data = robust_json_load(text)
    if not isinstance(data, dict):
        print(f"  [WARN] Still unparseable: {identifier}")
        return {}
    return {feat: normalize_feature_obj(data.get(feat, {})) for feat in FEATURE_NAMES}


def build_reconcile_user_content(visits_in_order):
    parts = []
    for i, v in enumerate(visits_in_order, 1):
        parts.append(f"VISIT_{i}_JSON:\n{json.dumps(v, ensure_ascii=False)}")
    return "\n\n".join(parts)


async def call_llm(client, system_prompt, user_content, semaphore):
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_content},
                    ],
                    temperature=0.0,
                    max_tokens=MAX_TOKENS,
                )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            msg = str(e).lower()
            retryable = any(s in msg for s in ["rate","429","timeout","temporar","overload","503","500","connection"])
            if not retryable or attempt == MAX_RETRIES - 1:
                print(f"  [ERROR] {e}")
                return ""
            await asyncio.sleep(0.8 * (2 ** attempt) + random.random() * 0.3)
    return ""


async def main_async():
    dotenv.load_dotenv(os.path.join(_ROOT, "scripts", ".env"))
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        print("Error: TOGETHER_API_KEY not found"); return

    reconc_path = os.path.join(_HERE, "pdf_reconc.json")
    feats_path  = os.path.join(_HERE, "pdf_feats.json")
    prompt_path = os.path.join(_HERE, "pdf_reconc_prompt.txt")

    with open(reconc_path) as f: reconc = json.load(f)
    with open(feats_path)  as f: feats  = json.load(f)
    with open(prompt_path) as f: system_prompt = f.read()

    MAX_VISITS = 10

    # Find all failed entries (empty visit or has RAW_MODEL_OUTPUT key)
    to_fix = []
    for pid, visits in reconc.items():
        raw_keys = [k for k in visits if k.endswith("_RAW_MODEL_OUTPUT")]
        empty_keys = [k for k in visits if not k.endswith("_RAW_MODEL_OUTPUT") and not visits[k]]

        failed_visits = set()
        for rk in raw_keys:
            failed_visits.add(rk.replace("_RAW_MODEL_OUTPUT", ""))
        for ek in empty_keys:
            failed_visits.add(ek)

        for vname in failed_visits:
            # vname is e.g. "Visit_6" — get visit number
            vnum = int(vname.split("_")[1])
            # Build cumulative input: V1..Vnum from feats
            pat_feats = feats.get(pid, {})
            visit_keys = [f"Visit_{n}" for n in range(1, MAX_VISITS+1) if pat_feats.get(f"Visit_{n}")]
            idx = next((i for i, vk in enumerate(visit_keys) if vk == vname), None)
            if idx is None:
                print(f"  [SKIP] {pid}/{vname} — not found in feats")
                continue
            cumulative = [pat_feats[vk] for vk in visit_keys[:idx+1]]
            user_content = build_reconcile_user_content(cumulative)
            to_fix.append((pid, vname, user_content))

    print(f"Entries to repair: {len(to_fix)}")
    for pid, vname, _ in to_fix:
        print(f"  {pid} / {vname}")
    print()

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    client = AsyncTogether(api_key=together_key)

    async def _run_one(pid, vname, user_content):
        raw = await call_llm(client, system_prompt, user_content, semaphore)
        merged = parse_cumulative_output(raw, f"{pid}_{vname}")
        return pid, vname, merged

    results = await asyncio.gather(*[_run_one(pid, v, uc) for pid, v, uc in to_fix])

    fixed = 0
    for pid, vname, merged in results:
        # Clean up RAW_MODEL_OUTPUT key if present
        raw_key = f"{vname}_RAW_MODEL_OUTPUT"
        if raw_key in reconc[pid]:
            del reconc[pid][raw_key]
        if merged:
            reconc[pid][vname] = merged
            print(f"  [OK]   {pid} / {vname}")
            fixed += 1
        else:
            reconc[pid][vname] = {}
            print(f"  [FAIL] {pid} / {vname} — still empty")

    with open(reconc_path, "w", encoding="utf-8") as f:
        json.dump(reconc, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {fixed}/{len(to_fix)} repaired. Saved to {reconc_path}")


if __name__ == "__main__":
    asyncio.run(main_async())
