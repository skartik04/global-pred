import os
import re
import json
import asyncio
import random
import aiohttp
import pandas as pd
from tqdm import tqdm
import dotenv

dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# ============ CONFIGURATION ============

# Just set the model — provider is auto-detected.
model = 'openai/gpt-oss-120b'
# model = 'gpt-4.1-mini'
# model = 'gpt-4o'
# model = 'gemini-2.5-flash'
# model = 'claude-sonnet-4-6'

# Which visit to process (1, 2, or 3)
visit_num = 1

# Input path (reconciled JSON only)
reconciled_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv_reconciled_gpt_oss.json')

# Prompt file
prompt_file = 'all_drugs.txt'

# Processing settings
temperature = 0.0
max_tokens = 4096
thinking_budget = 4000   # only used for Together AI
max_concurrency = 8
limit_patients = None    # Set to int to limit, None for all

# ========================================

DRUG_COLUMNS = [
    "carbamazepine", "clobazam", "clonazepam", "ethosuximide",
    "lamotrigine", "levetiracetam", "phenobarbital", "phenytoin",
    "topiramate", "valproate",
]

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
    name = model_name.lower().replace("/", "_").replace("-", "").replace(".", "")
    if len(name) > 20:
        name = name[:20]
    return name


def detect_provider(model_name: str) -> str:
    """Auto-detect provider from the model name."""
    m = model_name.lower()
    if "/" in m:
        return "together"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "gemini"
    raise ValueError(f"Cannot auto-detect provider for model '{model_name}'. "
                     "Use a recognizable model name or add a '/' prefix for Together AI.")


def parse_scores(content: str) -> dict:
    """Parse all 10 drug scores from response: DrugName: score = X.XX | reason: ..."""
    scores = {}
    for drug in DRUG_COLUMNS:
        match = re.search(rf'{drug}\s*:\s*score\s*=\s*([0-9.]+)', content, re.IGNORECASE)
        if match:
            try:
                scores[drug] = float(match.group(1))
            except ValueError:
                scores[drug] = None
        else:
            scores[drug] = None
    return scores


def parse_structured(thinking: str, content: str) -> dict:
    """Build structured dict from thinking + content."""
    clinical_reasoning = ""
    sec1_match = re.search(
        r'---\s*SECTION\s*1.*?---\s*(.*?)(?=---\s*SECTION\s*2|$)',
        content, re.DOTALL | re.IGNORECASE,
    )
    if sec1_match:
        clinical_reasoning = sec1_match.group(1).strip()

    drugs = {}
    for drug in DRUG_COLUMNS:
        match = re.search(rf'{drug}\s*:\s*(score\s*=\s*.+)', content, re.IGNORECASE)
        drugs[drug] = match.group(1).strip() if match else None

    return {"think": thinking, "reasoning": clinical_reasoning, "drugs": drugs}


def build_extracted_content(patient_id: str, visit_name: str, visit_features: dict, all_visits: dict) -> str:
    lines = [f"Patient ID: {patient_id}", f"Visit: {visit_name}", ""]
    visit_1_features = all_visits.get("Visit_1", {})

    for feat in FEATURE_NAMES:
        source = visit_1_features if feat == "Age_Years" else visit_features
        answer = source.get(feat, {}).get('Answer', 'Not available')
        reasoning = source.get(feat, {}).get('Reasoning', '')
        lines.append(f"{feat}: {answer} (Additional reasoning: {reasoning})")

    return "\n".join(lines)


def load_tasks(json_path: str, visit_num: int):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    visit_name = f"Visit_{visit_num}"
    tasks = []
    for pid, visits in data.items():
        visit_features = visits.get(visit_name, {})
        if not visit_features:
            continue
        content = build_extracted_content(pid, visit_name, visit_features, visits)
        tasks.append((pid, content))
    return tasks


# ---- Provider-specific API calls ----

async def call_together(
    session: aiohttp.ClientSession,
    together_key: str,
    system_prompt: str,
    user_content: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 6,
) -> tuple[str, str]:
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {together_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
    }

    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.post(url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            msg = data["choices"][0]["message"]
            reasoning = (msg.get("reasoning") or "").strip()
            content = (msg.get("content") or "").strip()
            return reasoning, content
        except Exception as e:
            msg_str = str(e).lower()
            retryable = any(s in msg_str for s in ["rate", "429", "timeout", "temporar", "overload", "503", "502", "500", "bad gateway", "connection"])
            if not retryable or attempt == max_retries - 1:
                print(f"[ERROR] {e}")
                return "", ""
            await asyncio.sleep(0.8 * (2 ** attempt) + random.random() * 0.3)
    return "", ""


async def call_openai(client, system_prompt: str, user_content: str, semaphore: asyncio.Semaphore, max_retries: int = 6) -> tuple[str, str]:
    async def _call():
        resp = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        return "", resp.choices[0].message.content.strip()

    for attempt in range(max_retries):
        try:
            async with semaphore:
                return await _call()
        except Exception as e:
            msg_str = str(e).lower()
            retryable = any(s in msg_str for s in ["rate", "429", "timeout", "temporar", "overload", "503", "500", "connection"])
            if not retryable or attempt == max_retries - 1:
                print(f"[ERROR] {e}")
                return "", ""
            await asyncio.sleep(0.8 * (2 ** attempt) + random.random() * 0.3)
    return "", ""


async def call_anthropic(client, system_prompt: str, user_content: str, semaphore: asyncio.Semaphore, max_retries: int = 6) -> tuple[str, str]:
    async def _call():
        resp = await asyncio.to_thread(
            lambda: client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
        )
        return "", resp.content[0].text.strip()

    for attempt in range(max_retries):
        try:
            async with semaphore:
                return await _call()
        except Exception as e:
            msg_str = str(e).lower()
            retryable = any(s in msg_str for s in ["rate", "429", "timeout", "temporar", "overload", "503", "500", "connection"])
            if not retryable or attempt == max_retries - 1:
                print(f"[ERROR] {e}")
                return "", ""
            await asyncio.sleep(0.8 * (2 ** attempt) + random.random() * 0.3)
    return "", ""


async def call_gemini(client, system_prompt: str, user_content: str, semaphore: asyncio.Semaphore, max_retries: int = 6) -> tuple[str, str]:
    from google.genai import types

    async def _call():
        resp = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model=model,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        )
        return "", resp.text.strip()

    for attempt in range(max_retries):
        try:
            async with semaphore:
                return await _call()
        except Exception as e:
            msg_str = str(e).lower()
            retryable = any(s in msg_str for s in ["rate", "429", "timeout", "temporar", "overload", "503", "500", "connection", "resource"])
            if not retryable or attempt == max_retries - 1:
                print(f"[ERROR] {e}")
                return "", ""
            await asyncio.sleep(0.8 * (2 ** attempt) + random.random() * 0.3)
    return "", ""


# ---- Main ----

async def main_async():
    provider = detect_provider(model)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    short_name = get_short_model_name(model)
    output_file = os.path.join(script_dir, f"drug/all_drugs/{short_name}_v{visit_num}.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    prompt_path = os.path.join(script_dir, prompt_file)
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    print(f"\n{'='*60}")
    print(f"ALL-DRUG PREDICTION")
    print(f"{'='*60}")
    print(f"Provider: {provider} (auto-detected)")
    print(f"Model:    {model}")
    print(f"Visit:    {visit_num}")
    print(f"Concurrency: {max_concurrency}")
    print(f"Output:   {output_file}")
    print(f"{'='*60}\n")

    # Init client
    client = None
    together_key = None

    if provider == 'together':
        together_key = os.getenv("TOGETHER_API_KEY")
        if not together_key:
            print("Error: TOGETHER_API_KEY not found")
            return

    elif provider == 'openai':
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found")
            return
        client = OpenAI(api_key=api_key)

    elif provider == 'anthropic':
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not found")
            return
        client = Anthropic(api_key=api_key)

    elif provider == 'gemini':
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY not found")
            return
        client = genai.Client(api_key=api_key)

    else:
        print(f"Error: Unknown provider '{provider}'. Use: openai | anthropic | gemini | together")
        return

    print("Loading tasks...")
    tasks = load_tasks(reconciled_json_path, visit_num)
    if limit_patients:
        tasks = tasks[:limit_patients]
        print(f"  Limited to {limit_patients} patients")
    print(f"Total patients: {len(tasks)}\n")

    # Run inference
    semaphore = asyncio.Semaphore(max_concurrency)
    results = {}
    pbar = tqdm(total=len(tasks), desc="Processing")

    async def _run_one_together(task, session):
        pid, user_content = task
        thinking, content = await call_together(session, together_key, system_prompt, user_content, semaphore)
        return pid, thinking, content

    async def _run_one_closed(task):
        pid, user_content = task
        if provider == 'openai':
            thinking, content = await call_openai(client, system_prompt, user_content, semaphore)
        elif provider == 'anthropic':
            thinking, content = await call_anthropic(client, system_prompt, user_content, semaphore)
        elif provider == 'gemini':
            thinking, content = await call_gemini(client, system_prompt, user_content, semaphore)
        return pid, thinking, content

    if provider == 'together':
        async with aiohttp.ClientSession() as session:
            for coro in asyncio.as_completed([_run_one_together(t, session) for t in tasks]):
                pid, thinking, content = await coro
                results[pid] = (thinking, content)
                pbar.update(1)
    else:
        for coro in asyncio.as_completed([_run_one_closed(t) for t in tasks]):
            pid, thinking, content = await coro
            results[pid] = (thinking, content)
            pbar.update(1)

    pbar.close()

    # Save structured JSON + CSV
    ordered_structured = {}
    parsed_data = []
    n_failed = 0

    for pid, _ in tasks:
        thinking, content = results.get(pid, ("", ""))
        ordered_structured[pid] = parse_structured(thinking, content)

        scores = parse_scores(content)
        row = {'patient_id': pid}
        row.update(scores)
        parsed_data.append(row)
        if all(v is None for v in scores.values()):
            n_failed += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ordered_structured, f, indent=2, ensure_ascii=False)

    csv_file = output_file.replace('.json', '.csv')
    df_out = pd.DataFrame(parsed_data, columns=['patient_id'] + DRUG_COLUMNS)
    df_out.to_csv(csv_file, index=False)

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"{'='*60}")
    print(f"Saved {len(results)} results:")
    print(f"  JSON: {output_file}")
    print(f"  CSV:  {csv_file}")
    if n_failed > 0:
        print(f"  Warning: {n_failed} patients had no scores parsed")
    print(f"{'='*60}\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
