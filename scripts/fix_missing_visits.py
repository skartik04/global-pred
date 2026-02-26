import os
import json
import asyncio
import random
import dotenv
import pandas as pd
from together import AsyncTogether

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TARGETS = [
    ("135_Best Betina", "Visit_2", "Best Betina"),
    ("24_Nangobi Jazmine", "Visit_3", "Nangobi Jazmine"),
    ("219_Magala Jovan", "Visit_2", "Magala Jovan"),
]

FEATURE_NAMES = [
    "Age_Years", "Sex_Female", "OnsetTimingYears", "SeizureFreq",
    "StatusOrProlonged", "CognitivePriority", "Risk_Factors", "SeizureType",
    "drug_clobazam", "drug_clonazepam", "drug_valproate", "drug_ethosuximide",
    "drug_levetiracetam", "drug_lamotrigine", "drug_phenobarbital",
    "drug_phenytoin", "drug_topiramate", "drug_carbamazepine",
]


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


def parse_response(response_text, identifier):
    parsed = {"id": identifier, "features": {}}
    if not response_text:
        return parsed
    raw = strip_code_fences(response_text)
    try:
        data = json.loads(raw)
    except Exception as e:
        print(f"[WARN] JSON parse failed for {identifier}: {e}")
        return parsed
    for feat in FEATURE_NAMES:
        feat_obj = data.get(feat, {})
        if isinstance(feat_obj, str):
            value, reasoning, supporting_text, confidence = feat_obj, "", "", ""
        elif isinstance(feat_obj, dict):
            value = feat_obj.get("value", "")
            reasoning = feat_obj.get("reasoning", "")
            supporting_text = feat_obj.get("supporting_text", "")
            confidence = feat_obj.get("confidence", "")
        else:
            value, reasoning, supporting_text, confidence = "", "", "", ""
        parsed["features"][feat] = {
            "Answer": value, "Reasoning": reasoning,
            "Supporting_Text": supporting_text, "Confidence": confidence,
        }
    return parsed


async def call_llm(client, system_prompt, user_content, max_retries=6):
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=8192,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            msg = str(e).lower()
            retryable = any(s in msg for s in ["rate", "429", "timeout", "temporar", "overload", "503", "500", "connection"])
            if not retryable or attempt == max_retries - 1:
                print(f"[ERROR] LLM call failed: {e}")
                return ""
            await asyncio.sleep(0.8 * (2 ** attempt) + random.random() * 0.3)
    return ""


def build_visit_content(row, visit_name):
    def sg(col):
        if col in row.index:
            v = row[col]
            return v if pd.notna(v) else ""
        return ""

    if visit_name == "Visit_2":
        return f"""Visit 2 (6 months follow-up):
6-Month Visit Date: {sg('Date of visit(6 months)')}
Patient Gender: {sg('Sex:')}
Medication Notes (6-month visit; may describe continuation or change): {sg('Second Entry(6 months)')}
Medication Prescription (6-month visit; may describe dosage change): {sg('Medication dosage and if there was a change in medication(6 months)')}
Additional Specifications: {sg('Specify')}

Note:
- Baseline age was recorded at the initial visit.
- This is approximately 6 months after baseline.
- Follow-up documentation may omit unchanged baseline information.
"""
    elif visit_name == "Visit_3":
        return f"""Visit 3 (12 months follow-up):
12-Month Visit Date: {sg('Date of visit(12 months)')}
Patient Gender: {sg('Sex:')}
Medication Notes (12-month visit; may describe continuation or change): {sg('Third Entry(12 months)')}
Medication Prescription (12-month visit; may describe dosage change): {sg('Medication dosage and if there was a change in medication(12 months)')}

Note:
- Baseline age was recorded at the initial visit.
- This is approximately 12 months after baseline.
- Follow-up documentation may omit unchanged baseline information.
"""
    return ""


async def main():
    dotenv.load_dotenv(os.path.join(SCRIPT_DIR, ".env"))
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        print("Error: TOGETHER_API_KEY not found")
        return

    with open(os.path.join(SCRIPT_DIR, "extract_prompt.txt"), "r", encoding="utf-8") as f:
        system_prompt = f.read()

    csv_path = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "combined_dataset.csv")
    df = pd.read_csv(csv_path, sep=";", engine="python", quotechar='"', doublequote=True, escapechar="\\")
    df = df.drop_duplicates(subset=["Name: ", "Date of visit(0 months)", "Date of visit(6 months)", "Date of visit(12 months)"])

    client = AsyncTogether(api_key=together_key)
    results = {}

    for patient_id, visit_name, name_fragment in TARGETS:
        rows = df[df["Name: "].str.contains(name_fragment, na=False)]
        if rows.empty:
            print(f"[WARN] {name_fragment} not found in CSV")
            continue
        row = rows.iloc[0]
        content = build_visit_content(row, visit_name)
        print(f"Running extraction for {patient_id} {visit_name}...")
        response = await call_llm(client, system_prompt, content)
        parsed = parse_response(response, f"{patient_id}_{visit_name}")
        results[patient_id] = results.get(patient_id, {})
        results[patient_id][visit_name] = parsed["features"]
        results[patient_id][f"{visit_name}_RAW"] = response
        print(f"  Done. Features extracted: {sum(1 for v in parsed['features'].values() if v.get('Answer',''))}/{len(FEATURE_NAMES)}")

    out_path = os.path.join(SCRIPT_DIR, "csv_missing_visits_fixed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
