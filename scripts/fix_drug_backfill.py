import json
import re
import os
import pandas as pd

# Paths are relative to script location (only_open folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_JSON_PATH = os.path.join(SCRIPT_DIR, "csv_feats_gpt_oss_trial.json")
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'global', 'data', 'combined_dataset.csv')
NEW_JSON_PATH = os.path.join(SCRIPT_DIR, "csv_backfilled_gpt_oss_trial.json")

MED_STATUS_COL = "Medication status"

VISIT1 = "Visit_1"
VISIT2 = "Visit_2"

DRUG_KEYS = [
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

# Visit_2 Exposure_before values that imply the drug existed before Visit_2
COPY_BACK_EXPOSURES = {"on_arrival", "past_tried"}

EXPOSURE_RE = re.compile(r"Exposure_before:\s*([a-z_]+)", re.I)
ACTION_RE = re.compile(r"Action_taken:\s*([a-z_]+)", re.I)

def norm(s) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and pd.isna(s):
        return ""
    return str(s).strip()

def norm_lower(s) -> str:
    return norm(s).lower()

def is_not_mentioned(ans: str) -> bool:
    v = norm_lower(ans)
    return v == "" or v.startswith("not mentioned")

def extract_exposure_action(answer: str):
    if not answer or is_not_mentioned(answer):
        return None, None
    e = EXPOSURE_RE.search(answer)
    a = ACTION_RE.search(answer)
    exposure = e.group(1).lower() if e else None
    action = a.group(1).lower() if a else None
    return exposure, action

def is_not_on_medication(med_status_val: str) -> bool:
    v = norm_lower(med_status_val)
    # tolerate common variants
    return v in {
        "not on medication",
        "not on meds",
        "not on med",
        "not on medications",
        "not on anti seizure medication",
        "not on antiseizure medication",
    }

def try_parse_record_id_from_json_key(patient_id: str):
    # Examples: "1_Nanyonga Aisha", "12_Some Name"
    s = norm(patient_id)
    if "_" not in s:
        return None
    prefix = s.split("_", 1)[0].strip()
    if prefix.isdigit():
        return int(prefix)
    return None

def safe_feat_answer(visit_feats: dict, key: str) -> str:
    obj = visit_feats.get(key, {})
    if isinstance(obj, dict):
        return obj.get("Answer", "")
    return ""

def set_backfill(visit_feats: dict, drug: str, as_previous: bool):
    if as_previous:
        visit_feats[drug] = {
            "Answer": "Exposure_before: past_tried; Action_taken: no_action",
            "Reasoning": "Backfilled: Visit 1 baseline meds were not documented; Visit 2 indicates this drug was used before Visit 2 but already stopped.",
            "Supporting_Text": "",
            "Confidence": "med",
        }
    else:
        visit_feats[drug] = {
            "Answer": "Exposure_before: on_arrival; Action_taken: continue",
            "Reasoning": "Backfilled: Visit 1 baseline meds were not documented; Visit 2 indicates this drug was already active going into Visit 2, so it was being continued at Visit 1.",
            "Supporting_Text": "",
            "Confidence": "med",
        }

def main():
    # Load JSON
    with open(OLD_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load CSV
    raw_df = pd.read_csv(
        CSV_PATH,
        sep=";",
        engine="python",
        quotechar='"',
        doublequote=True,
        escapechar="\\",
    )

    if MED_STATUS_COL not in raw_df.columns:
        raise ValueError(f"CSV missing required column: {MED_STATUS_COL}")

    # Build lookups so we can map JSON patient_id -> Medication status
    # This supports keys like "RecordID_Name" (example: "1_Nanyonga Aisha")
    by_record_id = {}
    by_key_recordid_name = {}
    by_name = {}

    has_record_id = "Record ID" in raw_df.columns
    has_name = "Name: " in raw_df.columns

    for i, row in raw_df.iterrows():
        med_status = norm(row.get(MED_STATUS_COL, ""))

        rid = None
        if has_record_id and pd.notna(row.get("Record ID")):
            try:
                rid = int(str(row["Record ID"]).strip())
            except Exception:
                rid = None

        name = ""
        if has_name and pd.notna(row.get("Name: ")):
            name = norm(row["Name: "])

        if rid is not None:
            by_record_id[rid] = med_status
            if name:
                by_key_recordid_name[f"{rid}_{name}"] = med_status

        if name:
            # last-write-wins is fine for your duplicated identical rows
            by_name[name] = med_status

    # Backfill Visit_1 from Visit_2 for patients where Medication status == Not on Medication
    patients_total = len(data)
    patients_found_medstatus = 0
    patients_gate_true = 0
    patients_changed = 0
    total_drugs_backfilled = 0
    missing_medstatus_examples = []

    for patient_id, visits in data.items():
        pid = norm(patient_id)

        med_status = None

        # 1) exact match on "RecordID_Name"
        if pid in by_key_recordid_name:
            med_status = by_key_recordid_name[pid]
        else:
            # 2) parse record id from json key and match Record ID column
            rid = try_parse_record_id_from_json_key(pid)
            if rid is not None and rid in by_record_id:
                med_status = by_record_id[rid]
            else:
                # 3) fallback: if json key is just a name
                if pid in by_name:
                    med_status = by_name[pid]

        if med_status is None:
            if len(missing_medstatus_examples) < 8:
                missing_medstatus_examples.append(pid)
            continue

        patients_found_medstatus += 1

        if not is_not_on_medication(med_status):
            continue

        patients_gate_true += 1

        if VISIT1 not in visits or VISIT2 not in visits:
            continue

        v1 = visits[VISIT1]
        v2 = visits[VISIT2]

        changed_this_patient = 0

        for drug in DRUG_KEYS:
            v1_ans = safe_feat_answer(v1, drug)
            if not is_not_mentioned(v1_ans):
                continue  # "if not mentioned then copy otherwise stfu"

            v2_ans = safe_feat_answer(v2, drug)
            exposure2, _ = extract_exposure_action(v2_ans)

            if exposure2 not in COPY_BACK_EXPOSURES:
                continue

            # Exposure_before from Visit_2 directly encodes what was true at Visit_1
            if exposure2 == "past_tried":
                set_backfill(v1, drug, as_previous=True)
            else:  # on_arrival
                set_backfill(v1, drug, as_previous=False)

            changed_this_patient += 1
            total_drugs_backfilled += 1

        if changed_this_patient > 0:
            patients_changed += 1

    os.makedirs(os.path.dirname(NEW_JSON_PATH), exist_ok=True)
    with open(NEW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Loaded patients: {patients_total}")
    print(f"Patients with med status found (by RecordID_Name or Record ID): {patients_found_medstatus}")
    print(f"Patients passing gate (Not on Medication): {patients_gate_true}")
    print(f"Patients changed: {patients_changed}")
    print(f"Total drug fields backfilled: {total_drugs_backfilled}")
    print(f"Saved: {NEW_JSON_PATH}")

    if missing_medstatus_examples:
        print("Examples of JSON patient_ids that could not be matched to CSV med status:")
        for x in missing_medstatus_examples:
            print("  ", x)

if __name__ == "__main__":
    main()
