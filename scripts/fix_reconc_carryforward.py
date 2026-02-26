import json
import re
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "csv_reconciled_gpt_oss_trial.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "csv_reconciled_gpt_oss.json")

DRUG_KEYS = [
    "drug_clobazam", "drug_clonazepam", "drug_valproate", "drug_ethosuximide",
    "drug_levetiracetam", "drug_lamotrigine", "drug_phenobarbital",
    "drug_phenytoin", "drug_topiramate", "drug_carbamazepine",
]

EXPOSURE_RE = re.compile(r"(Exposure_before:\s*)(\w+)", re.I)
ACTION_RE = re.compile(r"Action_taken:\s*(\w+)", re.I)

ACTIVE_ACTIONS = {"continue", "start"}


def get_last_segment(answer: str) -> str:
    """Extract the last Visit N: segment from a multi-visit answer."""
    parts = re.split(r"(?=\bVisit\s*\d+\s*:)", answer.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return ""
    return parts[-1]


def get_action(segment: str) -> str:
    m = ACTION_RE.search(segment)
    return m.group(1).lower() if m else ""


def get_visit_n_segment(answer: str, n: int) -> str:
    """Extract the Visit N segment from a multi-visit answer."""
    parts = re.split(r"(?=\bVisit\s*\d+\s*:)", answer.strip())
    for part in parts:
        part = part.strip()
        m = re.match(r"Visit\s*(\d+)\s*:", part)
        if m and int(m.group(1)) == n:
            return part
    return ""


def fix_exposure(segment: str, correct_exposure: str) -> str:
    """Replace Exposure_before value in a segment string."""
    return EXPOSURE_RE.sub(lambda m: m.group(1) + correct_exposure, segment, count=1)


def fix_answer(answer: str, visit_n: int, correct_exposure: str) -> str:
    """Replace the Exposure_before in the Visit N segment of a full answer string."""
    def replacer(part):
        part = part.strip()
        m = re.match(r"Visit\s*(\d+)\s*:", part)
        if m and int(m.group(1)) == visit_n:
            return fix_exposure(part, correct_exposure)
        return part

    parts = re.split(r"(?=\bVisit\s*\d+\s*:)", answer.strip())
    parts = [p.strip() for p in parts if p.strip()]
    fixed_parts = [replacer(p) for p in parts]
    return " ".join(fixed_parts)


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_fixed = 0
    patients_fixed = 0

    for pid, visits in data.items():
        patient_fixed = 0

        # Check V2: compare against V1 raw (no visit prefix, plain format)
        v1_feats = visits.get("Visit_1", {})
        v2_feats = visits.get("Visit_2", {})

        for dk in DRUG_KEYS:
            v1_ans = v1_feats.get(dk, {}).get("Answer", "")
            v2_ans = v2_feats.get(dk, {}).get("Answer", "")

            if not v1_ans or "not mentioned" in v1_ans.lower():
                continue
            if not v2_ans or "not mentioned" in v2_ans.lower():
                continue

            # V1 is raw (no Visit N: prefix) — parse directly
            v1_action = get_action(v1_ans)
            if v1_action not in ACTIVE_ACTIONS:
                continue

            # V2 should have Exposure_before: on_arrival in its Visit 2 segment
            v2_seg = get_visit_n_segment(v2_ans, 2)
            if not v2_seg:
                continue

            exp_m = EXPOSURE_RE.search(v2_seg)
            if not exp_m:
                continue
            current_exposure = exp_m.group(2).lower()

            if current_exposure != "on_arrival":
                fixed_ans = fix_answer(v2_ans, 2, "on_arrival")
                v2_feats[dk]["Answer"] = fixed_ans
                v2_feats[dk]["Reasoning"] += " [Post-processed: Exposure_before corrected to on_arrival at Visit 2 based on Visit 1 active status.]"
                patient_fixed += 1
                total_fixed += 1

        # Check V3: compare against V2 last segment
        v3_feats = visits.get("Visit_3", {})

        for dk in DRUG_KEYS:
            v2_ans = v2_feats.get(dk, {}).get("Answer", "")
            v3_ans = v3_feats.get(dk, {}).get("Answer", "")

            if not v2_ans or "not mentioned" in v2_ans.lower():
                continue
            if not v3_ans or "not mentioned" in v3_ans.lower():
                continue

            # Get V2's last segment action
            v2_last = get_last_segment(v2_ans)
            v2_action = get_action(v2_last)
            if v2_action not in ACTIVE_ACTIONS:
                continue

            # V3 should have Exposure_before: on_arrival in its Visit 3 segment
            v3_seg = get_visit_n_segment(v3_ans, 3)
            if not v3_seg:
                continue

            exp_m = EXPOSURE_RE.search(v3_seg)
            if not exp_m:
                continue
            current_exposure = exp_m.group(2).lower()

            if current_exposure != "on_arrival":
                fixed_ans = fix_answer(v3_ans, 3, "on_arrival")
                v3_feats[dk]["Answer"] = fixed_ans
                v3_feats[dk]["Reasoning"] += " [Post-processed: Exposure_before corrected to on_arrival at Visit 3 based on Visit 2 active status.]"
                patient_fixed += 1
                total_fixed += 1

        if patient_fixed > 0:
            patients_fixed += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Total patients processed: {len(data)}")
    print(f"Patients with fixes applied: {patients_fixed}")
    print(f"Total drug fields corrected: {total_fixed}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
