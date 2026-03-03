"""
Build ground-truth active drugs CSV from pdf_reconc.json.
Output: pdf_scripts/pdf_active_drugs_per_visit.csv
Format: pid, v1, v2, v3, ... vN  (semicolon-separated active drug names per visit)
"""
import os, re, json, csv

_HERE = os.path.dirname(os.path.abspath(__file__))

DRUG_FEATURE_NAMES = [
    "drug_clobazam", "drug_clonazepam", "drug_valproate", "drug_ethosuximide",
    "drug_levetiracetam", "drug_lamotrigine", "drug_phenobarbital",
    "drug_phenytoin", "drug_topiramate", "drug_carbamazepine",
]
DRUG_FEAT_TO_NAME = {f: f.replace("drug_", "") for f in DRUG_FEATURE_NAMES}


def extract_visit_segment(answer: str, visit_n: int) -> str:
    if not answer:
        return ""
    parts = re.split(r"(?=\bVisit\s*\d+\s*:)", answer.strip())
    for part in parts:
        part = part.strip()
        m = re.match(r"Visit\s*(\d+)\s*:\s*(.*)", part, re.DOTALL)
        if m and int(m.group(1)) == visit_n:
            return m.group(2).strip()
    # V1 fallback: no Visit N: markers at all → whole answer is V1
    if visit_n == 1 and not re.search(r"\bVisit\s*\d+\s*:", answer):
        return answer.strip()
    return ""


def parse_drug_segment(segment: str):
    if not segment:
        return "", ""
    s = segment.lower().strip()
    if s in ("not mentioned.", "not mentioned", ""):
        return "", ""

    exp_m = re.search(r"exposure_before:\s*(\w+)", s)
    act_m = re.search(r"action_taken:\s*(\w+)", s)
    if exp_m:
        return exp_m.group(1), act_m.group(1) if act_m else ""

    sh = re.match(r"^(on_arrival|past_tried|no_prior_exposure),?\s*(continue|start|stop|no_action)", s)
    if sh:
        return sh.group(1), sh.group(2)

    return "", ""


def get_active_drugs(visits: dict, visit_n: int) -> list:
    """Return sorted list of active drug names at visit_n."""
    visit_feats = visits.get(f"Visit_{visit_n}", {})
    active = []

    for feat in DRUG_FEATURE_NAMES:
        drug = DRUG_FEAT_TO_NAME[feat]
        answer = visit_feats.get(feat, {}).get("Answer", "")
        if not answer or "not mentioned" in answer.lower():
            continue

        seg = extract_visit_segment(answer, visit_n)
        if not seg:
            continue

        exposure, action = parse_drug_segment(seg)
        if not action:
            continue

        if action in ("continue", "start"):
            active.append(drug)
        elif action == "no_action" and exposure == "on_arrival":
            active.append(drug)
        # stop → not active; past_tried + no_action → not active

    return sorted(active)


def main():
    reconc_path = os.path.join(_HERE, "pdf_reconc.json")
    with open(reconc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Find max visit number across all patients
    max_v = 0
    for pid, visits in data.items():
        for k in visits:
            m = re.match(r"^Visit_(\d+)$", k)
            if m:
                max_v = max(max_v, int(m.group(1)))

    print(f"Patients: {len(data)}, Max visits: {max_v}")

    rows = []
    for pid, visits in data.items():
        row = {"pid": pid}
        # Find which visits exist for this patient
        present = sorted(
            [int(re.match(r"^Visit_(\d+)$", k).group(1))
             for k in visits if re.match(r"^Visit_(\d+)$", k)]
        )
        for vn in range(1, max_v + 1):
            col = f"v{vn}"
            if vn in present and visits.get(f"Visit_{vn}"):
                active = get_active_drugs(visits, vn)
                row[col] = "; ".join(active) if active else ""
            else:
                row[col] = None  # visit doesn't exist for this patient
        rows.append(row)

    # Sort by pid
    rows.sort(key=lambda r: r["pid"])

    out_path = os.path.join(_HERE, "pdf_active_drugs_per_visit.csv")
    cols = ["pid"] + [f"v{n}" for n in range(1, max_v + 1)]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to {out_path}")

    # Summary stats
    total_visits = sum(1 for r in rows for vn in range(1, max_v + 1) if r[f"v{vn}"] is not None)
    no_drug = sum(1 for r in rows for vn in range(1, max_v + 1) if r[f"v{vn}"] == "")
    print(f"Total visit slots: {total_visits}")
    print(f"Visits with 0 active drugs: {no_drug}")
    print(f"Visits with ≥1 active drug: {total_visits - no_drug}")


if __name__ == "__main__":
    main()
