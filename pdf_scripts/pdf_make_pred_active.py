"""
Build predicted active drugs CSV from pdf prediction output.
Output: pdf_scripts/pdf_pred_active_drugs_per_visit.csv
Format: pid, v1, v2, v3, ... vN
Each cell: pipe-separated options, each option = semicolon-separated active drug names.
e.g. "valproate|carbamazepine; valproate|valproate; levetiracetam"
"""
import os, re, json, csv

_HERE = os.path.dirname(os.path.abspath(__file__))

DRUG_COLUMNS = [
    "carbamazepine", "clobazam", "clonazepam", "ethosuximide",
    "lamotrigine", "levetiracetam", "phenobarbital", "phenytoin",
    "topiramate", "valproate",
]

PRED_PATH = os.path.join(_HERE, "drug", "openai_gptoss120b_all_visits_ext_options.json")


def get_active_from_option(opt: dict) -> list:
    """Return sorted list of active drug names from a predicted option."""
    active = []
    for d in opt.get("drugs", []):
        drug = d.get("drug", "").lower()
        action = d.get("action", "").lower()
        if drug in DRUG_COLUMNS and action in ("continue", "start"):
            active.append(drug)
    return sorted(active)


def main():
    with open(PRED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Find max visit number
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
        present = sorted(
            [int(re.match(r"^Visit_(\d+)$", k).group(1))
             for k in visits if re.match(r"^Visit_(\d+)$", k)]
        )
        for vn in range(1, max_v + 1):
            col = f"v{vn}"
            if vn not in present:
                row[col] = None
                continue
            vk = f"Visit_{vn}"
            res = visits.get(vk, {})
            # Build pipe-separated options
            option_strs = []
            for opt_num in [1, 2, 3]:
                opt = res.get(f"option_{opt_num}", {})
                if opt:
                    active = get_active_from_option(opt)
                    option_strs.append("; ".join(active) if active else "")
            row[col] = "|".join(option_strs) if option_strs else ""
        rows.append(row)

    rows.sort(key=lambda r: r["pid"])

    out_path = os.path.join(_HERE, "pdf_pred_active_drugs_per_visit.csv")
    cols = ["pid"] + [f"v{n}" for n in range(1, max_v + 1)]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to {out_path}")

    # Quick stats
    total = sum(1 for r in rows for vn in range(1, max_v + 1) if r[f"v{vn}"] is not None)
    empty = sum(1 for r in rows for vn in range(1, max_v + 1)
                if r[f"v{vn}"] is not None and all(o == "" for o in r[f"v{vn}"].split("|")))
    print(f"Total visit slots: {total}")
    print(f"Visits where all 3 options predicted 0 drugs: {empty}")


if __name__ == "__main__":
    main()
