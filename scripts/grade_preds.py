"""
grade_preds.py — Grading script for drug prediction pipeline.

For each patient × visit:
  - GT active set: drugs where action ∈ {continue, start} = drugs ON after the visit
  - Pred active set: drugs where predicted action ∈ {continue, start}
  - Match: GT active set == pred active set (actions ignored, just which drugs are active)

  GT rules (before converting to active set):
    - Backfilled entries included (treated same as any other entry)
    - stop only counted when Exposure_before == on_arrival
    - on_arrival + no_action → treated as continue (implicit)
  - Patients with empty GT (0 active drugs) are included; pred must also predict 0 to match.

Outputs:
  drug/all_drugs/grading_results.csv
"""

import os
import re
import json
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))

DRUG_FEATURE_NAMES = [
    "drug_clobazam", "drug_clonazepam", "drug_valproate", "drug_ethosuximide",
    "drug_levetiracetam", "drug_lamotrigine", "drug_phenobarbital",
    "drug_phenytoin", "drug_topiramate", "drug_carbamazepine",
]
DRUG_FEAT_TO_NAME = {f: f.replace("drug_", "") for f in DRUG_FEATURE_NAMES}

RECONC_PATH = os.path.join(_HERE, "csv_reconciled_gpt_oss.json")
PRED_TEMPLATE = os.path.join(_HERE, "drug", "all_drugs", "openai_gptoss120b_v{n}_ext_options.json")
OUT_CSV = os.path.join(_HERE, "drug", "all_drugs", "grading_results.csv")


# ── parsing helpers ──────────────────────────────────────────────────────────

def extract_visit_segment(answer: str, visit_n: int) -> str:
    """
    Extract the Visit N segment from a drug answer string.
    V1 answers are raw (no 'Visit N:' prefix) → return whole answer.
    V2/V3 answers have cumulative 'Visit 1: ... Visit 2: ...' format.
    """
    if not answer:
        return ""

    # Try to split on "Visit N:" markers
    parts = re.split(r'(?=\bVisit\s*\d+\s*:)', answer.strip())
    for part in parts:
        part = part.strip()
        m = re.match(r'Visit\s*(\d+)\s*:\s*(.*)', part, re.DOTALL)
        if m and int(m.group(1)) == visit_n:
            return m.group(2).strip()

    # V1 fallback: if no Visit N: markers at all, treat whole answer as V1
    if visit_n == 1 and not re.search(r'\bVisit\s*\d+\s*:', answer):
        return answer.strip()

    return ""


def parse_drug_segment(segment: str):
    """
    Parse a structured segment into (exposure, action).
    Handles both:
      - "Exposure_before: on_arrival; Action_taken: continue"
      - Shorthand "on_arrival, continue"
    Returns ('', '') if unparseable or empty/not-mentioned.
    """
    if not segment:
        return "", ""
    s = segment.lower().strip()
    if s in ("not mentioned.", "not mentioned", ""):
        return "", ""

    # Structured format
    exp_m = re.search(r'exposure_before:\s*(\w+)', s)
    act_m = re.search(r'action_taken:\s*(\w+)', s)
    if exp_m:
        exposure = exp_m.group(1)
        action = act_m.group(1) if act_m else ""
        return exposure, action

    # Shorthand
    sh = re.match(r'^(on_arrival|past_tried|no_prior_exposure),?\s*(continue|start|stop|no_action)', s)
    if sh:
        return sh.group(1), sh.group(2)

    return "", ""


# ── GT extraction ─────────────────────────────────────────────────────────────

def extract_gt(visits: dict, visit_n: int) -> dict:
    """
    Build ground-truth action dict for a visit.

    Rules:
      - Backfilled entries (Reasoning starts with 'Backfilled') → SKIP
        (these are inferred from later visits, not actual V1 clinical decisions)
      - action == 'continue'                     → include
      - action == 'start'                        → include
      - action == 'stop', exposure == 'on_arrival' → include
        (drug was genuinely active and the clinician stopped it)
      - action == 'stop', exposure != 'on_arrival' → SKIP
        (drug already inactive; not a real stop decision)
      - action == 'no_action', exposure == 'on_arrival' → include as 'continue'
        (drug was present, no documented change = implicitly continued)
      - action == 'no_action', exposure != 'on_arrival' → skip
      - not mentioned / empty → skip
    """
    gt = {}
    visit_feats = visits.get(f"Visit_{visit_n}", {})

    for feat in DRUG_FEATURE_NAMES:
        drug = DRUG_FEAT_TO_NAME[feat]
        feat_obj = visit_feats.get(feat, {})
        answer = feat_obj.get("Answer", "")

        if not answer or "not mentioned" in answer.lower():
            continue

        seg = extract_visit_segment(answer, visit_n)
        if not seg:
            continue

        exposure, action = parse_drug_segment(seg)
        if not action:
            continue

        if action == "no_action":
            # Only meaningful if drug was already on arrival (implicit continue)
            if exposure == "on_arrival":
                gt[drug] = "continue"
            continue

        if action == "stop" and exposure != "on_arrival":
            # Drug was already inactive — not a real clinical stop
            continue

        gt[drug] = action

    return gt


def count_active_after(visits: dict, visit_n: int) -> int:
    """
    Count drugs that are active AFTER visit N.
    Active = action is 'continue' or 'start', OR 'on_arrival; no_action' edge case.
    """
    visit_feats = visits.get(f"Visit_{visit_n}", {})
    count = 0

    for feat in DRUG_FEATURE_NAMES:
        answer = visit_feats.get(feat, {}).get("Answer", "")
        if not answer or "not mentioned" in answer.lower():
            continue

        seg = extract_visit_segment(answer, visit_n)
        if not seg:
            continue

        exposure, action = parse_drug_segment(seg)

        if action in ("continue", "start"):
            count += 1
        elif action == "no_action" and exposure == "on_arrival":
            # Drug was present, no documented change → still active
            count += 1

    return count


# ── grading ──────────────────────────────────────────────────────────────────

_VALID_ACTIONS = {"continue", "start", "stop"}
_ACTIVE_ACTIONS = {"continue", "start"}


def gt_to_active_set(gt: dict) -> frozenset:
    """GT active set: drug names where action is continue or start."""
    return frozenset(drug for drug, action in gt.items() if action in _ACTIVE_ACTIONS)


def option_to_active_set(option: dict) -> frozenset:
    """Pred active set: drug names where predicted action is continue or start."""
    drugs = option.get("drugs", []) if option else []
    return frozenset(
        d["drug"] for d in drugs
        if d.get("drug") and d.get("action") in _ACTIVE_ACTIONS
    )


def option_to_frozenset(option: dict) -> frozenset:
    """Keep for backwards compat (error analysis etc)."""
    drugs = option.get("drugs", []) if option else []
    return frozenset(
        (d["drug"], d["action"])
        for d in drugs
        if d.get("drug") and d.get("action") in _VALID_ACTIONS
    )


def fmt_drugs(option: dict) -> str:
    drugs = option.get("drugs", []) if option else []
    return "; ".join(
        f"{d['drug']}:{d['action']}"
        for d in drugs
        if d.get("action") in _VALID_ACTIONS
    )


def grade_patient(gt: dict, pred: dict) -> dict:
    gt_active = gt_to_active_set(gt)
    matched_option = None

    for n in [1, 2, 3]:
        opt = pred.get(f"option_{n}", {})
        if option_to_active_set(opt) == gt_active:
            matched_option = n
            break

    def active_str(opt):
        return "; ".join(sorted(option_to_active_set(opt))) if opt else ""

    return {
        "match": matched_option is not None,
        "matched_option": matched_option,
        "gt_active": "; ".join(sorted(gt_active)),
        "gt_size": len(gt_active),
        "option_1_active": active_str(pred.get("option_1")),
        "option_2_active": active_str(pred.get("option_2")),
        "option_3_active": active_str(pred.get("option_3")),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    with open(RECONC_PATH, "r", encoding="utf-8") as f:
        reconc = json.load(f)

    rows = []

    for visit_n in [1, 2, 3]:
        pred_path = PRED_TEMPLATE.format(n=visit_n)
        if not os.path.exists(pred_path):
            print(f"[WARN] Missing predictions: {pred_path}")
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)

        n_total = n_match = 0

        for pid, visits in reconc.items():
            if not visits.get(f"Visit_{visit_n}"):
                continue

            gt = extract_gt(visits, visit_n)

            pred = pred_data.get(pid, {})
            grade = grade_patient(gt, pred)
            gt_size = grade["gt_size"]
            mono_poly = "mono" if gt_size <= 1 else "poly"

            rows.append({
                "patient_id": pid,
                "visit": visit_n,
                "mono_poly": mono_poly,
                **grade,
            })

            n_total += 1
            if grade["match"]:
                n_match += 1

        pct = n_match / n_total * 100 if n_total else 0
        print(f"Visit {visit_n}: {n_match}/{n_total} matched  ({pct:.1f}%)")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("GRADING SUMMARY")
    print(f"{'='*60}")
    for visit_n in [1, 2, 3]:
        vdf = df[df["visit"] == visit_n]
        if len(vdf) == 0:
            continue
        acc = vdf["match"].mean() * 100
        print(f"\nVisit {visit_n}  (n={len(vdf)})  — overall: {acc:.1f}%")
        for mp in ["mono", "poly"]:
            sub = vdf[vdf["mono_poly"] == mp]
            if len(sub):
                print(f"  {mp:4s} (n={len(sub):3d}): {sub['match'].mean()*100:.1f}%")

    overall = df["match"].mean() * 100
    print(f"\nAll visits combined (n={len(df)}): {overall:.1f}%")
    print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()
