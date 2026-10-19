#!/usr/bin/env python3
"""
build_dataset.py - feeds the job postings we analysed in the chat into
extract_features.py and writes dataset.csv (one row per posting).

Run (both files in the same folder):

    python build_dataset.py
    python build_dataset.py --out my_dataset.csv --today 2026-10-14

Notes
-----
* The texts below were transcribed from your screenshots and pasted ads.
  Long "About the company" and benefits boilerplate was shortened; the parts
  that carry signals (pay, location, deadline, requirements) were kept.
* `label` is the verdict I gave in the chat: apply / verify / ignore.
  Edit any label you disagree with - that is YOUR ground truth for the model.
* To add a posting: append a dict to JOBS (title, company, label, text).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from dataset import dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import append_row, extract_features  # noqa: E402

JOBS = dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Build dataset.csv from the analysed postings.")
    ap.add_argument("--out", default="dataset.csv", help="output CSV (overwritten)")
    ap.add_argument("--today", default="2026-10-14", help="YYYY-MM-DD used for deadline checks")
    args = ap.parse_args()
    today = datetime.strptime(args.today, "%Y-%m-%d").date()

    out = Path(args.out)
    out.unlink(missing_ok=True)

    shown = ["unpaid_signal", "onsite_required", "hybrid", "occasional_visits",
             "location_restriction", "work_auth_required", "citizenship_required",
             "requires_student_status", "agency_signals", "us_centric_signal",
             "inr_signal", "is_contract"]

    print(f"{'#':>2}  {'label':6} {'company':26} {'pay':22} flags")
    print("-" * 110)
    for i, job in enumerate(JOBS, 1):
        text = job["text"].strip()
        feats, evidence = extract_features(text, today=today, title=job["title"])
        row = {"title": job["title"], "company": job["company"], "label": job["label"],
               **feats, "evidence_json": json.dumps(evidence, ensure_ascii=False)}
        append_row(out, row)

        if feats["has_salary"]:
            pay = f"{feats['salary_min']:,.0f}-{feats['salary_max']:,.0f} {feats['salary_currency']}"
            pay += f"/{feats['salary_period']}" if feats["salary_period"] else ""
        else:
            pay = "-"
        flags = [f for f in shown if feats[f]]
        if feats["deadline_passed"]:
            flags.append("deadline_passed")
        print(f"{i:>2}  {job['label']:6} {job['company'][:25]:26} {pay:22} {', '.join(flags)}")

    counts = Counter(j["label"] for j in JOBS)
    print("-" * 110)
    print(f"{len(JOBS)} postings -> {out}  |  labels: {dict(counts)}")


if __name__ == "__main__":
    main()
