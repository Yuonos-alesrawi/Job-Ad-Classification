#!/usr/bin/env python3
"""
extract_features.py - turn the raw text of a job posting into structured features.

Standard library only (no pip install needed).

Examples
--------
    python extract_features.py job.txt
    cat job.txt | python extract_features.py -
    python extract_features.py job.txt --today 2026-10-14 \
        --title "Junior Python Developer" --company "Acme" \
        --label ignore --append dataset.csv

Notes
-----
* Dates like 04.10.2026 or 04/10/2026 are read DAY-first (dd.mm.yyyy).
* Salary figures are returned as written (min/max/currency/period). The
  "monthly" values are rough: hour x 160, week x 4.33, year / 12.
* Every boolean flag also stores the text that triggered it ("evidence"),
  so a model (or you) can explain WHY a posting was flagged.
* Spelled-out numbers ("three years") are not parsed, only digits.
* The flags are hints, not verdicts. Read the evidence before trusting them.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from datetime import date, datetime
from pathlib import Path

# --------------------------------------------------------------------------
# 1. Configuration (edit freely)
# --------------------------------------------------------------------------

# Technologies you already know. Used for the "stack overlap" feature,
# which imitates LinkedIn's "matches X of Y skills" signal.
USER_STACK = {
    "python", "django", "django rest framework", "celery", "redis", "docker",
    "postgresql", "mysql", "sql", "git", "websocket", "rest api",
    "pandas", "numpy", "scikit-learn", "tensorflow",
}

# canonical name -> regex alternation (matched case-insensitively, whole word)
TECH_TERMS = {
    "python": r"python3?", "django": r"django",
    "django rest framework": r"django rest framework|drf",
    "flask": r"flask", "fastapi": r"fastapi", "java": r"java",
    "javascript": r"javascript|js", "typescript": r"typescript",
    "node.js": r"node\.?js", "react": r"react(?:\.js|js)?",
    "next.js": r"next\.?js", "vue": r"vue(?:\.js)?", "angular": r"angular",
    "php": r"php", "wordpress": r"wordpress", "golang": r"golang",
    "rust": r"rust", "c++": r"c\+\+", "c#": r"c\#", ".net": r"\.net|dotnet",
    "spring boot": r"spring boot|spring", "sql": r"sql", "nosql": r"nosql",
    "postgresql": r"postgres(?:ql)?", "mysql": r"mysql",
    "mongodb": r"mongo(?:db)?", "redis": r"redis", "kafka": r"kafka",
    "spark": r"spark", "docker": r"docker", "kubernetes": r"kubernetes|k8s",
    "aws": r"aws|amazon web services", "azure": r"azure",
    "gcp": r"gcp|google cloud", "terraform": r"terraform",
    "ci/cd": r"ci/cd|ci-cd|cicd", "git": r"git(?:hub|lab)?",
    "rest api": r"rest(?:ful)?\s+apis?|restful", "graphql": r"graphql",
    "websocket": r"websockets?", "celery": r"celery",
    "langchain": r"langchain", "langgraph": r"langgraph", "rag": r"rag",
    "llm": r"llms?|large language models?", "pytorch": r"pytorch",
    "tensorflow": r"tensorflow", "scikit-learn": r"scikit-learn|sklearn",
    "pandas": r"pandas", "numpy": r"numpy", "power bi": r"power ?bi",
    "tableau": r"tableau", "airflow": r"airflow", "linux": r"linux",
    "agile": r"agile|scrum",
}
TECH_RE = {
    name: re.compile(rf"(?<![\w+#.-])(?:{pat})(?![\w+#])", re.I)
    for name, pat in TECH_TERMS.items()
}

# Boolean signals: name -> list of regexes (case-insensitive).
# English first, a few Arabic phrases at the end of some lists.
SIGNALS: dict[str, list[str]] = {
    "unpaid_signal": [
        r"\bunpaid\b", r"\bno (?:stipend|salary|pay)\b",
        r"not in a position to offer a conventional salary",
        r"\bequity[- ]only\b", r"\bwithout (?:pay|salary|compensation)\b",
        r"\bvolunteer\b", r"بدون (?:أجر|راتب)", r"غير مدفوع",
    ],
    "equity_mention": [r"\bequity\b", r"\bstock options?\b"],
    "remote_claim": [
        r"\bremote(?:ly)?\b", r"work from (?:anywhere|home)",
        r"\b100% remote\b", r"عن بعد",
    ],
    "onsite_required": [
        r"\bon[- ]?site\b", r"\boffice[- ]based\b", r"\bin[- ]person\b",
        r"\bin[- ]office (?:work|presence|days?|required|position|role)",
        r"work(?:ing)? (?:from|in|at) (?:the |our )?office",
        r"الدوام من مقر الشركة",
    ],
    "hybrid": [
        r"\bhybrid\b", r"remote work is available for up to half",
        r"\b(?:\d|two|three)\s+days?\s+(?:per week\s+)?(?:in|at)\s+(?:the\s+)?office",
    ],
    "occasional_visits": [
        r"occasional (?:client )?(?:visits|site visits|travel)",
        r"(?:some|few) in[- ]office", r"team collaboration sessions",
        r"office visits",
    ],
    "location_restriction": [
        r"\b(?:must|need to|have to)\s+be\s+(?:based|located|resident|living|residing)\s+in\b",
        r"\bapplicants?\s+must\s+be\s+based\b",
        r"\bmust\s+(?:reside|live)\s+in\b",
        r"\bonly\s+(?:applicants?|candidates?)\s+(?:from|in|based|located)\b",
        r"\b(?:UK|US|EU|UAE|KSA)[- ]based\b", r"\blocal candidates\b",
    ],
    "work_auth_required": [
        r"authori[sz]ed to work", r"work authori[sz]ation", r"right to work",
        r"eligible to work", r"\bW-?2\b", r"\bE-?Verify\b",
        r"security clearance",
    ],
    "no_visa_sponsorship": [
        r"(?:no|not|unable to|cannot|can't|won't|will not|do not|don't)\s+(?:\w+\s+){0,3}sponsor",
    ],
    "visa_mention": [r"\bvisa\b", r"\bsponsorship\b"],
    "us_centric_signal": [
        r"\b401\s?\(?k\)?", r"\bnational (?:pay )?range\b",
        r"health insurance (?:premiums?|reimbursement)", r"\bveteran status\b",
        r"\bEEO\b", r"\bat[- ]will\b",
    ],
    "citizenship_required": [
        r"\b(?:hold|holding|possess)\w*\s+(?:\w+\s+){0,2}citizenship",
        r"citizenship\s+(?:is\s+)?(?:required|mandatory|needed)",
        r"\b(?:must|need to)\s+be\s+(?:an?\s+)?(?:\w+\s+){0,2}citizens?\b",
        r"\b(?:US|U\.S\.|UK|EU|Canadian|American|British|German)\s+citizens?\b",
        r"الجنسية",
    ],
    "requires_student_status": [
        r"mandatory internship", r"part of (?:a|your|the) (?:school|university|academic|study|studies|degree)",
        r"currently (?:enrolled|pursuing|studying)", r"must be (?:a )?(?:current )?student",
        r"\bstudents? only\b", r"pursuing (?:a|an) (?:undergraduate|bachelor|master|degree)",
        r"for students as part of",
    ],
    "agency_signals": [
        r"hiring for (?:one of )?our clients?",
        r"on behalf of (?:our|a|one of our) clients?",
        r"we are (?:a |an )?(?:recruitment|staffing|talent|executive search) (?:agency|firm|company|partner)",
        r"global leader in the .{0,100}? industry",
        r"regardless of background, experience, or prior employment history",
        r"confidential (?:client|company|employer)",
        r"(?:is|are) working with (?:a|an)\s+[\w\s,-]{0,60}?(?:business|company|startup|client)\s+to\s+recruit",
    ],
    "is_contract": [
        r"\bfreelance\b", r"\bcontract basis\b", r"\bcontract (?:role|position)\b",
        r"\bcontract[- ]to[- ]hire\b", r"\bhourly contract\b", r"\bper[- ]task\b",
        r"\b1099\b", r"\btype:\s*(?:hourly )?contract", r"متعاقد",
    ],
    "level_intern": [r"\bintern(?:ship)?\b", r"\btrainee\b"],
    "level_junior": [
        r"\bjunior\b", r"entry[- ]level", r"\bfresh(?:er|\s+grad(?:uate)?)\b",
        r"\bnew grad\b", r"\bgraduate\b",
    ],
    "level_senior": [
        r"\bsenior\b", r"\blead\b", r"\bprincipal\b", r"\bstaff\b",
        r"\bfounding\b", r"\barchitect\b",
    ],
    "inr_signal": [r"₹", r"\bINR\b", r"\brupees?\b", r"\blakhs?\b", r"\bLPA\b"],
}
SIGNAL_RE = {k: [re.compile(p, re.I) for p in v] for k, v in SIGNALS.items()}

HYPE_PATTERNS = [
    r"not a corporate", r"no coffee runs", r"rock ?stars?", r"\bninjas?\b",
    r"\bgurus?\b", r"fast[- ]paced", r"game[- ]chang", r"exciting opportunity",
    r"unique opportunity", r"cutting[- ]edge", r"\bfrontier\b", r"world[- ]class",
    r"hit the ground running", r"\bpassionate\b", r"\bownership\b",
    r"if you(?:'ve| have) read this far",
]
HYPE_RE = [re.compile(p, re.I) for p in HYPE_PATTERNS]
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u26A1]")

# --------------------------------------------------------------------------
# 2. Text normalisation
# --------------------------------------------------------------------------

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_BIDI = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069]")


def normalize(text: str) -> str:
    """Unicode-normalise, strip bidi marks, convert Arabic-Indic digits."""
    text = unicodedata.normalize("NFKC", text)
    text = _BIDI.sub("", text).translate(_DIGITS)
    return text.replace("\r\n", "\n")


# --------------------------------------------------------------------------
# 3. Salary
# --------------------------------------------------------------------------

CUR = (r"(?:£|\$|€|₹|(?<![A-Za-z])(?:USD|GBP|EUR|INR|EGP|AED|SAR|JOD|CAD|AUD)\b)")
NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
# reject "$40M", "$2 billion", and partial numbers produced by backtracking
TAIL = r"(?![\d.,]\d)(?!\s?(?:[mMbB]\b|million|billion))"
SEP = r"\s?(?:-|–|—|to|and)\s?"

# (kind, regex) in priority order; overlapping matches are discarded.
MONEY_PATTERNS = [
    ("range_before", rf"({CUR})\s?({NUM})\s?([kK])?{SEP}(?:{CUR}\s?)?({NUM})\s?([kK])?{TAIL}"),
    ("range_after", rf"({NUM})\s?([kK])?{SEP}({NUM})\s?([kK])?\s?({CUR})"),
    ("single_before", rf"({CUR})\s?({NUM})\s?([kK])?{TAIL}"),
    ("single_after", rf"({NUM})\s?([kK])?\s?({CUR})"),
]
MONEY_RE = [(k, re.compile(p, re.I)) for k, p in MONEY_PATTERNS]
SYMBOLS = {"£": "GBP", "$": "USD", "€": "EUR", "₹": "INR"}

PERIODS = [
    ("hour", r"/\s?(?:hr|hour)\b|per hour|hourly|an hour|بالساعة"),
    ("month", r"/\s?(?:mo|month)\b|per month|monthly|a month|شهري|/\s?شهر"),
    ("year", r"/\s?(?:yr|year)\b|per (?:year|annum)|annual|yearly|a year|p\.a\.|\bpa\b|سنوي"),
    ("week", r"/\s?(?:wk|week)\b|per week|weekly"),
]
TO_MONTHLY = {"hour": 160, "week": 4.33, "month": 1, "year": 1 / 12}


def _amount(num: str, k: str | None) -> float:
    return float(num.replace(",", "")) * (1000 if k else 1)


def extract_salary(text: str) -> dict:
    mentions = []  # (start, end, lo, hi, currency)
    taken: list[tuple[int, int]] = []
    for kind, rx in MONEY_RE:
        for m in rx.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            g = m.groups()
            if kind == "range_before":
                cur, lo, hi = g[0], _amount(g[1], g[2]), _amount(g[3], g[4])
                if g[4] and not g[2]:
                    lo *= 1000  # "25-35k"
            elif kind == "range_after":
                cur, lo, hi = g[4], _amount(g[0], g[1]), _amount(g[2], g[3])
                if g[3] and not g[1]:
                    lo *= 1000
            elif kind == "single_before":
                cur, lo = g[0], _amount(g[1], g[2])
                hi = lo
            else:
                cur, lo = g[2], _amount(g[0], g[1])
                hi = lo
            cur = SYMBOLS.get(cur, cur.upper())
            mentions.append((m.start(), m.end(), min(lo, hi), max(lo, hi), cur))
            taken.append((m.start(), m.end()))

    empty = {
        "has_salary": False, "salary_min": None, "salary_max": None,
        "salary_currency": None, "salary_period": None,
        "salary_monthly_min": None, "salary_monthly_max": None,
    }
    if not mentions:
        return empty

    mentions.sort()
    cluster = [mentions[0]]
    for m in mentions[1:]:
        if m[4] == cluster[0][4] and m[0] - cluster[-1][1] <= 45:
            cluster.append(m)
    lo = min(m[2] for m in cluster)
    hi = max(m[3] for m in cluster)
    cur = cluster[0][4]

    window = text[cluster[-1][1]: cluster[-1][1] + 30].lower()
    period = next((p for p, rx in PERIODS if re.search(rx, window)), None)
    out = dict(empty, has_salary=True, salary_min=lo, salary_max=hi,
               salary_currency=cur, salary_period=period)
    if period:
        out["salary_monthly_min"] = round(lo * TO_MONTHLY[period])
        out["salary_monthly_max"] = round(hi * TO_MONTHLY[period])
    return out


# --------------------------------------------------------------------------
# 4. Deadline and experience
# --------------------------------------------------------------------------

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
DATE_PATTERNS = [
    ("dmy_num", re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")),
    ("ymd_iso", re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")),
    ("d_month_y", re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTH_RE})\.?,?\s+(\d{{4}})\b", re.I)),
    ("month_d_y", re.compile(rf"\b({MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)),
]
DEADLINE_CTX = re.compile(
    r"deadline|apply\s+(?:by|before|until)|applications?\s+(?:close|due|until|by)"
    r"|closing date|closes?\b|due (?:date|by)|before|until|ينتهي|آخر موعد|الموعد النهائي",
    re.I,
)


def _to_date(kind: str, g: tuple) -> date | None:
    try:
        if kind == "dmy_num":
            d, m, y = int(g[0]), int(g[1]), int(g[2])
            if m > 12 and d <= 12:
                d, m = m, d
        elif kind == "ymd_iso":
            y, m, d = int(g[0]), int(g[1]), int(g[2])
        elif kind == "d_month_y":
            d, m, y = int(g[0]), MONTHS[g[1].lower()], int(g[2])
        else:
            m, d, y = MONTHS[g[0].lower()], int(g[1]), int(g[2])
        return date(y, m, d)
    except (ValueError, KeyError):
        return None


def extract_deadline(text: str, today: date) -> dict:
    found = []
    for kind, rx in DATE_PATTERNS:
        for m in rx.finditer(text):
            context = text[max(0, m.start() - 80): m.start()]
            if DEADLINE_CTX.search(context):
                d = _to_date(kind, m.groups())
                if d:
                    found.append((m.start(), d))
    if not found:
        return {"deadline": None, "deadline_passed": None, "days_until_deadline": None}
    d = min(found)[1]  # first one in the text
    return {
        "deadline": d.isoformat(),
        "deadline_passed": d < today,
        "days_until_deadline": (d - today).days,
    }


def extract_experience(text: str) -> dict:
    cands = []
    for m in re.finditer(
        r"(\d{1,2})\s*(?:\+|to|-|–|—)?\s*(\d{1,2})?\s*\+?\s*(?:years?|yrs?)\b", text, re.I
    ):
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else None
        ctx = text[max(0, m.start() - 60): m.end() + 60].lower()
        cands.append((lo, hi, "experience" in ctx or "خبرة" in ctx))
    pool = [c for c in cands if c[2]] or cands
    if not pool:
        return {"experience_years_min": None, "experience_years_max": None}
    lo, hi, _ = pool[0]
    return {"experience_years_min": lo, "experience_years_max": hi}


# --------------------------------------------------------------------------
# 5. Main extraction
# --------------------------------------------------------------------------

def extract_features(text: str, today: date | None = None, title: str = "") -> tuple[dict, dict]:
    """Return (features, evidence). `evidence` maps each flag to up to 3 matched snippets."""
    today = today or date.today()
    text = normalize(text)
    scan = f"{title}\n{text}" if title else text

    feats: dict = {"word_count": len(text.split()), "char_count": len(text)}
    evidence: dict[str, list[str]] = {}

    # boolean signals
    counts: dict[str, int] = {}
    for name, rxs in SIGNAL_RE.items():
        hits = [m.group(0).strip() for rx in rxs for m in rx.finditer(scan)]
        feats[name] = bool(hits)
        counts[name] = len(hits)
        if hits:
            evidence[name] = hits[:3]
    feats["agency_signal_count"] = counts["agency_signals"]

    # structured fields
    feats.update(extract_salary(text))
    feats.update(extract_deadline(text, today))
    feats.update(extract_experience(text))

    # technologies
    found = sorted(name for name, rx in TECH_RE.items() if rx.search(scan))
    overlap = [t for t in found if t in USER_STACK]
    feats["tech_count"] = len(found)
    feats["tech_terms"] = found
    feats["stack_overlap_count"] = len(overlap)
    feats["stack_overlap_ratio"] = round(len(overlap) / len(found), 2) if found else 0.0

    # style / marketing
    hype = [rx.pattern for rx in HYPE_RE if rx.search(text)]
    feats["hype_count"] = len(hype)
    feats["emoji_count"] = len(EMOJI_RE.findall(text))
    feats["exclamation_count"] = text.count("!")

    return feats, evidence


# --------------------------------------------------------------------------
# 6. CLI
# --------------------------------------------------------------------------

def _csv_value(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, list):
        return "|".join(v)
    return "" if v is None else v


def append_row(path: Path, row: dict) -> None:
    fields = list(row)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
        if header != fields:
            sys.exit(f"error: columns in {path} differ from the current feature set. "
                     "Use a new CSV file (or add the missing columns).")
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: _csv_value(v) for k, v in row.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract features from a job posting.")
    ap.add_argument("file", nargs="?", default="-", help="text file with the posting, or - for stdin")
    ap.add_argument("--today", help="YYYY-MM-DD (default: system date)")
    ap.add_argument("--title", default="", help="job title (optional, helps level detection)")
    ap.add_argument("--company", default="", help="company name (stored in the CSV only)")
    ap.add_argument("--label", default="", help="your verdict: apply / verify / ignore (for training data)")
    ap.add_argument("--append", metavar="CSV", help="append one row to this CSV dataset")
    ap.add_argument("--quiet", action="store_true", help="do not print the JSON")
    args = ap.parse_args()

    if args.file == "-":
        if sys.stdin.isatty():
            print("Paste the job text, then press Ctrl-D (Ctrl-Z then Enter on Windows):", file=sys.stderr)
        text = sys.stdin.read()
    else:
        text = Path(args.file).read_text(encoding="utf-8")

    today = datetime.strptime(args.today, "%Y-%m-%d").date() if args.today else date.today()
    feats, evidence = extract_features(text, today=today, title=args.title)

    if not args.quiet:
        print(json.dumps({"features": feats, "evidence": evidence},
                         ensure_ascii=False, indent=2, default=str))

    if args.append:
        row = {"title": args.title, "company": args.company, "label": args.label, **feats,
               "evidence_json": json.dumps(evidence, ensure_ascii=False)}
        append_row(Path(args.append), row)
        print(f"appended to {args.append}", file=sys.stderr)


if __name__ == "__main__":
    main()