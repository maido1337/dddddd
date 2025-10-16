#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cv_matcher.py
Izstrādātājs: (tu)
Mērķis: salīdzināt JD ar katru no 3 CV, ģenerēt JSON (HR fokuss) un īsu MD pārskatu.
Atbalsta: Gemini Flash 2.5 (izmantojot call_gemini placeholder) vai lokālu fallback.
"""

import os
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from collections import Counter, defaultdict

# Konfigurācija
INPUT_DIR = Path("sample_inputs")
OUTPUT_DIR = Path("outputs")
PROMPT_DIR = Path("prompts")
OUTPUT_DIR.mkdir(exist_ok=True)
PROMPT_DIR.mkdir(exist_ok=True)

JD_FILE = INPUT_DIR / "jd.txt"
CV_FILES = [INPUT_DIR / f"cv{i}.txt" for i in (1,2,3)]

# ---------- Helper utilities ----------
def read_text(path: Path) -> str:
    if not path.exists():
        print(f"[WARN] {path} not found.")
        return ""
    return path.read_text(encoding="utf-8")

def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

# ---------- Gemini call (placeholder) ----------
def call_gemini(prompt: str, temperature: float = 0.0, max_tokens: int = 1200):
    """
    Placeholder to call Gemini Flash 2.5.
    Implementē savu klientu šeit (example: Google Vertex AI Generative API / or other).
    Šī funkcija gaida, ka Gemini atbildēs ar JSON stringu (precīzi tajā formātā, ko pieprasa projekts).
    Ja nevēlies vai nevari izsaukt Gemini, atgriez None -> fallback tiks izmantots.
    """
    # --- EXAMPLE: If you have a function to call Gemini, uncomment and return parsed JSON
    #
    # import requests
    # headers = {"Authorization": f"Bearer {os.getenv('GEMINI_API_KEY')}"}
    # payload = {"model":"gemini-flash-2.5","prompt":prompt,"temperature":temperature}
    # r = requests.post(os.getenv("GEMINI_ENDPOINT"), json=payload, headers=headers, timeout=60)
    # r.raise_for_status()
    # return r.json()
    #
    # For now, return None to use local fallback by default.
    return None

# ---------- Local fallback analyser ----------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9āčēģīķļņōŗšūž\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_terms(text: str, min_len=3):
    text = normalize(text)
    words = [w for w in text.split() if len(w) >= min_len]
    return words

def local_matcher(jd_text: str, cv_text: str):
    """
    Vienkāršs svaru/atslēgvārdu un frāžu salīdzinājums, lai uzģenerētu JSON atbildi.
    Nav tik labs kā LLM, bet darbojas lokāli kā fallback.
    """
    jd_norm = normalize(jd_text)
    cv_norm = normalize(cv_text)
    jd_terms = extract_terms(jd_text)
    cv_terms = extract_terms(cv_text)

    jd_counter = Counter(jd_terms)
    cv_counter = Counter(cv_terms)

    # svarīgākie JD atslēgvārdi — top 30
    top_jd = [t for t,_ in jd_counter.most_common(30)]

    matches = []
    missing = []
    match_weight = 0.0
    total_possible = 0.0

    for term in top_jd:
        total_possible += 1.0
        if term in cv_counter:
            matches.append(term)
            match_weight += 1.0
        else:
            missing.append(term)

    # vienkāršs semantiskais saderības mērs (difflib on whole text)
    seq_sim = SequenceMatcher(None, jd_norm, cv_norm).ratio()  # 0..1

    # kombinē skaitli 0..100
    score = int(round(( (match_weight / max(1,total_possible)) * 0.7 + seq_sim * 0.3 ) * 100))

    # verdict
    if score >= 75:
        verdict = "strong match"
    elif score >= 45:
        verdict = "possible match"
    else:
        verdict = "not a match"

    # strengths = top matching phrases (take top matched terms)
    strengths = matches[:10]

    # missing requirements: top missing JD terms (first 8)
    missing_reqs = missing[:8]

    summary = f"Lokālais novērtējums: {score}/100. {len(strengths)} no galvenajiem JD atslēgvārdiem atrasti."

    json_out = {
        "match_score": score,
        "summary": summary,
        "strengths": strengths,
        "missing_requirements": missing_reqs,
        "verdict": verdict
    }
    return json_out

# ---------- render report ----------
def json_to_markdown_report(candidate_name: str, jd_title: str, json_obj):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    md = []
    md.append(f"# Pārskats: {candidate_name}")
    md.append(f"_Generēts: {now}_\n")
    md.append(f"**JD:** {jd_title}\n")
    md.append(f"**Match score:** **{json_obj.get('match_score', 'N/A')}**\n")
    md.append(f"**Verdict:** {json_obj.get('verdict','')}\n")
    md.append("## Kopsavilkums")
    md.append(json_obj.get("summary",""))
    md.append("\n## Spēcīgākās puses (no CV, kas saskan ar JD)")
    if json_obj.get("strengths"):
        for s in json_obj["strengths"]:
            md.append(f"- {s}")
    else:
        md.append("- Nav identificētas spēcīgas atbilstības.")
    md.append("\n## Trūkst / Nepietiekami redzams JD (rekomendētā uzmanība)")
    if json_obj.get("missing_requirements"):
        for m in json_obj["missing_requirements"]:
            md.append(f"- {m}")
    else:
        md.append("- Nav identificētu trūkumu.")
    md.append("\n## Ieteikums")
    md.append(f"- **{json_obj.get('verdict','')}**")
    return "\n".join(md)

# ---------- main flow ----------
def process_one(jd_text: str, cv_path: Path, idx: int):
    cv_text = read_text(cv_path)
    candidate_name = cv_path.stem
    prompt_path = PROMPT_DIR / f"prompt_{candidate_name}.md"

    # create prompt
    prompt_md = f"# Gemini prompt for candidate {candidate_name}\n\n" \
                "## Job description (JD)\n\n" \
                f"{jd_text}\n\n" \
                "## Candidate CV\n\n" \
                f"{cv_text}\n\n" \
                "### Instructions to model (IMPORTANT):\n" \
                "Return ONLY a JSON object exactly in this structure:\n\n" \
                "{\n" \
                '  "match_score": 0-100,\n' \
                '  "summary": "short Latvian summary",\n' \
                '  "strengths": ["..."],\n' \
                '  "missing_requirements": ["..."],\n' \
                '  "verdict": "strong match | possible match | not a match"\n' \
                "}\n\n" \
                "Score should reflect how well the CV matches main JD requirements. Be concise and HR-focused."
    save_text(prompt_path, prompt_md)

    # call Gemini
    gemini_resp = call_gemini(prompt_md, temperature=0.0)
    if gemini_resp:
        # Expect gemini_resp to be JSON or string that is JSON
        if isinstance(gemini_resp, str):
            try:
                resp_json = json.loads(gemini_resp)
            except Exception as e:
                print(f"[WARN] Could not parse Gemini response as JSON: {e}")
                resp_json = None
        elif isinstance(gemini_resp, dict):
            resp_json = gemini_resp
        else:
            resp_json = None
    else:
        resp_json = None

    # fallback local analyzer
    if resp_json is None:
        print(f"[INFO] Using local fallback analysis for {candidate_name}")
        resp_json = local_matcher(jd_text, cv_text)

    # save json & report
    out_json_path = OUTPUT_DIR / f"{candidate_name}.json"
    save_json(out_json_path, resp_json)

    report_md = json_to_markdown_report(candidate_name, "Job description", resp_json)
    out_md_path = OUTPUT_DIR / f"{candidate_name}_report.md"
    save_text(out_md_path, report_md)

    print(f"[OK] Processed {candidate_name}: saved {out_json_path} and {out_md_path}")

def main():
    jd_text = read_text(JD_FILE)
    if not jd_text:
        print("[ERROR] JD file missing or empty. Exiting.")
        sys.exit(1)

    for i, cv in enumerate(CV_FILES, start=1):
        process_one(jd_text, cv, i)

if __name__ == "__main__":
    main()
