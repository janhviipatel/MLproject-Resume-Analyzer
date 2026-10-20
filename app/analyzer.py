"""Core analysis engine: framework-independent so it is easy to test."""
import json
import re
from pathlib import Path

from typing import Optional

import numpy as np

from .embedder import Embedder
from .features import first_word, hand_features, weak_phrases_in, ACTION_VERBS
from .parser import BULLET
from .nlp import KeywordExtractor, keyword_gap, readability, rewrite, infer_sections, explain
from .train import load_models, read_csv

DATA = Path(__file__).parent / "data"
SKILLS = json.loads((DATA / "skills.json").read_text(encoding="utf-8"))

SECTIONS = {
    "Summary": re.compile(r"^(professional\s+)?(summary|profile|objective|about)\b", re.I),
    "Experience": re.compile(r"^(work\s+|professional\s+)?(experience|employment)\b", re.I),
    "Education": re.compile(r"^education\b", re.I),
    "Skills": re.compile(r"^(technical\s+|core\s+)?skills\b", re.I),
    "Projects": re.compile(r"^projects\b", re.I),
}


def _compile():
    out = {}
    for name, aliases in SKILLS.items():
        alts = "|".join(re.escape(a) for a in sorted(set(aliases + [name.lower()]), key=len, reverse=True))
        out[name] = re.compile(rf"(?<![\w+#.]){'(?:' + alts + ')'}(?![\w+#])", re.I)
    return out


PATTERNS = _compile()


def extract_skills(text: str) -> list:
    return [name for name, rx in PATTERNS.items() if rx.search(text)]


def split_requirements(jd: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+|\n+|;", jd)
    out = []
    for p in parts:
        p = BULLET.sub("", p).strip(" -\t")
        if len(p.split()) >= 4:
            out.append(p)
    return out[:25]


def get_bullets(resume: str) -> list:
    bullets = [BULLET.sub("", l).strip() for l in resume.split("\n") if BULLET.match(l)]
    if bullets:
        return bullets
    # No bullet glyphs: fall back to sentence-like lines that are not headings.
    return [l.strip() for l in resume.split("\n") if len(l.split()) >= 7][:30]


def bullet_reasons(text: str) -> list:
    reasons = []
    if not re.search(r"\d", text):
        reasons.append("No number or scale")
    weak = weak_phrases_in(text)
    if weak:
        reasons.append(f"Weak phrase: “{weak[0]}”")
    if first_word(text) not in ACTION_VERBS:
        reasons.append("Doesn’t open with an action verb")
    if len(text.split()) < 6:
        reasons.append("Too short to show impact")
    return reasons


class Engine:
    def __init__(self):
        rows = read_csv("bullets.csv") + read_csv("roles.csv")
        corpus = [r["text"] for r in rows] + [(DATA / n).read_text(encoding="utf-8") for n in ("sample_resume.txt", "sample_job.txt")]
        self.embedder = Embedder(corpus)
        self.models = load_models()
        self.clf = self.models["bullet"]
        self.kw = KeywordExtractor(corpus, {a.lower() for al in SKILLS.values() for a in al} | {n.lower() for n in SKILLS})

    def _roles(self, resume: str) -> dict:
        rp, sp = self.models["role"], self.models["seniority"]
        pr = rp.predict_proba([resume])[0]
        top = np.argsort(pr)[::-1][:3]
        sr = sp.predict_proba([resume])[0]
        return {
            "roles": [{"role": str(rp.classes_[i]), "prob": round(float(pr[i]), 2)} for i in top],
            "seniority": {"label": str(sp.classes_[int(np.argmax(sr))]),
                          "probs": {str(c): round(float(p), 2) for c, p in zip(sp.classes_, sr)}},
            "confidence": round(float(pr.max()), 2),
        }

    def analyze(self, resume: str, jd: str = "") -> dict:
        resume, jd = resume.strip(), (jd or "").strip()
        lines = [l.strip() for l in resume.split("\n")]
        words = len(resume.split())

        # structure (headings) + content-based inference for resumes without headings
        sections = [k for k, rx in SECTIONS.items() if any(rx.match(l) for l in lines)]
        inferred = infer_sections(lines)
        sections_inferred = [k for k, v in inferred.items() if v and k not in sections]
        contact = {
            "email": bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", resume)),
            "phone": bool(re.search(r"\d{3}[\s).-]*\d{3}[\s.-]*\d{4}", resume)),
            "link": bool(re.search(r"linkedin\.com|github\.com", resume, re.I)),
        }
        present = len(set(sections) | set(k for k in sections_inferred if k != "Summary" or True))
        structure = round(100 * (0.6 * min(present, 5) / 5 + 0.4 * sum(contact.values()) / 3))

        # bullet quality + explanations + rewrites
        bullets = get_bullets(resume)
        bullet_out, quality = [], 0
        if bullets:
            proba = self.clf.predict_proba(bullets)[:, 1]
            quality = round(100 * float(np.mean(proba)))
            why = explain(self.clf, bullets)
            for b, p, w in zip(bullets, proba, why):
                p = float(p)
                bullet_out.append({"text": b, "quality": round(p, 2), "confidence": round(abs(p - 0.5) * 2, 2),
                                   "reasons": bullet_reasons(b), "why": w,
                                   "rewrite": rewrite(b) if p < 0.75 else None})

        read = readability(resume, bullets)
        role_info = self._roles(resume)

        # job fit: skills taxonomy + weighted keywords + semantic requirement matching
        r_skills = extract_skills(resume)
        fit, req_out, j_skills, matched, missing, kws, kscore = None, [], [], [], [], [], None
        if jd:
            j_skills = extract_skills(jd)
            matched = [s for s in j_skills if s in r_skills]
            missing = [s for s in j_skills if s not in r_skills]
            skill_cov = len(matched) / len(j_skills) if j_skills else None
            kws = self.kw.extract(jd)
            kscore = keyword_gap(resume, kws)
            reqs = split_requirements(jd)
            evidence = bullets + [l for l in lines if len(l.split()) >= 4 and l not in bullets]
            sem = None
            if reqs and evidence:
                scaled = self.embedder.scale(self.embedder.similarity(reqs, evidence))
                best = scaled.argmax(axis=1)
                for i, r in enumerate(reqs):
                    s = float(scaled[i, best[i]])
                    level = "strong" if s >= 0.7 else "partial" if s >= 0.4 else "gap"
                    req_out.append({"requirement": r, "evidence": evidence[best[i]] if s >= 0.4 else None,
                                    "score": round(s, 2), "level": level})
                sem = float(np.mean([r["score"] for r in req_out]))
            comps = [c for c in (skill_cov, sem, kscore) if c is not None]
            fit = round(100 * sum(comps) / len(comps)) if comps else 0

        # overall
        if fit is not None:
            parts = [(quality, 0.30), (structure, 0.15), (read["score"], 0.10), (fit, 0.45)]
        else:
            parts = [(quality, 0.55), (structure, 0.30), (read["score"], 0.15)]
        score = round(sum(v * w for v, w in parts) / sum(w for _, w in parts))

        terms = matched + [k["term"] for k in kws if k.get("present")]
        weak_found = sorted({w for b in bullets for w in weak_phrases_in(b)})
        return {
            "score": score,
            "scores": {"fit": fit, "quality": quality, "structure": structure, "readability": read["score"],
                       "keywords": None if kscore is None else round(100 * kscore)},
            "words": words,
            "engine": self.embedder.name,
            "sections": sections,
            "sections_inferred": sections_inferred,
            "contact": contact,
            "skills": {"resume": r_skills, "job": j_skills, "matched": matched, "missing": missing},
            "keywords": kws,
            "requirements": req_out,
            "bullets": bullet_out,
            "readability": read,
            **role_info,
            "low_confidence": role_info["confidence"] < 0.4,
            "highlights": {"weak": weak_found, "terms": terms},
            "resume_text": resume,
            "tips": self._tips(bullet_out, missing, sections, sections_inferred, contact, words, fit, req_out, read, role_info),
        }

    @staticmethod
    def _tips(bullets, missing, sections, inferred, contact, words, fit, reqs, read, role):
        tips = []
        weak = [b for b in bullets if b["quality"] < 0.5]
        if weak:
            tips.append({"level": "bad", "text": f"{len(weak)} of {len(bullets)} bullets read as weak. See the Rewrite tab for starting points."})
        if missing:
            tips.append({"level": "bad", "text": "Job skills not found on your resume: " + ", ".join(missing[:6]) + ". Add them only if they\u2019re true."})
        gaps = [r for r in reqs if r["level"] == "gap"]
        if gaps:
            tips.append({"level": "mid", "text": f"{len(gaps)} job requirements have no matching evidence. Add a bullet that shows each one."})
        for s in ("Experience", "Education", "Skills"):
            if s not in sections:
                tips.append({"level": "mid" if s in inferred else "bad", "text": f"Add a clearly labeled \u201c{s}\u201d heading" + (" (the content looks present, but ATS parsers rely on headings)." if s in inferred else ".")})
        if not all(contact.values()):
            tips.append({"level": "mid", "text": "Your contact line is missing an email, phone number, or LinkedIn/GitHub link."})
        if read["buzzwords"]:
            tips.append({"level": "mid", "text": "Cut buzzwords (" + ", ".join(read["buzzwords"][:3]) + ") and show the result instead."})
        if read["first_person"] or read["passive_voice"]:
            tips.append({"level": "mid", "text": "Avoid first-person pronouns and passive voice; start bullets with a verb."})
        if words > 750:
            tips.append({"level": "mid", "text": f"At {words} words this is long. Aim for one page, about 400 to 650 words."})
        if role["confidence"] < 0.4:
            tips.append({"level": "mid", "text": "Your resume doesn\u2019t clearly signal one target role. A focused summary and skills line helps."})
        if not weak and bullets:
            tips.append({"level": "good", "text": "Your bullets are specific and measurable. Keep that pattern."})
        if fit is not None and fit >= 75:
            tips.append({"level": "good", "text": "Strong match with this job description."})
        return tips
