"""Lightweight NLP helpers: keywords, readability, rewrites, section inference, explanations."""
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

from .features import ACTION_VERBS, first_word, weak_phrases_in

EXTRA_STOP = set("""looking candidate ideal role join team teams experience years year strong ability skills work working
using use write ship scale own collaborate plus preferred required requirements responsibilities qualifications will must should including etc able new""".split())
STOP = list(ENGLISH_STOP_WORDS | EXTRA_STOP)
STOP_SET = set(STOP)
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.]{1,}[a-zA-Z0-9+#]\b|\b[a-zA-Z]\b"


class KeywordExtractor:
    """RAKE-style phrases scored by TF x rarity in a background corpus.

    Phrases never span stopwords or punctuation, and known skill terms get a boost.
    """

    def __init__(self, corpus: List[str], known: Optional[set] = None):
        bg = TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2), token_pattern=TOKEN).fit(corpus)
        self.idf = dict(zip(bg.get_feature_names_out(), bg.idf_))
        self.default = float(np.median(bg.idf_)) * 1.3
        self.known = known or set()

    def extract(self, jd: str, top: int = 14) -> List[dict]:
        counts = Counter()
        for seg in re.split(r"[,;:!?()\n/]+|\.(?=\s|$)", jd.lower()):
            run = []
            for w in seg.split() + [None]:
                w = re.sub(r"^[^a-z0-9+#]+|[^a-z0-9+#]+$", "", w) if w else None
                if w and w not in STOP_SET and len(w) > 1:
                    run.append(w)
                    continue
                for n in (1, 2):
                    for i in range(len(run) - n + 1):
                        counts[" ".join(run[i:i + n])] += 1
                run = []
        scored = []
        for t, c in counts.items():
            w = (1 + np.log(c)) * self.idf.get(t, self.default) * (1.25 if " " in t else 1.0)
            if t in self.known:
                w *= 2.0
            scored.append((t, w))
        scored.sort(key=lambda x: -x[1])
        chosen = []
        for t, w in scored:
            if any(t in c[0].split() or c[0] in t.split() for c in chosen):
                continue
            chosen.append((t, w))
            if len(chosen) == top:
                break
        mx = max((w for _, w in chosen), default=1)
        return [{"term": t, "weight": round(float(w / mx), 2)} for t, w in chosen]


def keyword_gap(resume: str, keywords: List[dict]) -> Optional[float]:
    """Weighted coverage 0..1 (1 = every important JD term is in the resume)."""
    if not keywords:
        return None
    low = resume.lower()
    tot = got = 0.0
    for k in keywords:
        k["present"] = bool(re.search(r"(?<![a-z0-9])" + re.escape(k["term"]) + r"(?![a-z0-9])", low))
        tot += k["weight"]
        got += k["weight"] if k["present"] else 0
    return got / tot if tot else None


BUZZ = ["synergy", "go-getter", "results-driven", "team player", "hard working", "hard-working", "detail-oriented",
        "self-starter", "think outside the box", "passionate", "motivated", "proven track record", "fast-paced",
        "dynamic", "go getter", "quick learner"]
PASSIVE = re.compile(r"\b(was|were|been|being|is|are)\s+(\w+ed|\w+en)\b", re.I)
FIRST = re.compile(r"\b(i|my|me|myself)\b", re.I)


def readability(text: str, bullets: List[str]) -> dict:
    units = bullets or [l for l in text.split("\n") if len(l.split()) >= 5]
    lens = [len(u.split()) for u in units] or [0]
    avg = sum(lens) / len(lens)
    low = text.lower()
    buzz = [b for b in BUZZ if b in low]
    passive = len(PASSIVE.findall(text))
    first = len(FIRST.findall(text))
    score = 100 - max(0, avg - 22) * 2.5 - max(0, 8 - avg) * 3 - 8 * len(buzz) - 6 * passive - 5 * first
    return {"score": int(max(0, min(100, round(score)))), "avg_bullet_words": round(avg, 1), "buzzwords": buzz,
            "passive_voice": passive, "first_person": first, "long_bullets": sum(1 for n in lens if n > 30)}


GERUND = {"maintaining": "Maintained", "managing": "Managed", "writing": "Wrote", "fixing": "Fixed", "developing": "Developed",
          "testing": "Tested", "building": "Built", "supporting": "Supported", "handling": "Handled", "creating": "Created",
          "designing": "Designed", "running": "Ran", "analyzing": "Analyzed", "monitoring": "Monitored", "updating": "Updated",
          "preparing": "Prepared", "coordinating": "Coordinated", "implementing": "Implemented"}
LEAD = re.compile(r"^(responsible for|worked on|worked with|helped with|helped|assisted with|assisted|participated in|"
                  r"duties included|tasked with|involved in|in charge of|contributed to|supported)\s+", re.I)


def rewrite(bullet: str) -> Optional[str]:
    """Template rewrite with placeholders. Never invents numbers or facts."""
    rest = LEAD.sub("", bullet.strip().rstrip("."))
    words = rest.split()
    verb = None
    if words and words[0].lower() in GERUND:
        verb, rest = GERUND[words[0].lower()], " ".join(words[1:])
    elif first_word(rest) in ACTION_VERBS:
        verb = None
    else:
        verb = "[Action verb]"
    out = f"{verb} {rest}" if verb else rest
    if not re.search(r"\d", out):
        out += ", [improving/reducing] [metric] by [X%]"
    return out[0].upper() + out[1:]


DEGREE = re.compile(r"\b(b\.?s\.?|b\.?tech|b\.?e\.?|m\.?s\.?|m\.?tech|mba|bachelor|master|ph\.?d|university|college|institute)\b", re.I)
DATES = re.compile(r"\b(19|20)\d\d\s*[-–—to]+\s*((19|20)\d\d|present|current)\b", re.I)


def infer_sections(lines: List[str]) -> Dict[str, int]:
    """Heuristic line classifier: finds content types even when headings are missing."""
    found = {"Education": 0, "Experience": 0, "Skills": 0, "Summary": 0}
    body = [l for l in lines if l]
    for i, l in enumerate(body):
        if DEGREE.search(l):
            found["Education"] += 1
        if DATES.search(l):
            found["Experience"] += 1
        if l.count(",") >= 3 and len(l.split()) <= 28 and not re.search(r"\d{4}|@", l):
            found["Skills"] += 1
        if i < 8 and len(l.split()) >= 12 and not l.startswith("-"):
            found["Summary"] += 1
    return found


HAND_NAMES = ["has number", "has metric", "action verb", "weak phrase", "length"]


def explain(pipe, texts: List[str], k: int = 3) -> List[dict]:
    """Per-text top contributing features (coef x value) for the bullet classifier."""
    feats, clf = pipe.named_steps["features"], pipe.named_steps["clf"]
    names = list(feats.transformer_list[0][1].get_feature_names_out()) + HAND_NAMES
    X = feats.transform(texts)
    X = X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)
    coef = clf.coef_[0]
    out = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        pairs = sorted(((names[j], float(v * coef[j])) for j, v in zip(row.indices, row.data)), key=lambda p: p[1])
        out.append({"weak": [n for n, c in pairs[:k] if c < -0.05], "strong": [n for n, c in pairs[::-1][:k] if c > 0.05]})
    return out
