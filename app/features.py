"""Hand-crafted features for the bullet-quality classifier.

Kept in its own module so the trained pipeline can be unpickled anywhere.
"""
import re
import numpy as np

ACTION_VERBS = set("""led built designed developed launched reduced increased improved created managed
delivered automated optimized optimised implemented analyzed analysed drove shipped scaled mentored architected
owned streamlined negotiated grew cut saved wrote engineered deployed trained established generated migrated
introduced coordinated boosted secured resolved raised lifted cutting leading building""".split())

WEAK_PHRASES = ["responsible for", "worked on", "worked with", "helped", "assisted", "participated in",
                "duties included", "tasked with", "involved in", "various", "familiar with",
                "part of", "contributed to", "in charge of", "other duties", "team player"]

_NUM = re.compile(r"\d")
_IMPACT = re.compile(r"(\d\s?%|\$\s?\d|\d+\s?[kKmMxX]\b|\d+\s?(?:x|times|percent|points))")


def first_word(text: str) -> str:
    m = re.match(r"\W*([A-Za-z\-]+)", text)
    return m.group(1).lower() if m else ""


def weak_phrases_in(text: str):
    low = text.lower()
    return [p for p in WEAK_PHRASES if p in low]


def hand_features(texts) -> np.ndarray:
    rows = []
    for t in texts:
        words = len(t.split())
        rows.append([
            1.0 if _NUM.search(t) else 0.0,
            1.0 if _IMPACT.search(t) else 0.0,
            1.0 if first_word(t) in ACTION_VERBS else 0.0,
            1.0 if weak_phrases_in(t) else 0.0,
            min(words, 30) / 30.0,
        ])
    return np.array(rows)
