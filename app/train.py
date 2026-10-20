"""Train all models.   python -m app.train

 - bullet quality (strong/weak)      data/bullets.csv
 - target role (8 classes)           data/roles.csv
 - seniority (junior/mid/senior)     data/roles.csv
Saved under app/models/. All training data in the repo is small and largely synthetic.
"""
from pathlib import Path
import csv

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from .features import hand_features

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODEL_FILES = {"bullet": "bullet_clf.joblib", "role": "role_clf.joblib", "seniority": "seniority_clf.joblib"}


def bullet_pipeline() -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("hand", FunctionTransformer(hand_features, validate=False)),
        ])),
        ("clf", LogisticRegression(C=4.0, max_iter=1000, class_weight="balanced")),
    ])


def text_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
        ("clf", LogisticRegression(C=8.0, max_iter=2000)),
    ])


def read_csv(name: str):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_datasets() -> dict:
    b, r = read_csv("bullets.csv"), read_csv("roles.csv")
    return {
        "bullet": ([x["text"] for x in b], np.array([int(x["label"]) for x in b])),
        "role": ([x["text"] for x in r], np.array([x["role"] for x in r])),
        "seniority": ([x["text"] for x in r], np.array([x["seniority"] for x in r])),
    }


def pipeline_for(name: str) -> Pipeline:
    return bullet_pipeline() if name == "bullet" else text_pipeline()


def train_all(verbose: bool = True) -> dict:
    MODELS.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, (X, y) in load_datasets().items():
        pipe = pipeline_for(name).fit(X, y)
        joblib.dump(pipe, MODELS / MODEL_FILES[name])
        out[name] = pipe
        if verbose:
            print(f"trained {name}: {len(X)} rows")
    if verbose:
        print("Run `python -m app.evaluate` for held-out metrics.")
    return out


def load_models() -> dict:
    out = {}
    for name, fn in MODEL_FILES.items():
        try:
            out[name] = joblib.load(MODELS / fn)
        except Exception:
            return train_all(verbose=False)
    return out


if __name__ == "__main__":
    train_all()
