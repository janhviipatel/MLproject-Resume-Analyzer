"""Held-out evaluation.   python -m app.evaluate   -> prints and writes app/models/metrics.json

Stratified 75/25 split per model, plus 5-fold cross-validated accuracy. NOTE: most data here is
synthetic and template-generated, so scores will look very high. They show the pipeline works,
not how it performs on real resumes.
"""
import json

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .train import MODELS, load_datasets, pipeline_for


def evaluate() -> dict:
    report = {}
    for name, (X, y) in load_datasets().items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        pipe = pipeline_for(name).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        labels = [str(c) for c in pipe.classes_]
        cv = cross_val_score(pipeline_for(name), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1))
        report[name] = {
            "rows": len(X), "test_rows": len(yte), "labels": labels,
            "report": classification_report(yte, pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(yte, pred, labels=pipe.classes_).tolist(),
            "cv_accuracy_mean": round(float(cv.mean()), 3), "cv_accuracy_std": round(float(cv.std()), 3),
            "note": "Mostly synthetic data: optimistic. Re-evaluate on real labeled resumes.",
        }
        r = report[name]["report"]
        print(f"\n== {name} ==  accuracy {r['accuracy']:.3f}  macro-F1 {r['macro avg']['f1-score']:.3f}  cv {cv.mean():.3f}±{cv.std():.3f}")
        print("labels", labels)
        for row in report[name]["confusion_matrix"]:
            print(" ", row)
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / "metrics.json").write_text(json.dumps(report, indent=2))
    print("\nwrote", MODELS / "metrics.json")
    return report


if __name__ == "__main__":
    evaluate()
