import os
os.environ["RESUMELENS_EMBEDDER"] = "lsa"  # fast and offline

from pathlib import Path
from app.analyzer import Engine, extract_skills, split_requirements, get_bullets
from app.nlp import readability, rewrite, keyword_gap, infer_sections
from app.parser import normalize

D = Path(__file__).parent.parent / "app" / "data"
RESUME = (D / "sample_resume.txt").read_text()
JOB = (D / "sample_job.txt").read_text()
engine = Engine()


def test_skill_extraction():
    s = extract_skills("Built APIs in Python and C++ on AWS with Docker and JavaScript.")
    assert {"Python", "C++", "AWS", "Docker", "JavaScript"} <= set(s)
    assert "Java" not in extract_skills("JavaScript only")


def test_bullet_normalisation():
    assert normalize("• Built a thing\n● Led a team").startswith("- Built")
    assert len(get_bullets("- one two three four five six\n- seven eight")) == 2


def test_classifier_separates_strong_from_weak():
    p = engine.clf.predict_proba(["Reduced latency by 40% by adding a Redis cache", "Responsible for various tasks"])[:, 1]
    assert p[0] > 0.5 > p[1]


def test_full_analysis_with_job():
    r = engine.analyze(RESUME, JOB)
    assert 0 <= r["score"] <= 100 and r["engine"] == "lsa+tfidf"
    assert "Snowflake" in r["skills"]["matched"] and "Kubernetes" in r["skills"]["missing"]
    assert r["keywords"] and r["scores"]["keywords"] is not None
    weak = [b for b in r["bullets"] if "Responsible" in b["text"]][0]
    strong = [b for b in r["bullets"] if "Led migration" in b["text"]][0]
    assert weak["quality"] < strong["quality"] and weak["rewrite"] and strong["rewrite"] is None
    assert len(r["roles"]) == 3 and r["seniority"]["label"] in {"junior", "mid", "senior"}


def test_analysis_without_job():
    r = engine.analyze(RESUME)
    assert r["scores"]["fit"] is None and r["requirements"] == [] and r["keywords"] == []


def test_requirement_splitting():
    assert len(split_requirements(JOB)) >= 4


def test_readability_flags_buzzwords():
    r = readability("I am a results-driven team player. Reports were written by me.", [])
    assert "team player" in r["buzzwords"] and r["first_person"] >= 1 and r["passive_voice"] >= 1


def test_rewrite_never_invents_numbers():
    out = rewrite("Responsible for maintaining ETL pipelines")
    assert out.startswith("Maintained ETL pipelines") and "[X%]" in out
    assert "[X%]" not in rewrite("Cut costs by 30% by right-sizing servers")


def test_keyword_gap_weighted():
    kws = [{"term": "kubernetes", "weight": 1.0}, {"term": "python", "weight": 0.5}]
    assert abs(keyword_gap("I use Python daily", kws) - (0.5 / 1.5)) < 1e-6


def test_infer_sections_without_headings():
    f = infer_sections(["B.S. Computer Science, State University, 2020", "Acme Corp 2021 - Present", "Python, SQL, Docker, Git, AWS"])
    assert f["Education"] and f["Experience"] and f["Skills"]
