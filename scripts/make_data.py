"""Generate SYNTHETIC training data (template-based, seeded).

    python scripts/make_data.py

Writes app/data/roles.csv and extends app/data/bullets.csv. Everything produced here is
synthetic and only meant to bootstrap the models. Replace with real labeled data when you have it.
"""
import csv
import random
from pathlib import Path

rng = random.Random(7)
DATA = Path(__file__).resolve().parent.parent / "app" / "data"

ROLES = {
 "Data Scientist": (["Python", "scikit-learn", "statistics", "A/B testing", "pandas", "SQL", "XGBoost", "feature engineering", "Jupyter", "hypothesis testing", "forecasting"],
                    ["built predictive models for churn and demand", "ran experiments and analyzed results", "presented insights to stakeholders", "cleaned and modeled large datasets"]),
 "Data Engineer": (["Python", "SQL", "Airflow", "Spark", "Kafka", "Snowflake", "dbt", "ETL", "data warehouse", "AWS", "BigQuery"],
                   ["built batch and streaming pipelines", "modeled warehouse tables", "monitored data quality and freshness", "orchestrated workflows"]),
 "Backend Developer": (["Python", "Java", "Node.js", "REST APIs", "PostgreSQL", "Redis", "Docker", "microservices", "FastAPI", "Spring", "message queues"],
                       ["designed and shipped REST APIs", "optimized database queries", "wrote unit and integration tests", "maintained production services"]),
 "Frontend Developer": (["JavaScript", "TypeScript", "React", "HTML", "CSS", "Vue", "accessibility", "Webpack", "responsive design", "Jest", "Figma"],
                        ["built responsive user interfaces", "implemented design systems", "improved page performance", "collaborated with designers"]),
 "DevOps Engineer": (["Docker", "Kubernetes", "Terraform", "AWS", "CI/CD", "Jenkins", "Linux", "monitoring", "Ansible", "GitHub Actions", "Prometheus"],
                     ["automated deployments and infrastructure", "managed clusters and cloud costs", "improved reliability and on-call response", "built CI/CD pipelines"]),
 "ML Engineer": (["Python", "PyTorch", "TensorFlow", "MLOps", "model serving", "Docker", "feature stores", "NLP", "computer vision", "Kubernetes", "LLMs"],
                 ["trained and deployed deep learning models", "built model serving and monitoring", "optimized inference latency", "productionized research prototypes"]),
 "Product Manager": (["roadmap", "stakeholder management", "user research", "OKRs", "Agile", "A/B testing", "prioritization", "Jira", "go-to-market", "analytics", "strategy"],
                     ["defined product requirements and roadmap", "prioritized backlog with engineering", "ran user interviews and experiments", "aligned stakeholders on launches"]),
 "Data Analyst": (["SQL", "Excel", "Tableau", "Power BI", "dashboards", "reporting", "Python", "KPIs", "data cleaning", "statistics", "storytelling"],
                  ["built dashboards and weekly reports", "analyzed trends and KPIs", "wrote SQL for ad hoc questions", "presented findings to business teams"]),
}
LEVELS = {
 "junior": (["Junior", "Entry-level", "Associate"], ["0-2 years of experience", "1 year of experience", "recent graduate with internship experience"],
            ["Assisted senior teammates", "Learning best practices", "Completed internships and academic projects", "Contributed to small features"]),
 "mid": (["", "Mid-level", ""], ["3-5 years of experience", "4 years of experience", "5 years of experience"],
         ["Owned features end to end", "Delivered projects independently", "Collaborated across teams", "Improved existing systems"]),
 "senior": (["Senior", "Lead", "Staff"], ["8+ years of experience", "10 years of experience", "7+ years of experience"],
            ["Led teams and mentored engineers", "Set technical strategy and architecture", "Drove cross-org initiatives", "Hired and coached the team"]),
}

with open(DATA / "roles.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text", "role", "seniority", "source"])
    for role, (skills, tasks) in ROLES.items():
        for lvl, (titles, years, stories) in LEVELS.items():
            for _ in range(8):
                text = (f"{rng.choice(titles)} {role} with {rng.choice(years)}. "
                        f"Skills: {', '.join(rng.sample(skills, 5))}. "
                        f"{rng.choice(stories)}; {rng.choice(tasks)} and {rng.choice(tasks)}.")
                w.writerow([" ".join(text.split()), role, lvl, "synthetic"])

# ---- bullets
existing = []
with open(DATA / "bullets.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        existing.append((r["text"], int(r["label"]), r.get("source") or "seed"))
seen = {t for t, _, _ in existing}

things = ["API latency", "page load time", "infrastructure costs", "customer churn", "report generation time", "deployment time", "support tickets", "onboarding time", "query time", "model error rate", "manual effort", "bug count"]
methods = ["adding caching", "rewriting slow queries", "automating the workflow", "redesigning the process", "introducing monitoring", "migrating to a managed service", "tuning the model", "refactoring core modules"]
gains = ["conversion rate", "user retention", "test coverage", "throughput", "model accuracy", "revenue", "adoption", "uptime"]
builds = ["a data pipeline", "a recommendation engine", "an internal dashboard", "a REST API", "a mobile feature", "a monitoring system", "a forecasting model", "a CI/CD pipeline", "a search service", "an A/B testing framework"]
scale = ["serving {n}K users", "processing {n}M records daily", "used by {n} teams", "handling {n}K requests per minute", "saving ${n}K per year"]
strong_verbs = ["Built", "Designed", "Developed", "Launched", "Deployed", "Engineered", "Implemented", "Created"]
weak_open = ["Responsible for", "Worked on", "Helped with", "Assisted with", "Participated in", "Involved in", "Duties included", "Tasked with", "Contributed to", "In charge of", "Supported"]
weak_task = ["the team's projects", "various tasks", "daily operations", "customer requests", "the company website", "reports and documentation", "testing and bug fixes", "meetings and planning", "data entry", "new features", "the backend", "internal tools"]

def n(): return rng.choice([2, 3, 5, 8, 12, 15, 20, 30, 40, 60])
new = []
for _ in range(400):
    k = rng.random()
    if k < .4: t = f"{rng.choice(['Reduced', 'Cut', 'Lowered', 'Decreased'])} {rng.choice(things)} by {n()}% by {rng.choice(methods)}"
    elif k < .65: t = f"{rng.choice(['Increased', 'Improved', 'Boosted', 'Grew'])} {rng.choice(gains)} by {n()}% through {rng.choice(methods)}"
    else: t = f"{rng.choice(strong_verbs)} {rng.choice(builds)} {rng.choice(scale).format(n=n())}"
    if t not in seen: seen.add(t); new.append((t, 1, "synthetic"))
    if sum(1 for x in new if x[1] == 1) >= 70: break
for _ in range(400):
    t = f"{rng.choice(weak_open)} {rng.choice(weak_task)}"
    if rng.random() < .25: t = rng.choice(["Hard working team player", "Good communication skills", "Quick learner and self-starter", "Detail-oriented professional"])
    if t not in seen: seen.add(t); new.append((t, 0, "synthetic"))
    if sum(1 for x in new if x[1] == 0) >= 70: break

with open(DATA / "bullets.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["text", "label", "source"])
    for row in existing + new: w.writerow(row)
print("bullets", len(existing) + len(new), "roles", 8 * 3 * 8)
