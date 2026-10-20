# ResumeLens

Resume analyzer: FastAPI + scikit-learn backend, plain HTML/CSS/JS frontend (no build step, no CDN).
Python 3.9+ compatible and light enough for an older laptop. No GPU or API key.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.train          # trains the 3 classifiers (also happens automatically on first start)
uvicorn app.main:app         # add --reload --reload-dir app while developing
```

Open http://127.0.0.1:8000 and click **Load sample**.

```bash
python -m app.evaluate       # held-out metrics -> app/models/metrics.json (also served at /api/metrics)
python -m pytest             # tests (set RESUMELENS_EMBEDDER=tfidf to skip LSA)
python scripts/make_data.py  # regenerate the synthetic datasets (seeded)
```

## What each model does

| Piece | Method | Data |
|---|---|---|
| Bullet quality | TF-IDF + hand features (numbers, action verb, weak phrase, length) → logistic regression | `bullets.csv`, 200 rows |
| Per-bullet explanations | Top feature contributions (coefficient × value) from that classifier | same model |
| Target role (8 roles) | TF-IDF → logistic regression, top-3 with probabilities | `roles.csv`, 192 rows |
| Seniority (junior/mid/senior) | TF-IDF → logistic regression | `roles.csv` |
| Keyword extraction | RAKE-style phrases weighted by TF × rarity in a background corpus, boosted for known skills | no training |
| Keyword gap | Weighted share of important job terms present in the resume | n/a |
| Semantic fit | Best available: sentence-transformers → **LSA** (TF-IDF + TruncatedSVD, blended with TF-IDF) → TF-IDF | LSA fits on the bundled corpus |
| Rewrites | Rule/template based with `[X%]`-style placeholders, never invents facts | n/a |
| Readability | Bullet length, buzzwords, passive voice, first-person pronouns | n/a |
| Section inference | Heuristic line classifier (degrees, date ranges, comma-heavy skill lines) for resumes without headings | n/a |

Install `sentence-transformers` to upgrade semantic matching automatically.

## API

`POST /api/analyze` (form: `resume_text`, `job_text`, optional `resume_file`), `GET /api/sample`, `GET /api/metrics`, `GET /api/health`.

Original fields are unchanged (`score`, `scores.{fit,quality,structure}`, `skills`, `requirements`, `bullets[].{text,quality,reasons}`, `tips`, `engine`, ...).
New fields: `scores.keywords`, `scores.readability`, `keywords[]` (term, weight, present), `readability`, `roles[]`, `seniority`, `confidence`,
`low_confidence`, `sections_inferred`, `highlights`, `resume_text`, and per bullet: `confidence`, `why` (model signals), `rewrite`.

## Frontend

Light/dark theme (system default + remembered toggle), drag-and-drop upload, paste mode, skeleton loaders, animated score ring,
inline SVG radar chart, tabs (Overview, Skills, Requirements, Bullets, Rewrite, Resume), bullet filter/sort, inline bullet edit with
re-analyze, score delta vs the previous run, run history in `localStorage` with side-by-side compare, JSON export, print-to-PDF,
sticky mobile action bar.

## Limits (read this)

- **Training data is synthetic.** `bullets.csv` (60 hand-written seed rows + 140 generated) and `roles.csv` (all generated) come from templates.
  Held-out scores in `metrics.json` are ~100% because train and test come from the same templates. They show the pipeline works,
  **not** how well it performs on real resumes. Replace with real, labeled data and re-run `python -m app.evaluate` first.
- Role and seniority predictions are rough signals. The UI warns when confidence is low (< 40%).
- Skills come from a hand-built taxonomy (`app/data/skills.json`); extend it for your domain.
- Scanned PDFs need OCR. Scores are editing guidance, not a prediction of any real ATS.
- Not run end to end in the build environment: the API and frontend were not exercised in a browser there. If something breaks, check the terminal for a traceback.
