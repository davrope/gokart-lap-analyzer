# GoKart Coaching Platform

A multi-session Streamlit app for kart telemetry coaching using Garmin `.FIT` files.

The platform now supports:
- Landing page with authentication
- Multi-track and multi-attempt workflow
- Persisted FIT uploads (Supabase Storage)
- Persisted analytics (Supabase Postgres)
- Attempt-level deep analysis (overview + advanced curve pipeline)
- Track-level historical progress dashboards

## Core workflow
1. Login via email magic link.
2. Upload a FIT file and attach it to a track (new or existing).
3. Re-open any attempt and analyze it in detail.
4. Review track history to compare attempts and identify improvement trends.

## Tech stack
- Python
- Streamlit
- Plotly
- pandas / numpy
- fitparse
- scipy
- Supabase (Auth + Postgres + Storage)

## Project structure

```text
gokart-lap-analyzer/
├── app.py
├── analysis/
├── services/
├── ui/
├── pages/
│   ├── 1_Upload_Attempt.py
│   ├── 2_Attempt_Analysis.py
│   ├── 3_Track_History.py
│   └── 4_Further_Analysis.py
├── db/
│   └── migrations/
├── lap_methods/
├── tests/
├── requirements.txt
└── TESTING.md
```

## Getting started

```bash
git clone <repo-url>
cd gokart-lap-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Supabase setup

1. Create a Supabase project.
2. Run SQL migration in Supabase SQL editor:
   - `db/migrations/20260212_000001_multi_session.sql`
3. Configure secrets/environment:
   - Local `.env` (auto-loaded by app): copy from `.env.example`
   - Streamlit Cloud secrets: copy from `.streamlit/secrets.toml.example`

```toml
SUPABASE_URL = "https://<project>.supabase.co"
SUPABASE_ANON_KEY = "<anon-key>"
APP_BASE_URL = "https://<your-streamlit-app-url>"
MULTI_ATTEMPT_MODE = "true"
```

```bash
cp .env.example .env
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

## Run app

```bash
streamlit run app.py
```

## Testing

Automated tests:

```bash
export FIT_TEST_FILE="/absolute/path/to/your-session.fit"
.venv/bin/python -m unittest discover -s tests -v
```

Detailed methodology and smoke scenarios are in `TESTING.md`.
