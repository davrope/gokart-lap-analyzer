# Testing Methodology

This project should be tested at three levels before adding new features.

## 1) Fast contract tests (every change)

Goal: catch breakages in core data contracts quickly.

What we validate:
- `gps_gate.run(...)` returns the expected keys and coherent shapes.
- Distance metrics are added correctly (`lap_distance_m`, `lap_distance_km`).
- Further-analysis service returns stable schemas and non-empty recommendations.

Command:

```bash
export FIT_TEST_FILE="/absolute/path/to/your-session.fit"
.venv/bin/python -m unittest discover -s tests -v
```

Notes:
- `FIT_TEST_FILE` is required and should point to a local `.fit` file not tracked in git.
- This avoids leaking activity file names in repository code/history.

## 2) Streamlit smoke test (before merge/release)

Goal: ensure UI flow still works with real data.

Steps:
1. Run app:
   ```bash
   .venv/bin/streamlit run app.py
   ```
2. Upload your local `.fit` file (not tracked in git).
3. Confirm main page renders:
   - summary metrics
   - lap table and downloads
   - plots (if enabled)
4. Click `Open further analysis`.
5. Confirm further page renders:
   - lap KPIs and speed chart
   - lap overview table
   - recommendations list
6. Click `Back to main page`.

Acceptance criteria:
- No exceptions in terminal/UI.
- Navigation works both ways.
- Tables and charts are populated when laps exist.

## 3) Regression checks for future curve analytics

When curve detection is added, extend tests to include:
- Per-curve schema contract (required columns and dtypes).
- At least one deterministic benchmark:
  - same FIT file
  - expected curve count in a reasonable range
  - stable aggregate metrics (e.g., mean apex speed tolerance band)
- Recommendation rules tied to measurable conditions.

Guideline:
- Prefer range-based assertions over exact floating-point equality.
- Keep one realistic FIT file in repo for repeatable checks.
- Run Level 1 tests before every commit that changes `lap_methods/`, `analysis/`, `pages/`, or `app.py`.
