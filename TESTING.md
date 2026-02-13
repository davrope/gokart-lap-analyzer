# Testing Methodology

This project now has two testing layers:
- telemetry/analysis contract testing
- platform workflow testing (auth, repositories, persistence payloads)

## 1) Fast automated tests (every change)

Run all tests:

```bash
export FIT_TEST_FILE="/absolute/path/to/your-session.fit"
.venv/bin/python -m unittest discover -s tests -v
```

### FIT-based tests
- `tests/test_pipeline_with_fit.py`
- Validates GPS gate output contract, distance metrics, and further-analysis contract.
- Requires `FIT_TEST_FILE` to avoid hardcoded FIT names in source.

### Platform tests (no external services)
- `tests/test_auth_service.py`: auth flow behavior with mocks.
- `tests/test_repositories.py`: repository interface behavior with fake query chains.
- `tests/test_analysis_persistence.py`: payload mapping from analysis outputs to DB rows.

## 2) Manual smoke tests (before release)

## A. Auth + landing
1. Start app: `streamlit run app.py`
2. Confirm landing page renders.
3. Request magic link and verify login callback works.

## B. Upload workflow
1. Open `Upload Attempt` page.
2. Create a new track.
3. Upload a `.fit` file and process.
4. Confirm attempt is saved and marked processed.

## C. Attempt analysis
1. Open `Attempt Analysis` page.
2. Select track + attempt.
3. Verify both tabs render:
   - `Overview`
   - `Advanced`
4. Confirm no errors in plots and tables.

## D. Track history
1. Open `Track History` page.
2. Confirm timeline chart appears.
3. Confirm consistency trend appears.
4. Confirm heatmap and latest-vs-best comparisons render.

## E. Security / data isolation
1. Login as user A and create track/attempt.
2. Login as user B.
3. Confirm user B cannot view user A records.

## Acceptance criteria
- Existing single-attempt analytics still produce stable outputs.
- Multi-attempt persistence works end-to-end.
- Historical dashboards are populated from saved attempt data.
- No cross-user data leakage with RLS.
