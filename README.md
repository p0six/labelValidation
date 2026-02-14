# README.md - Documentation

## AI-Powered Alcohol Label Verification App

Prototype for TTB compliance team. Uses Gemini 1.5 Flash for fast, accurate label extraction and matching.

### Setup
1. Clone repo: `git clone <repo-url>`
2. Install deps: `pip install -r requirements.txt`
3. Add Gemini key to `.streamlit/secrets.toml`
4. Run locally: `streamlit run app.py`

### Deployment to Streamlit Community Cloud
1. Fork/connect repo to Streamlit Cloud (app.streamlit.io)
2. Set secrets in Streamlit dashboard (GEMINI_API_KEY)
3. Auto-deploys on push to main. GitHub Actions runs tests first.

### Testing
- Uses pytest with mocking for AI calls.
- Run locally: `pytest tests/ -v`
- CI runs tests on every push/PR; deploy only if tests pass.
- Coverage: Focuses on preprocessing, extraction (mocked), and comparison logic. Add more tests as features grow.

Assumptions in tests:
- Mocks Gemini API to avoid real calls/costs.
- Uses dummy images for preprocessing tests.
- Expand with integration tests if needed (e.g., via streamlit.testing, but overkill for prototype).

### Test Images
Generate with AI tools or use real photos. Place in `test_images/` for manual testing.

### Assumptions/Trade-offs
- Focus on core fields: Brand, ABV, Warning.
- Preprocessing handles common issues (glare, angle).
- Batch assumes CSV has exact filenames.
- No direct COLA integration (per specs).
- Speed: ~2-4s per image.

Demo GIF: (Add your own)
