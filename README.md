# AI-Powered Alcohol Label Verification

**Prototype for accelerating TTB compliance review**
#### Author: Michael Romero

An interactive Streamlit application that leverages multimodal large language models to extract and verify critical information from alcohol beverage labels:
- Brand name  
- Alcohol by volume (ABV / proof)  
- Government health warning statement  

The tool compares extracted values against user-provided expectations and presents clear match/fail verdicts with confidence scores and explanatory notes.

## TODO:
- Improve the UI, making uploaded images visible, perhaps annotating failures directly on the image.
- Create additional tests, especially for edge cases (e.g., very blurry images, unusual label designs).
- Improve batch submission user experience
- Investigate additional LLM's beyond Gemini and OpenAI, such as Claude or Llama 3, to compare performance and accuracy.
- Validate build steps
---

## Core Features
- **Single-label verification**  
  Upload → real-time preview → enter expected values → AI analysis → verdict + detailed field-by-field comparison table
- **Batch processing**  
  Upload folder of images (or ZIP archive) + CSV of expected values → process all labels → downloadable results CSV
- **Image preprocessing**  
  OpenCV-based correction for glare, skew, contrast, and common label photography issues
- **Typical latency**  
  2–6 seconds per label (OpenAI), 8-12 seconds (Gemini 2.5 Pro)

## Technology Stack

- UI & orchestration: Streamlit  
- Vision + reasoning:  GPT-4o (primary) / Gemini 2.5 Pro (alternative)
- Image handling: OpenCV  
- Testing: pytest (LLM calls mocked)  
- CI/CD: GitHub Actions → Streamlit Cloud auto-deploy

## Quick Start (Local)

### 1. Clone & enter directory
```bash
git clone https://github.com/p0six/labelValidation.git
cd labelValidation
```
### 2. Virtual environment
```
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```
### 3. Install dependencies
```pip install -r requirements.txt```

### 4. Add API keys (create/edit .streamlit/secrets.toml)
```
OPENAI_API_KEY = "sk-..."
GEMINI_API_KEY = "your-key-here"
```

### 5. Run (OpenAI by default)
```streamlit run app.py```

#### Or force Gemini-compatible mode:
```LLM_PROVIDER=gemini streamlit run app.py```

-----

## Deployment to Streamlit Community Cloud
1. Fork/connect repo to Streamlit Cloud (app.streamlit.io)
2. Set secrets in Streamlit dashboard (GEMINI_API_KEY, OPENAI_API_KEY)
3. If you prefer to use Gemini ('openai' is the default), set environment variable 'LLM_PROVIDER' to 'gemini'
4. Auto-deploys on push to main. GitHub Actions runs tests first. 

## Testing
- Uses pytest with mocking for AI calls.
- Run locally: `pytest tests/ -v`, or from within your IDE.
- CI runs tests on every push/PR. All commits to 'main' are deployed. (limitation of Streamlit)
- Coverage: Currently focused on label extraction for both Gemini and OpenAI.

#### Assumptions in tests:
- Mocks Gemini/OpenAI API to avoid real calls/costs.
- Uses dummy images for preprocessing tests.

### Test Images
Generate with AI tools or use real photos. Place in `test_images/` for manual testing.

## Assumptions/Trade-offs
- Focus on core fields: Brand, ABV, Warning.
- Preprocessing handles common issues (glare, angle).
- Batch assumes CSV has exact filenames.
- Speed: ~2-6s per image.
