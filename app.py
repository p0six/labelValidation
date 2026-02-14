# app.py - AI-Powered Alcohol Label Verification App (updated for google-genai SDK 2026)

import streamlit as st
import base64
import io
import json
import os
import zipfile
from datetime import datetime

from google import genai
from google.genai import types
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ────────────────────────────────────────────────
# CONFIG & SECRETS
# ────────────────────────────────────────────────

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

if not GEMINI_API_KEY and not os.getenv("PYTEST_CURRENT_TEST"):
    st.error("Gemini API key not found. Please add GEMINI_API_KEY to Streamlit secrets or environment variables.")
    st.stop()

# Initialize client once
client = genai.Client(api_key=GEMINI_API_KEY)

# The extraction prompt (same as before)
EXTRACTION_PROMPT = """
You are an expert TTB label compliance analyst with 20 years experience.

Extract EXACTLY these fields from the alcohol label image. 
Return ONLY valid JSON, no extra text.

{
  "brand_name": "exact text of primary brand name, prefer largest/prominent text",
  "class_type": "e.g. Kentucky Straight Bourbon Whiskey, Gin, etc.",
  "alcohol_content": "exact alcohol statement, e.g. 45% Alc./Vol. (90 Proof)",
  "net_contents": "e.g. 750 mL or 1 Liter",
  "government_warning_full": "the complete government warning text exactly as it appears",
  "warning_header_is_all_caps": true/false (is "GOVERNMENT WARNING:" literally in all capitals?),
  "warning_header_is_bold": true/false (does it appear significantly bolder than surrounding text?),
  "extracted_confidence": "high/medium/low",
  "notes": "any important observations (e.g. text is curved, glare present, creative placement)"
}
"""


# ────────────────────────────────────────────────
# IMAGE PREPROCESSING (OpenCV)
# ────────────────────────────────────────────────

def preprocess_image(image_bytes):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return image_bytes  # fallback if decode fails

    # Auto-contrast (CLAHE on grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Sharpen
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Simple glare reduction
    _, thresh = cv2.threshold(sharpened, 240, 255, cv2.THRESH_TOZERO_INV)

    # Convert back to BGR for consistency
    enhanced_color = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode('.jpg', enhanced_color)
    return buffer.tobytes()


# ────────────────────────────────────────────────
# LABEL EXTRACTION (Gemini 1.5 Flash via new SDK)
# ────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Analyzing label...")
def extract_label_data(image_bytes: bytes, preprocess: bool = True) -> dict | None:
    try:
        processed_bytes = preprocess_image(image_bytes) if preprocess else image_bytes

        base64_image = base64.b64encode(processed_bytes).decode("utf-8")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                EXTRACTION_PROMPT,
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }
                }
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )

        text = response.text.strip()
        if not text.startswith("{"):
            st.warning(f"Gemini did not return valid JSON. Raw response:\n{text[:200]}...")
            return None

        return json.loads(text)

    except Exception as e:
        st.error(f"Error during Gemini extraction: {str(e)}")
        return None


# ────────────────────────────────────────────────
# FIELD COMPARISON LOGIC
# ────────────────────────────────────────────────

def compare_fields(extracted: dict, expected_brand: str, expected_abv: str, expected_warning: str):
    results = []

    # Brand (case-insensitive)
    brand_match = extracted.get("brand_name", "").lower() == expected_brand.lower()
    results.append({
        "field": "Brand Name",
        "from_label": extracted.get("brand_name", "—"),
        "from_app": expected_brand,
        "match": "✅" if brand_match else "❌",
        "notes": "Case-insensitive"
    })

    # ABV (normalize some common variations)
    abv_label = extracted.get("alcohol_content", "").replace("°", " Proof").strip()
    abv_expected = expected_abv.strip()
    abv_match = abv_label == abv_expected
    results.append({
        "field": "Alcohol Content",
        "from_label": abv_label,
        "from_app": abv_expected,
        "match": "✅" if abv_match else "❌",
        "notes": "Exact match after normalization"
    })

    # Government Warning
    warning_text_match = extracted.get("government_warning_full", "").strip() == expected_warning.strip()
    header_caps = extracted.get("warning_header_is_all_caps", False)
    header_bold = extracted.get("warning_header_is_bold", False)

    warning_issues = []
    if not header_caps:
        warning_issues.append("Header not ALL CAPS")
    if not header_bold:
        warning_issues.append("Header not bold")

    results.append({
        "field": "Government Warning",
        "from_label": extracted.get("government_warning_full", "—")[:100] + "..." if len(
            extracted.get("government_warning_full", "")) > 100 else extracted.get("government_warning_full", "—"),
        "from_app": expected_warning[:100] + "..." if len(expected_warning) > 100 else expected_warning,
        "match": "✅" if warning_text_match and header_caps and header_bold else "❌",
        "notes": "; ".join(warning_issues) if warning_issues else "Exact match + formatting"
    })

    overall_ok = all(r["match"] == "✅" for r in results)
    verdict = "APPROVED" if overall_ok else "REJECTED – Issues found"
    confidence = extracted.get("extracted_confidence", "—")
    notes = extracted.get("notes", "—")

    return results, verdict, confidence, notes


# ────────────────────────────────────────────────
# HELPER: Process one image for batch/single
# ────────────────────────────────────────────────

def process_single_image(image_bytes, filename, expected_brand, expected_abv, expected_warning):
    extracted = extract_label_data(image_bytes)
    if not extracted:
        return {
            "filename": filename,
            "verdict": "ERROR",
            "confidence": "—",
            "notes": "Extraction failed"
        }, None

    results_table, verdict, confidence, notes = compare_fields(
        extracted, expected_brand, expected_abv, expected_warning
    )
    return {
        "filename": filename,
        "verdict": verdict,
        "confidence": confidence,
        "notes": notes
    }, results_table


# ────────────────────────────────────────────────
# STREAMLIT UI
# ────────────────────────────────────────────────

st.set_page_config(page_title="TTB Label Verifier", layout="wide")
st.title("AI-Powered Alcohol Label Verification")
st.caption("Prototype using Gemini 1.5 Flash – Fast label matching for TTB compliance")

tab_single, tab_batch = st.tabs(["Single Label", "Batch Processing"])

with tab_single:
    st.header("Verify Single Label")

    uploaded_file = st.file_uploader("Upload label image", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)
    with col1:
        expected_brand = st.text_input("Expected Brand Name", "OLD TOM DISTILLERY")
        expected_abv = st.text_input("Expected Alcohol Content", "45% Alc./Vol. (90 Proof)")
    with col2:
        expected_warning = st.text_area(
            "Expected Government Warning",
            value="GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS.",
            height=140
        )

    if st.button("Verify Label", type="primary", use_container_width=True):
        if uploaded_file:
            image_bytes = uploaded_file.read()
            with st.spinner("Preprocessing & analyzing label..."):
                summary, detail_table = process_single_image(
                    image_bytes, uploaded_file.name,
                    expected_brand, expected_abv, expected_warning
                )

                st.subheader("Result")
                st.markdown(f"**Verdict:** {summary['verdict']}")
                st.markdown(f"**Confidence:** {summary['confidence']}")
                st.markdown(f"**Notes:** {summary['notes']}")

                if detail_table:
                    df = pd.DataFrame(detail_table)
                    st.dataframe(
                        df.style.map(
                            lambda v: "background-color: #d4edda; color: #155724;" if v == "✅" else
                            "background-color: #f8d7da; color: #721c24;" if v == "❌" else "",
                            subset=["match"]
                        ),
                        use_container_width=True
                    )
        else:
            st.info("Please upload an image first.")

with tab_batch:
    st.header("Batch Verification")
    st.markdown("Upload multiple images (or ZIP) + CSV with expected values")

    files = st.file_uploader("Images or ZIP", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)
    csv_file = st.file_uploader("Expected values CSV", type=["csv"])

    if st.button("Process Batch", type="primary", use_container_width=True):
        if not files or not csv_file:
            st.warning("Please provide both images/ZIP and CSV.")
        else:
            with st.spinner("Processing batch..."):
                try:
                    expected_df = pd.read_csv(csv_file)
                    results = []

                    image_files = []
                    if len(files) == 1 and files[0].name.lower().endswith(".zip"):
                        with zipfile.ZipFile(files[0], "r") as z:
                            for name in z.namelist():
                                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                                    image_files.append((name, z.read(name)))
                    else:
                        for f in files:
                            image_files.append((f.name, f.read()))

                    for filename, img_bytes in image_files:
                        row = expected_df[expected_df["filename"] == filename]
                        if row.empty:
                            results.append({
                                "filename": filename,
                                "verdict": "SKIPPED",
                                "confidence": "—",
                                "notes": "No matching row in CSV"
                            })
                            continue

                        summary, _ = process_single_image(
                            img_bytes, filename,
                            row["expected_brand"].iloc[0],
                            row["expected_abv"].iloc[0],
                            row.get("expected_warning", "").iloc[0]
                        )
                        results.append(summary)

                    if results:
                        df = pd.DataFrame(results)
                        st.subheader("Batch Results")
                        st.dataframe(df, use_container_width=True)

                        csv_data = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Results CSV",
                            csv_data,
                            file_name=f"label_verification_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No images processed.")
                except Exception as e:
                    st.error(f"Batch processing failed: {str(e)}")

st.markdown("---")
st.caption("Prototype – Uses Gemini 1.5 Flash + OpenCV preprocessing. Not for production use.")
