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

is_ci_or_test = (
    os.getenv("GITHUB_ACTIONS") == "true"      # GitHub Actions
    or os.getenv("PYTEST_CURRENT_TEST") is not None  # pytest
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not is_ci_or_test:
    # Only enforce secrets.toml lookup in local/development Streamlit runs
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", GEMINI_API_KEY)
    if not GEMINI_API_KEY:
        st.error("Gemini API key not found. Please add GEMINI_API_KEY to Streamlit secrets or environment variables.")
        st.stop()
else:
    # In CI / tests: use dummy or env var (your mocks never hit real API anyway)
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = "dummy-key-for-ci-tests"

# Conditional decorator: real cache in Streamlit, no-op during pytest
if os.getenv("PYTEST_CURRENT_TEST") is None:
    # Normal Streamlit run → use caching
    cache_decorator = st.cache_data(
        ttl=3600,
        show_spinner="Analyzing label..."
    )
else:
    # Running under pytest → skip caching entirely (no runtime warning)
    def cache_decorator(func):
        return func

# Initialize client once
client = genai.Client(api_key=GEMINI_API_KEY)

# The extraction prompt (same as before)
EXTRACTION_PROMPT = """
You are an expert TTB label compliance analyst.

Your ONLY task is to extract text EXACTLY as it appears on the label — do NOT correct spelling, do NOT fix typos, do NOT guess missing letters, do NOT hallucinate words.

If a letter is unclear, damaged, or partially visible due to glare/curve/reflection, transcribe it as best you can see it — even if it looks wrong. Do NOT assume the intended word.

Each field within the image can span multiple lines. If the line that follows an incomplete field contains the text that would complete the expected field, include it in the same field.

We are ONLY concerned with the brand name, alcohol content, and warning label. The warning label header "GOVERNMENT WARNING:" must be in all caps. The rest of the warning text is not case sensitive.

Match failures should be explained clearly, verbosely, in the 'Notes' field for the user.

Return ONLY valid JSON. No explanations, no corrections.
{
  "brand_name": "exact text of primary brand name, prefer largest/prominent text",
  "class_type": "exact class/type designation",
  "alcohol_content": "exact alcohol content statement as written",
  "net_contents": "exact net contents text",
  "government_warning_full": "the COMPLETE government warning text EXACTLY as it appears — character for character, line by line, including any typos or formatting errors visible in the image",
  "warning_header_is_all_caps": true/false,
  "warning_header_is_bold": true/false,
  "extracted_confidence": "high/medium/low",
  "notes": "only observations about image quality or reading difficulty — do NOT include any text corrections here"
}
"""


# ────────────────────────────────────────────────
# IMAGE PREPROCESSING (OpenCV)
# ────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes, enable_cylindrical_unwrap: bool = True, max_dim: int = 1024) -> bytes:
    """
    Enhanced preprocessing:
    - Decode image
    - Perspective correction (if quad detected)
    - Approximate cylindrical dewarping (stretch compressed edges)
    - Contrast enhancement + sharpening
    """
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

        # Resize to max dimension while preserving aspect ratio
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Step 1: Grayscale & edge detection for contour finding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Step 2: Find largest quadrilateral contour (hopefully label boundary)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback early
        return _enhance_and_encode(gray)

    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    if len(approx) == 4:
        # Perspective correction: warp to rectangle
        pts = approx.reshape(4, 2).astype(np.float32)
        # Sort points roughly: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst_pts = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst_pts)
        warped = cv2.warpPerspective(img, M, (max_width, max_height))
    else:
        warped = img  # No good quad → skip perspective

    # Step 3: Approximate cylindrical unwrap (simple horizontal stretch model)
    if enable_cylindrical_unwrap and warped.shape[1] > 100:
        h, w = warped.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

        # Assume cylinder axis is vertical → compress edges horizontally
        # Simple model: stretch outer columns more (inverse of compression)
        center_x = w / 2
        radius_factor = 1.2  # tune: higher = more aggressive unwrap (1.1–1.5 typical)
        offset = (map_x - center_x) * (radius_factor - 1) * (1 - np.abs(map_x - center_x) / (w / 2))

        map_x_unwarped = (map_x + offset).astype(np.float32)
        map_y_unwarped = map_y.astype(np.float32)

        unwrapped = cv2.remap(warped, map_x_unwarped, map_y_unwarped, cv2.INTER_LINEAR)
    else:
        unwrapped = warped

    # Step 4: Final enhancement
    return _enhance_and_encode(unwrapped)


def _enhance_and_encode(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    _, buffer = cv2.imencode('.jpg', sharpened, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return buffer.tobytes()


# ────────────────────────────────────────────────
# LABEL EXTRACTION (Gemini 1.5 Flash via new SDK)
# ────────────────────────────────────────────────
# @cache_decorator - disabling caching - unnecessary for testing and causing an annoying warning with mock client. Can re-enable in production.
def extract_label_data(image_bytes: bytes, preprocess: bool = True) -> dict | None:
    try:
        processed_bytes = preprocess_image(image_bytes) if preprocess else image_bytes

        base64_image = base64.b64encode(processed_bytes).decode("utf-8")

        response = client.models.generate_content(
            model="gemini-2.5-pro",  # or your preferred model
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
                top_p=0.0,
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
