# app.py - AI-Powered Alcohol Label Verification App (updated for google-genai SDK 2026 + OpenAI support)

import streamlit as st
import base64
import io
import json
import os
import sys
import zipfile
from datetime import datetime

from google import genai
from google.genai import types
from openai import OpenAI
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ────────────────────────────────────────────────
# CONFIG & SECRETS
# ────────────────────────────────────────────────

is_ci_or_test = (
    os.getenv("GITHUB_ACTIONS") == "true" or
    os.getenv("PYTEST_CURRENT_TEST") is not None or
    "pytest" in sys.modules
)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Default to openai (faster); set to "gemini" to swap

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not is_ci_or_test:
    if LLM_PROVIDER == "gemini":
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", GEMINI_API_KEY)
        if not GEMINI_API_KEY:
            st.error("Gemini API key not found. Please add GEMINI_API_KEY to Streamlit secrets or environment variables.")
            st.stop()
    elif LLM_PROVIDER == "openai":
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", OPENAI_API_KEY)
        if not OPENAI_API_KEY:
            st.error("OpenAI API key not found. Please add OPENAI_API_KEY to Streamlit secrets or environment variables.")
            st.stop()
else:
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = "dummy-key-for-ci-tests"
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = "dummy-key-for-ci-tests"

# Conditional decorator: real cache in Streamlit, no-op during pytest
if os.getenv("PYTEST_CURRENT_TEST") is None:
    cache_decorator = st.cache_data(ttl=3600, show_spinner="Analyzing label...")
else:
    def cache_decorator(func):
        return func

# Initialize clients
if LLM_PROVIDER == "gemini":
    client = genai.Client(api_key=GEMINI_API_KEY)
elif LLM_PROVIDER == "openai":
    client = OpenAI(api_key=OPENAI_API_KEY)

# The extraction prompt (same as before)
EXTRACTION_PROMPT = """
You are an expert TTB label compliance analyst.

Your ONLY task is to extract text EXACTLY as it appears on the label — do NOT correct spelling, do NOT fix typos, do NOT guess missing letters, do NOT hallucinate words.

If a letter is unclear, damaged, or partially visible due to glare/curve/reflection, transcribe it as best you can see it — even if it looks wrong. Do NOT assume the intended word.

Each field within the image can span multiple lines. If the line that follows an incomplete field contains the text that would complete the expected field, include it in the same field. As one example, if the brand name is split across two lines, with "OLD TOM" on one line and "DISTILLERY" on the next line, you should combine both lines into the single "brand_name" field in your output.

The text from each field may not always appear on the same line, and may be split across multiple lines. Use your best judgment to determine if adjacent lines should be combined into the same field based on the expected values and the visual layout of the text.

As one example, the text "45% Alc./Vol. (90 Proof)" may be split across two lines on the label, with "45% Alc./Vol." on one line and "(90 Proof)" on the next line. In this case, you should combine both lines into the single "alcohol_content" field in your output.

We are ONLY concerned with the brand name, alcohol content, and warning label. The warning label header "GOVERNMENT WARNING:" must be in all caps, and be at least SLIGHTLY bolder than the rest of the warning text. If the header is not in all caps, or if it is not visually distinguishable as a header (e.g. bolded), this should be noted in the output.

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
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _enhance_and_encode(gray)

    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
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
        warped = img

    if enable_cylindrical_unwrap and warped.shape[1] > 100:
        h, w = warped.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        center_x = w / 2
        radius_factor = 1.2
        offset = (map_x - center_x) * (radius_factor - 1) * (1 - np.abs(map_x - center_x) / (w / 2))
        map_x_unwarped = (map_x + offset).astype(np.float32)
        map_y_unwarped = map_y.astype(np.float32)
        unwrapped = cv2.remap(warped, map_x_unwarped, map_y_unwarped, cv2.INTER_LINEAR)
    else:
        unwrapped = warped

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
# LABEL EXTRACTION (LLM-agnostic)
# ────────────────────────────────────────────────

@cache_decorator
def extract_label_data(image_bytes: bytes, preprocess: bool = True) -> dict | None:
    try:
        processed_bytes = preprocess_image(image_bytes) if preprocess else image_bytes
        base64_image = base64.b64encode(processed_bytes).decode("utf-8")

        if LLM_PROVIDER == "gemini":
            response = client.models.generate_content(
                model="gemini-2.5-pro",
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
        elif LLM_PROVIDER == "openai":
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            text = response.choices[0].message.content.strip()
        else:
            st.error(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
            return None

        if not text.startswith("{"):
            st.warning(f"LLM did not return valid JSON. Raw response:\n{text[:200]}...")
            return None

        return json.loads(text)

    except Exception as e:
        st.error(f"Error during extraction: {str(e)}")
        return None

# ────────────────────────────────────────────────
# FIELD COMPARISON LOGIC
# ────────────────────────────────────────────────

def compare_fields(extracted: dict, expected_brand: str, expected_abv: str, expected_warning: str):
    results = []

    brand_match = extracted.get("brand_name", "").lower() == expected_brand.lower()
    results.append({
        "field": "Brand Name",
        "from_label": extracted.get("brand_name", "—"),
        "from_app": expected_brand,
        "match": "✅" if brand_match else "❌",
        "notes": "Case-insensitive"
    })

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
        "from_label": extracted.get("government_warning_full", "—")[:100] + "..." if len(extracted.get("government_warning_full", "")) > 100 else extracted.get("government_warning_full", "—"),
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

import streamlit as st
import pandas as pd
from datetime import datetime
import zipfile
# Assuming these are defined elsewhere:
# from your_module import process_single_image, LLM_PROVIDER

st.set_page_config(page_title="TTB Label Verifier", layout="wide")

# Reduce top padding / vacant space above the first content
st.markdown(
    """
    <style>
        .stApp > header {
            display: none;           /* hides the empty header bar completely if not needed */
        }
        section.main .block-container {
            padding-top: 0.5rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("AI-Powered Alcohol Label Verification")
st.caption(f"Prototype using {LLM_PROVIDER.capitalize()} – Fast label matching for TTB compliance")

tab_single, tab_batch = st.tabs(["Single Label", "Batch Processing"])

with tab_single:
    st.header("Verify Single Label")

    # Always use three columns for stable layout
    col_left, col_middle, col_right = st.columns([2, 2, 3])

    # ── Left column ──
    with col_left:
        # Callback for immediate reaction to file selection
        def on_new_file_uploaded():
            # Clear old verification results so right column resets
            st.session_state.pop("verification_results", None)

        # File uploader OUTSIDE the form → can have on_change
        uploaded_file = st.file_uploader(
            "Upload label image",
            type=["jpg", "jpeg", "png"],
            help="Drag & drop or click to select a label photo",
            on_change=on_new_file_uploaded,
            key="single_label_uploader"   # stable key recommended
        )

        # ── The rest of the inputs + button stay inside the form ──
        with st.form(key="single_label_form", clear_on_submit=True):
            expected_brand = st.text_input(
                "Expected Brand Name",
                value="OLD TOM DISTILLERY"
            )
            expected_abv = st.text_input(
                "Expected Alcohol Content",
                value="45% Alc./Vol. (90 Proof)"
            )
            expected_warning = st.text_area(
                "Expected Government Warning",
                value="GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS.",
                height=140
            )

            verify_button = st.form_submit_button(
                "Verify Label",
                type="primary",
                use_container_width=True
            )

        st.caption("Upload a new image to clear previous results and see preview instantly")

    # ── Handle immediate preview when file is selected ──
    with col_middle:
        if uploaded_file is not None:
            # Load bytes only when necessary (change detection via name + size)
            current_key = f"preview_{uploaded_file.name}_{uploaded_file.size}"
            if (
                "uploaded_image_bytes" not in st.session_state
                or st.session_state.get("preview_key") != current_key
            ):
                with st.spinner("Loading preview..."):
                    image_bytes = uploaded_file.read()
                    st.session_state["uploaded_image_bytes"] = image_bytes
                    st.session_state["uploaded_filename"] = uploaded_file.name
                    st.session_state["preview_key"] = current_key
                # No rerun here — let natural page render continue

        if "uploaded_image_bytes" in st.session_state:
            st.subheader("Uploaded Label")
            header_col, btn_col = st.columns([5, 1])
            st.image(
                st.session_state["uploaded_image_bytes"],
                use_column_width=True
            )
            with header_col:
                st.caption(st.session_state["uploaded_filename"])
            with btn_col:
                st.download_button(
                    label="↓",
                    data=st.session_state["uploaded_image_bytes"],
                    file_name=st.session_state["uploaded_filename"],
                    mime="image/jpeg",
                    key="download_single_label",
                    type="secondary"
                )

        else:
            st.info("Upload an image to see preview here", icon="🖼️")

    # ── Right column: Results ──
    with col_right:
        # Verification logic (runs only when form is submitted)
        if verify_button:
            if uploaded_file is not None:
                with st.spinner("Preprocessing & analyzing label..."):
                    # Use already loaded bytes if available, otherwise read again
                    if "uploaded_image_bytes" in st.session_state:
                        image_bytes = st.session_state["uploaded_image_bytes"]
                    else:
                        image_bytes = uploaded_file.read()
                        st.session_state["uploaded_image_bytes"] = image_bytes
                        st.session_state["uploaded_filename"] = uploaded_file.name

                    summary, detail_table = process_single_image(
                        image_bytes,
                        uploaded_file.name,
                        expected_brand,
                        expected_abv,
                        expected_warning
                    )

                    st.session_state["verification_results"] = {
                        "summary": summary,
                        "detail_table": detail_table
                    }

                st.rerun()
            else:
                st.warning("Please upload an image first.", icon="⚠️")

        # Display results if they exist
        if "verification_results" in st.session_state:
            results = st.session_state["verification_results"]
            summary = results["summary"]
            detail_table = results.get("detail_table", [])

            st.subheader("Verification Result")
            st.markdown(f"**Verdict:** {summary.get('verdict', '—')}")
            st.markdown(f"**Confidence:** {summary.get('confidence', '—')}")
            st.markdown(f"**Notes:** {summary.get('notes', '—')}")

            if detail_table:
                df = pd.DataFrame(detail_table)
                # your styling and dataframe code here (unchanged)
                desired_order = ["match", "notes"] + [col for col in df.columns if col not in ["match", "notes"]]
                df = df[desired_order]

                styled_df = df.style.map(
                    lambda v: "background-color: #d4edda; color: #155724; font-weight: bold;" if v == "✅" else
                              "background-color: #f8d7da; color: #721c24; font-weight: bold;" if v == "❌" else "",
                    subset=["match"]
                )

                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    column_config={
                        "match": st.column_config.TextColumn("Match", width="small"),
                        "notes": st.column_config.TextColumn("Notes", width="medium"),
                        "field": st.column_config.TextColumn("Field", width="medium"),
                        "from_label": st.column_config.TextColumn("From Label", width="large"),
                        "from_app": st.column_config.TextColumn("From App", width="large"),
                    },
                    hide_index=True,
                )
        else:
            st.info("Verification results will appear here after you click Verify.", icon="🔍")

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
                    # st.write("CSV columns found:", list(expected_df.columns))
                    # st.write("First few rows:", expected_df.head().to_dict(orient="records"))
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
