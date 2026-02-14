# tests/test_app.py
"""
Pytest suite for the alcohol label verification app.
Focuses on:
- Image preprocessing
- Field comparison logic
- Mocked Gemini extraction

Run with: pytest tests/ -v
"""

import json
import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Import functions from your main app (adjust path if needed)
from app import (
    preprocess_image,
    compare_fields,
    extract_label_data,  # we'll mock the Gemini part
    EXTRACTION_PROMPT,   # if you want to check prompt
)


# ────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────

@pytest.fixture
def sample_image_bytes():
    """Create a small synthetic image as bytes (black square with white text)"""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.putText(
        img,
        "OLD TOM DISTILLERY",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        "45% Alc./Vol. (90 Proof)",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        "GOVERNMENT WARNING: ...",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


@pytest.fixture
def mock_extracted_data():
    """Typical successful Gemini extraction output"""
    return {
        "brand_name": "OLD TOM DISTILLERY",
        "class_type": "Straight Bourbon Whiskey",
        "alcohol_content": "45% Alc./Vol. (90 Proof)",
        "net_contents": "750 mL",
        "government_warning_full": (
            "GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD "
            "NOT DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF "
            "BIRTH DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR "
            "ABILITY TO DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS."
        ),
        "warning_header_is_all_caps": True,
        "warning_header_is_bold": True,
        "extracted_confidence": "high",
        "notes": "Clear image, no glare",
    }


# ────────────────────────────────────────────────
# Tests: Preprocessing
# ────────────────────────────────────────────────

def test_preprocess_image_returns_bytes(sample_image_bytes):
    result = preprocess_image(sample_image_bytes)
    assert isinstance(result, bytes)
    assert len(result) > 100  # some reasonable minimum size


def test_preprocess_image_handles_invalid_input():
    invalid = b"not an image"
    result = preprocess_image(invalid)
    assert isinstance(result, bytes)
    # Should return input or fallback bytes
    assert len(result) > 0


# ────────────────────────────────────────────────
# Tests: Field Comparison
# ────────────────────────────────────────────────

def test_compare_fields_perfect_match(mock_extracted_data):
    results, verdict, confidence, notes = compare_fields(
        mock_extracted_data,
        expected_brand="old tom distillery",  # case-insensitive
        expected_abv="45% Alc./Vol. (90 Proof)",
        expected_warning=mock_extracted_data["government_warning_full"],
    )

    assert verdict == "APPROVED"
    assert confidence == "high"
    assert all(r["match"] == "✅" for r in results)
    assert "Exact match" in results[-1]["notes"]


def test_compare_fields_brand_case_insensitive(mock_extracted_data):
    results, verdict, _, _ = compare_fields(
        mock_extracted_data,
        expected_brand="Old Tom DISTILLERY",
        expected_abv="45% Alc./Vol. (90 Proof)",
        expected_warning=mock_extracted_data["government_warning_full"],
    )
    brand_result = next(r for r in results if r["field"] == "Brand Name")
    assert brand_result["match"] == "✅"


def test_compare_fields_warning_formatting_fail(mock_extracted_data):
    extracted = mock_extracted_data.copy()
    extracted["warning_header_is_all_caps"] = False
    extracted["warning_header_is_bold"] = False

    results, verdict, _, _ = compare_fields(
        extracted,
        "OLD TOM DISTILLERY",
        "45% Alc./Vol. (90 Proof)",
        extracted["government_warning_full"],
    )

    assert verdict == "REJECTED – Issues found"
    warning_result = next(r for r in results if r["field"] == "Government Warning")
    assert warning_result["match"] == "❌"
    assert "not ALL CAPS" in warning_result["notes"]


# ────────────────────────────────────────────────
# Tests: Gemini extraction (mocked)
# ────────────────────────────────────────────────

@pytest.mark.asyncio  # if your code becomes async in future
def test_extract_label_data_mocked(mocker, sample_image_bytes):
    # Mock the entire client.models.generate_content call
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {
            "brand_name": "TEST BRAND",
            "alcohol_content": "40% ABV",
            "government_warning_full": "GOVERNMENT WARNING: test text",
            "warning_header_is_all_caps": True,
            "warning_header_is_bold": True,
            "extracted_confidence": "high",
            "notes": "Mocked test",
        }
    )

    mock_generate = mocker.patch(
        "app.client.models.generate_content", return_value=mock_response
    )

    # Call the function
    result = extract_label_data(sample_image_bytes, preprocess=False)

    assert isinstance(result, dict)
    assert result["brand_name"] == "TEST BRAND"
    assert mock_generate.call_count == 1

    # Optional: verify that prompt was passed correctly
    call_args = mock_generate.call_args[1]
    assert "contents" in call_args
    assert len(call_args["contents"]) == 2  # prompt + image
    assert call_args["config"].temperature == 0.0
    assert call_args["config"].response_mime_type == "application/json"


def test_extract_label_data_handles_non_json(mocker, sample_image_bytes):
    mock_response = MagicMock()
    mock_response.text = "Not JSON at all"

    mocker.patch("app.client.models.generate_content", return_value=mock_response)

    result = extract_label_data(sample_image_bytes, preprocess=False)
    assert result is None  # your code should return None on invalid JSON
