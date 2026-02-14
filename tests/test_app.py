# tests/test_app.py - Pytest suite for the app

import pytest
from unittest.mock import patch, MagicMock
import json
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.abspath('..')) # or the specific path to the directory
from app import preprocess_image, extract_label_data, compare_fields  # Import from your app.py

# Sample mock extraction data (simulates Gemini output)
MOCK_EXTRACTION = {
    "brand_name": "OLD TOM DISTILLERY",
    "class_type": "Kentucky Straight Bourbon Whiskey",
    "alcohol_content": "45% Alc./Vol. (90 Proof)",
    "net_contents": "750 mL",
    "government_warning_full": "GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS.",
    "warning_header_is_all_caps": True,
    "warning_header_is_bold": True,
    "extracted_confidence": "high",
    "notes": "Clear image"
}

@pytest.fixture
def sample_image_bytes():
    # Create a simple black image as bytes
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def test_preprocess_image(sample_image_bytes):
    processed = preprocess_image(sample_image_bytes)
    assert isinstance(processed, bytes)
    assert len(processed) > 0  # Ensure something was returned

@patch('app.genai')  # Mock the entire genai module
def test_extract_label_data(mock_genai, sample_image_bytes):
    # Setup mocks
    mock_file = MagicMock()
    mock_genai.upload_file.return_value = mock_file
    
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(MOCK_EXTRACTION)
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model
    
    result = extract_label_data(sample_image_bytes, preprocess=False)
    assert result == MOCK_EXTRACTION
    mock_genai.upload_file.assert_called_once()
    mock_model.generate_content.assert_called_once()
    mock_file.delete.assert_called_once()

def test_compare_fields_match():
    expected_brand = "old tom distillery"  # Case insensitive
    expected_abv = "45% Alc./Vol. (90 Proof)"
    expected_warning = MOCK_EXTRACTION["government_warning_full"]
    
    results, verdict, confidence, notes = compare_fields(
        MOCK_EXTRACTION, expected_brand, expected_abv, expected_warning
    )
    
    assert verdict == "APPROVED"
    assert all(r['match'] == '✅' for r in results)
    assert confidence == "high"
    assert notes == "Clear image"

def test_compare_fields_mismatch():
    expected_brand = "Wrong Brand"
    expected_abv = "50% Alc./Vol. (100 Proof)"
    expected_warning = "Wrong warning text"
    
    mock_extract = MOCK_EXTRACTION.copy()
    mock_extract["warning_header_is_all_caps"] = False
    mock_extract["warning_header_is_bold"] = False
    
    results, verdict, _, _ = compare_fields(
        mock_extract, expected_brand, expected_abv, expected_warning
    )
    
    assert verdict == "REJECTED - Issues found"
    assert all(r['match'] == '❌' for r in results)

def test_compare_fields_abv_normalization():
    mock_extract = MOCK_EXTRACTION.copy()
    mock_extract["alcohol_content"] = "45% Alc/Vol (90°))"  # With degree symbol
    
    results, _, _, _ = compare_fields(
        mock_extract, "OLD TOM DISTILLERY", "45% Alc./Vol. (90 Proof)", MOCK_EXTRACTION["government_warning_full"]
    )
    
    abv_result = next(r for r in results if r['field'] == 'Alcohol Content')
    assert abv_result['match'] == '✅'  # Should match after normalization
