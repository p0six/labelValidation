# tests/test_app.py
"""
Pytest suite for the alcohol label verification app.
Uses separate classes to isolate Gemini and OpenAI provider testing.
Each class reloads app.py with the correct LLM_PROVIDER.
"""

import importlib
import json
import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Initial import (will be reloaded in fixtures)
import app


# ────────────────────────────────────────────────
# Common Fixtures
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
    """Typical successful LLM extraction output"""
    return {
        "brand_name": "TEST BRAND",
        "class_type": "Straight Bourbon Whiskey",
        "alcohol_content": "40% ABV",
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
        "notes": "Mocked test",
    }


# ────────────────────────────────────────────────
# Gemini Provider Tests
# ────────────────────────────────────────────────

class TestGeminiProvider:
    @pytest.fixture(autouse=True)
    def setup_gemini(self):
        """Reload app.py with LLM_PROVIDER = 'gemini'"""
        original_env = os.environ.get("LLM_PROVIDER")
        os.environ["LLM_PROVIDER"] = "gemini"
        importlib.reload(app)
        yield
        # Cleanup
        if original_env is not None:
            os.environ["LLM_PROVIDER"] = original_env
        else:
            os.environ.pop("LLM_PROVIDER", None)
        importlib.reload(app)

    def test_extract_label_data_mocked(self, mocker, sample_image_bytes, mock_extracted_data):
        mock_response = MagicMock()
        mock_response.text = json.dumps(mock_extracted_data)

        mock_call = mocker.patch(
            "app.client.models.generate_content",
            return_value=mock_response
        )

        result = app.extract_label_data(sample_image_bytes, preprocess=False)

        assert isinstance(result, dict)
        assert result["brand_name"] == mock_extracted_data["brand_name"]
        assert mock_call.call_count == 1

    def test_extract_label_data_handles_non_json(self, mocker, sample_image_bytes):
        mock_response = MagicMock()
        mock_response.text = "Not JSON at all"

        mocker.patch("app.client.models.generate_content", return_value=mock_response)

        result = app.extract_label_data(sample_image_bytes, preprocess=False)
        assert result is None


# ────────────────────────────────────────────────
# OpenAI Provider Tests
# ────────────────────────────────────────────────

class TestOpenAIProvider:
    @pytest.fixture(autouse=True)
    def setup_openai(self):
        """Reload app.py with LLM_PROVIDER = 'openai'"""
        original_env = os.environ.get("LLM_PROVIDER")
        os.environ["LLM_PROVIDER"] = "openai"
        importlib.reload(app)
        yield
        # Cleanup
        if original_env is not None:
            os.environ["LLM_PROVIDER"] = original_env
        else:
            os.environ.pop("LLM_PROVIDER", None)
        importlib.reload(app)

    def test_extract_label_data_mocked(self, mocker, sample_image_bytes, mock_extracted_data):
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps(mock_extracted_data)

        mock_call = mocker.patch(
            "app.client.chat.completions.create",
            return_value=mock_completion
        )

        result = app.extract_label_data(sample_image_bytes, preprocess=False)

        assert isinstance(result, dict)
        assert result["brand_name"] == mock_extracted_data["brand_name"]
        assert mock_call.call_count == 1

    def test_extract_label_data_handles_non_json(self, mocker, sample_image_bytes):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Not JSON at all"

        mocker.patch("app.client.chat.completions.create", return_value=mock_response)

        result = app.extract_label_data(sample_image_bytes, preprocess=False)
        assert result is None
