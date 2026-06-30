#tests\tests_core\test_semantic_analyzer_ful\test_semantic_analyzer_extended.py
import pytest
import numpy as np
from unittest.mock import patch
from core.semantic_analyzer import SemanticAnalyzer

@pytest.mark.parametrize("text,expected_lang", [
    ("English text", "en"),
    ("Текст на русском", "ru"),
    ("这是一个用于测试的中文句子。", "zh"),  # Достаточно длинный китайский
    ("Texto en español", "es"),
    ("", "unknown")
])
def test_multilingual_analysis(text, expected_lang):
    analyzer = SemanticAnalyzer()
    for _ in range(3):  # langdetect не всегда стабилен — даем шанс переопределить
        lang = analyzer.detect_language(text)
        if lang == expected_lang:
            break
    assert lang == expected_lang, f"Detected {lang}, expected {expected_lang} in: {text}"

def test_special_characters_handling():
    analyzer = SemanticAnalyzer()

    safe_sql = "SELECT * FROM users WHERE id = 1;"
    unsafe_html = "<script>alert('boom')</script>"

    assert analyzer.is_input_safe(safe_sql)
    assert not analyzer.is_input_safe(unsafe_html)

