#tests\tests_core\tests_orchestration\bridge_synthesizer_ful\test_bridge_synthesizer_end.py

import pytest
from unittest.mock import MagicMock
from core.models import Thought
from core.orchestration.bridge_synthesizer import BridgeSynthesizer

class TestBridgeSynthesizerLanguageModel:
    def test_language_model_generation(self):
        """Тест генерации контента через language_model"""
        # Мокируем language_model чтобы возвращать предсказуемый результат
        mock_lm = MagicMock()
        mock_lm.return_value = [{"generated_text": "  Mocked bridge thought  "}]
        
        synthesizer = BridgeSynthesizer(
            session_topic="test topic",
            language_model=mock_lm
        )
        
        test_thoughts = [
            Thought(content="Thought 1", voice="melody", coherence=0.8),
            Thought(content="Thought 2", voice="harmony", coherence=0.7)
        ]
        dissonance_matrix = [[0, 0.8], [0.8, 0]]
        
        result = synthesizer.generate(test_thoughts, dissonance_matrix)
        
        # Проверяем что language_model был вызван с правильным prompt
        mock_lm.assert_called_once()
        call_args = mock_lm.call_args[0][0]
        assert "Thought 1" in call_args
        assert "Thought 2" in call_args
        assert "test topic" in call_args
        
        # Проверяем что результат был правильно обработан (strip())
        assert result.content == "Mocked bridge thought"
        assert result.coherence == pytest.approx(1.0)  # max_d + 0.2 = 0.8 + 0.2 = 1.0
        
    def test_language_model_empty_output(self):
        """Тест обработки пустого вывода language_model"""
        mock_lm = MagicMock()
        mock_lm.return_value = [{"generated_text": ""}]
        
        synthesizer = BridgeSynthesizer(
            session_topic="test",
            language_model=mock_lm
        )
        
        test_thoughts = [
            Thought(content="A", voice="melody", coherence=0.5),
            Thought(content="B", voice="harmony", coherence=0.6)
        ]
        dissonance_matrix = [[0, 0.9], [0.9, 0]]
        
        result = synthesizer.generate(test_thoughts, dissonance_matrix)
        
        # Проверяем что используется fallback контент при пустом выводе
        assert result.content == "Bridge between 'A' and 'B' on 'test'"
        assert result.coherence == pytest.approx(1.0)
        
    def test_language_model_with_whitespace(self):
        """Тест обработки вывода с пробельными символами"""
        mock_lm = MagicMock()
        mock_lm.return_value = [{"generated_text": "  \n  Test output  \t  "}]
        
        synthesizer = BridgeSynthesizer(
            session_topic="test",
            language_model=mock_lm
        )
        
        test_thoughts = [
            Thought(content="X", voice="melody", coherence=0.7),
            Thought(content="Y", voice="harmony", coherence=0.8)
        ]
        dissonance_matrix = [[0, 0.7], [0.7, 0]]
        
        result = synthesizer.generate(test_thoughts, dissonance_matrix)
        
        # Проверяем что лишние пробелы удалены
        assert result.content == "Test output"
        assert result.coherence == pytest.approx(0.9)  # 0.7 + 0.2 = 0.9