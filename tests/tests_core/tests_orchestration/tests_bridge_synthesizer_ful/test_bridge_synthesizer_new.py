#tests\bridge_synthesizer_ful\test_bridge_synthesizer_new.py

import pytest
import math
import numpy as np
from core.models import Thought
from core.orchestration.bridge_synthesizer import BridgeSynthesizer

class TestBridgeSynthesizer:
    @pytest.fixture
    def synthesizer(self):
        return BridgeSynthesizer(session_topic="test_topic")

    def test_less_than_two_thoughts_raises_error(self, synthesizer):
        """Проверка строки 25: исключение при < 2 мыслях"""
        with pytest.raises(ValueError, match="BridgeSynthesizer требует минимум 2 мысли"):
            synthesizer.generate(
                [Thought(content="single thought", voice="voice", coherence=0.5)], 
                [[]]
            )

    def test_handles_nan_in_dissonance_matrix(self, synthesizer):
        """Проверка строки 34: обработка NaN в матрице диссонанса"""
        thoughts = [
            Thought(content="thought 1", voice="voice", coherence=0.5),
            Thought(content="thought 2", voice="voice", coherence=0.5)
        ]
        dissonance = [[0, np.nan], [np.nan, 0]]
        
        result = synthesizer.generate(thoughts, dissonance)
        
        assert isinstance(result, Thought)

    def test_metadata_fallback_to_indices(self, synthesizer):
        """Проверка строк 51-52: fallback на индексы если нет id в metadata"""
        thoughts = [
            Thought(content="thought 1", voice="voice", coherence=0.5),  # без metadata
            Thought(content="thought 2", voice="voice", coherence=0.5, metadata={"other_field": "value"})  # metadata без id
        ]
        dissonance = [[0, 0.8], [0.8, 0]]
        
        result = synthesizer.generate(thoughts, dissonance)
        
        assert result.metadata["bridge_of"] == [0, 1]
        assert "dissonance" in result.metadata

    def test_metadata_uses_ids_when_present(self, synthesizer):
        """Дополнительный тест: проверка использования id когда они есть"""
        thoughts = [
            Thought(content="thought 1", voice="voice", coherence=0.5, metadata={"id": "id1"}),
            Thought(content="thought 2", voice="voice", coherence=0.5, metadata={"id": "id2"})
        ]
        dissonance = [[0, 0.8], [0.8, 0]]
        
        result = synthesizer.generate(thoughts, dissonance)
        
        assert result.metadata["bridge_of"] == ["id1", "id2"]

    def test_generate_without_language_model(self):
        """Проверка строк 53-54: генерация контента без language_model"""
        synthesizer = BridgeSynthesizer(session_topic="test_topic")  # Не передаем language_model
        thoughts = [
            Thought(content="first thought", voice="voice", coherence=0.5),
            Thought(content="second thought", voice="voice", coherence=0.5)
        ]
        dissonance = [[0, 0.9], [0.9, 0]]
        
        result = synthesizer.generate(thoughts, dissonance)
        
        # Проверяем, что контент сгенерирован по шаблону (без модели)
        expected_content = (
            f"Bridge between 'first thought' and 'second thought' "
            f"on 'test_topic'"
        )
        assert result.content == expected_content
        assert result.metadata["dissonance"] == 0.9