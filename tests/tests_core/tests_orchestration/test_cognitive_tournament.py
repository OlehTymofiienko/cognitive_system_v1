#tests\test_cognitive_tournament.py

import pytest
from core.models import Thought
from unittest.mock import MagicMock
from core.orchestration.cognitive_tournament import CognitiveTournament

@pytest.fixture
def sample_thoughts():
    return [
        Thought("This is good", "melody", 0.5),
        Thought("I am not sure", "bass", 0.6),
        Thought("Absolutely yes", "counterpoint", 0.7),
        Thought("No way", "melody", 0.4)
    ]

def test_no_tournament_below_threshold(sample_thoughts):
    tour = CognitiveTournament("Test", depth_threshold=5)
    out = tour.run(sample_thoughts, depth=3)
    assert out == sample_thoughts

def test_tournament_elimination_and_synthesis(sample_thoughts):
    tour = CognitiveTournament("Test", depth_threshold=2)
    out = tour.run(sample_thoughts, depth=4)
    
    assert len(out) == 3
    pro, con, synth = out
    
    # Более точные проверки чемпионов
    assert pro.content == "Absolutely yes"  # max coherence в pro
    assert con.content == "I am not sure"   # max coherence в con
    
    # Проверяем структуру синтеза
    assert "Bridge" in synth.content
    assert "Absolutely yes" in synth.content
    assert "I am not sure" in synth.content
    assert synth.voice == "melody"
    assert synth.coherence == pytest.approx(0.65)  # (0.7 + 0.6)/2

def test_synthesis_with_language_model(sample_thoughts):
    """Тестируем использование language_model при синтезе (строка 53)"""
    # Создаем mock для language_model
    mock_lm = MagicMock()
    mock_lm.return_value = [{"generated_text": "Synthetic conclusion from model"}]
    
    # Создаем турнир с mock language_model
    tour = CognitiveTournament(
        session_topic="Test", 
        depth_threshold=2,
        language_model=mock_lm
    )
    
    # Запускаем турнир
    out = tour.run(sample_thoughts, depth=4)
    
    # Проверяем что language_model был использован
    pro, con, synth = out
    assert synth.content == "Synthetic conclusion from model"
    mock_lm.assert_called_once()


