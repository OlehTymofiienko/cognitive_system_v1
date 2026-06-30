#tests\test_orchestra_behavior.py

import pytest
import logging
from unittest.mock import MagicMock, patch
from system.orchestra_behavior import OrchestraBehavior, get_latest_text

@pytest.fixture
def orchestra_fixture():
    """Универсальная фикстура для всех тестов"""
    mock_orchestra = MagicMock()
    mock_orchestra.tick_interval = 5.0
    mock_orchestra.add_thought.return_value = "test_voice"
    
    semantic = MagicMock()
    behavior = OrchestraBehavior(
        orchestra_mgr=mock_orchestra,
        semantic_analyzer=semantic,
        counterpoint_threshold=0.3,
        dissonance_threshold=0.75,
        coh_window=5,
        low_coh_delta=0.05,
        high_coh_delta=0.2,
        min_tick=1.0,
        max_tick=10.0
    )
    return behavior, mock_orchestra, semantic

# Тесты для вспомогательной функции
def test_get_latest_text():
    assert get_latest_text({"voices": {"melody": [{"text": "test"}]}}, "melody") == "test"
    assert get_latest_text({"voices": {}}, "missing", "default") == "default"

# Базовые тесты
def test_orchestra_behavior_init(orchestra_fixture):
    behavior, _, _ = orchestra_fixture
    assert behavior.cp_th == 0.3
    assert behavior.dis_th == 0.75
    assert behavior.min_tick == 1.0

# Параметризованные тесты
@pytest.mark.parametrize("weights,should_trigger", [
    ([0.1, 0.2], True),
    ([0.4, 0.5], False)
])
def test_handle_counterpoint(orchestra_fixture, weights, should_trigger):
    behavior, mock_orchestra, _ = orchestra_fixture
    state = {
        "voices": {
            "counterpoint": [{"weight": w} for w in weights],
            "melody": [{"text": "Melody"}],
            "bass": [{"text": "Bass"}]
        }
    }
    
    behavior._handle_counterpoint(state)
    
    if should_trigger:
        mock_orchestra.add_thought.assert_called_once()
        assert "tension" in mock_orchestra.add_thought.call_args[0][0]
    else:
        mock_orchestra.add_thought.assert_not_called()

@pytest.mark.parametrize("dissonance,expected_tick", [
    (0.8, 6.0),
    (0.5, 5.0)
])
def test_handle_dissonance(orchestra_fixture, dissonance, expected_tick):
    behavior, mock_orchestra, _ = orchestra_fixture
    state = {
        "voices": {
            "melody": [{"text": "M"}],
            "bass": [{"text": "B"}]
        },
        "dissonance": {
            "names": ["melody", "bass"],
            "matrix": [[0, dissonance], [dissonance, 0]]
        }
    }
    
    behavior._handle_dissonance(state)
    
    if dissonance > behavior.dis_th:
        mock_orchestra.add_thought.assert_called_once()
    assert mock_orchestra.tick_interval == expected_tick

# Интеграционные тесты
def test_apply_integration(orchestra_fixture):
    behavior, mock_orchestra, _ = orchestra_fixture
    test_state = {
        "voices": {
            "counterpoint": [{"weight": 0.1}],
            "melody": [{"text": "M"}],
            "bass": [{"text": "B"}]
        },
        "dissonance": {
            "names": ["melody", "bass"],
            "matrix": [[0, 0.8], [0.8, 0]]
        }
    }
    
    behavior.apply(test_state, [0.5, 0.7, 0.4])
    assert mock_orchestra.add_thought.call_count == 2
    assert mock_orchestra.tick_interval == 6.0

# Edge-кейсы
def test_missing_voices(orchestra_fixture):
    behavior, mock_orchestra, _ = orchestra_fixture
    state = {
        "dissonance": {
            "names": ["other"],
            "matrix": [[0]]
        }
    }
    behavior._handle_dissonance(state)
    mock_orchestra.add_thought.assert_not_called()

# Тесты логирования
def test_logging_in_behavior(orchestra_fixture, caplog):
    behavior, _, _ = orchestra_fixture
    caplog.set_level(logging.INFO)
    
    behavior.apply(
        state={
            "voices": {
                "counterpoint": [{"weight": 0.1}],
                "melody": [{"text": "M"}],
                "bass": [{"text": "B"}]
            },
            "dissonance": {
                "names": ["melody", "bass"],
                "matrix": [[0, 0.8], [0.8, 0]]
            }
        },
        coherence_history=[0.5, 0.7, 0.4]
    )
    
    assert "Weak counterpoint" in caplog.text
    assert "High dissonance (0.80)" in caplog.text

@pytest.mark.parametrize("history_length,should_adapt", [
    (3, False),  # Меньше coh_window (5)
    (5, True),   # Равно coh_window
    (7, True)    # Больше coh_window
])
def test_adapt_tick_interval_history_length(orchestra_fixture, history_length, should_adapt):
    """Тест обработки разной длины истории когерентности"""
    behavior, mock_orchestra, _ = orchestra_fixture
    initial_tick = mock_orchestra.tick_interval
    history = [0.5] * history_length
    
    behavior._adapt_tick_interval(history)
    
    if should_adapt:
        # Проверяем, что значение изменилось
        assert mock_orchestra.tick_interval != initial_tick
    else:
        # Проверяем, что значение осталось прежним
        assert mock_orchestra.tick_interval == initial_tick

@pytest.mark.parametrize("delta,expected_tick,expected_log", [
    (0.02, 4.0, "Low coherence"),      # Низкая дельта - уменьшаем интервал
    (0.30, 6.0, "High coherence"),     # Высокая дельта - увеличиваем интервал 
    (0.10, 5.0, None)                  # Средняя дельта - без изменений
])
def test_adapt_tick_interval_deltas(orchestra_fixture, delta, expected_tick, expected_log, caplog):
    """Тест различных сценариев изменения когерентности"""
    behavior, mock_orchestra, _ = orchestra_fixture
    caplog.set_level(logging.INFO)
    
    # Создаем историю с нужной дельтой
    history = [0.5, 0.5 + delta/2, 0.5 - delta/2, 0.5 + delta/2, 0.5]
    
    behavior._adapt_tick_interval(history)
    
    assert mock_orchestra.tick_interval == expected_tick
    if expected_log:
        assert expected_log in caplog.text
    else:
        assert "coherence Δ" not in caplog.text

def test_adapt_tick_interval_boundaries(orchestra_fixture):
    """Тест граничных значений интервала"""
    behavior, mock_orchestra, _ = orchestra_fixture
    
    # Проверяем минимальное значение
    mock_orchestra.tick_interval = 1.5
    behavior._adapt_tick_interval([0.5]*5)  # Дельта 0
    assert mock_orchestra.tick_interval == 1.0  # Не меньше min_tick
    
    # Проверяем максимальное значение
    mock_orchestra.tick_interval = 9.5
    behavior._adapt_tick_interval([0.1, 0.9, 0.1, 0.9, 0.1])  # Большая дельта
    assert mock_orchestra.tick_interval == 10.0  # Не больше max_tick