import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from core.orchestra import (
    BaseOrchestra,
    TemporalThought,
    SimpleOrchestra,
    log_error
)

# Фикстура для тестирования SimpleOrchestra
@pytest.fixture
def orchestra():
    """Фикстура, создающая экземпляр SimpleOrchestra с mock для embed_fn"""
    embed_fn = MagicMock(return_value=np.array([1.0, 0.5, 0.25]))
    return SimpleOrchestra(embed_fn)

### 1. Тесты для log_error() ###
def test_log_error_with_exception(capsys):
    """Тестируем log_error с передачей исключения"""
    try:
        1/0
    except Exception as e:
        log_error("Test error", e)
    
    captured = capsys.readouterr()
    assert "ERROR: Test error" in captured.err
    assert "Exception: division by zero" in captured.err

def test_log_error_without_exception(capsys):
    """Тестируем log_error без исключения"""
    log_error("Test error")
    captured = capsys.readouterr()
    assert "ERROR: Test error" in captured.err
    assert "Exception:" not in captured.err

### 2. Тесты для TemporalThought ###
def test_temporal_thought_validation():
    """Тестируем валидацию half_life"""
    with pytest.raises(ValueError, match="must be positive"):
        TemporalThought("test", np.array([1]), half_life=0)

def test_temporal_thought_zero_age():
    """Тестируем нулевой возраст"""
    thought = TemporalThought("test", np.array([1]))
    assert thought.weight() == 1.0  # Вес сразу после создания

### 3. Тесты для SimpleOrchestra ###
@pytest.mark.parametrize("emb,expected_msg", [
    (None, "empty result"),
    (np.array([]), "empty result"),
    (np.array([np.nan, 1.0]), "contains NaN or Inf"),
    (np.array([1.0, np.inf]), "contains NaN or Inf")
])
def test_add_thought_invalid_embeddings(orchestra, emb, expected_msg):
    """Тестируем обработку невалидных эмбеддингов"""
    orchestra.embed_fn.return_value = emb

    with pytest.raises(RuntimeError) as exc_info:
        orchestra.add_thought("test")
    
    error_text = str(exc_info.value)
    
    # Проверяем, что сообщение содержит ожидаемый фрагмент
    assert expected_msg in error_text
    assert "Failed to process text" in error_text  # Убедимся, что обертка catch работает

def test_add_thought_embedding_exception(orchestra):
    """Тестируем исключение в embed_fn"""
    orchestra.embed_fn.side_effect = ValueError("Embedding failed")
    with pytest.raises(RuntimeError, match="Failed to process text"):
        orchestra.add_thought("test")

def test_dissonance_matrix_zero_weights(orchestra):
    """Тестируем нулевые веса мыслей"""
    # Создаем мысль с нулевым весом
    thought = TemporalThought("test", np.array([1, 0, 0]))
    with patch.object(thought, 'weight', return_value=0.0):
        orchestra.voices['melody'].append(thought)
        names, matrix = orchestra.calculate_dissonance_matrix()
        assert len(names) == 0

def test_dissonance_matrix_nan_handling(orchestra):
    """Тестируем обработку NaN в эмбеддингах"""
    # Добавляем мысли с проблемными эмбеддингами
    orchestra.voices['melody'].append(
        TemporalThought("nan_thought", np.array([np.nan, 1.0]))
    )
    orchestra.voices['counterpoint'].append(
        TemporalThought("inf_thought", np.array([1.0, np.inf]))
    )
    
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert len(names) == 0

def test_full_workflow(orchestra):
    """Тестируем полный цикл работы"""
    # 1. Добавляем валидные мысли
    orchestra.embed_fn.side_effect = [
        np.array([1.0, 0.0]),  # melody
        np.array([0.5, 0.5]),   # counterpoint
        np.array([0.0, 1.0])    # bass
    ]
    
    assert orchestra.add_thought("melody") == "melody"
    assert orchestra.add_thought("counter") == "counterpoint"
    assert orchestra.add_thought("bass") == "bass"
    
    # 2. Проверяем матрицу диссонансов
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert len(names) == 3
    assert matrix.shape == (3, 3)
    assert np.all(np.diag(matrix) == 0)  # Диагональ должна быть 0

def test_add_thought_with_nan_embedding(orchestra):
    """Тестируем явно добавление мысли с NaN эмбеддингом"""
    orchestra.embed_fn.return_value = np.array([np.nan, 1.0])
    with pytest.raises(RuntimeError, match="contains NaN or Inf"):
        orchestra.add_thought("test")

def test_add_thought_with_inf_embedding(orchestra):
    """Тестируем явно добавление мысли с Inf эмбеддингом"""
    orchestra.embed_fn.return_value = np.array([1.0, np.inf])
    with pytest.raises(RuntimeError, match="contains NaN or Inf"):
        orchestra.add_thought("test")

def test_add_thought_nan_embedding_raises_error(orchestra):
    """Тестируем, что NaN-эмбеддинг вызывает исключение"""
    orchestra.embed_fn.return_value = np.array([np.nan, 0.5, 1.0])
    
    with pytest.raises(RuntimeError, match="Embedding contains NaN or Inf values"):
        orchestra.add_thought("test")