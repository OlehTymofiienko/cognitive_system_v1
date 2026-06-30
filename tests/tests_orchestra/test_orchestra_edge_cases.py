# tests/test_orchestra_edge_cases.py
import pytest
import numpy as np
from collections import deque
from unittest.mock import MagicMock
from core.orchestra import SimpleOrchestra

from core.context_manager import ContextManager

def test_trigger_context_manager():
    cm = ContextManager()
    dummy_contexts = [{"coherence": 0.5}, {"coherence": 0.7}]
    result = cm.get_coherence(dummy_contexts)
    assert result == pytest.approx(0.6, 0.01)


@pytest.fixture
def mock_embed_fn():
    """Фикстура для мока функции эмбеддинга"""
    mock = MagicMock()
    # Возвращаем фиксированный вектор для тестов
    mock.return_value = np.array([1.0, 0.0])
    return mock

def test_cosine_similarity_edge_cases(mock_embed_fn):
    """Тест edge-кейсов для вычисления косинусной схожести"""
    # Инициализируем SimpleOrchestra с мокнутой функцией эмбеддинга
    orchestra = SimpleOrchestra(embed_fn=mock_embed_fn)
    
    # 1. Тест с нулевыми векторами
    zero_vec = np.zeros(5)
    assert orchestra._cosine(zero_vec, zero_vec) == 0.0
    
    # 2. Тест с одним нулевым вектором
    assert orchestra._cosine(np.array([1.0, 0.0]), zero_vec[:2]) == 0.0
    
    # 3. Тест с идентичными векторами
    vec = np.array([0.5, 0.5])
    assert orchestra._cosine(vec, vec) == pytest.approx(1.0, 0.01)

def test_empty_voice_handling(mock_embed_fn):
    """Тест обработки пустых голосов при расчете матрицы диссонансов"""
    # Инициализируем SimpleOrchestra
    orchestra = SimpleOrchestra(embed_fn=mock_embed_fn)
    
    # Имитируем пустой голос, используя deque из collections
    orchestra.voices['melody'] = deque(maxlen=3)
    
    # Проверяем, что метод не падает с пустыми голосами
    names, matrix = orchestra.calculate_dissonance_matrix()
    
    # Проверяем ожидаемые результаты
    assert len(names) == 0  # Нет активных голосов
    assert matrix.shape == (0, 0)  # Пустая матрица

def test_cosine_nan_protection(mock_embed_fn):
    orchestra = SimpleOrchestra(embed_fn=mock_embed_fn)
    invalid_vec = np.array([np.nan, np.nan])
    valid_vec = np.array([1.0, 0.0])
    result = orchestra._cosine(invalid_vec, valid_vec)
    assert result == 0.0  # защита от NaN

def test_cosine_dimension_mismatch(mock_embed_fn):
    orchestra = SimpleOrchestra(embed_fn=mock_embed_fn)
    with pytest.raises(ValueError):
        orchestra._cosine(np.array([1.0]), np.array([1.0, 0.0]))

def test_add_thought_edge_cases(mock_embed_fn):
    """Тест обработки крайних случаев при добавлении мысли"""
    orchestra = SimpleOrchestra(embed_fn=mock_embed_fn)
    
    # 1. Первая мысль должна установить ключ
    voice = orchestra.add_thought("test")
    assert orchestra.key is not None
    
    # 2. Очень длинный текст
    long_text = "a" * 10000
    orchestra.add_thought(long_text)  # Не должен падать
