import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch
from core.orchestra import SimpleOrchestra, TemporalThought, log_error, BaseOrchestra

@pytest.fixture
def mock_embed_fn():
    def embed(text: str) -> np.ndarray:
        return np.array([ord(c) for c in text[:5]])
    return embed

@pytest.fixture
def orchestra(mock_embed_fn):
    return SimpleOrchestra(mock_embed_fn)

def test_base_orchestra_abstract_methods():
    """Тест абстрактных методов BaseOrchestra (строки 25-29)"""
    with pytest.raises(TypeError):
        BaseOrchestra()  # Нельзя создать экземпляр абстрактного класса

def test_orchestra_initialization(orchestra):
    """Тест инициализации (строки 19-22)"""
    assert orchestra.key is None
    assert set(orchestra.voices.keys()) == {'melody', 'counterpoint', 'bass'}
    assert orchestra.voices['melody'].maxlen == 4
    assert orchestra.voices['counterpoint'].maxlen == 3
    assert orchestra.voices['bass'].maxlen == 2

def test_add_thought_invalid_input(orchestra):
    """Тест обработки невалидного ввода (строки 45, 54)"""
    with pytest.raises(ValueError, match="Text cannot be empty"):
        orchestra.add_thought("")
    
    with pytest.raises(ValueError, match="must be a string"):
        orchestra.add_thought(None)
    
    with pytest.raises(ValueError, match="must be a string"):
        orchestra.add_thought(123)

def test_add_thought_embedding_failure(orchestra):
    """Тест ошибок эмбеддинга (строки 66-70)"""
    with patch.object(orchestra, 'embed_fn', side_effect=Exception("Embedding failed")):
        with pytest.raises(RuntimeError, match="Failed to process text"):
            orchestra.add_thought("test")
    
    with patch.object(orchestra, 'embed_fn', return_value=np.array([])):
        with pytest.raises(RuntimeError, match="empty result"):
            orchestra.add_thought("test")

def test_first_thought_sets_key(orchestra):
    """Тест установки ключа первой мыслью (строка 89)"""
    assert orchestra.key is None
    orchestra.add_thought("first thought")
    assert orchestra.key is not None

def test_voice_assignment_logic(orchestra):
    """Тест логики распределения по голосам"""
    # Первая мысль устанавливает ключ
    orchestra.add_thought("base thought")
    
    with patch.object(orchestra, '_calculate_dynamic_threshold', return_value=0.7), \
         patch.object(orchestra, '_cosine') as mock_cosine:
        
        test_cases = [
            (0.71, 'melody'),      # Выше порога
            (0.70, 'melody'),      # Теперь и на пороге - melody
            (0.69, 'counterpoint'), # Чуть ниже порога
            (0.42, 'counterpoint'), # Граница counterpoint
            (0.41, 'bass'),        # Ниже границы counterpoint
        ]
        
        for sim, expected_voice in test_cases:
            mock_cosine.return_value = sim
            voice = orchestra.add_thought(f"test {sim}")
            assert voice == expected_voice, \
                f"При similarity={sim} ожидался {expected_voice}, получен {voice}"
            
def test_voice_boundary_conditions(orchestra):
    """Тест граничных значений распределения"""
    orchestra.add_thought("base") 

    with patch.object(orchestra, '_calculate_dynamic_threshold', return_value=1.0), \
         patch.object(orchestra, '_cosine') as mock_cosine:

        test_cases = [
            (1.01, 'melody'),
            (1.00, 'melody'),    # Ровно threshold
            (0.99, 'counterpoint'),
            (0.60, 'counterpoint'),  # Ровно threshold*0.6
            (0.59, 'bass')
        ]

        for sim, expected_voice in test_cases:
            mock_cosine.return_value = sim
            voice = orchestra.add_thought(f"test {sim}")
            assert voice == expected_voice, \
                f"При similarity={sim} ожидался {expected_voice}, получен {voice}"
            
def test_empty_orchestra(orchestra):
    """Тест поведения с пустым оркестром"""
    assert orchestra.get_coherence() == 0.0
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert len(names) == 0

def test_temporal_thought_weight():
    """Тест временного веса мысли (строки 138, 146)"""
    thought = TemporalThought("test", np.array([1, 2, 3]), half_life=10)
    
    # Проверяем начальный вес
    assert thought.weight() == 1.0
    
    # Эмулируем старение
    with patch('time.time', return_value=time.time() + 10):
        assert thought.weight() == pytest.approx(0.5, abs=0.01)  # ~0.5 после полураспада
    
    with patch('time.time', return_value=time.time() + 20):
        assert thought.weight() == pytest.approx(0.25, abs=0.01)  # ~0.25 после двух полураспадо

def test_dissonance_matrix_edge_cases(orchestra):
    """Тест edge cases для матрицы диссонансов (строки 186, 188, 194-198)"""
    # Пустой оркестр
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert len(names) == 0
    assert matrix.shape == (0, 0)
    
    # Только один голос
    orchestra.add_thought("single thought")
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert len(names) == 1
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 0.0  # Диссонанс с самим собой всегда 0

def test_cosine_similarity_edge_cases(orchestra):
    """Тест edge cases для _cosine (строки 229-234)"""
    # Нулевой вектор
    zero_vec = np.array([0, 0, 0])
    assert orchestra._cosine(zero_vec, zero_vec) == 0.0
    
    # Один нулевой вектор
    normal_vec = np.array([1, 2, 3])
    assert orchestra._cosine(zero_vec, normal_vec) == 0.0
    
    # NaN в векторе
    nan_vec = np.array([np.nan, 1, 2])
    assert orchestra._cosine(nan_vec, normal_vec) == 0.0

def test_log_error(capsys):
    """Тест функции логирования ошибок"""
    exc = ValueError("test error")
    log_error("test message", exc)
    
    captured = capsys.readouterr()
    assert "LOG_ERROR CALLED" in captured.out
    assert "ERROR: test message" in captured.err
    assert "Exception: test error" in captured.err

