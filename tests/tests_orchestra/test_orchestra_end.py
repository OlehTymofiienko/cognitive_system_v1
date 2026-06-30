#tests\test_orchestra_end.py

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import time
from core.orchestra import SimpleOrchestra, TemporalThought, log_error, BaseOrchestra

# Фикстуры и вспомогательные функции
def mock_embed_fixed_factory(embeddings):
    """Мок для функции эмбеддинга с фиксированными значениями"""
    counter = 0
    def _embed(_):
        nonlocal counter
        if counter >= len(embeddings):
            raise IndexError("No more embeddings")
        emb = embeddings[counter]
        counter += 1
        return emb
    return _embed

def test_orchestra_initialization():
    """Тест инициализации SimpleOrchestra"""
    embed_fn = MagicMock(return_value=np.array([1,0,0]))
    orch = SimpleOrchestra(embed_fn)
    
    assert orch.key is None
    assert set(orch.voices.keys()) == {'melody', 'counterpoint', 'bass'}
    assert orch.voices['melody'].maxlen == 4
    assert orch.voices['counterpoint'].maxlen == 3
    assert orch.voices['bass'].maxlen == 2
    assert orch.embed_fn == embed_fn

def test_add_thought_invalid_input():
    """Тест обработки невалидного ввода в add_thought"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    
    with pytest.raises(ValueError, match="Text must be a string"):
        orch.add_thought(None)
    
    with pytest.raises(ValueError, match="Text cannot be empty"):
        orch.add_thought("")
    
    with pytest.raises(ValueError, match="Text must be a string"):
        orch.add_thought(123)

def test_embedding_failures():
    """Тест обработки ошибок при получении эмбеддингов"""
    orch = SimpleOrchestra(lambda x: None)
    with pytest.raises(RuntimeError, match="empty result"):
        orch.add_thought("test")
    
    orch = SimpleOrchestra(lambda x: np.array([]))
    with pytest.raises(RuntimeError, match="empty result"):
        orch.add_thought("test")
    
    orch = SimpleOrchestra(lambda x: 1/0)
    with pytest.raises(RuntimeError, match="Failed to process text"):
        orch.add_thought("test")

def test_first_thought_sets_key():
    """Тест установки ключа первой мыслью"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    assert orch.key is None
    orch.add_thought("first")
    assert np.array_equal(orch.key, np.array([1,0,0]))

def test_voice_assignment_full_logic():
    """Полный тест логики распределения по голосам"""
    embs = [
        np.array([1.0, 0.0, 0.0]),  # base (ключ)
        np.array([1.0, 0.0, 0.0]),  # melody (sim=1.0)
        np.array([0.6, 0.8, 0.0]),  # counterpoint (sim=0.6)
        np.array([0.0, 1.0, 0.0])   # bass (sim=0.0)
    ]
    embs = [emb/np.linalg.norm(emb) for emb in embs]
    
    orch = SimpleOrchestra(mock_embed_fixed_factory(embs))
    orch.add_thought("base")  # Использует первый эмбеддинг
    
    with patch.object(orch, '_calculate_dynamic_threshold', return_value=0.7):
        voices = [orch.add_thought(f"t{i}") for i in range(3)]  # Использует остальные 3
        assert voices == ['melody', 'counterpoint', 'bass']

def test_exception_handling(capsys):
    """Тест обработки исключений в add_thought"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    
    with patch.object(orch, '_cosine', side_effect=Exception("Test error")):
        with pytest.raises(RuntimeError, match="Failed to process text"):
            orch.add_thought("test")
    
    captured = capsys.readouterr()
    assert "ERROR: Failed to process text" in captured.err
    assert "Exception: Test error" in captured.err

def test_empty_voice_handling():
    """Тест обработки пустых голосов при расчете матрицы диссонансов"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    names, matrix = orch.calculate_dissonance_matrix()
    assert names == []
    assert matrix.shape == (0, 0)

def test_single_voice_dissonance():
    """Тест матрицы диссонансов с одним голосом"""
    # Мокаем weight() чтобы возвращал ненулевое значение
    with patch('core.orchestra.TemporalThought.weight', return_value=1.0):
        orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
        orch.add_thought("test")
        names, matrix = orch.calculate_dissonance_matrix()
        assert names == ['melody']
        assert matrix.shape == (1, 1)
        assert matrix[0,0] == 0.0

def test_temporal_thought_edge_cases():
    """Тест граничных случаев поведения TemporalThought"""
    
    # 1. Проверка недопустимых значений half_life
    with pytest.raises(ValueError, match="half_life must be positive"):
        TemporalThought("test", np.array([1, 0, 0]), half_life=0)

    with pytest.raises(ValueError, match="half_life must be positive"):
        TemporalThought("test", np.array([1, 0, 0]), half_life=-10)

    # 2. Базовое значение времени
    test_time = 1_000_000.0

    # 3. Сразу после создания — вес должен быть равен 1.0
    with patch('time.time', return_value=test_time):
        thought = TemporalThought("test", np.array([1, 0, 0]), half_life=60)
        assert thought.weight() == pytest.approx(1.0, abs=1e-8)

    # 4. Спустя один half-life — вес должен быть ~0.5
    with patch('time.time', return_value=test_time):
        thought = TemporalThought("test", np.array([1, 0, 0]), half_life=60)

    with patch('time.time', return_value=test_time + 60):
        assert thought.weight() == pytest.approx(0.5, abs=1e-2)

    # 5. При отрицательном возрасте — вес всё равно должен быть 1.0 (по текущей логике)
    with patch('time.time', return_value=test_time):
        thought = TemporalThought("test", np.array([1, 0, 0]), half_life=60)

    with patch('time.time', return_value=test_time - 100):
        assert thought.weight() == pytest.approx(1.0)

    # 6. Спустя много времени — вес должен быть близок к нулю
    with patch('time.time', return_value=test_time):
        thought = TemporalThought("test", np.array([1, 0, 0]), half_life=60)

    with patch('time.time', return_value=test_time + 100_000):
        assert thought.weight() == pytest.approx(0.0, abs=1e-6)

def test_temporal_thought_weight_calculation():
    """Точная проверка расчета веса"""
    test_time = 1000000.0
    with patch('time.time', return_value=test_time):
        thought = TemporalThought("test", np.array([1,0,0]), half_life=60)
        
    with patch('time.time', return_value=test_time + 30):
        assert thought.weight() == pytest.approx(0.7071, abs=1e-4)
    
    with patch('time.time', return_value=test_time + 60):
        assert thought.weight() == pytest.approx(0.5, abs=1e-4)
    
    with patch('time.time', return_value=test_time + 120):
        assert thought.weight() == pytest.approx(0.25, abs=1e-4)

def test_temporal_thought_comprehensive():
    """Всесторонний тест TemporalThought"""
    # 1. Проверка недопустимых значений half_life
    with pytest.raises(ValueError):
        TemporalThought("test", np.array([1,0,0]), half_life=0)
    
    with pytest.raises(ValueError):
        TemporalThought("test", np.array([1,0,0]), half_life=-1)

    # 2. Фиксируем время тестирования
    base_time = 1_000_000.0
    
    # 3. Создаём объект в "замороженном" времени
    with patch('time.time', return_value=base_time):
        thought = TemporalThought("test", np.array([1,0,0]), half_life=60)
        assert thought.weight() == 1.0  # Исходный вес

    # 4. Проверяем в разные моменты времени
    test_cases = [
        (0, 1.0),       # Сразу
        (30, 0.7071),   # Половина half-life (√0.5)
        (60, 0.5),      # Ровно half-life
        (90, 0.3535),   # Полтора half-life
        (120, 0.25),    # Два half-life
        (-10, 1.0),     # Отрицательное время
        (1e6, 0.0)      # Очень большое время
    ]
    
    for offset, expected in test_cases:
        with patch('time.time', return_value=base_time + offset):
            assert thought.weight() == pytest.approx(expected, abs=1e-4)

def test_cosine_edge_cases():
    """Тест edge-случаев для _cosine"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    
    # Нулевые векторы
    assert orch._cosine(np.array([0,0,0]), np.array([0,0,0])) == 0.0
    
    # NaN в векторах
    assert orch._cosine(np.array([np.nan, 0, 0]), np.array([1,0,0])) == 0.0
    
    # Разная длина векторов
    with pytest.raises(ValueError):
        orch._cosine(np.array([1,0]), np.array([1,0,0]))

def test_dissonance_matrix_empty_voices():
    """Тест пустой матрицы диссонансов"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    names, matrix = orch.calculate_dissonance_matrix()
    assert names == []
    assert matrix.shape == (0, 0)

def test_cosine_similarity_errors():
    """Тест обработки ошибок в _cosine"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    
    # NaN в векторах
    assert orch._cosine(np.array([np.nan, 0, 0]), np.array([1,0,0])) == 0.0
    
    # Нулевые векторы
    assert orch._cosine(np.array([0,0,0]), np.array([0,0,0])) == 0.0

def test_full_coverage():
    """Комплексный тест для полного покрытия"""
    # Мокаем weight() для избежания нулевых весов
    with patch('core.orchestra.TemporalThought.weight', return_value=1.0):
        # Тест инициализации
        embed_fn = MagicMock(return_value=np.array([1,0,0]))
        orch = SimpleOrchestra(embed_fn)
        
        # Тест добавления мысли
        with patch.object(orch, '_calculate_dynamic_threshold', return_value=0.7):
            voice = orch.add_thought("test")
            assert voice in ['melody', 'counterpoint', 'bass']
        
        # Тест coherence
        assert orch.get_coherence() == pytest.approx(1/9, abs=0.01)
        
        # Тест матрицы диссонансов
        names, matrix = orch.calculate_dissonance_matrix()
        assert len(names) == 1
        assert matrix.shape == (1, 1)

def test_orchestra_initialization():
    embed_fn = MagicMock(return_value=np.array([1,0,0]))
    orch = SimpleOrchestra(embed_fn)
    
    assert orch.key is None
    assert set(orch.voices.keys()) == {'melody', 'counterpoint', 'bass'}
    assert orch.voices['melody'].maxlen == 4
    assert orch.voices['counterpoint'].maxlen == 3
    assert orch.voices['bass'].maxlen == 2

def test_add_thought_input_validation():
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    
    with pytest.raises(ValueError, match="Text must be a string"):
        orch.add_thought(None)
    
    with pytest.raises(ValueError, match="Text cannot be empty"):
        orch.add_thought("")
    
    with pytest.raises(ValueError, match="Text must be a string"):
        orch.add_thought(123)

def test_embedding_error_handling():
    # 1. Ошибка в функции эмбеддинга
    orch = SimpleOrchestra(lambda x: 1/0)
    with pytest.raises(RuntimeError, match="Failed to process text"):
        orch.add_thought("test")
    
    # 2. Пустой эмбеддинг
    orch = SimpleOrchestra(lambda x: np.array([]))
    with pytest.raises(RuntimeError, match="empty result"):
        orch.add_thought("test")
    
    # 3. None вместо эмбеддинга
    orch = SimpleOrchestra(lambda x: None)
    with pytest.raises(RuntimeError, match="empty result"):
        orch.add_thought("test")

def test_empty_dissonance_matrix():
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    names, matrix = orch.calculate_dissonance_matrix()
    assert names == []
    assert matrix.shape == (0, 0)

def test_log_error_output(capsys):
    exc = Exception("test error")
    log_error("test message", exc)
    
    captured = capsys.readouterr()
    assert "ERROR: test message" in captured.err
    assert "Exception: test error" in captured.err
    assert "LOG_ERROR CALLED" in captured.out

def test_orchestra_empty_dissonance():
    """Тест пустой матрицы диссонансов"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    names, matrix = orch.calculate_dissonance_matrix()
    assert names == []
    assert matrix.shape == (0, 0)

def test_empty_orchestra_behavior():
    """Тест поведения пустого оркестра"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    assert orch.get_coherence() == 0.0
    names, matrix = orch.calculate_dissonance_matrix()
    assert names == []
    assert matrix.shape == (0, 0)

def test_voice_overflow():
    """Тест переполнения голосов"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    with patch('core.orchestra.TemporalThought.weight', return_value=1.0):
        for i in range(5):  # melody maxlen=4
            orch.add_thought(f"test{i}")
        assert len(orch.voices['melody']) == 4

def test_base_orchestra_instantiation_error():
    from core.orchestra import BaseOrchestra

    class IncompleteOrchestra(BaseOrchestra):
        pass  # не реализует abstract методы

    with pytest.raises(TypeError):
        IncompleteOrchestra()




