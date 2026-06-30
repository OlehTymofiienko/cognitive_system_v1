# tests\tests_orchestra\test_orhestra_ful.py

import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch
from core.orchestra import BaseOrchestra, TemporalThought, SimpleOrchestra

# Фикстура для тестирования SimpleOrchestra
@pytest.fixture
def orchestra():
    embed_fn = MagicMock(return_value=np.array([1.0, 0.5, 0.25]))
    return SimpleOrchestra(embed_fn)

# Тесты для BaseOrchestra (абстрактного класса)
def test_base_orchestra_abstract_methods_01():
    """Тестируем, что абстрактные методы действительно требуют реализации"""
    with pytest.raises(TypeError):
        BaseOrchestra()  # строка 45 - попытка создать экземпляр абстрактного класса

    class TestOrchestra(BaseOrchestra):
        def add_thought(self, text):
            return "test"
        
        def get_coherence(self):
            return 0.5
            
        def calculate_dissonance_matrix(self):
            return [], np.array([])
    
    # Проверяем, что реализованный класс работает
    orchestra = TestOrchestra()
    assert orchestra.add_thought("test") == "test"

# Тесты для TemporalThought
def test_temporal_thought_invalid_half_life():
    """Тестируем обработку недопустимого half_life"""
    with pytest.raises(ValueError, match="must be positive"): 
        TemporalThought("test", np.array([1]), half_life=0)

# Тесты для обработки ошибок в SimpleOrchestra.add_thought()
def test_add_thought_invalid_input(orchestra):
    """Тестируем обработку недопустимых входных данных"""
    with pytest.raises(ValueError, match="must be a string"):
        orchestra.add_thought(None)
    
    with pytest.raises(ValueError, match="cannot be empty"):
        orchestra.add_thought("")

def test_add_thought_embedding_failure(orchestra):
    """Тестируем обработку ошибок при получении эмбеддинга"""
    orchestra.embed_fn.side_effect = RuntimeError("Embedding failed")
    
    with pytest.raises(RuntimeError, match="Failed to process text"):  # строка 66-102
        orchestra.add_thought("test")

def test_add_thought_empty_embedding(orchestra):
    """Тестируем обработку пустого эмбеддинга"""
    orchestra.embed_fn.return_value = np.array([])
    
    with pytest.raises(RuntimeError, match="empty result"):  # строка 69
        orchestra.add_thought("test")

# Тесты для обработки исключений в calculate_dissonance_matrix
def test_calculate_dissonance_matrix_empty(orchestra):
    """Тестируем случай, когда нет валидных голосов"""
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert len(names) == 0
    assert matrix.shape == (0, 0)  # строка 297-298

def test_calculate_dissonance_matrix_with_invalid_embeddings(orchestra):
    """Тестируем обработку невалидных эмбеддингов"""
    # Добавляем мысли с проблемными эмбеддингами
    orchestra.voices['melody'].append(
        TemporalThought("test", np.array([np.nan, np.inf]))
    )
    
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert len(names) == 0  # строка 334-336 - голос должен быть пропущен

# Тесты для edge cases в calculate_dissonance_matrix
def test_calculate_dissonance_matrix_single_voice(orchestra):
    """Тестируем случай с одним голосом"""
    orchestra.add_thought("test")
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert len(names) == 1
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 0.0  # диссонанс с самим собой должен быть 0

def test_calculate_dissonance_matrix_error_handling(orchestra, monkeypatch, caplog):
    """Тестируем обработку ошибок при вычислении матрицы"""
    # Добавляем тестовые данные напрямую, минуя add_thought
    melody_thought = TemporalThought("melody", np.array([1.0, 0.0, 0.0]))
    counter_thought = TemporalThought("counter", np.array([0.5, 0.5, 0.0]))
    
    orchestra.voices['melody'].append(melody_thought)
    orchestra.voices['counterpoint'].append(counter_thought)
    
    # Проверяем начальное состояние
    assert len(orchestra.voices['melody']) == 1
    assert len(orchestra.voices['counterpoint']) == 1
    
    # Подменяем cosine_similarity чтобы вызвать ошибку
    def mock_cosine_similarity(*args, **kwargs):
        raise ValueError("Simulation error")
    
    monkeypatch.setattr("core.orchestra.cosine_similarity", mock_cosine_similarity)
    
    # В реальном коде ошибка обрабатывается внутри метода
    names, matrix = orchestra.calculate_dissonance_matrix()
    
    # Проверяем, что метод вернул матрицу с максимальными диссонансами
    assert len(names) == 2
    assert np.all(matrix == 1.0)  # При ошибке должна быть единичная матрица
    assert "Matrix calculation error" in caplog.text  # Проверяем логирование

def test_calculate_dissonance_matrix_with_zero_weights(orchestra):
    """Тестируем случай, когда веса мыслей становятся нулевыми"""
    # Добавляем старую мысль, вес которой будет почти нулевым
    old_thought = TemporalThought("old", np.array([1.0, 0.0, 0.0]), half_life=0.0001)
    orchestra.voices['melody'].append(old_thought)
    
    # Даем время для "старения" мысли
    import time
    time.sleep(0.1)
    
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert len(names) == 0  # голос должен быть пропущен из-за нулевых весов

def test_calculate_dissonance_matrix_nan_handling(orchestra):
    """Тестируем обработку NaN в эмбеддингах"""
    # Добавляем мысль с NaN в эмбеддинге
    orchestra.voices['melody'].append(
        TemporalThought("nan_thought", np.array([np.nan, 0, 0]))
    )
    
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert len(names) == 0  # Голос с NaN должен быть пропущен

def test_add_thought_exception_logging(orchestra, caplog, capsys):
    """Тестируем логирование исключений при добавлении мысли через embed_fn"""
    import logging
    logger = logging.getLogger("core.orchestra")
    logger.setLevel(logging.DEBUG)

    # Убедимся, что у логгера есть handler, чтобы caplog смог отловить
    if not logger.hasHandlers():
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        logger.addHandler(stream_handler)

    # Подготовка: функция эмбеддинга выбрасывает исключение
    orchestra.embed_fn.side_effect = RuntimeError("Embedding failed")

    with caplog.at_level(logging.ERROR, logger="core.orchestra"):
        with pytest.raises(RuntimeError, match="Failed to process text"):
            orchestra.add_thought("test")

    # Проверяем, что лог об ошибке присутствует
    assert any("Failed to process text" in record.message and record.levelno == logging.ERROR
               for record in caplog.records)

    # Проверяем вывод в stderr (print внутри log_error)
    captured = capsys.readouterr()
    assert "LOG_ERROR CALLED" in captured.out
    assert "ERROR: Failed to process text" in captured.err
    assert "Exception: Embedding failed" in captured.err

def test_temporal_thought_zero_age():
    """Тестируем случай, когда возраст мысли равен нулю"""
    thought = TemporalThought("test", np.array([1.0]))
    assert thought.weight() == 1.0  # вес должен быть 1.0 сразу после создания

def test_temporal_thought_edge_cases():
    """Тестируем крайние случаи TemporalThought"""
    # Тест для строки 51 - проверка half_life
    with pytest.raises(ValueError, match="must be positive"):
        TemporalThought("test", np.array([1]), half_life=0)
    
    # Тест для строки 60 - нулевой возраст
    thought = TemporalThought("test", np.array([1]))
    assert thought.weight() == 1.0  # Вес должен быть 1.0 сразу после создания

def test_add_thought_error_handling(orchestra, mocker):
    """Тестируем полную обработку ошибок в add_thought() (строки 72-108)"""
    # 1. Тест ошибки получения эмбеддинга
    orchestra.embed_fn.side_effect = RuntimeError("Embedding failed")
    with pytest.raises(RuntimeError, match="Failed to process text"):
        orchestra.add_thought("test")
    
    # 2. Тест пустого эмбеддинга
    orchestra.embed_fn.side_effect = None
    orchestra.embed_fn.return_value = np.array([])
    with pytest.raises(RuntimeError, match="empty result"):
        orchestra.add_thought("test")
    
    # 3. Тест ошибки в распределении по голосам
    orchestra.embed_fn.return_value = np.array([1, 0, 0])
    mocker.patch.object(orchestra, '_calculate_dynamic_threshold', side_effect=ValueError("Threshold error"))
    with pytest.raises(RuntimeError, match="Failed to process text"):
        orchestra.add_thought("test")

def test_calculate_dissonance_matrix_edge_cases(orchestra):
    """Тестируем крайние случаи calculate_dissonance_matrix()"""
    # 1. Тест с одним голосом и одной мыслью
    thought = TemporalThought("test", np.array([1, 0, 0]))
    orchestra.voices['melody'].append(thought)
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert len(names) == 1
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 0  # Диссонанс с самим собой должен быть 0

    # 2. Тест с нулевыми весами - нужно очистить предыдущие данные
    orchestra.voices['melody'].clear()
    old_thought = TemporalThought("old", np.array([1, 0, 0]), half_life=0.0001)
    orchestra.voices['melody'].append(old_thought)
    time.sleep(0.1)
    
    # Мокируем weight() чтобы гарантировать нулевой вес
    with patch.object(old_thought, 'weight', return_value=0.0):
        names, matrix = orchestra.calculate_dissonance_matrix()
        assert len(names) == 0  # Голос должен быть пропущен

def test_base_orchestra_abstract_methods_02():
    """Тестируем абстрактные методы BaseOrchestra (строка 35)"""
    class TestOrchestra(BaseOrchestra):
        def add_thought(self, text):
            return "test"
        
        def calculate_dissonance_matrix(self):
            return [], np.array([])
    
    # Проверяем, что get_coherence() требует реализации
    with pytest.raises(TypeError):
        TestOrchestra().get_coherence()

def test_base_orchestra_abstract_methods():
    """Тестируем абстрактные методы BaseOrchestra"""
    with pytest.raises(TypeError):
        BaseOrchestra()  # Проверяем, что класс действительно абстрактный
        
    class TestOrchestra(BaseOrchestra):
        def add_thought(self, text): 
            return "test"
        def get_coherence(self): 
            return 0.5
        def calculate_dissonance_matrix(self): 
            return [], np.zeros((0, 0))  # Используем zeros вместо array([])
    
    # Проверяем реализованные методы
    orchestra = TestOrchestra()
    assert orchestra.add_thought("test") == "test"
    assert orchestra.get_coherence() == 0.5
    
    # Для сравнения массивов используем np.array_equal
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert names == []
    assert matrix.shape == (0, 0)

def test_temporal_thought_validation():
    """Тестируем валидацию в TemporalThought"""
    # Проверка half_life > 0
    with pytest.raises(ValueError, match="must be positive"):
        TemporalThought("test", np.array([1]), half_life=0)
    
    # Проверка веса для только что созданной мысли
    thought = TemporalThought("test", np.array([1]))
    assert thought.weight() == 1.0

def test_orchestra_empty_cases(orchestra):
    """Тестируем краевые случаи с пустым оркестром"""
    # Проверка calculate_dissonance_matrix с пустым оркестром
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert len(names) == 0
    assert matrix.shape == (0, 0)
    
    # Проверка get_coherence с пустым оркестром
    assert orchestra.get_coherence() == 0.0

def test_log_error_with_exception(capsys):
    """Тестируем log_error с передачей исключения"""
    from core.orchestra import log_error
    try:
        1/0
    except Exception as e:
        log_error("Test error", e)
    
    captured = capsys.readouterr()
    assert "LOG_ERROR CALLED" in captured.out
    assert "ERROR: Test error" in captured.err
    assert "Exception: division by zero" in captured.err

def test_log_error_without_exception(capsys):
    """Тестируем log_error без исключения"""
    from core.orchestra import log_error
    log_error("Test error")
    
    captured = capsys.readouterr()
    assert "LOG_ERROR CALLED" in captured.out
    assert "ERROR: Test error" in captured.err
    assert "Exception:" not in captured.err

def test_add_thought_error_paths(orchestra, mocker, caplog):
    """Тестируем все возможные ошибки в add_thought()"""
    # 1. Ошибка в embed_fn
    orchestra.embed_fn.side_effect = ValueError("Embedding error")
    with pytest.raises(RuntimeError, match="Failed to process text"):
        orchestra.add_thought("test")
    assert "Embedding error" in caplog.text

    # 2. Пустой эмбеддинг
    orchestra.embed_fn.side_effect = None
    orchestra.embed_fn.return_value = np.array([])
    with pytest.raises(RuntimeError, match="empty result"):
        orchestra.add_thought("test")
    
    # 3. None эмбеддинг
    orchestra.embed_fn.return_value = None
    with pytest.raises(RuntimeError, match="empty result"):
        orchestra.add_thought("test")

    # 4. Ошибка при расчете порога
    orchestra.embed_fn.return_value = np.array([1, 0, 0])
    mocker.patch.object(orchestra, '_calculate_dynamic_threshold', 
                       side_effect=RuntimeError("Threshold error"))
    with pytest.raises(RuntimeError, match="Failed to process text"):
        orchestra.add_thought("test")

def test_nan_handling_in_dissonance_matrix(orchestra):
    """Тестируем обработку NaN в calculate_dissonance_matrix()"""
    # 1. NaN в эмбеддинге
    orchestra.voices['melody'].append(
        TemporalThought("nan_thought", np.array([np.nan, 1.0, 0.0]))
    )
    
    # 2. Inf в эмбеддинге
    orchestra.voices['counterpoint'].append(
        TemporalThought("inf_thought", np.array([1.0, np.inf, 0.0]))
    )
    
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert len(names) == 0  # Оба голоса должны быть пропущены

def test_empty_dissonance_matrix(orchestra):
    """Тестируем возврат пустой матрицы"""
    # 1. Все голоса пустые
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert names == []
    assert matrix.shape == (0, 0)
    
    # 2. Голоса есть, но все эмбеддинги невалидные
    orchestra.voices['melody'].append(
        TemporalThought("bad_thought", np.array([np.nan, np.nan])))
    names, matrix = orchestra.calculate_dissonance_matrix()
    assert names == []
    assert matrix.shape == (0, 0)

def test_temporal_thought_edge_cases():
    """Тестируем крайние случаи TemporalThought"""
    # Строка 51 - проверка half_life
    with pytest.raises(ValueError, match="must be positive"):
        TemporalThought("test", np.array([1]), half_life=0)
    
    # Строка 60 - нулевой возраст
    thought = TemporalThought("test", np.array([1]))
    assert thought.weight() == 1.0  # Вес должен быть 1.0 сразу после создания

def test_base_orchestra_abstract_methods():
    """Тестируем абстрактные методы (строка 35)"""
    with pytest.raises(TypeError):
        BaseOrchestra()  # Нельзя создать экземпляр абстрактного класса
        
    class TestOrchestra(BaseOrchestra):
        def add_thought(self, text): return "test"
        def get_coherence(self): return 0.5
        def calculate_dissonance_matrix(self): return [], np.array([])
    
    # Проверяем реализацию
    orchestra = TestOrchestra()
    assert orchestra.add_thought("test") == "test"

def test_add_thought_log_error_with_none_embedding(orchestra, caplog, capsys):
    orchestra.embed_fn.return_value = None

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="Failed to process text"):
        orchestra.add_thought("test")

    captured = capsys.readouterr()
    assert "LOG_ERROR CALLED" in captured.out
    assert "Failed to get text embedding: empty result" in captured.err

def test_add_thought_embed_fn_raises_exception_logged(orchestra, caplog, capsys):
    orchestra.embed_fn.side_effect = ValueError("Synthetic failure")

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="Failed to process text"):
        orchestra.add_thought("test")

    captured = capsys.readouterr()
    assert "LOG_ERROR CALLED" in captured.out
    assert "ERROR: Failed to process text" in captured.err
    assert "Exception: Synthetic failure" in captured.err

def test_add_thought_nan_embedding_raises_error(orchestra):
    """Тестируем, что NaN-эмбеддинг вызывает исключение"""
    orchestra.embed_fn.return_value = np.array([np.nan, 0.5, 1.0])
    
    with pytest.raises(RuntimeError, match="Embedding contains NaN or Inf values"):
        orchestra.add_thought("test")

def test_calculate_dissonance_matrix_avg_emb_error_logged(orchestra, caplog):
    orchestra.voices['melody'].append(
        TemporalThought("voice1", np.array([np.nan, np.nan, np.nan]))
    )

    with caplog.at_level("ERROR"):
        names, matrix = orchestra.calculate_dissonance_matrix()

    assert names == []
    assert matrix.shape == (0, 0)
    assert "Non-finite embedding" in caplog.text



