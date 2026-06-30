import numpy as np
import pytest
from collections import deque
import logging
from unittest.mock import patch
from core.orchestra import BaseOrchestra, TemporalThought, SimpleOrchestra


def test_cosine_nan_handling(orchestra_instance):
    """Проверяем обработку NaN в _cosine()."""
    # 1. Создаем тестовые векторы
    nan_vector = np.array([np.nan, 0, 0])
    valid_vector = np.array([1, 0, 0])
    
    # 2. Вызываем метод
    result = orchestra_instance._cosine(nan_vector, valid_vector)
    
    # 3. Проверяем результат
    assert result == 0.0
    assert not np.isnan(result)

def test_dynamic_threshold_calculation(populated_orchestra):
    """Проверяем расчет динамического порога."""
    # 1. Мокаем _cosine для предсказуемых результатов
    with patch.object(populated_orchestra, '_cosine', return_value=0.5):
        threshold = populated_orchestra._calculate_dynamic_threshold()
        assert threshold == 0.5

@pytest.mark.parametrize("sim_values", [
    [0.9, 0.5, 0.3],
    [0.8, 0.7, 0.6],
])
def test_dynamic_threshold_variations_explicit(sim_values):
    """
    Вместо «подмешивания» в существующие голоса, мы явно создаём ровно по одной TemporalThought 
    в каждом из трёх голосов, и патчим _cosine так, чтобы возвращались наши sim_values.
    Тогда sims = sim_values, и 33-й перцентиль даёт ожидаемый результат.
    """
    # 1) Создаём оркестр и задаём key, чтобы _calculate_dynamic_threshold не вернул дефолт 0.7
    orch = SimpleOrchestra(lambda _: np.array([1.0, 0.0, 0.0]))
    orch.key = np.array([1.0, 0.0, 0.0])

    # 2) Заполняем ровно по одной мысли в каждом голосе
    #    Сам emb не важен, потому что _cosine мы патчим
    t1 = TemporalThought("a", np.array([1,0,0]))
    t2 = TemporalThought("b", np.array([1,0,0]))
    t3 = TemporalThought("c", np.array([1,0,0]))
    orch.voices['melody'] = deque([t1])
    orch.voices['counterpoint'] = deque([t2])
    orch.voices['bass'] = deque([t3])

    # 3) Патчим _cosine: он будет вызван ровно три раза, по одному на каждую мысль
    with patch.object(orch, '_cosine', side_effect=sim_values):
        result = orch._calculate_dynamic_threshold()

    # 4) Проверяем, что результат совпадает с 33-м перцентилем sim_values
    expected = float(np.percentile(sim_values, 33))
    assert pytest.approx(expected, rel=1e-2) == result

def test_dynamic_threshold_edge_cases(orchestra_instance):
    """Проверяем edge-кейсы расчёта порога."""
    # Пустые similarity
    with patch.object(orchestra_instance, '_cosine', return_value=0.5), \
         patch.object(orchestra_instance, 'voices', {'melody': []}):
        assert orchestra_instance._calculate_dynamic_threshold() == 0.7
    
    # Одно значение
    with patch.object(orchestra_instance, '_cosine', return_value=0.3):
        orchestra_instance.voices = {'melody': [TemporalThought("test", np.array([1]))]}
        assert orchestra_instance._calculate_dynamic_threshold() == 0.3

def test_empty_voices_threshold(orchestra_instance):
    """Проверяем дефолтный порог при пустых голосах."""
    assert orchestra_instance._calculate_dynamic_threshold() == 0.7

def test_add_thought_exception_handling(orchestra_instance, capsys):
    """
    Проверяем, что при ошибке внутри add_thought()
    срабатывает log_error и RuntimeError.
    """
    # Мокаем embed_fn, чтобы оно упало
    orchestra_instance.embed_fn = lambda text: (_ for _ in ()).throw(RuntimeError("Boom"))
    
    with pytest.raises(RuntimeError, match="Failed to process text"):
        orchestra_instance.add_thought("trigger")

    captured = capsys.readouterr()
    # Код всегда сначала вызывает log_error → "LOG_ERROR CALLED"
    assert "LOG_ERROR CALLED" in captured.out
    assert "ERROR: Failed to process text" in captured.err

def test_add_thought_edge_cases(orchestra_instance):
    """Проверяем обработку крайних случаев."""
    # Пустая строка
    with pytest.raises(ValueError):
        orchestra_instance.add_thought("")
    
    # Не строка
    with pytest.raises(ValueError):
        orchestra_instance.add_thought(None)

def test_add_thought_empty_embedding(orchestra_instance):
    """Проверяем обработку пустого эмбеддинга."""
    with patch.object(orchestra_instance, 'embed_fn', return_value=np.array([])):
        with pytest.raises(RuntimeError, match="empty result"):
            orchestra_instance.add_thought("test")

@pytest.mark.parametrize(
    "sim, threshold, expected_voice",
    [
        (0.9, 0.8, 'melody'),         # sim >= threshold
        (0.6, 0.8, 'counterpoint'),   # threshold*0.6 <= sim < threshold
        (0.2, 0.8, 'bass'),           # sim < threshold*0.6
        (0.8, 0.8, 'melody'),         # пограничный случай: sim == threshold
        (0.48, 0.8, 'counterpoint'),  # пограничный нижний: sim == threshold*0.6
        (0.47, 0.8, 'bass'),          # чуть ниже порога counterpoint
    ]
)
def test_add_thought_voice_parametrized(orchestra_instance, sim, threshold, expected_voice, caplog):
    """Проверяем распределение по голосам при разных sim и threshold."""
    
    logger = logging.getLogger("core.orchestra")
    logger.setLevel(logging.DEBUG)
    
    with patch.object(orchestra_instance, '_calculate_dynamic_threshold', return_value=threshold):
        with patch.object(orchestra_instance, '_cosine', return_value=sim):
            with caplog.at_level(logging.DEBUG):
                voice = orchestra_instance.add_thought(f"thought_{sim}")
                
                assert voice == expected_voice
                assert f"voice={expected_voice}" in caplog.text
                assert f"similarity: {sim:.2f}" in caplog.text

def test_first_thought_logging(orchestra_instance, caplog):
    """Проверяем логирование первой мысли."""
    # Настраиваем логгер
    logger = logging.getLogger("core.orchestra")
    logger.setLevel(logging.INFO)
    
    with caplog.at_level(logging.INFO):
        orchestra_instance.add_thought("First thought")
        assert any("Initialized orchestra key" in record.message for record in caplog.records)

def test_cosine_zero_vectors(orchestra_instance):
    """Проверяем обработку нулевых векторов."""
    zero_vector = np.array([0, 0, 0])
    assert orchestra_instance._cosine(zero_vector, zero_vector) == 0.0

def test_temporal_thought_invalid_half_life():
    """Проверяем обработку невалидного half_life."""
    with pytest.raises(ValueError):
        TemporalThought("test", np.array([1, 0]), half_life=0)

def test_temporal_thought_weight():
    """Проверяем расчет веса мысли с учетом времени."""
    thought = TemporalThought("test", np.array([1, 0]), half_life=10)
    
    with patch('time.time', return_value=thought.birth_time + 10):  # Прошло ровно half_life
        assert pytest.approx(thought.weight(), 0.01) == 0.5
    
    with patch('time.time', return_value=thought.birth_time + 20):  # Прошло 2*half_life
        assert pytest.approx(thought.weight(), 0.01) == 0.25

def test_dissonance_matrix(populated_orchestra):
    """Проверяем расчет матрицы диссонансов."""
    names, matrix = populated_orchestra.calculate_dissonance_matrix()
    assert isinstance(names, list)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(names), len(names))

def test_dissonance_matrix_zero_weights(populated_orchestra):
    """Проверяем обработку нулевых весов при расчете матрицы."""
    # Делаем все веса нулевыми
    with patch.object(TemporalThought, 'weight', return_value=0.0):
        names, matrix = populated_orchestra.calculate_dissonance_matrix()
        assert isinstance(matrix, np.ndarray)
        assert not np.isnan(matrix).any()

def test_base_orchestra_abstract_methods():
    """Проверяем, что абстрактные методы вызывают ошибки."""
    with pytest.raises(TypeError) as excinfo:
        BaseOrchestra()
    assert "Can't instantiate abstract class" in str(excinfo.value)
    
    # Альтернативно, можно создать подкласс для тестирования
    class TestOrchestra(BaseOrchestra):
        def add_thought(self, text): pass
        def get_coherence(self): pass
        def calculate_dissonance_matrix(self): pass
    
    # Проверяем, что теперь можно создать экземпляр
    assert isinstance(TestOrchestra(), BaseOrchestra)

def test_log_error_output(capsys):
    """Проверяем вывод функции log_error."""
    from core.orchestra import log_error
    try:
        1/0
    except Exception as e:
        log_error("Test error", e)
    
    captured = capsys.readouterr()
    assert "LOG_ERROR CALLED" in captured.out
    assert "Test error" in captured.err
    assert "division by zero" in captured.err  # Проверяем текст ошибки

def test_cosine_dimension_mismatch(orchestra_instance):
    """Проверяем обработку несовпадающих размерностей."""
    a = np.array([1, 0])
    b = np.array([1, 0, 0])
    with pytest.raises(ValueError, match="not aligned"):
        orchestra_instance._cosine(a, b)



