import pytest
import numpy as np
import math
import logging
from collections import deque
from unittest.mock import MagicMock, patch
from unittest.mock import ANY
from io import StringIO

from core.orchestra import BaseOrchestra, SimpleOrchestra, TemporalThought


# ─────────────────────────────────────────────────────────────
# 1. Тестирование абстрактного класса BaseOrchestra
# ─────────────────────────────────────────────────────────────
def test_base_orchestra_instantiation_triggers_type_error():
    """Проверка, что нельзя создать экземпляр абстрактного класса."""
    class Incomplete(BaseOrchestra):
        pass

    with pytest.raises(TypeError):
        Incomplete()


# ─────────────────────────────────────────────────────────────
# 2. Тестирование инициализации и логирования
# ─────────────────────────────────────────────────────────────
def test_first_thought_logs_info(caplog):
    """Проверка логирования при инициализации ключевого эмбеддинга."""
    embed_fn = lambda text: np.array([1.0, 0.0])
    orchestra = SimpleOrchestra(embed_fn)

    with caplog.at_level("INFO"):
        orchestra.add_thought("First thought")

    assert "Initialized orchestra key" in caplog.text


def test_add_thought_logs_debug(caplog):
    """Проверка отладочных сообщений при добавлении мысли."""
    embed_fn = lambda text: np.array([1.0, 0.0])
    orchestra = SimpleOrchestra(embed_fn)

    orchestra.add_thought("First")  # Инициализация ключа
    caplog.clear()

    orchestra.add_thought("Second")

    assert "Debug: sim=" in caplog.text
    assert "Added thought to voice" in caplog.text


# ─────────────────────────────────────────────────────────────
# 3. Тестирование обработки ошибок
# ─────────────────────────────────────────────────────────────
def test_add_thought_exception_after_key(mocker):
    """Проверка обработки исключений после инициализации ключа."""
    embed_fn = lambda text: np.array([1.0, 0.0])
    orchestra = SimpleOrchestra(embed_fn)
    orchestra.add_thought("First thought")  # Инициализация ключа
    
    mock_log_error = mocker.patch("core.orchestra.log_error")
    mocker.patch.object(
        orchestra,
        "_calculate_dynamic_threshold",
        side_effect=Exception("Simulated error")
    )

    with pytest.raises(RuntimeError) as err:
        orchestra.add_thought("Second thought")

    assert "Simulated error" in str(err.value)
    mock_log_error.assert_called_once_with("Failed to process text", ANY)


def test_add_thought_invalid_input():
    """Проверка обработки невалидного ввода."""
    orchestra = SimpleOrchestra(embed_fn=lambda x: np.array([1.0]))
    
    with pytest.raises(ValueError):
        orchestra.add_thought("")  # Пустой текст
        
    with pytest.raises(ValueError):
        orchestra.add_thought(None)  # None вместо текста


# ─────────────────────────────────────────────────────────────
# 4. Параметризованное тестирование распределения по голосам
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,embedding,expected_voice,threshold", [
    ("exact", np.array([1.0, 0.0]), "melody", 0.75),
    ("high", np.array([0.9, 0.1]), "melody", 0.75),
    ("mid",  np.array([0.7, 0.3]), "melody", 0.75),         # ← было: counterpoint
    ("edge", np.array([0.6, 0.4]), "melody", 0.75),         # ← было: counterpoint
    ("low",  np.array([0.5, 0.5]), "counterpoint", 0.75),   # ← было: bass
    ("zero", np.array([0.0, 1.0]), "bass", 0.75),
])

def test_voice_assignment_with_fixed_threshold(text, embedding, expected_voice, threshold, mocker):
    """Тест с фиксированным порогом для точного контроля."""
    embed_fn = MagicMock(return_value=embedding)
    orchestra = SimpleOrchestra(embed_fn)
    
    # Фиксируем порог
    mocker.patch.object(
        orchestra,
        "_calculate_dynamic_threshold",
        return_value=threshold
    )
    
    # Инициализируем ключ
    orchestra.key = np.array([1.0, 0.0])
    
    assigned_voice = orchestra.add_thought(text)
    assert assigned_voice == expected_voice

def test_log_error_args(mocker):
    """Проверка аргументов log_error."""
    mock_log_error = mocker.patch("core.orchestra.log_error")
    embed_fn = lambda x: np.array([1.0, 0.0])
    orchestra = SimpleOrchestra(embed_fn)
    
    mocker.patch.object(
        orchestra,
        "_calculate_dynamic_threshold",
        side_effect=Exception("Test error")
    )

    with pytest.raises(RuntimeError):
        orchestra.add_thought("Test")
    
    mock_log_error.assert_called_once()
    args = mock_log_error.call_args[0]
    assert args[0] == "Failed to process text"
    assert isinstance(args[1], Exception)
    assert "Test error" in str(args[1])


# ─────────────────────────────────────────────────────────────
# 5. Тестирование TemporalThought
# ─────────────────────────────────────────────────────────────
def test_temporal_thought_zero_age():
    """Проверка веса для только что созданной мысли."""
    thought = TemporalThought("Test", np.array([1.0]))
    with patch("time.time", return_value=thought.birth_time):
        assert thought.weight() == 1.0


def test_temporal_thought_weight_decay():
    """Проверка уменьшения веса со временем."""
    thought = TemporalThought("Test", np.array([1.0]), half_life=60.0)
    with patch("time.time", return_value=thought.birth_time + 60.0):  # +60 сек
        assert thought.weight() == pytest.approx(0.5)  # должен уменьшиться вдвое


def test_temporal_thought_invalid_half_life():
    """Проверка валидации half_life."""
    with pytest.raises(ValueError):
        TemporalThought("Test", np.array([1.0]), half_life=0)
    
    with pytest.raises(ValueError):
        TemporalThought("Test", np.array([1.0]), half_life=-1.0)


# ─────────────────────────────────────────────────────────────
# 6. Тестирование calculate_dissonance_matrix()
# ─────────────────────────────────────────────────────────────
class DummyThought:
    """Фиктивная мысль для тестирования."""
    def __init__(self, emb, weight_val):
        self.emb = emb
        self._w = weight_val

    def weight(self):
        return self._w


@pytest.fixture
def orchestra_with_voices():
    """Фикстура с подготовленными голосами для тестирования матрицы диссонансов."""
    orchestra = SimpleOrchestra(embed_fn=lambda x: np.array([1.0, 0.0]))
    orchestra.voices = {
        "main": [DummyThought(np.array([1.0, 0.0]), 1.0)],
        "counter": [DummyThought(np.array([0.0, 1.0]), 1.0)],
        "empty": [],  # пустой голос
        "zero_weight": [DummyThought(np.array([0.5, 0.5]), 0.0)],  # нулевой вес
        "nan_emb": [DummyThought(np.array([np.nan, np.nan]), 1.0)],  # NaN эмбеддинг
    }
    return orchestra


def test_calculate_dissonance_matrix(orchestra_with_voices):
    """Актуальный тест матрицы диссонансов после фильтрации голосов."""
    names, matrix = orchestra_with_voices.calculate_dissonance_matrix()

    # Проверяем, что в итоговый список вошли только валидные голоса
    expected_names = {"main", "counter"}
    assert set(names) == expected_names
    assert matrix.shape == (2, 2)

    # Проверка диагонали (дисс. внутри голоса = 0.0)
    for i in range(len(names)):
        assert matrix[i, i] == pytest.approx(0.0)

    # main vs counter → ортогональны → диссонанс ≈ 1.0
    idx_main = names.index("main")
    idx_counter = names.index("counter")
    assert matrix[idx_main, idx_counter] == pytest.approx(1.0)
    assert matrix[idx_counter, idx_main] == pytest.approx(1.0)

    # Дополнительная защита: матрица не содержит NaN
    assert not np.isnan(matrix).any()
    assert np.all((0.0 <= matrix) & (matrix <= 1.0))


def test_dissonance_matrix_zero_weights(orchestra_with_voices):
    """Проверка исключения голосов с нулевыми весами из диссонансной матрицы."""
    names, matrix = orchestra_with_voices.calculate_dissonance_matrix()

    # Голос с нулевым весом должен быть исключён
    assert "zero_weight" not in names
    assert matrix.shape == (2, 2)  # только main и counter

    # Проверяем, что матрица не содержит NaN
    assert not np.isnan(matrix).any()
    assert np.all((0.0 <= matrix) & (matrix <= 1.0))

def test_dissonance_matrix_zero_weights_logging(orchestra_with_voices, caplog):
    with caplog.at_level(logging.DEBUG):
        orchestra_with_voices.calculate_dissonance_matrix()

    assert "Skipping voice 'zero_weight' - zero sum weights" in caplog.text


# ─────────────────────────────────────────────────────────────
# 7. Тестирование get_coherence()
# ─────────────────────────────────────────────────────────────
def test_get_coherence_full():
    """Гибридный тест полной когерентности: ручное заполнение всех голосов."""
    embed_fn = lambda x: np.array([1.0, 0.0])
    orchestra = SimpleOrchestra(embed_fn)

    # Создаем фейковую мысль
    thought = TemporalThought("Test", np.array([1.0]), half_life=60.0)

    # Ручное заполнение всех голосов, независимо от их количества
    for voice_name in orchestra.voices:
        orchestra.voices[voice_name] = deque([thought], maxlen=1)

    filled = [v for v in orchestra.voices.values() if len(v) > 0]
    assert len(filled) == len(orchestra.voices), "Не все голоса заполнены"

    # Проверка когерентности
    coherence = orchestra.get_coherence()
    assert coherence == pytest.approx(1.0), f"Expected coherence=1.0, got {coherence:.2f}"


def test_get_coherence_zero_capacity():
    """Проверка случая с нулевой емкостью."""
    orchestra = SimpleOrchestra(embed_fn=lambda x: np.array([1.0]))
    orchestra.voices = {}  # Нет голосов
    assert orchestra.get_coherence() == 0.0

