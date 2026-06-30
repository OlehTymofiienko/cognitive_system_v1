# tests\tests_orchestra\test_orchestra_full_02.py

import numpy as np
import pytest
from unittest.mock import MagicMock
from core.orchestra import BaseOrchestra, TemporalThought, SimpleOrchestra
import logging
import time

# Настройка логгера для тестов
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("core.orchestra")

@pytest.fixture
def orchestra():
    """Фикстура для создания тестового оркестра"""
    embed_fn = MagicMock(return_value=np.array([1.0, 0.5, 0.25]))
    return SimpleOrchestra(embed_fn)

# Тесты для TemporalThought
class TestTemporalThought:
    def test_validation_half_life(self):
        """Тестируем валидацию half_life (строка 51)"""
        with pytest.raises(ValueError, match="must be positive"):
            TemporalThought("test", np.array([1]), half_life=0)

    def test_weight_zero_age(self):
        """Тестируем вес при нулевом возрасте (строка 60)"""
        thought = TemporalThought("test", np.array([1]))
        assert thought.weight() == 1.0  # Вес сразу после создания

    def test_weight_after_time(self):
        """Тестируем изменение веса со временем"""
        thought = TemporalThought("test", np.array([1]), half_life=1.0)
        time.sleep(1.5)
        assert 0 < thought.weight() < 1.0  # Вес должен уменьшиться

# Тесты для SimpleOrchestra.add_thought()
class TestAddThought:
    @pytest.mark.parametrize("input_data,expected_error", [
    (None, "Text must be a string"),  # NoneType
    (np.array([]), "Text must be a string"),  # ndarray
    (42, "Text must be a string"),  # int
    (object(), "Text must be a string"),  # объект
    ("", "cannot be empty"),  # пустая строка
])
    def test_error_handling(self, orchestra, input_data, expected_error):
        """Тестируем обработку ошибок валидации входа"""
        with pytest.raises(ValueError, match=expected_error):
            orchestra.add_thought(input_data)

    def test_embedding_exception(self, orchestra):
        """Тестируем исключения в embed_fn"""
        orchestra.embed_fn.side_effect = Exception("Embedding failed")
        with pytest.raises(RuntimeError, match="Failed to process text"):
            orchestra.add_thought("test")

    def test_successful_add(self, orchestra):
        """Тестируем успешное добавление мысли"""
        orchestra.embed_fn.return_value = np.array([1.0, 0.0, 0.0])
        voice = orchestra.add_thought("valid thought")
        assert voice in ['melody', 'counterpoint', 'bass']
        assert len(orchestra.voices[voice]) == 1

# Тесты для SimpleOrchestra.calculate_dissonance_matrix()
class TestDissonanceMatrix:
    def test_empty_matrix(self, orchestra):
        """Тестируем пустую матрицу (строки 308-309)"""
        names, matrix = orchestra.calculate_dissonance_matrix()
        assert names == []
        assert matrix.shape == (0, 0)

    def test_nan_handling(self, orchestra):
        """Тестируем обработку NaN в эмбеддингах"""
        orchestra.voices['melody'].append(
            TemporalThought("test", np.array([np.nan, 1.0])))
        names, matrix = orchestra.calculate_dissonance_matrix()
        assert len(names) == 0

    def test_single_voice(self, orchestra):
        """Тестируем матрицу с одним голосом"""
        orchestra.voices['melody'].append(
            TemporalThought("test", np.array([1.0, 0.5])))
        names, matrix = orchestra.calculate_dissonance_matrix()
        assert len(names) == 1
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 0  # Диссонанс с самим собой

    def test_multiple_voices(self, orchestra):
        """Тестируем матрицу с несколькими голосами"""
        orchestra.voices['melody'].append(
            TemporalThought("melody", np.array([1.0, 0.0])))
        orchestra.voices['bass'].append(
            TemporalThought("bass", np.array([0.5, 0.5])))

        names, matrix = orchestra.calculate_dissonance_matrix()
        assert len(names) == 2
        assert matrix.shape == (2, 2)
        assert np.all(np.diag(matrix) == 0)  # Диссонанс с самим собой должен быть 0
        assert 0 < matrix[0, 1] <= 1.0  # Допускаем граничное значение


# Интеграционные тесты
class TestIntegration:
    def test_full_workflow(self, orchestra):
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

    def test_coherence_calculation(self, orchestra):
        """Тестируем расчет когерентности"""
        assert orchestra.get_coherence() == 0.0  # Пустой оркестр
        
        orchestra.embed_fn.return_value = np.array([1.0, 0.0])
        orchestra.add_thought("test")
        assert 0 < orchestra.get_coherence() < 1.0