# tests/test_orchestra_additional.py
import pytest
import numpy as np
from unittest.mock import MagicMock
from core.orchestra import SimpleOrchestra, TemporalThought

from core.context_manager import ContextManager

def test_trigger_context_manager():
    cm = ContextManager()
    dummy_contexts = [{"coherence": 0.5}, {"coherence": 0.7}]
    result = cm.get_coherence(dummy_contexts)
    assert result == pytest.approx(0.6, 0.01)


class TestOrchestraAdditional:
    @pytest.fixture
    def mock_embed_fn(self):
        """Фикстура для мока функции эмбеддинга"""
        def _embed(text: str) -> np.ndarray:
            # Возвращаем фиксированный вектор для тестов
            return np.array([1.0, 0.0, 0.0]) if text == "text1" else np.array([0.0, 1.0, 0.0])
        return MagicMock(side_effect=_embed)

    def test_matrix_generation_edge_cases(self, mock_embed_fn):
        """Тест генерации матрицы для пустых входов"""
        orchestra = SimpleOrchestra(embed_fn=mock_embed_fn)
        # Тестируем с пустыми голосами
        names, matrix = orchestra.calculate_dissonance_matrix()
        assert names == []
        assert matrix.shape == (0, 0)

    def test_voice_allocation_special_cases(self, mock_embed_fn):
        """Тест распределения голосов для особых случаев"""
        orchestra = SimpleOrchestra(embed_fn=mock_embed_fn)
        
        # 1. Тест с None (должен вызывать ValueError)
        with pytest.raises(ValueError):
            orchestra.add_thought(None)
            
        # 2. Тест с пустой строкой (должен вызывать ValueError)
        with pytest.raises(ValueError):
            orchestra.add_thought("")
        
        # 3. Тест первого добавления (должен установить key)
        voice = orchestra.add_thought("text1")
        assert voice in ['melody', 'counterpoint', 'bass']
        assert orchestra.key is not None
        
        # 4. Тест с нестроковым вводом
        with pytest.raises(ValueError):
            orchestra.add_thought(123)

    def test_dissonance_matrix_calculation(self, mock_embed_fn):
        """Тест правильности расчета матрицы диссонансов"""
        orchestra = SimpleOrchestra(embed_fn=mock_embed_fn)
        
        # Добавляем тестовые мысли
        orchestra.add_thought("text1")  # Вектор [1, 0, 0]
        orchestra.add_thought("text2")  # Вектор [0, 1, 0]
        
        names, matrix = orchestra.calculate_dissonance_matrix()
        
        # Проверяем структуру результата
        assert len(names) == 2
        assert matrix.shape == (2, 2)
        
        # Проверяем расчет диссонанса: 1 - cosine_similarity([1,0,0], [0,1,0]) = 1 - 0 = 1
        assert matrix[0, 1] == pytest.approx(1.0, 0.01)
        assert matrix[1, 0] == pytest.approx(1.0, 0.01)  # Матрица должна быть симметричной

    def test_coherence_calculation(self, mock_embed_fn):
        """Тест расчета когерентности"""
        orchestra = SimpleOrchestra(embed_fn=mock_embed_fn)
        
        # Изначально когерентность 0
        assert orchestra.get_coherence() == 0.0
        
        # После добавления мысли
        orchestra.add_thought("text1")
        assert orchestra.get_coherence() == pytest.approx(1/9, 0.01)  # 1 слот из 9 возможных

    def test_thought_lifecycle(self, mock_embed_fn):
        """Полный тест жизненного цикла мысли в оркестре"""
        orchestra = SimpleOrchestra(embed_fn=mock_embed_fn)

        # Добавляем первую мысль
        voice = orchestra.add_thought("Важная мысль")
        
        # Проверка корректного голосового распределения
        assert voice in orchestra.voices
        assert len(orchestra.voices[voice]) == 1

        # Проверка самой мысли
        thought = orchestra.voices[voice][0]
        assert isinstance(thought, TemporalThought)
        assert thought.text == "Важная мысль"

        # Проверка корректного веса (возраст < 1 секунда, вес ≈ 1)
        weight = thought.weight()
        assert 0.95 <= weight <= 1.0

        # Инициализация ключа
        assert orchestra.key is not None
        assert isinstance(orchestra.key, np.ndarray)

        # Повторное добавление — убедимся, что мысль добавилась в нужный голос
        second_voice = orchestra.add_thought("text2")
        assert second_voice in orchestra.voices
        assert len(orchestra.voices[second_voice]) == 1 or len(orchestra.voices[second_voice]) == 2

        # Проверка когерентности после двух мыслей
        coherence = orchestra.get_coherence()
        assert coherence == pytest.approx(2 / 9, 0.01)  # 2 занятого слота из 9

        # Проверка матрицы диссонансов
        names, matrix = orchestra.calculate_dissonance_matrix()
        assert len(names) >= 1
        assert matrix.shape == (len(names), len(names))
