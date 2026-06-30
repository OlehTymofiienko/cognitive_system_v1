#tests\test_orchestra.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from core.orchestra import SimpleOrchestra, TemporalThought, BaseOrchestra
from core.context_manager import ContextManager  # или любая публичная функция
from numpy.testing import assert_allclose

def test_trigger_context_manager():
    cm = ContextManager()
    dummy_contexts = [{"coherence": 0.5}, {"coherence": 0.7}]
    result = cm.get_coherence(dummy_contexts)
    assert result == pytest.approx(0.6, 0.01)

def mock_embed_fixed_factory(embeddings):
    """
    Возвращает функцию, которая поочерёдно отдаёт векторы из списка embeddings.
    """
    counter = {'i': 0}
    def _inner(_: str) -> np.ndarray:
        emb = embeddings[counter['i']]
        counter['i'] += 1
        return emb
    return _inner

def test_voice_allocation_fixed_vectors():
    """Проверяем корректное распределение по голосам с учетом нормализации"""
    # 1. Создаем тестовые векторы (уже нормализованные)
    embs = [
        np.array([1.0, 0.0, 0.0]),  # Базовый вектор (ключ)
        np.array([0.8, 0.6, 0.0]),  # Melody (similarity=0.8)
        np.array([0.6, 0.8, 0.0]),  # Counterpoint (similarity=0.6)
        np.array([0.0, 1.0, 0.0])   # Bass (similarity=0.0)
    ]
    embs = [emb/np.linalg.norm(emb) for emb in embs]  # Нормализуем
    
    # 2. Создаем оркестр с моком для эмбеддингов
    embed_mock = MagicMock(side_effect=embs)
    orch = SimpleOrchestra(embed_mock)
    
    # 3. Первая мысль устанавливает ключ [1,0,0]
    orch.add_thought("base")
    
    # 4. Проверяем расчет similarity
    assert orch._cosine(embs[1], orch.key) == pytest.approx(0.8, abs=1e-7)
    assert orch._cosine(embs[2], orch.key) == pytest.approx(0.6, abs=1e-7)
    assert orch._cosine(embs[3], orch.key) == pytest.approx(0.0, abs=1e-7)
    
    # 5. Фиксируем порог и тестируем распределение
    with patch.object(orch, '_calculate_dynamic_threshold', return_value=0.7):
        # Добавляем оставшиеся мысли
        voices = [orch.add_thought(f"t{i}") for i in range(1, 4)]
        
        # Проверяем распределение
        assert voices == ['melody', 'counterpoint', 'bass'], \
            f"Получено распределение: {voices}"

def test_voice_boundaries_1():
    """Проверка точных границ распределения"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    
    test_cases = [
        (0.71, 'melody'),      # Выше порога (0.7)
        (0.70, 'melody'),      # На пороге
        (0.69, 'counterpoint'), # Чуть ниже порога
        (0.42, 'counterpoint'), # Граница counterpoint (0.7*0.6=0.42)
        (0.41, 'bass')         # Ниже границы counterpoint
    ]
    
    with patch.object(orch, '_calculate_dynamic_threshold', return_value=0.7), \
         patch.object(orch, '_cosine') as mock_cosine:
        for sim, expected in test_cases:
            mock_cosine.return_value = sim
            voice = orch.add_thought("test")
            assert voice == expected, f"При similarity={sim} ожидался {expected}"

def test_voice_boundaries_2():
    """Проверка точных границ распределения"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    
    test_cases = [
        (0.81, 'melody'),
        (0.80, 'melody'),
        (0.79, 'counterpoint'),
        (0.48, 'counterpoint'),
        (0.47, 'bass')
    ]
    
    with patch.object(orch, '_calculate_dynamic_threshold', return_value=0.8), \
         patch.object(orch, '_cosine') as mock_cosine:
        for sim, expected in test_cases:
            mock_cosine.return_value = sim
            voice = orch.add_thought("test")
            assert voice == expected, f"При similarity={sim} ожидался {expected}"

def test_get_coherence_empty_and_filled():
    """
    Для пустого оркестра get_coherence() == 0.0.
    После трёх мыслей (по одному в каждом голосе) coherence = 3/9.
    """
    dummy = lambda x: np.zeros(3, dtype=np.float32)
    orch = SimpleOrchestra(dummy)
    assert orch.get_coherence() == pytest.approx(0.0, abs=1e-6)

    # Заселяем все голоса фиксированными эмбеддингами
    embs = [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.7, 0.7, 0.0]) / np.linalg.norm([0.7, 0.7, 0.0]),
    np.array([0.0, 1.0, 0.0])
]
    orch = SimpleOrchestra(mock_embed_fixed_factory(embs))
    for i in range(3):
        orch.add_thought(f"x{i}")

    # Ёмкость: 4+3+2 = 9, занято 3
    assert orch.get_coherence() == pytest.approx(3/9, rel=1e-3)

def test_dissonance_matrix_fixed():
    """
    Проверяем, что матрица диссонансов = 1 - cosine:
      melody vs counterpoint ≈ 1 - 0.7 = 0.3
      melody vs bass        = 1 - 0   = 1.0
    """
    embs = [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.7, 0.7, 0.0]) / np.linalg.norm([0.7, 0.7, 0.0]),
    np.array([0.0, 1.0, 0.0])
]
    orch = SimpleOrchestra(mock_embed_fixed_factory(embs))
    for i in range(3):
        orch.add_thought(f"z{i}")

    names, M = orch.calculate_dissonance_matrix()
    idx = {name: i for i, name in enumerate(names)}

    # Убедимся, что все три голоса присутствуют
    assert set(names) == {'melody', 'counterpoint', 'bass'}

    # Точность по 0.01
    assert M[idx['melody'], idx['counterpoint']] == pytest.approx(0.3, abs=1e-2)
    assert M[idx['melody'], idx['bass']] == pytest.approx(1.0, abs=1e-6)
    assert M[idx['melody'], idx['counterpoint']] == pytest.approx(1 - 0.7, abs=1e-2)  # ≈ 0.3

def test_temporal_thought_weight_decay():
    """
    Проверяем экспоненциальное затухание веса:
    полураспад=1 → через ~1 с вес ≈0.5.
    """
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tt = TemporalThought("hi", emb, half_life=1.0)

    w0 = tt.weight()
    assert 0.9 < w0 <= 1.0

    time.sleep(1.1)
    w1 = tt.weight()
    assert w1 == pytest.approx(0.5, rel=1e-1)

def test_simple_orchestra_is_base_orchestra():
    """SimpleOrchestra реализует BaseOrchestra."""
    dummy = lambda x: np.zeros(3, dtype=np.float32)
    orch = SimpleOrchestra(dummy)
    assert isinstance(orch, BaseOrchestra)

def test_similarity_calculation():
    """Проверка расчета косинусной схожести"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    orch.add_thought("base")  # Ключ = [1,0,0]

    test_vectors = [
        ([1,0,0], 1.0),       # Точное совпадение
        ([0.7,0.7,0], 0.707), # 45 градусов
        ([0,1,0], 0.0),       # Ортогональный
        ([-1,0,0], -1.0)      # Противоположный
    ]
    
    for vec, expected_sim in test_vectors:
        vec = np.array(vec)
        vec = vec/np.linalg.norm(vec)  # Нормализуем тестовый вектор
        sim = orch._cosine(vec, orch.key)
        assert sim == pytest.approx(expected_sim, abs=0.01)

def test_debug_similarity():
    """Временный тест для отладки расчетов"""
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.7, 0.0, 0.0]) / np.linalg.norm([0.7, 0.0, 0.0])
    sim = np.dot(emb1, emb2)
    print(f"Similarity between [1,0,0] and [0.7,0,0]: {sim:.4f}")
    assert sim == pytest.approx(1.0, abs=1e-6)  # Должно быть 1.0 после нормализации

def test_voice_assignment_with_angles():
    """Проверка распределения по углам между векторами"""
    orch = SimpleOrchestra(lambda x: np.array([1,0,0]))
    
    # Углы: 0°, 30°, 90° (similarity: 1.0, ~0.866, 0.0)
    test_cases = [
        ([1,0,0], 'melody'),
        ([0.866,0.5,0], 'counterpoint'),
        ([0,1,0], 'bass')
    ]
    
    with patch.object(orch, '_calculate_dynamic_threshold', return_value=0.9):
        for vec, expected in test_cases:
            vec = np.array(vec)/np.linalg.norm(vec)
            with patch.object(orch, 'embed_fn', return_value=vec):
                voice = orch.add_thought("test")
                assert voice == expected

from core.orchestra import BaseOrchestra

def test_force_base_orchestra_methods():
    assert BaseOrchestra.add_thought.__isabstractmethod__ is True
    assert BaseOrchestra.get_coherence.__isabstractmethod__ is True

def test_base_orchestra_instantiation_error():
    from core.orchestra import BaseOrchestra

    class IncompleteOrchestra(BaseOrchestra):
        pass  # не реализует abstract методы

    with pytest.raises(TypeError):
        IncompleteOrchestra()

def test_base_orchestra_instantiation_triggers_abstract_01():
    from core.orchestra import BaseOrchestra

    class Stub(BaseOrchestra):
        pass  # не реализует методы

    with pytest.raises(TypeError):  # абстрактные методы будут проверяться
        Stub()

def test_base_orchestra_instantiation_triggers_abstract_02():
    from core.orchestra import BaseOrchestra

    class Incomplete(BaseOrchestra):
        pass

    with pytest.raises(TypeError):
        Incomplete()





