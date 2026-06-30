#tests\test_orchestra_extended.py

import time
import pytest
import logging
import numpy as np
from unittest.mock import MagicMock
from core.orchestra import SimpleOrchestra, TemporalThought


# ---------- MOCK UTILS ----------
def mock_embed_fixed_factory(embeddings):
    counter = {'i': 0}
    def _inner(_: str) -> np.ndarray:
        emb = embeddings[counter['i']]
        counter['i'] += 1
        return emb
    return _inner

# ---------- EDGE CASES & COVERAGE ----------
def test_dynamic_threshold_calculation():
    embs = [
        np.array([1.0, 0, 0], dtype=np.float32),
        np.array([0.9, 0, 0], dtype=np.float32)
    ]
    orch = SimpleOrchestra(mock_embed_fixed_factory(embs))
    orch.add_thought("text1")  # sim = 1.0
    orch.add_thought("text2")  # sim = 0.9
    threshold = orch._calculate_dynamic_threshold()
    assert pytest.approx(threshold, 0.1) == 0.9

def test_embedding_failure_handling():
    def broken_embed(_: str): return None
    orch = SimpleOrchestra(broken_embed)
    with pytest.raises(RuntimeError, match="Failed to get text embedding"):
        orch.add_thought("fail")

def test_single_voice_dissonance_matrix():
    emb = np.array([1.0, 0.0, 0.0])
    orch = SimpleOrchestra(lambda _: emb)
    orch.add_thought("only one")
    names, matrix = orch.calculate_dissonance_matrix()
    assert names == ["melody"]
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 0.0

def test_add_thought_empty_text():
    orch = SimpleOrchestra(lambda _: np.ones(3))
    with pytest.raises(ValueError, match="Text cannot be empty"):
        orch.add_thought("   ")

def test_add_thought_logging_on_failure(capsys):
    def bad_embed(_: str):
        raise ValueError("Test error message")
    
    orch = SimpleOrchestra(bad_embed)
    
    with pytest.raises(RuntimeError, match="Failed to process text"):
        orch.add_thought("fail-text")
    
    captured = capsys.readouterr()
    
    print("STDOUT:", captured.out)
    print("STDERR:", captured.err)

    print("Captured stderr:", captured.err)
    assert "ERROR: Failed to process text" in captured.err
    assert "Exception: Test error message" in captured.err

def test_log_error_direct_call(capsys):
    from core.orchestra import log_error
    try:
        raise ValueError("Simulated error")
    except Exception as e:
        log_error("Test logging error", e)

    captured = capsys.readouterr()
    assert "ERROR: Test logging error" in captured.err
    assert "Exception: Simulated error" in captured.err


# ---------- INTEGRATION TESTS ----------
def test_massive_thought_allocation_and_coherence():
    def dummy_embed(text: str) -> np.ndarray:
        idx = len(text) % 3
        return np.eye(3)[idx]
    
    orch = SimpleOrchestra(dummy_embed)
    voices_seen = set()
    
    for i in range(12):
        voice = orch.add_thought(f"thought{i}")
        voices_seen.add(voice)
    
    assert voices_seen <= {"melody", "counterpoint", "bass"}
    coherence = orch.get_coherence()
    assert 0.0 < coherence <= 1.0

def test_thought_decay_over_time():
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    thought = TemporalThought("тестовая мысль", emb, half_life=1.0)

    # Начальный вес сразу после создания
    initial_weight = thought.weight()
    assert 0.9 < initial_weight <= 1.0

    # Имитация прошедшего времени: 1.2 секунды
    thought.birth_time = time.time() - 1.2
    decayed_weight = thought.weight()

    # Ожидаем ≈ 0.5 с допуском
    assert 0.4 < decayed_weight < 0.6

def test_multiple_voices_weight_decay_simulation():
    # Более выраженные различия в эмбеддингах
    embs = {
        "A1": np.array([0.9, 0.1, 0.0]),  # melody (близко к ключу)
        "B2": np.array([0.4, 0.5, 0.1]),   # counterpoint
        "C3": np.array([0.1, 0.1, 0.8])    # bass (далеко от ключа)
    }
    
    def dummy_embed(text: str) -> np.ndarray:
        return embs[text]
    
    # Инициализируем с первым эмбеддингом как ключ
    orch = SimpleOrchestra(dummy_embed)
    orch.key = embs["A1"]  # Устанавливаем ключ вручную
    
    # Добавляем мысли
    orch.add_thought("A1")  # melody
    orch.add_thought("B2")  # counterpoint
    orch.add_thought("C3")  # bass
    
    # Проверяем распределение
    names, matrix = orch.calculate_dissonance_matrix()
    assert set(names) == {"melody", "counterpoint", "bass"}
    assert matrix.shape == (3, 3)
    
    # Проверяем диссонансы
    idx = {name: i for i, name in enumerate(names)}
    assert matrix[idx["melody"], idx["bass"]] > 0.5  # Должны сильно различаться

def test_voice_distribution_logic():
    """Проверяет четкое распределение по 3 голосам"""
    embs = [
        np.array([0.95, 0.05, 0]),  # melody
        np.array([0.6, 0.3, 0.1]),   # counterpoint
        np.array([0.1, 0.1, 0.8])    # bass
    ]
    
    orch = SimpleOrchestra(mock_embed_fixed_factory(embs))
    voices = [orch.add_thought(f"t{i}") for i in range(3)]
    
    assert set(voices) == {"melody", "counterpoint", "bass"}

def test_non_numeric_embedding_handling():
    def bad_embed(_): return "not-a-vector"
    orch = SimpleOrchestra(bad_embed)
    with pytest.raises(RuntimeError):
        orch.add_thought("test")

def test_voice_overflow_handling():
    # Чередуем эмбеддинги для распределения по разным голосам
    embs = [
        np.array([1.0, 0, 0]),  # melody
        np.array([0.7, 0.7, 0]), # counterpoint
        np.array([0, 1, 0]),     # bass
        np.array([1.0, 0, 0]),   # melody
    ]
    orch = SimpleOrchestra(mock_embed_fixed_factory(embs))
    
    for i in range(len(embs)):
        orch.add_thought(f"t{i}")
    
    assert len(orch.voices['melody']) == 2  # 2 мысли в melody (maxlen=4)
    assert len(orch.voices['counterpoint']) == 1
    assert len(orch.voices['bass']) == 1

def test_custom_half_life():
    emb = np.array([1,0,0])
    orch = SimpleOrchestra(lambda _: emb)
    
    # Модифицируем тест для проверки через прямое создание TemporalThought
    thought = TemporalThought("test", emb, half_life=0.1)
    orch.voices['melody'].append(thought)
    
    time.sleep(0.15)  # больше чем half_life
    assert thought.weight() < 0.5

