# tests/test_context_manager.py

import pytest
import asyncio
import time
from core.context_manager import ContextManager

@pytest.fixture
def cm():
    return ContextManager()

def test_add_context_valid(cm, monkeypatch):
    # Готовим мысль с нужными полями
    thought = {
        "content": "Hello world",
        "language": "en",
        "trust_score": 0.5,
        "impulse": {"type": "test"}
    }
    # Заставляем анализатор возвращать именно "X"
    monkeypatch.setattr(
        cm.semantic_analyzer,
        "extract_core_concept",
        lambda txt: "X"
    )
    result = cm.add_context(thought)
    assert result == "X"
    # Проверяем, что в active_contexts появился новый контекст
    assert cm.active_contexts[-1]["core_concept"] == "x"

def test_add_context_invalid(cm):
    with pytest.raises(ValueError):
        cm.add_context(None)

def test_get_coherence_empty(cm):
    assert cm.get_coherence([]) == 0.0

def test_get_coherence_values(cm):
    contexts = [
        {"coherence": 0.5},
        {"coherence": 1.5},
        {"coherence": 1.0},
    ]
    # среднее (0.5 + 1.5 + 1.0) / 3 = 1.0
    assert cm.get_coherence(contexts) == pytest.approx(1.0)

def test_get_dissonance_empty(cm):
    assert cm.get_dissonance([]) == 0.0

def test_get_dissonance_list(cm):
    matrix = [
        [0, 1],
        [1, 0]
    ]
    # ожидаем максимальную диссонансную пару = 1
    assert cm.get_dissonance(matrix) == 1

def test_tick_and_export(cm):
    ctx = {"core_concept": "Y"}
    cm.add_context(ctx)
    cm.tick()
    state = cm.export_state()
    assert "active_contexts" in state
    assert "history" in state
    # после tick в истории должен появиться хотя бы один снимок
    assert isinstance(state["history"], list) and len(state["history"]) > 0

# Тест конкурентного доступа
@pytest.mark.asyncio
async def test_concurrent_context_access():
    manager = ContextManager(max_contexts=100)

    async def add(i):
        thought = {
            "content": f"context {i}",
            "trust_score": 0.9,
            "language": "en",
            "impulse": {"type": "test"}
        }
        manager.add_context(thought)

    tasks = [add(i) for i in range(100)]
    await asyncio.gather(*tasks)

    assert len(manager.active_contexts) == 100
    assert all("core_concept" in ctx for ctx in manager.active_contexts)

# Тест очистки устаревших данных
def test_context_expiration():
    manager = ContextManager(ttl=0.1)  # 100ms TTL
    manager.add_context({
    "content": "temp value",
    "trust_score": 0.9,
    "language": "en",
    "impulse": {"type": "test"}
})
    time.sleep(0.2)
    assert "temp" not in manager.active_contexts

@pytest.mark.asyncio
async def test_concurrent_context_access_with_ttl_validation():
    ttl = 1.0  # 1 секунда
    manager = ContextManager(max_contexts=150, ttl=ttl)

    async def add(i):
        thought = {
            "content": f"context_{i}",
            "trust_score": 0.9,
            "language": "en",
            "impulse": {"type": "test"}
        }
        manager.add_context(thought)

    tasks = [add(i) for i in range(100)]
    await asyncio.gather(*tasks)

    # ⏳ Подождем, чтобы включить TTL
    time.sleep(0.5)
    manager.tick()

    # 🎯 Проверка уникальности core_concept
    concepts = [ctx["core_concept"] for ctx in manager.active_contexts]
    assert len(set(concepts)) == len(concepts), "core_concept содержит дубликаты"

    # ⏱️ Проверка актуальности timestamp
    now = time.time()
    for ctx in manager.active_contexts:
        age = now - ctx["timestamp"]
        assert age <= ttl, f"Контекст устарел: age={age:.3f} > ttl={ttl}"

    # ✅ Проверка, что TTL применился
    before_cleanup = len(manager.active_contexts)
    time.sleep(0.6)  # превысим TTL
    manager.tick()
    after_cleanup = len(manager.active_contexts)
    assert after_cleanup < before_cleanup, "TTL не удалил устаревшие контексты"