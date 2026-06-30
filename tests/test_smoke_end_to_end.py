import pytest
import random
from core.models import Impulse
from core.orchestration.conductor_optimizer import ConductorOptimizer
from core.orchestration.meta_conductor import MetaConductor

@pytest.mark.asyncio
async def test_full_optimization_cycle():
    # Подготовка тестовых данных
    meta = MetaConductor(session_topic="Cognitive Architecture")
    optimizer = ConductorOptimizer(meta)
    
    # Генерация тестовых импульсов
    test_impulses = [
        Impulse(
            type=random.choice(["exploratory", "reflective", "integrative"]),
            intensity=random.uniform(0.3, 1.0),
            complexity=random.uniform(2.0, 8.0)
        )
        for _ in range(5)
    ]
    
    # Выполнение оптимизации
    metrics = await optimizer.optimize(test_impulses)
    
    # Проверка основных метрик
    assert 0.0 <= metrics["avg_coherence"] <= 1.0
    assert 0.0 <= metrics["avg_dissonance"] <= 1.0
    assert 0.0 <= metrics["avg_off_topic_ratio"] <= 1.0
    assert 0.0 <= metrics["bridge_threshold"] <= 1.0
    assert 1 <= metrics["tournament_depth"] <= optimizer._max_depth
    assert metrics["execution_time"] > 0
    
    # Проверка истории
    assert len(metrics["history"]) == len(test_impulses)
    for entry in metrics["history"]:
        assert "off_topic_ratio" in entry
        assert "execution_time" in entry
        assert "timestamp" in entry