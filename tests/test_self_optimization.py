import pytest
from core.models import Impulse
from core.orchestration.meta_conductor import MetaConductor
from core.orchestration.conductor_optimizer import ConductorOptimizer

@pytest.mark.asyncio
async def test_self_optimization():
    # Создаем MetaConductor (может потребоваться мок для language_model)
    meta = MetaConductor(session_topic="TestOptimization")
    
    # Устанавливаем начальные параметры
    meta.bridge_threshold = 0.5
    meta.tournament_depth = 3
    
    # Создаем оптимизатор
    optimizer = ConductorOptimizer(
        meta,
        target_coherence=0.7,
        diss_target=0.4,
        max_depth=5
    )
    
    # Создаем тестовые импульсы
    impulses = [
        Impulse(type="exploratory", intensity=0.8, complexity=6.0),
        Impulse(type="conflict", intensity=0.6, complexity=7.0),
    ]
    
    # Вызываем асинхронный метод optimize
    metrics = await optimizer.optimize(impulses)
    
    # Проверяем, что метрики рассчитаны и параметры изменились
    assert metrics["avg_coherence"] > 0
    assert metrics["avg_dissonance"] >= 0
    assert metrics["bridge_threshold"] != 0.5 or metrics["tournament_depth"] != 3
    # Мы не можем точно предсказать направление изменения, 
    # но ожидаем, что они обновились
    assert 0 <= metrics["bridge_threshold"] <= 1
    assert 1 <= metrics["tournament_depth"] <= 5