import pytest
from core.models import Impulse
from core.orchestration.meta_conductor import MetaConductor

@pytest.mark.asyncio
async def test_orchestrate_complex():
    conductor = MetaConductor("AI Ethics")
    complex_impulse = Impulse(type="exploratory", intensity=0.8, complexity=7.5)

    results = await conductor.orchestrate(complex_impulse)
    # Должны быть как минимум три голоса
    assert len(results) >= 3
    # В списке результатов найдётся мысль с voice='bass'
    assert any(t.voice == 'bass' for t in results)

@pytest.mark.asyncio
async def test_meta_conductor_with_bridge():
    # создаём импульс с достаточной сложностью, чтобы пошла параллельная оркестровка
    # complexity > 5.0, чтобы пойти в _parallel_orchestration
    impulse = Impulse(type="conflict", intensity=3.0, complexity=7.0)

    # правильно инстанцируем класс и занижаем порог
    meta = MetaConductor(session_topic="IntegrationTest", language_model=None)
    meta.bridge_threshold = 0.0  # любой диссонанс даст мостовую мысль
    meta.tournament_depth = 10   # число мыслей (4) ≤ порога → турнир не запустится

    thoughts = await meta.orchestrate(impulse)

    # среди мыслей должен быть мостовой, у которого в metadata есть bridge_of
    assert any("bridge_of" in t.metadata for t in thoughts), (
        "BridgeSynthesizer не сработал, bridge-мысль не найдена"
    )

    # и общее число мыслей = 3 «голосовые» + 1 мостовая
    assert len(thoughts) == 4