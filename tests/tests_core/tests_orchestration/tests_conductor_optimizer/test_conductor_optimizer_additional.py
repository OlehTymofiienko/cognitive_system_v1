# tests\tests_core\tests_orchestration\tests_conductor_optimizer\test_conductor_optimizer_additional.py

import pytest

from core.impulse_engine import Impulse
from core.orchestration.conductor_optimizer import ConductorOptimizer
from core.orchestration.conductor_optimizer import _run_coroutine_in_thread

# Мок для плоских списков мыслей
class AutoDummyMeta:
    def __init__(self, thoughts):
        self._thoughts = thoughts
        self.bridge_threshold = 0.5
        self.tournament_depth = 3
        self.calculate_off_topic_ratio = lambda _: 0.0
        self.target = 0.75
        self._alpha = 0.1
        self._beta = 0.1
        self._diss_target = 0.3
        self._max_depth = 10
        self._parallel_orchestration = None

    async def orchestrate(self, impulse):
        return [t.copy() for t in self._thoughts], None


@pytest.mark.asyncio
async def test_optimize_none_input():
    meta = AutoDummyMeta([])
    opt = ConductorOptimizer(meta)
    result = await opt.optimize(None)
    assert result["avg_coherence"] == 0.0
    assert result["history"] == []


@pytest.mark.asyncio
async def test_optimize_invalid_type():
    meta = AutoDummyMeta([])
    opt = ConductorOptimizer(meta)
    result = await opt.optimize("invalid")
    assert result["avg_coherence"] == 0.0
    assert result["history"] == []


class FailingMeta:
    def __init__(self):
        self.bridge_threshold = 0.5
        self.tournament_depth = 3
        self.calculate_off_topic_ratio = lambda _: 0.0
        self.target = 0.75
        self._alpha = 0.1
        self._beta = 0.1
        self._diss_target = 0.3
        self._max_depth = 10

    async def orchestrate(self, impulse):
        raise ValueError("boom!")


@pytest.mark.asyncio
async def test_orchestrate_exception_handling():
    opt = ConductorOptimizer(FailingMeta())
    result = await opt.optimize([Impulse(type="reflective", intensity=1.0)])
    assert result["avg_coherence"] == 0.0
    assert result["history"] == []


class EmptyMeta:
    def __init__(self):
        self.bridge_threshold = 0.5
        self.tournament_depth = 3
        self.calculate_off_topic_ratio = lambda _: 0.0
        self.target = 0.75
        self._alpha = 0.1
        self._beta = 0.1
        self._diss_target = 0.3
        self._max_depth = 10

    async def orchestrate(self, impulse):
        return [], None


@pytest.mark.asyncio
async def test_optimize_no_thoughts_returned():
    opt = ConductorOptimizer(EmptyMeta())
    result = await opt.optimize([Impulse(type="reflective", intensity=1.0)])
    assert result["avg_coherence"] == 0.0
    assert result["history"] == []
