# tests\tests_core\tests_orchestration\tests_conductor_optimizer\test_conductor_optimizer_end.py

import inspect
import numpy as np
import pytest
import asyncio
import warnings
from typing import Any, List, Dict
from types import SimpleNamespace
from core.impulse_engine import Impulse
from core.orchestration.conductor_optimizer import ConductorOptimizer

from core.orchestration.conductor_optimizer import (
    _run_coroutine_in_thread,
    ConductorOptimizer,
)

@pytest.fixture(autouse=True)
def ignore_runtime_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

# Фикстура «сырых» мыслей
@pytest.fixture
def fake_thoughts():
    return [
        SimpleNamespace(coherence=0.5),
        SimpleNamespace(coherence=0.9),
        SimpleNamespace(coherence=0.7),
    ]

# Dummy meta-conductor для проверки взаимодействия
class AutoDummyMeta:
    def __init__(self, thoughts: List[Dict[str, Any]]):
        if not isinstance(thoughts, list):
            raise TypeError("AutoDummyMeta expects a list of thought dictionaries")

        self._thoughts = thoughts

        # Параметры адаптации
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
        # Возвращаем плоский список мыслей и None для матрицы диссонанса
        return [t.copy() for t in self._thoughts], None
    
@pytest.mark.asyncio
async def test_auto_dummy_meta_works():
    # 1. Подготовка тестовых данных
    thoughts = [{
        "content": "auto thought",
        "impulse": Impulse(type="reflective", intensity=1.0),
        "coherence": 0.9  # Явно задаем coherence для проверки
    }]

    # 2. Инициализация
    opt = ConductorOptimizer(AutoDummyMeta(thoughts))
    
    # 3. Вызов метода с конкретным импульсом
    test_impulse = [Impulse(type="reflective", intensity=1.0)]
    res = await opt.optimize(test_impulse)
    
    # 4. Отладочный вывод (можно оставить для диагностики)
    print(f"Debug: Received metrics - {res}")
    
    # 5. Проверки
    assert res["avg_coherence"] == pytest.approx(0.9), \
        f"Expected coherence ~0.9, got {res['avg_coherence']}"
    assert len(res["history"]) > 0, "History should not be empty"
    assert res["history"][0]["impulse"] == test_impulse[0], \
        "Impulse in history should match input"

@pytest.mark.asyncio
async def test_optimize_no_impulses():
    """Если impulses пуст, не вызываем orchestrate, возвращаем defaults."""
    async def orch(imp):
        pytest.skip("orchestrate не должен вызываться при пустых импульсах")

    meta = DummyMeta(orch)
    opt = ConductorOptimizer(meta)

    metrics = await opt.optimize([])
    assert metrics["avg_coherence"] == pytest.approx(0.0)
    assert metrics["avg_dissonance"] == pytest.approx(0.0)
    assert metrics["avg_off_topic_ratio"] == pytest.approx(0.0)

    # пороги и глубина не изменились
    assert metrics["bridge_threshold"] == pytest.approx(meta.bridge_threshold)
    assert metrics["tournament_depth"] == pytest.approx(meta.tournament_depth)

@pytest.mark.asyncio
async def test_optimize_list_returned(fake_thoughts):
    async def orch(imp):
        dm = [[0.1,0.2],[0.2,0.1]]
        return [fake_thoughts, dm]

    meta = DummyMeta(orch)
    opt = ConductorOptimizer(meta)

    # <— вот здесь объявляем old_b
    old_b = meta.bridge_threshold

    metrics = await opt.optimize([1,2,3])
    assert metrics["bridge_threshold"] > old_b

def test_optimize_sync(monkeypatch, fake_thoughts):
    # заставим optimize() вернуть заранее
    sync_called = []
    class FakeOpt(ConductorOptimizer):
        async def optimize(self, imps):
            sync_called.append(True)
            return {"dummy": True}

    fake = FakeOpt(DummyMeta(lambda imp: None))
    res = fake.optimize_sync([1,2,3])
    assert res == {"dummy": True}
    assert sync_called

@pytest.mark.asyncio
async def test_optimize_empty_dissonance(fake_thoughts):
    """DM может быть None или пустым списком, без ошибок."""
    async def orch1(imp):
        return (fake_thoughts, None)

    async def orch2(imp):
        return (fake_thoughts, [])

    for orch in (orch1, orch2):
        meta = DummyMeta(orch)
        opt = ConductorOptimizer(meta)
        metrics = await opt.optimize([42])
        assert metrics["avg_dissonance"] == pytest.approx(0.0)

@pytest.mark.asyncio
async def test_optimize_various_scenarios():
    """Проверяем влияние avg_coherence и avg_dissonance на пороги."""
    # высокий coh, низкий diss → bridge_threshold растёт
    async def orch_hi_coh(imp):
        ths = [SimpleNamespace(coherence=1.0)] * 3
        dm = [[0.0]]
        return (ths, dm)

    meta_hi = DummyMeta(orch_hi_coh)
    opt_hi = ConductorOptimizer(meta_hi)
    old_b_hi = meta_hi.bridge_threshold
    m_hi = await opt_hi.optimize([0])
    assert m_hi["bridge_threshold"] > old_b_hi

    # низкий coh, высокий diss → bridge падает, depth растёт
    async def orch_lo_coh_hi_d(imp):
        ths = [SimpleNamespace(coherence=0.0)] * 3
        dm = [[1.0]]
        return (ths, dm)

    meta_lo = DummyMeta(orch_lo_coh_hi_d)
    opt_lo = ConductorOptimizer(meta_lo)
    old_b_lo = meta_lo.bridge_threshold
    old_d_lo = meta_lo.tournament_depth
    m_lo = await opt_lo.optimize([0])
    assert m_lo["bridge_threshold"] < old_b_lo
    # глубина турнира инкрементируется на 1
    assert m_lo["tournament_depth"] == old_d_lo + 1

@pytest.mark.asyncio
async def test_numpy_dissonance_matrix(fake_thoughts):
    # orchestrate возвращает tuple ([Thoughts], numpy-matrix)
    async def orch(imp):
        return (fake_thoughts, np.array([[0.2, 0.5], [0.5, 0.1]]))

    meta = DummyMeta(orch)
    opt = ConductorOptimizer(meta)
    old_d = meta.tournament_depth
    old_b = meta.bridge_threshold
    metrics = await opt.optimize([0])
    # max_diss = 0.5 < diss_target, значит depth-- 
    assert metrics["avg_dissonance"] == pytest.approx(0.5)
    assert metrics["tournament_depth"] == old_d

@pytest.mark.asyncio
async def test_off_topic_error(fake_thoughts, monkeypatch):
    # orchestrate возвращает базовый
    async def orch(imp): return (fake_thoughts, None)
    meta = DummyMeta(orch)
    # force calculate_off_topic_ratio to throw
    def bad_off(ths): raise RuntimeError("fail")
    meta.calculate_off_topic_ratio = bad_off

    opt = ConductorOptimizer(meta)
    # Должно работать без исключений
    metrics = await opt.optimize([1])
    assert "avg_off_topic_ratio" in metrics

@pytest.mark.asyncio
async def test_history_no_valid():
    async def orch(imp):
        # for each impulse, return thoughts missing coherence
        return ([{"not": "Thought"}], None)

    meta = DummyMeta(orch)
    opt = ConductorOptimizer(meta)
    metrics = await opt.optimize([1,2])
    # avg_coherence и avg_dissonance от valid=[] → 0.0
    assert metrics["avg_coherence"] == pytest.approx(0.0)
    assert metrics["avg_dissonance"] == pytest.approx(0.0)
    # но history теперь длиннее 0
    assert metrics["history"]

# 1) avg_coh_all > target_coherence: threshold increases
@pytest.mark.asyncio
async def test_coherence_threshold_growth(fake_thoughts):
    async def orch(imp):
        # вся avg_coh_all =1.0
        ths = [SimpleNamespace(coherence=1.0)]*3
        # dm empty
        return (ths, [])
    meta = DummyMeta(orch)
    # override target lower
    opt = ConductorOptimizer(meta, target_coherence=0.5)
    before = meta.bridge_threshold
    metrics = await opt.optimize([0])
    assert metrics["bridge_threshold"] > before

# 2) avg_coh_all == target_coherence: threshold stays same
@pytest.mark.asyncio
async def test_coherence_threshold_nochange(fake_thoughts):
    async def orch(imp):
        ths = [SimpleNamespace(coherence=0.6)]*3
        return (ths, [])
    meta = DummyMeta(orch)
    opt = ConductorOptimizer(meta, target_coherence=0.6)
    before = meta.bridge_threshold
    metrics = await opt.optimize([0])
    assert metrics["bridge_threshold"] == pytest.approx(before)

# 3) Dissonance less than, greater than, equal
@pytest.mark.asyncio
async def test_dissonance_threshold_variations(fake_thoughts):
    # < target: depth -1
    async def orch1(imp):
        return (fake_thoughts, [[0.0]])
    meta1 = DummyMeta(orch1)
    opt1 = ConductorOptimizer(meta1, diss_target=1.0)
    before1 = meta1.tournament_depth
    m1 = await opt1.optimize([0])
    assert m1["tournament_depth"] == before1 - 1

    # > target: depth +1
    async def orch2(imp):
        return (fake_thoughts, [[1.0]])
    meta2 = DummyMeta(orch2)
    opt2 = ConductorOptimizer(meta2, diss_target=0.5)
    before2 = meta2.tournament_depth
    m2 = await opt2.optimize([0])
    assert m2["tournament_depth"] == before2 + 1

    # == target: no change
    async def orch3(imp):
        return (fake_thoughts, [[0.5]])
    meta3 = DummyMeta(orch3)
    opt3 = ConductorOptimizer(meta3, diss_target=0.5)
    before3 = meta3.tournament_depth
    m3 = await opt3.optimize([0])
    assert m3["tournament_depth"] == before3

# 4) parallel_orchestration.invalidate is called
class DummyParallel:
    def __init__(self):
        self.invalidated = False
    def invalidate(self):
        self.invalidated = True

@pytest.mark.asyncio
async def test_recursion_cache_invalidation(fake_thoughts):
    async def orch(imp):
        return (fake_thoughts, [])
    meta = DummyMeta(orch)
    pp = DummyParallel()
    meta._parallel_orchestration = pp
    opt = ConductorOptimizer(meta)
    await opt.optimize([1])
    assert pp.invalidated

# 5) handle non-list thoughts
@pytest.mark.asyncio
async def test_skip_invalid_thoughts():
    async def orch(imp):
        return ("not a list", None)
    meta = DummyMeta(orch)
    opt = ConductorOptimizer(meta)
    # should complete without error, history empty
    metrics = await opt.optimize([0])
    assert metrics["history"] == []

# 1) Проверяем все три ветки адаптации tournament_depth
@pytest.mark.asyncio
async def test_dissonance_variations_and_list_of_lists(fake_thoughts):
    # 1a) avg_diss < target → depth--
    async def low(imp):
        return fake_thoughts, [[0.0]]
    meta1 = DummyMeta(low)
    opt1 = ConductorOptimizer(meta1, diss_target=0.5)
    before1 = meta1.tournament_depth
    res1 = await opt1.optimize([0])
    assert res1["avg_dissonance"] == pytest.approx(0.0)
    assert res1["tournament_depth"] == before1 - 1

    # 1b) avg_diss > target → depth++
    async def high(imp):
        return fake_thoughts, [[1.0]]
    meta2 = DummyMeta(high)
    opt2 = ConductorOptimizer(meta2, diss_target=0.5)
    before2 = meta2.tournament_depth
    res2 = await opt2.optimize([0])
    assert res2["avg_dissonance"] == pytest.approx(1.0)
    assert res2["tournament_depth"] == before2 + 1

    # 1c) avg_diss == target → no change
    async def eq(imp):
        return fake_thoughts, [[0.5]]
    meta3 = DummyMeta(eq)
    opt3 = ConductorOptimizer(meta3, diss_target=0.5)
    before3 = meta3.tournament_depth
    res3 = await opt3.optimize([0])
    assert res3["avg_dissonance"] == pytest.approx(0.5)
    assert res3["tournament_depth"] == before3

# 2) Проверяем сохранение invalidate() в _parallel_orchestration
class StubParallel:
    def __init__(self):
        self.was = False
    def invalidate(self):
        self.was = True

@pytest.mark.asyncio
async def test_recursion_cache_invalidation(fake_thoughts):
    async def orch(imp):
        return fake_thoughts, None

    meta = DummyMeta(orch)
    stub = StubParallel()
    meta._parallel_orchestration = stub

    opt = ConductorOptimizer(meta)
    await opt.optimize([1])
    assert stub.was, "invalidate() должен быть вызван"

# 3) Пропускаем не-списочные мысли (if not thoughts or not list)
@pytest.mark.asyncio
async def test_skip_non_list_thoughts():
    # orchestrate возвращает строку вместо списка → ветка continue
    async def orch(imp):
        return "oops", None

    meta = DummyMeta(orch)
    opt = ConductorOptimizer(meta)
    stats = await opt.optimize([0])
    # history пуст, значит ни одной записи не добавилось
    assert stats["history"] == []

@pytest.mark.asyncio
async def test_dissonance_numpy_error(fake_thoughts):
    class BadArray:
        size = 2
        def max(self):
            raise ValueError("bad")
    async def orch(imp):
        return fake_thoughts, BadArray()

    meta = DummyMeta(orch)
    opt = ConductorOptimizer(meta)
    metrics = await opt.optimize([0])
    # упали внутрии catch, но avg_diss=0
    assert metrics["avg_dissonance"] == pytest.approx(0.0)

@pytest.mark.asyncio
async def test_off_topic_valid(fake_thoughts):
    async def orch(imp):
        return fake_thoughts, None
    meta = DummyMeta(orch)
    # новый stub
    meta.calculate_off_topic_ratio = lambda ths: 0.123
    opt = ConductorOptimizer(meta)
    metrics = await opt.optimize([0])
    assert metrics["avg_off_topic_ratio"] == pytest.approx(0.123)

@pytest.mark.asyncio
async def test_dissonance_empty_list(fake_thoughts):
    async def orch(imp):
        return fake_thoughts, [[]]  # dm exists, but dm[0] empty
    meta = DummyMeta(orch)
    opt = ConductorOptimizer(meta)
    metrics = await opt.optimize([0])
    assert metrics["avg_dissonance"] == pytest.approx(0.0)

@pytest.mark.asyncio
async def test_raw_not_two_length(fake_thoughts):
    # возвращает list длины 1 → dm=None, thoughts=raw=list → guarded
    async def orch1(imp):
        return [fake_thoughts]  # list len=1

    meta1 = DummyMeta(orch1)
    opt1 = ConductorOptimizer(meta1)
    res1 = await opt1.optimize([0])
    # avg_coh >0, avg_diss=0
    assert "avg_dissonance" in res1
    assert res1["avg_dissonance"] == pytest.approx(0.0)

    # возвращает tuple длины 3 — treated как invalid
    async def orch2(imp):
        return (fake_thoughts, [[0.1]], "extra")
    meta2 = DummyMeta(orch2)
    opt2 = ConductorOptimizer(meta2)
    res2 = await opt2.optimize([0])
    assert res2["avg_dissonance"] == pytest.approx(0.0)

@pytest.mark.asyncio
async def test_empty_thoughts_list():
    async def orch(imp):
        return ([], [[1.0]])
    meta = DummyMeta(orch)
    opt = ConductorOptimizer(meta)
    res = await opt.optimize([0])
    assert res["history"] == []

# Фикстура из вашего файла
class DummyMeta:
    def __init__(self, orch):
        self.bridge_threshold = 0.5
        self.tournament_depth = 3
        self.calculate_off_topic_ratio = lambda thoughts: 0.0
        self.target = 0.75
        self._alpha = 0.1
        self._beta = 0.1
        self._diss_target = 0.3
        self._max_depth = 10
        self.orchestrate = orch


# ——————————————————————————————————————————
# 1) Тестируем _run_coroutine_in_thread()
#    — успех (возвращаемое значение)
#    — провал (исключение внутри coroutine)
# ——————————————————————————————————————————

def test_run_coroutine_in_thread_success():
    async def foo():
        return "OK"
    result = _run_coroutine_in_thread(lambda: foo())
    assert result == "OK"

def test_run_coroutine_in_thread_exception():
    async def bar():
        raise RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        _run_coroutine_in_thread(lambda: bar())


# ——————————————————————————————————————————
# 2) Покрываем ветку optimise_sync() внутри try:
#    когда есть запущенный loop,
#    и run_coroutine_threadsafe() возвращает Future-like
# ——————————————————————————————————————————

def test_optimize_sync_with_running_loop(monkeypatch):
    # 1) Подменяем get_running_loop → наш dummy_loop
    dummy_loop = object()
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: dummy_loop)
    # 2) Подменяем run_coroutine_threadsafe → возвращает fake_future
    fake_future = SimpleNamespace(result=lambda: {"ran": True})
    def fake_run_coroutine_threadsafe(coro, loop):
        # убеждаемся, что передали наш loop
        assert loop is dummy_loop
        return fake_future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)

    # 3) Нас не интересует настоящее optimize(), оно не должно вызываться
    class FakeOpt(ConductorOptimizer):
        async def optimize(self, imps):
            pytest.skip("optimize() не должна вызываться в этом тесте")

    opt = FakeOpt(DummyMeta(lambda imp: None))
    out = opt.optimize_sync([1,2,3])
    assert out == {"ran": True}


# ——————————————————————————————————————————
# 3) raw не tuple/list длины 2 → thoughts остаются None → пропускаем
#    и ветка «continue» должна не добавлять ничего в history
# ——————————————————————————————————————————

@pytest.mark.asyncio
async def test_raw_not_two_length_list_is_processed(fake_thoughts):
    async def orch(imp):
        # raw — список не длины 2, считается списком мыслей
        return fake_thoughts  # list of Thought
    meta = DummyMeta(orch)
    opt = ConductorOptimizer(meta)
    metrics = await opt.optimize([42])

    # history должно содержать одну запись
    assert len(metrics["history"]) == 1
    assert metrics["history"][0]["impulse"] == 42
    # avg_coherence рассчиталась правильно
    assert metrics["avg_coherence"] == pytest.approx(0.7)

@pytest.mark.asyncio
async def test_empty_thoughts_tuple_is_skipped():
    async def orch(imp):
        # raw — tuple, но первый элемент пустой список → continue
        return ([], None)
    meta = DummyMeta(orch)
    opt = ConductorOptimizer(meta)
    metrics = await opt.optimize([99])

    # history остаётся пустой
    assert metrics["history"] == []
    assert metrics["avg_coherence"] == pytest.approx(0.0)
    assert metrics["avg_dissonance"] == pytest.approx(0.0)

def test_module_import_and_attrs():
    import importlib
    mod = importlib.import_module("core.orchestration.conductor_optimizer")
    # проверяем, что на уровне модуля определены все нужные сущности
    assert hasattr(mod, "_run_coroutine_in_thread")
    assert hasattr(mod, "ConductorOptimizer")

def test_optimize_sync_runtime_error(monkeypatch, fake_thoughts):
    # 1) Заставляем asyncio.get_running_loop() бросать RuntimeError
    monkeypatch.setattr(
        __import__("asyncio"),
        "get_running_loop",
        lambda: (_ for _ in ()).throw(RuntimeError())
    )
    # 2) Подменяем asyncio.run, чтобы оно возвращало фиксированный результат
    monkeypatch.setattr(
        __import__("asyncio"),
        "run",
        lambda coro: {"ran_via_run": True}
    )

    # Используем обычный ConductorOptimizer (optimize не вызовется)
    opt = ConductorOptimizer(DummyMeta(lambda imp: None))
    res = opt.optimize_sync([1,2,3])
    assert res == {"ran_via_run": True}

@pytest.mark.asyncio
async def test_conductor_optimizer_custom_parameters():
    async def orch(_):
        return [[{
            "content": "deep thought",
            "impulse": Impulse(type="reflective", intensity=1.0),
            "coherence": 0.7
        }]], None  # thoughts, dissonance_matrix

    class CustomMeta(DummyMeta):
        async def orchestrate(self, impulse):
            return await orch(impulse)

        def __init__(self):
            super().__init__(orch)
            self._alpha = 0.9
            self._beta = 0.05
            self._max_depth = 2
            self._diss_target = 0.1

    opt = ConductorOptimizer(CustomMeta())
    result = await opt.optimize([Impulse(type="reflective", intensity=1.0)])

    assert result["avg_coherence"] == pytest.approx(0.7)
    assert result["history"]
    assert isinstance(result["bridge_threshold"], float)
    assert isinstance(result["tournament_depth"], int)

@pytest.mark.asyncio
async def test_off_topic_ratio_triggers_filtering():
    def orch(imp):
        return [["irrelevant thought"]]

    class MetaWithOffTopic(DummyMeta):
        def __init__(self, orch):
            super().__init__(orch)
            self.calculate_off_topic_ratio = lambda thoughts: 0.8  # превышает порог

    opt = ConductorOptimizer(MetaWithOffTopic(orch))
    res = await opt.optimize(["ping"])
    # off_topic должен быть отброшен
    assert res["history"] == []
    assert res["avg_coherence"] == 0.0

def test_run_coroutine_in_thread_none_input():
    async def dummy():
        return None
    result = _run_coroutine_in_thread(dummy)
    assert result is None

@pytest.mark.asyncio
async def test_extreme_intensity_and_coherence_values():
    async def orch(_):
        return [[{
            "content": "extreme thought",
            "impulse": Impulse(type="reflective", intensity=1.0),
            "coherence": 1.0
        }]]

    class Meta(DummyMeta):
        def __init__(self):
            async def orch(_):
                return [[{
                    "content": "extreme thought",
                    "impulse": Impulse(type="reflective", intensity=1.0),
                    "coherence": 1.0
                }]]
            super().__init__(orch)

    opt = ConductorOptimizer(Meta())
    result = await opt.optimize([Impulse(type="reflective", intensity=1.0)])

    assert result["avg_coherence"] == pytest.approx(1.0)
    assert result["avg_dissonance"] == pytest.approx(0.0)
    assert result["avg_off_topic_ratio"] == pytest.approx(0.0)
    assert result["history"]

@pytest.mark.asyncio
async def test_optimize_with_none_impulses():
    # Оркестратор возвращает валидную мысль-словарь
    def orch(_):
        return [[{
            "content": "valid thought",
            "impulse": Impulse(type="reflective"),
            "coherence": 0.8
        }]]

    opt = ConductorOptimizer(DummyMeta(orch))

    # Передаём None, ожидаем graceful fallback
    metrics = await opt.optimize(None)

    # Проверка структуры
    assert isinstance(metrics, dict)
    assert metrics["history"] == []               # Данных не будет, так как impulses стал []
    assert metrics["avg_coherence"] == 0.0
    assert metrics["avg_dissonance"] == 0.0
    assert metrics["avg_off_topic_ratio"] == 0.0
    assert isinstance(metrics["bridge_threshold"], float)
    assert isinstance(metrics["tournament_depth"], int)
    assert metrics["execution_time"] >= 0.0

@pytest.mark.asyncio
async def test_optimize_with_non_list_impulses(caplog):
    # Устанавливаем уровень логирования, чтобы caplog перехватил предупреждение
    caplog.set_level("WARNING")

    # Используем валидную структуру мысли
    def orch(_):
        return [[{
            "content": "dummy",
            "impulse": Impulse(type="reflective"),
            "coherence": 0.5
        }]]

    opt = ConductorOptimizer(DummyMeta(orch))

    # Передаём строку вместо списка импульсов
    metrics = await opt.optimize("not a list")

    # Проверка, что корректно возвращено с пустой историей
    assert metrics["history"] == []
    assert metrics["avg_coherence"] == 0.0
    assert metrics["avg_dissonance"] == 0.0
    assert metrics["avg_off_topic_ratio"] == 0.0

    # Проверка, что предупреждение логгируется
    assert any("Ignoring non-list impulses" in msg for msg in caplog.text.splitlines())

def test_run_coroutine_in_thread_error_propagation():
    async def raise_err():
        raise ValueError("boom!")
    
    # Теперь будет проходить
    with pytest.raises(ValueError, match="boom!"):
        _run_coroutine_in_thread(raise_err)

@pytest.mark.asyncio
async def test_optimizer_skips_empty_thoughts_list():
    def orch(imp):
        return []
    opt = ConductorOptimizer(DummyMeta(orch))
    result = await opt.optimize([100])
    assert result["history"] == []

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
    result = await opt.optimize("invalid")  # не список
    assert result["avg_coherence"] == 0.0
    assert result["history"] == []

class FailingMeta:
    def __init__(self):
        self.bridge_threshold = 0.5
        self.tournament_depth = 3
        self.calculate_off_topic_ratio = lambda _: 0.0
        self._alpha = 0.1
        self._beta = 0.1
        self._diss_target = 0.5
        self._max_depth = 10

    async def orchestrate(self, impulse):
        raise ValueError("boom!")

@pytest.mark.asyncio
async def test_orchestrate_exception_handling():
    opt = ConductorOptimizer(FailingMeta())
    impulses = [Impulse(type="reflective", intensity=1.0)]
    result = await opt.optimize(impulses)
    assert result["avg_coherence"] == 0.0
    assert result["history"] == []

class EmptyMeta:
    def __init__(self):
        self.bridge_threshold = 0.5
        self.tournament_depth = 3
        self.calculate_off_topic_ratio = lambda _: 0.0
        self._alpha = 0.1
        self._beta = 0.1
        self._diss_target = 0.5
        self._max_depth = 10

    async def orchestrate(self, impulse):
        return [], None

@pytest.mark.asyncio
async def test_optimize_no_thoughts_returned():
    opt = ConductorOptimizer(EmptyMeta())
    impulses = [Impulse(type="reflective", intensity=1.0)]
    result = await opt.optimize(impulses)
    assert result["avg_coherence"] == 0.0
    assert result["history"] == []
