import asyncio
import logging
import statistics
import time
from datetime import datetime
from threading import Thread
from typing import Any, Optional, Callable, List, Dict
from typing import Any, Callable, Coroutine, TypeVar, List
from core.models import Impulse, Thought
from core.orchestration.meta_conductor import MetaConductor
from core.orchestration.utils import calculate_dissonance_matrix

logger = logging.getLogger(__name__)
T = TypeVar('T')

def _run_coroutine_in_thread(make_coro: Callable[[], Coroutine[Any, Any, T]]) -> T:
    result: List[T] = []
    exc: List[BaseException] = []

    def runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            coro = make_coro()
            if not asyncio.iscoroutine(coro):
                raise TypeError("make_coro must return a coroutine")
            task = asyncio.ensure_future(coro, loop=loop)
            res = loop.run_until_complete(task)
            result.append(res)
        except Exception as e:
            logger.exception("Error in coroutine execution")
            exc.append(e)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    t = Thread(target=runner, daemon=True)
    t.start()
    t.join()

    if exc:
        raise exc[0] from None
    if not result:
        raise RuntimeError("Coroutine didn't return any result")
    return result[0]


class ConductorOptimizer:
    def __init__(
        self,
        meta: MetaConductor,
        target_coherence: float = 0.6,
        diss_target: float = 0.5,
        alpha: float = 0.5,
        beta: float = 0.3,
        max_depth: int = 10
    ):
        self.meta = meta
        self.target = target_coherence
        self._diss_target = diss_target
        self._alpha = alpha
        self._beta = beta
        self._max_depth = max_depth
        self.history: List[Dict[str, Any]] = []

    async def optimize(self, impulses: Optional[List[Impulse]]) -> Dict[str, Any]:
        """
        Асинхронная оптимизация MetaConductor:
        собирает метрики (coherence, dissonance, off_topic, time)
        и настраивает bridge_threshold и tournament_depth.
        Также содержит защиту от некорректных входов и устойчивое поведение.
        """
        if impulses is None:
            logger.warning("Received None impulses, defaulting to empty list")
            impulses = []
        elif not isinstance(impulses, list):
            logger.warning("Ignoring non-list impulses")
            return {
                "avg_coherence": 0.0,
                "avg_dissonance": 0.0,
                "avg_off_topic_ratio": 0.0,
                "bridge_threshold": self.meta.bridge_threshold,
                "tournament_depth": self.meta.tournament_depth,
                "execution_time": 0.0,
                "history": []
            }

        t_start = time.perf_counter()

        # Инвалидация RecursionCache (если применимо)
        po = getattr(self.meta, "_parallel_orchestration", None)
        if po and hasattr(po, "invalidate"):
            po.invalidate()
            logger.debug("Recursion cache invalidated")

        self.history.clear()
        for imp in impulses:
            t_imp = time.perf_counter()
            try:
                raw = await self.meta.orchestrate(imp)
            except Exception as e:
                logger.error(f"Error orchestrating {imp}: {e}")
                continue

            # Поддержка форматов возврата
            if isinstance(raw, (tuple, list)) and len(raw) == 2:
                thoughts, dm = raw
            else:
                thoughts = raw
                dm = None

            if not thoughts or not isinstance(thoughts, list):
                continue

            # 1) Coherence
            coh_vals = []
            for item in thoughts:
                if isinstance(item, list):
                    # Обрабатываем вложенные списки мыслей
                    for t in item:
                        if isinstance(t, dict) and isinstance(t.get("coherence"), (int, float)):
                            coh_vals.append(t["coherence"])
                        elif hasattr(t, "coherence") and isinstance(getattr(t, "coherence"), (int, float)):
                            coh_vals.append(t.coherence)
                else:
                    # Обрабатываем одиночные мысли
                    if isinstance(item, dict) and isinstance(item.get("coherence"), (int, float)):
                        coh_vals.append(item["coherence"])
                    elif hasattr(item, "coherence") and isinstance(getattr(item, "coherence"), (int, float)):
                        coh_vals.append(item.coherence)

            avg_coh = sum(coh_vals) / len(coh_vals) if coh_vals else 0.0

            # 2) Dissonance
            max_d = 0.0
            if dm is not None:
                size = getattr(dm, "size", None)
                try:
                    if isinstance(size, int) and size > 0 and hasattr(dm, "max"):
                        max_d = float(dm.max())
                    elif isinstance(dm, list) and dm and isinstance(dm[0], list):
                        max_d = max(max(row) for row in dm)
                except Exception as err:
                    logger.error(f"Dissonance max error: {err}")

            # 3) Off-topic ratio
            off_ratio = 0.0
            if hasattr(self.meta, "calculate_off_topic_ratio"):
                try:
                    off_ratio = float(self.meta.calculate_off_topic_ratio(thoughts))
                except Exception as err:
                    logger.error(f"Off-topic calc error: {err}")

            # 4) Запись истории
            self.history.append({
                "impulse": imp,
                "coherence": avg_coh,
                "dissonance": max_d,
                "off_topic_ratio": off_ratio,
                "execution_time": time.perf_counter() - t_imp,
                "timestamp": datetime.utcnow()
            })

        # Если ничего не накопилось
        if not self.history:
            total_time = time.perf_counter() - t_start
            logger.warning("No data collected in optimization")
            return {
                "avg_coherence": 0.0,
                "avg_dissonance": 0.0,
                "avg_off_topic_ratio": 0.0,
                "bridge_threshold": self.meta.bridge_threshold,
                "tournament_depth": self.meta.tournament_depth,
                "execution_time": total_time,
                "history": []
            }

        # Успешные записи
        valid = [h for h in self.history if "coherence" in h]
        avg_coh_all = statistics.mean(h["coherence"] for h in valid) if valid else 0.0
        avg_diss_all = statistics.mean(h["dissonance"] for h in valid) if valid else 0.0
        avg_off_all = statistics.mean(h["off_topic_ratio"] for h in valid) if valid else 0.0

        # Адаптация порогов
        orig_b = self.meta.bridge_threshold
        orig_d = self.meta.tournament_depth

        if avg_coh_all < self.target:
            new_b = orig_b - self._alpha * (self.target - avg_coh_all)
        else:
            new_b = orig_b + self._beta * (avg_coh_all - self.target)
        self.meta.bridge_threshold = max(0.0, min(1.0, new_b))

        if avg_diss_all > self._diss_target:
            new_d = min(orig_d + 1, self._max_depth)
        elif avg_diss_all < self._diss_target:
            new_d = max(orig_d - 1, 1)
        else:
            new_d = orig_d
        self.meta.tournament_depth = new_d

        total_time = time.perf_counter() - t_start
        logger.info(
            "Optimized: bridge %.3f→%.3f, depth %d→%d, coh=%.3f, diss=%.3f, off=%.3f, time=%.3fs",
            orig_b,
            self.meta.bridge_threshold,
            orig_d,
            self.meta.tournament_depth,
            avg_coh_all,
            avg_diss_all,
            avg_off_all,
            total_time
        )

        return {
            "avg_coherence":       avg_coh_all,
            "avg_dissonance":      avg_diss_all,
            "avg_off_topic_ratio": avg_off_all,
            "bridge_threshold":    self.meta.bridge_threshold,
            "tournament_depth":    self.meta.tournament_depth,
            "execution_time":      total_time,
            "history":             self.history,
        }

    def optimize_sync(self, impulses: List[Impulse]) -> Dict[str, Any]:
        """
        Синхронная обёртка для optimize().
        Поддерживает существующий event loop и asyncio.run().
        """
        try:
            loop = asyncio.get_running_loop()
            fut = asyncio.run_coroutine_threadsafe(self.optimize(impulses), loop)
            return fut.result()
        except RuntimeError:
            return asyncio.run(self.optimize(impulses))
