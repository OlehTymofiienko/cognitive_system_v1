#!/usr/bin/env python3
# system/orchestra_behavior.py

import logging
from typing import List

logger = logging.getLogger(__name__)

def get_latest_text(state: dict, voice_name: str, fallback: str = "") -> str:
        """
        Возвращает текст последнего элемента голосовой линии voice_name.
        Если голос пуст — возвращает fallback.
        """
        voice_list = state["voices"].get(voice_name, [])
        return voice_list[-1]["text"] if voice_list else fallback

class OrchestraBehavior:
    """
    Поведенческая логика для OrchestraContextManager:
      1) Генерирует контрапункт, если в counterpoint пусто или слабые мысли
      2) Вмешивается при сильном диссонансе между melody и bass
      3) Адаптирует tick_interval по динамике coherence
    """

    def __init__(
        self,
        orchestra_mgr,
        semantic_analyzer,
        counterpoint_threshold: float = 0.3,
        dissonance_threshold: float = 0.75,
        coh_window: int = 5,
        low_coh_delta: float = 0.05,
        high_coh_delta: float = 0.2,
        min_tick: float = 1.0,
        max_tick: float = 10.0,
    ):
        self.orchestra = orchestra_mgr
        self.semantic = semantic_analyzer
        self.cp_th = counterpoint_threshold
        self.dis_th = dissonance_threshold
        self.coh_window = coh_window
        self.low_delta = low_coh_delta
        self.high_delta = high_coh_delta
        self.min_tick = min_tick
        self.max_tick = max_tick

    def apply(self, state: dict, coherence_history: List[float]):
        """
        Применяет три правила поведения к текущему состоянию orchestра:
          5.1 — слабый counterpoint
          5.2 — высокая диссонансность melody–bass
          5.3 — адаптивный tick_interval по coherence_history
        """
        self._handle_counterpoint(state)
        self._handle_dissonance(state)
        self._adapt_tick_interval(coherence_history)

    def _handle_counterpoint(self, state: dict):
        """
        Обрабатывает слабый контрапункт — добавляет вопрос, если веса низкие.
        """
        cp_list = state["voices"].get("counterpoint", [])
        weights = [t["weight"] for t in cp_list]

        if not weights or max(weights) < self.cp_th:
            mel  = get_latest_text(state, "melody", "a melodic intention")
            bass = get_latest_text(state, "bass", "a foundational silence")

            prompt = f"What tension exists between “{mel}” and “{bass}”?"

            voice = self.orchestra.add_thought(prompt)
            logger.info(f"[Behavior] Weak counterpoint ⇒ added question to '{voice}'")

    def _handle_dissonance(self, state: dict):
        names = state["dissonance"]["names"]
        mat   = state["dissonance"]["matrix"]
        try:
            i_mel = names.index("melody")
            i_bas = names.index("bass")
        except ValueError:
            return

        dis = mat[i_mel][i_bas]
        if dis > self.dis_th:
            # предлагаем мостик между melody и bass
            mel   = state["voices"]["melody"][-1]["text"]
            bass  = state["voices"]["bass"][-1]["text"]
            prompt = (
                f"What idea can reconcile “{mel}” with “{bass}”?"
            )
            voice = self.orchestra.add_thought(prompt)
            # при сильном диссонансе замедляем ход тиков
            old_t = self.orchestra.tick_interval
            new_t = min(self.max_tick, old_t + 1.0)
            self.orchestra.tick_interval = new_t
            logger.info(
                f"[Behavior] High dissonance ({dis:.2f}) ⇒ added to '{voice}', "
                f"tick_interval {old_t:.1f}→{new_t:.1f}"
            )

    def _adapt_tick_interval(self, coherence_history: List[float]):
        if len(coherence_history) < self.coh_window:
            return

        window = coherence_history[-self.coh_window :]  
        delta = max(window) - min(window)

        old_t = self.orchestra.tick_interval
        # если coherence мало меняется — ускоряем ход (уменьшаем интервал)
        if delta < self.low_delta:
            new_t = max(self.min_tick, old_t - 1.0)
            self.orchestra.tick_interval = new_t
            logger.info(
                f"[Behavior] Low coherence Δ={delta:.3f} ⇒ tick_interval {old_t:.1f}→{new_t:.1f}"
            )
        # если coherence резко растёт — замедляем ход (увеличиваем интервал)
        elif delta > self.high_delta:
            new_t = min(self.max_tick, old_t + 1.0)
            self.orchestra.tick_interval = new_t
            logger.info(           
                f"[Behavior] High coherence Δ={delta:.3f} ⇒ tick_interval {old_t:.1f}→{new_t:.1f}"
            )
