# core/orchestra_context.py

import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .orchestra import SimpleOrchestra


class OrchestraContextManager:
    """
    Обёртка над SimpleOrchestra для управления когнитивным ритмом,
    мониторинга когерентности и диссонансов, а также подготовки данных
    для визуализации голосов.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        tick_interval: float = 5.0
    ) -> None:
        """
        Args:
            embed_fn: Функция, возвращающая эмбеддинг текста.
            tick_interval: Интервал (в секундах) для автоматического «тика»—
                           пересчёта метрик и отправки в систему.
        """
        self.orchestra = SimpleOrchestra(embed_fn)
        self.tick_interval = tick_interval
        self._last_tick = time.time()

    def add_thought(self, text: str) -> str:
        """
        Принимает новую мысль, прокидывает в оркестр и возвращает
        имя голоса, в который мысль попала.

        Returns:
            'melody', 'counterpoint' или 'bass'
        """
        voice = self.orchestra.add_thought(text)
        return voice

    def get_coherence(self) -> float:
        """
        Возвращает текущую когерентность (0.0–1.0).
        """
        return self.orchestra.get_coherence()

    def get_dissonance_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Возвращает имена голосов и матрицу их диссонансов.
        """
        return self.orchestra.calculate_dissonance_matrix()

    def tick(self) -> None:
        """
        Периодически пересчитывает метрики и сбрасывает счётчик.
        Вызывать в основном цикле или по таймеру.
        """
        now = time.time()
        if now - self._last_tick < self.tick_interval:
            return 

        coherence = self.get_coherence()
        names, dissonance = self.get_dissonance_matrix()

        # Здесь можно отправить метрики дальше:
        # например, в лог, в внешний монитор, в UI для визуализации
        print(f"[Orchestra Tick] coherence={coherence:.3f}")
        for i, ni in enumerate(names):
            row = ", ".join(f"{d:.2f}" for d in dissonance[i])
            print(f"  {ni}: [{row}]")

        self._last_tick = now

    def export_state(self) -> Dict[str, Any]:
        """
        Подготавливает «снимок» состояния оркестра:
          - список мыслей в каждом голосе с их весами,
          - текущую когерентность,
          - имена голосов и матрицу диссонансов.
        """
        voices_snapshot = {}
        for name, queue in self.orchestra.voices.items():
            voices_snapshot[name] = [
                {"text": t.text, "weight": t.weight()}
                for t in queue
            ]

        coherence = self.get_coherence()
        names, dissonance = self.get_dissonance_matrix()

        return {
            "voices": voices_snapshot,
            "coherence": coherence,
            "dissonance": {
                "names": names,
                "matrix": dissonance.tolist()
            }
        }
