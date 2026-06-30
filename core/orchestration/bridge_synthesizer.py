#core\orchestration\bridge_synthesizer.py

from typing import Any, List, Optional
from core.models import Thought
import math

class BridgeSynthesizer:
    def __init__(self,
                 session_topic: str,
                 threshold: float = 0.7,
                 language_model: Optional[Any] = None):
         self.session_topic = session_topic
         self.threshold = threshold
         self.language_model = language_model

    def generate(
        self,
        thoughts: List[Thought],
        dissonance_matrix: List[List[float]]
    ) -> Thought:
        """
        1) Находит пару мыслей с максимальным диссонансом.
        2) Генерирует мостовую мысль.
        3) Возвращает Thought c metadata["bridge_of"] и metadata["dissonance"].
        """
        if len(thoughts) < 2:
            raise ValueError("BridgeSynthesizer требует минимум 2 мысли")

        # поиск пары с max диссонансом
        max_d = 0.0
        best_pair = (0, 1)
        for i in range(len(thoughts)):
            for j in range(i+1, len(thoughts)):
                d = dissonance_matrix[i][j]
                if math.isnan(d):
                    d = 0.0
                if d > max_d:
                    max_d, best_pair = d, (i, j)

        i1, i2 = best_pair
        t1, t2 = thoughts[i1], thoughts[i2]

        # составляем prompt
        prompt = (
            f"Свяжи мысли:\n"
            f"1) {t1.content}\n"
            f"2) {t2.content}\n"
            f"в контексте «{self.session_topic}»."
        )

        # генерируем контент
        if self.language_model:
            out = self.language_model(prompt, max_new_tokens=50, do_sample=False)[0]["generated_text"]
            content = out.strip()
            if not content:  # Добавляем проверку на пустую строку
                content = f"Bridge between '{t1.content}' and '{t2.content}' on '{self.session_topic}'"
        else:
            content = f"Bridge between '{t1.content}' and '{t2.content}' on '{self.session_topic}'"

        # собираем metadata
        source_ids = [
            t1.metadata.get("id", i1),
            t2.metadata.get("id", i2)
        ]

        return Thought(
            content=content,
            voice="melody",
            coherence=min(1.0, max_d + 0.2),
            metadata={"bridge_of": source_ids, "dissonance": max_d},
            origin="BridgeSynthesizer"
        )
