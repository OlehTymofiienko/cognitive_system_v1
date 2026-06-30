# core/orchestration/meta_conductor.py
from __future__ import annotations
import asyncio
from typing import List, Optional
from transformers import TextGenerationPipeline

from typing import List, Optional, Union

from core.models import Thought, Impulse
from core.orchestration.voice_conductor import VoiceConductor
from core.orchestration.utils import calculate_dissonance_matrix
from core.orchestration.bridge_synthesizer import BridgeSynthesizer
from core.processing.hybrid_processing_pool import HybridProcessingPool
from core.orchestration.recursion_cache import RecursionCache

import random

class MetaConductor:
    def __init__(
        self,
        session_topic: str,
        language_model: Optional[TextGenerationPipeline] = None
    ):
        self.session_topic = session_topic
        self.bridge_threshold = 0.7
        self.language_model = language_model

        self.hybrid_pool = HybridProcessingPool(self.session_topic)
        self.tournament_depth = 3  # настраиваемый порог

    def calculate_off_topic_ratio(self, thoughts: List[Thought]) -> float:
        """Расчет процента off-topic мыслей (заглушка для тестов)"""
        if not thoughts:
            return 0.0
        
        # Реальная имплементация будет использовать semantic-анализ
        # Для тестов возвращаем случайное значение
        return random.uniform(0.1, 0.3)  # Заглушка

    async def _simple_processing(self, impulse: Impulse) -> Union[Thought, List[Thought]]:
        return await self.hybrid_pool.process(impulse)
        
    async def orchestrate(self, impulse: Impulse) -> List[Thought]:
        if impulse.complexity > 5.0:
            return await self._parallel_orchestration(impulse)
        return [await self._simple_processing(impulse)]
    
    @RecursionCache()

    async def _parallel_orchestration(self, impulse: Impulse) -> List[Thought]:
        """
        Параллельная оркестрация:
        1) Запуск трёх VoiceConductor в async-параллели.
        2) Расчёт матрицы диссонансов и генерация bridge-thought, если threshold пройден.
        3) Запуск cognitive_tournament при превышении tournament_depth.
        """
        # 1) Генерация мыслей от каждого голоса
        voices = ['melody', 'counterpoint', 'bass']
        conductors = [VoiceConductor(v, self.session_topic) for v in voices]
        thoughts = await asyncio.gather(*(vc.process(impulse) for vc in conductors))

        # 2) Расчёт диссонанса
        from core.orchestration.utils import calculate_dissonance_matrix
        names, dm = calculate_dissonance_matrix(thoughts)
        max_d = max(max(row) for row in dm) if dm.size else 0.0

        # 3) BridgeSynthesizer при превышении порога
        if max_d >= self.bridge_threshold:
            from core.orchestration.bridge_synthesizer import BridgeSynthesizer
            bs = BridgeSynthesizer(
                session_topic=self.session_topic,
                language_model=self.language_model
            )
            thoughts.append(bs.generate(thoughts, dm))

        # 4) CognitiveTournament, если число мыслей > tournament_depth
        if len(thoughts) > self.tournament_depth:
            from core.orchestration.cognitive_tournament import CognitiveTournament
            tournament = CognitiveTournament(
                session_topic=self.session_topic,
                depth_threshold=self.tournament_depth,
                language_model=self.language_model
            )
            thoughts = tournament.run(thoughts, depth=len(thoughts))

        return thoughts

    async def _simple_processing(self, impulse: Impulse) -> Thought:
        return await self.hybrid_pool.process(impulse)
