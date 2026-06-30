
from dataclasses import replace
import asyncio
from typing import Union, List
from core.models import Impulse, Thought
from core.orchestration.voice_conductor import VoiceConductor


class HybridProcessingPool:

    def __init__(self, session_topic: str):
        self.session_topic = session_topic
        #self.meta_conductor = MetaConductor(session_topic=session_topic)
        # убрали хранимый экземпляр, будем подгружать MetaConductor по необходимости

    async def process(self, impulse: Impulse) -> Union[Thought, List[Thought]]:
        # 1) Системный приоритет — напрямую к MetaConductor
        if impulse.priority == "system":
            return await self._system_priority(impulse)

        # 2) Dreamlike → фантазийная ветка
        if impulse.type == "dreamlike":
            return await self._fantasy_branch(impulse)

        # 3) Внешняя система
        if impulse.priority == "external":
            return await self._delegate_processing(impulse)

        # 4) Локальный CPU для exploratory с невысокой сложностью
        if impulse.type == "exploratory" and impulse.complexity < 5.0:
            return await self._local_cpu_processing(impulse)

        # 5) Асинхронный параллелизм для интенсивных импульсов
        if impulse.intensity >= 0.7:
            return await self._async_parallel(impulse)

        # 6) Фоллбэк
        return await self._default(impulse)  

    async def _local_cpu_processing(self, impulse: Impulse) -> Thought:
        return Thought(
            content=f"Locally processed {impulse.type} impulse",
            voice="melody",
            coherence=0.6
        )

    async def _async_parallel(self, impulse: Impulse) -> List[Thought]:
        # реальное распараллеливание через VoiceConductor
        voices = ['melody', 'counterpoint', 'bass']
        conductors = [VoiceConductor(v, self.session_topic) for v in voices]
        tasks = [vc.process(impulse) for vc in conductors]
        return await asyncio.gather(*tasks)

    async def _delegate_processing(self, impulse: Impulse) -> Thought:
        return Thought(
            content=f"Delegated {impulse.type} impulse externally",
            voice="bass",
            coherence=0.5
        )

    async def _default(self, impulse: Impulse) -> Thought:
        return Thought(   
            content=f"Default processing of {impulse.type}",
            voice="melody",
            coherence=0.5
        )

    async def _fantasy_branch(self, impulse: Impulse) -> Thought:
        return Thought(
            content=f"A wondrous dream emerges about {self.session_topic}",
            voice="melody",
            coherence=0.65
        )

    async def _system_priority(self, impulse: Impulse) -> Union[Thought, List[Thought]]:
        # локальный импорт, чтобы не было цикла
        from core.orchestration.meta_conductor import MetaConductor

        # клонируем импульс, сбрасываем флаг system → internal
        safe_impulse = replace(impulse, priority="internal")

        mc = MetaConductor(session_topic=self.session_topic,
                           language_model=None)
        result = await mc.orchestrate(safe_impulse)

        # если получили список из одного элемента — возвращаем просто Thought
        if isinstance(result, list) and len(result) == 1:
            return result[0]
        return result