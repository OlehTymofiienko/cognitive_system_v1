import asyncio
from typing import Tuple, Dict, Any, List
from core.models import Thought, Impulse
from core.orchestration.meta_conductor import MetaConductor
from core.orchestration.voice_line_manager import VoiceLineManager

class SessionManager:
    def __init__(self, session_topic: str, language_model=None):
        self.session_topic = session_topic
        self.meta = MetaConductor(session_topic, language_model)

    async def run(self, impulse: Impulse) -> Tuple[str, Dict[str, Any], List[Thought]]:
        # Оркестрация
        thoughts: List[Thought] = await self.meta.orchestrate(impulse)

        # Построение линии мыслей
        manager = VoiceLineManager(self.session_topic)
        for t in thoughts:
            manager.add_thought(t)

        script = manager.get_script()
        data = manager.export_json()
        # manager.draw(...)  # при необходимости

        return script, data, thoughts

    def run_sync(self, impulse: Impulse) -> Tuple[str, Dict[str, Any], List[Thought]]:
        return asyncio.run(self.run(impulse))
