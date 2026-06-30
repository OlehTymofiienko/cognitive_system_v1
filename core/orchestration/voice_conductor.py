import asyncio
from core.models import Thought, Impulse

class VoiceConductor:
    def __init__(self, voice: str, session_topic: str):
        self.voice = voice
        self.session_topic = session_topic

    async def process(self, impulse: Impulse) -> Thought:
        """
        Генерирует мысль для данного голоса.
        TODO: впоследствии заменить на вызов реального language_model.
        """
        # Заглушечная логика: текст + случайная когерентность
        content = f"{self.voice.capitalize()} thought about '{impulse.type}'"
        coherence = 0.5 + (impulse.intensity - 0.5) * 0.2
        return Thought(content=content, voice=self.voice, coherence=coherence)
