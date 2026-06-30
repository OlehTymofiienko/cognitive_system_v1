import pytest
from core.models import Impulse
from core.orchestration.voice_conductor import VoiceConductor

@pytest.mark.asyncio
async def test_voice_conductor_process():
    session_topic = "AI Cohesion"
    for voice in ['melody', 'counterpoint', 'bass']:
        vc = VoiceConductor(voice, session_topic)
        impulse = Impulse(type="exploratory", intensity=0.6, complexity=2.0)
        thought = await vc.process(impulse)

        # Проверяем, что мысль соответствует голосу
        assert thought.voice == voice
        assert voice.capitalize() in thought.content
        # Когерентность в диапазоне [0.3, 0.7]
        assert 0.3 <= thought.coherence <= 0.7
