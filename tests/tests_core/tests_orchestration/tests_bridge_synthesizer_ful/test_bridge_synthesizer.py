#tests\bridge_synthesizer_ful\test_bridge_synthesizer.py

from core.models import Thought
from core.orchestration.bridge_synthesizer import BridgeSynthesizer

def test_bridge_synthesizer_generates_bridge():
    topic = "AI Ethics"
    bs = BridgeSynthesizer(topic, threshold=0.5)
    
    # две мысльи с сильным конфликтом
    t1 = Thought("A", "melody", 0.5)
    t2 = Thought("B", "bass",   0.5)
    matrix = [
        [0.0, 0.9],
        [0.9, 0.0]
    ]
    
    bridge = bs.generate([t1, t2], matrix)
    assert "Bridge between" in bridge.content
    assert topic in bridge.content
    assert bridge.coherence >= 0.8
    assert bridge.voice == 'melody'
