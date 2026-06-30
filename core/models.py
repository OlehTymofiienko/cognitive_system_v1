#core\models.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class Thought:
    content: str
    voice: str                # 'melody' | 'counterpoint' | 'bass'
    coherence: float
    metadata: Dict = field(default_factory=dict)
    # откуда эта мысль пришла (для тестов и отладки)
    origin: Optional[str] = None
    #emb: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class Impulse:
    type: str
    intensity: float
    complexity: float
    priority: str = "internal"  # либо 'internal', либо 'external'
