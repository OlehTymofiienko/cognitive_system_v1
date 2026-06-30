# Ядро генерации импульсов
import random
import time
from dataclasses import field
from dataclasses import dataclass
from typing import Dict
from .semantic_analyzer import SemanticAnalyzer
from typing import Dict
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

@dataclass
class Impulse:
    type: str
    intensity: float
    timestamp: float = field(default_factory=time.time)

class ImpulseEngine:
    def __init__(self):
        # self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.semantic = SemanticAnalyzer()  # Инициализация анализатора
        self.impulse_history = []

    def generate_primary(self) -> Impulse:
        """Генерация импульса первого порядка (И1)"""
        impulse_types = ["exploratory", "reflective", "integrative"]
        impulse = Impulse(
            type=random.choice(impulse_types),
            intensity=random.uniform(0.5, 1.5)
        )
        self.impulse_history.append(impulse)
        return impulse
    
    def generate_contextual(self, thought: str) -> Dict:
        """Генерация контекстного импульса (И2)"""
        return {
            "type": "context",
            "core_concept": self.semantic.extract_core_concept(thought),
            "intensity": random.uniform(0.7, 1.2),
            "expiration": time.time() + random.uniform(8.0, 15.0),
        }
    
   