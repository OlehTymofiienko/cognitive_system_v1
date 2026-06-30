import time
import sys
import logging
from typing import Dict
from typing import Optional
from core.context_manager import ContextManager
from dataclasses import asdict


class ThoughtValidator:
    """Оценивает достоверность мыслей на основе источника и контекста"""
    
    def __init__(self):
        self.context_manager: Optional[ContextManager] = None
        # Базовые веса доверия к компонентам системы
        self.trust_weights = {
            'impulse_engine': 0.9,
            'context_manager': 0.7,
            'external_input': 0.5,
            'intuition': 0.8,
            'working_memory': 0.6
        }
        
        # Динамические поправки (обновляются в runtime)
        self.dynamic_weights: Dict[str, float] = {}
    
    def update_trust(self, component: str, delta: float):
        """Корректирует вес доверия к компоненту"""
        if component in self.trust_weights:
            new_weight = max(0.1, min(1.0, self.trust_weights[component] + delta))
            self.dynamic_weights[component] = new_weight
    
    def get_trust(self, component: str) -> float:
        """Возвращает актуальный вес доверия с логированием неизвестных источников"""
        if component not in self.trust_weights and component not in self.dynamic_weights:
            logging.warning(f"Unknown source: {component} - using default trust")
        return self.dynamic_weights.get(
            component,
            self.trust_weights.get(component, 0.5)
        )
        
    def validate_thought(self, thought: dict) -> float:
        """Оценивает достоверность мысли в диапазоне [0.3, 1.0]."""
        # 1. Проверка входа
        if not isinstance(thought, dict):
            logging.warning("Invalid thought type - expected dict")
            return 0.3

        content = thought.get('content', '').strip()
        if not content:
            logging.warning("Empty thought content")
            return 0.3

        # 2. Определение источника
        source = thought.get('source') or (
            'impulse_engine' if 'impulse' in thought else
            'context_manager' if 'context' in thought else
            'external_input'
        )

        raw_trust = self.get_trust(source)
        base_trust = min(0.8, raw_trust + 0.1)

        # 3. Импульс: dataclass или dict
        impulse = thought.get('impulse')
        if impulse:
            if hasattr(impulse, '__dataclass_fields__'):
                intensity = getattr(impulse, 'intensity', 0.5)
            elif isinstance(impulse, dict):
                intensity = impulse.get('intensity', 0.5)
            else:
                intensity = 0.5
        else:
            intensity = 0.5

        intensity = max(0.1, min(1.0, intensity))
        coherence = thought.get('coherence', 0.5)

        # 4. Контекстная проверка
        anomaly_flags = 0
        if self.context_manager and getattr(self.context_manager, 'active_contexts', []):
            last_ctx = self.context_manager.active_contexts[-1].get('core_concept', '')
            if f"not {last_ctx}".lower() in content.lower():
                anomaly_flags += 1
                logging.warning(f"Context contradiction: '{last_ctx}' vs '{content}'")

        # 5. Итоговая формула
        trust_score = base_trust * (0.6 * intensity + 0.4 * coherence)

        # 6. Штраф за аномалии
        if anomaly_flags > 0:
            penalty = max(0.5, 1.0 - 0.1 * anomaly_flags)
            trust_score *= penalty

        return max(0.3, min(1.0, trust_score))

