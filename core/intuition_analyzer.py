# Анализ интуитивных прорывов

from typing import Dict

class IntuitionAnalyzer:
    def __init__(self):
        self.deviation_threshold = 0.7
        self.resonance_threshold = 0.6
    
    def analyze(self, thought_graph, current_thought: Dict) -> bool:
        """Определяет, является ли мысль интуитивным прорывом"""
        if len(thought_graph.graph) < 3:
            return False
        
        # Проверяем отклонение от предыдущих мыслей
        last_thoughts = list(thought_graph.graph.nodes(data=True))[-3:]
        deviation_score = self._calculate_deviation(current_thought, last_thoughts)
        
        return deviation_score > self.deviation_threshold
    
    def _calculate_deviation(self, thought, last_thoughts) -> float:
        """Вычисляет степень отклонения от предыдущего контекста"""
        # Заглушка - в реальности используем эмбеддинги
        return sum(1 for t in last_thoughts 
                 if t[1]['thought'] != thought) / len(last_thoughts)