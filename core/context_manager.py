# core/context_manager.py

import time
import torch
import random
from collections import deque
from typing import Deque, Dict, Any, Optional

from core.semantic_analyzer import SemanticAnalyzer
from core.orchestra_context import OrchestraContextManager


class ContextManager:
    def __init__(self, max_contexts: int = 5, tick_interval: float = 10.0, ttl: Optional[float] = None):
        """
        Улучшенный менеджер контекстов с:
        - языковой валидацией
        - временным весом контекстов
        - автоматической очисткой
        - интеграцией SimpleOrchestra
        """
        # 1) Ваш SemanticAnalyzer
        self.semantic_analyzer = SemanticAnalyzer()

        # 2) Инициализируем OrchestraContextManager
        #    передаём сюда метод get_embedding
        self.orchestra_mgr = OrchestraContextManager(
            embed_fn=self.semantic_analyzer.get_embedding,
            tick_interval=tick_interval
        )

        # остальной конструктор…
        self.active_contexts: Deque[Dict[str, Any]] = deque(maxlen=max_contexts)
        self.context_switch_threshold = 0.65
        self.min_concept_length = 5
        self._ttl = ttl  # если None — очистка по maxlen, если задан — по времени

        # история снимков состояния контекстов
        self.history: list = []

    def handle_user_message(self, text: str) -> None:
        # прокидываем мысль в оркестр
        voice = self.orchestra_mgr.add_thought(text)
        print(f"→ мысль попала в голос «{voice}»")
        # далее ваша логика…

    def main_loop(self):
        while True:
            self.orchestra_mgr.tick()
            time.sleep(1.0)

    def should_apply_context(self, current_coherence: float) -> bool:
        """Умное решение о смене контекста с учетом когерентности"""
        if not self.active_contexts:
            return True

        # Динамические коэффициенты
        coherence_factor = 1.0 - current_coherence  # Чем ниже когерентность, тем чаще смена
        novelty_factor = random.uniform(0.2, 0.8)   # Элемент случайности
        time_factor = self._get_time_factor()       # Учет времени жизни контекста

        return (coherence_factor + novelty_factor + time_factor) > 1.5

    def _get_time_factor(self) -> float:
        """Возвращает коэффициент 'устаревания' контекста"""
        if not self.active_contexts:
            return 0.0

        last_ctx_age = time.time() - self.active_contexts[-1].get('timestamp', 0)
        return min(1.0, last_ctx_age / 300)  # Нормализуем до 1.0 после 5 минут

    def add_context(self, thought: Dict[str, Any]) -> Optional[str]:
        """Добавляет контекст с проверкой доверия и структуры мысли"""
        # валидируем структуру мысли
        if not isinstance(thought, dict):
            raise ValueError("Thought must be a dict")

        trust_score = thought.get('trust_score', 0.5)
        if trust_score < 0.3:  # Не добавляем низкодостоверные контексты
            return None

        if thought.get('language') != 'en':
            return None

        concept = self.semantic_analyzer.extract_core_concept(thought['content'])
        if concept and concept != "undefined":
            # Очистка устаревших контекстов перед добавлением
            if len(self.active_contexts) >= self.active_contexts.maxlen:
                self._remove_oldest_context()

            new_ctx = {
                "core_concept": concept.lower(),
                "timestamp": time.time(),
                "source": thought.get('impulse', {}).get('type', 'unknown')
            }
            self.active_contexts.append(new_ctx)
            return concept

        return None

    def _remove_oldest_context(self) -> None:
        """Удаляет самый старый контекст с проверкой"""
        if self.active_contexts:
            self.active_contexts.popleft()

    def update_current_context(self, new_data: Dict[str, Any]) -> None:
        """
        Обновляет текущий контекст с валидацией:
        - Не позволяет "испортить" хороший контекст
        - Сохраняет временную метку
        """
        if not self.active_contexts:
            thought = {
                'trust_score': new_data.get('trust_score', 0.85),
                'weight': new_data.get('weight', 0.9),
                'coherence': new_data.get('coherence', 1.0),
                'intensity': new_data.get('intensity', 1.1),
                'content': new_data.get('content', ''),
                'language': new_data.get('language', 'en'),
                **new_data
            }
            self.add_context(thought)
            return

        current = self.active_contexts[-1]
        
        # Обновление core_concept при наличии и допустимой длине
        if (
            'core_concept' in new_data
            and len(new_data['core_concept'].split()) >= 1
            and len(new_data['core_concept']) >= self.min_concept_length
        ):
            current['core_concept'] = new_data['core_concept'].lower()

        # Обновление остальных полей (без потери существующих)
        for key, value in new_data.items():
            if key != 'core_concept':
                current[key] = value

        # Сохраняем временную метку
        current['timestamp'] = time.time()

    def is_valid_concept(self, concept: str) -> bool:
        """Проверяет, что концепт подходит для контекста"""
        return (
            concept
            and concept != "undefined"
            and len(concept.split()) >= 1
            and len(concept) >= self.min_concept_length
            and all(c.isalpha() or c.isspace() for c in concept)
        )

    def get_current_context_vector(self) -> Optional[torch.Tensor]:
        """Возвращает эмбеддинг текущего контекста"""
        if not self.active_contexts:
            return None
        return self.semantic_analyzer.get_embedding(
            self.active_contexts[-1]['core_concept']
        )
    
    def _remove_expired_contexts(self) -> None:
        """Удаляет контексты, устаревшие по TTL"""
        if self._ttl is None:
            return

        now = time.time()
        self.active_contexts = deque(
            [ctx for ctx in self.active_contexts if now - ctx["timestamp"] <= self._ttl],
            maxlen=self.active_contexts.maxlen
        )

    def tick(self) -> None:
        """Сохраняет снимок текущего состояния и очищает устаревшие контексты"""
        self._remove_expired_contexts()
        snapshot = {
            "active_contexts": list(self.active_contexts),
            "timestamp": time.time()
        }
        self.history.append(snapshot)

    def export_state(self) -> Dict[str, Any]:
        """Экспортирует состояние менеджера контекстов"""
        return {
            "active_contexts": list(self.active_contexts),
            "history": self.history
        }

    def get_coherence(self, contexts: list) -> float:
        """Средняя когерентность списка контекстов"""
        if not contexts:
            return 0.0
        total = sum(c.get("coherence", 0.0) for c in contexts)
        return total / len(contexts)

    def get_dissonance(self, matrix: list) -> float:
        """Максимальная диссонансная пара в матрице"""
        if not matrix:
            return 0.0
        rows = [row for row in matrix if row]
        if not rows:
            return 0.0
        return max(max(row) for row in rows)
