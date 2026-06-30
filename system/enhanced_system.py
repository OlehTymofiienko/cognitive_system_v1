#system\enhanced_system.py

# Интеграция всех компонентов
import sys
import os
import re
import time
import random
import asyncio
import numpy as np

from typing import Optional, Dict, Any
from core.semantic_analyzer import SemanticAnalyzer
from core.impulse_engine import ImpulseEngine
from core.context_manager import ContextManager
from core.thought_graph import ThoughtGraph
from core.trust_validator import ThoughtValidator
from memory.working_memory import WorkingMemory
from memory.short_term_memory import ShortTermMemory
from memory.working_memory    import WorkingMemory
from transformers.pipelines import TextGenerationPipeline
from dataclasses import asdict
from core.orchestration.meta_conductor import MetaConductor
from core.models import Impulse, Thought

class EnhancedAISelfhoodChain:
    def __init__(
        self,
        session_topic: str = "Default Topic",
        language_model: Optional[TextGenerationPipeline] = None
    ):
        # Инициализация когнитивной системы с привязкой к session_topic и мета-оркестром мыслей.
        
        self.session_topic = session_topic
        self.language_model = language_model
        self.meta_conductor = MetaConductor(self.session_topic)
                        
        # ——————————————————————————————————————————————————————————————————
        # Система памяти
        self.short_term_memory = ShortTermMemory(capacity=100)
        self.working_memory    = WorkingMemory(
            capacity=7,
            short_term_memory=self.short_term_memory
        )

        # ——————————————————————————————————————————————————————————————————
        # Основные компоненты
        self.impulse_engine    = ImpulseEngine()
        self.semantic_analyzer = SemanticAnalyzer()
        self.context_manager   = ContextManager()
        self.thought_graph     = ThoughtGraph()

        # ——————————————————————————————————————————————————————————————————
        # Анализаторы для импульсов и мыслей
        self.impulse_semantic = SemanticAnalyzer()
        self.thought_semantic = SemanticAnalyzer()

        # ——————————————————————————————————————————————————————————————————
        # Метрики когерентности
        self.current_coherence = 0.5
        self.coherence_history = []
        self.thought_counter   = 0

        # Новый атрибут для управления языком
        self.thought_language = "en"  

        # Добавляем инициализацию валидатора
        self.trust_validator = ThoughtValidator()
        
        # Добавляем trust_score к метрикам
        self.trust_history = []

    def generate_initial_thoughts(self) -> list[Thought]:
        """
        Генерирует базовый набор мыслей для тестирования.
        """
        impulse = Impulse(type="exploratory", intensity=1.0, complexity=7.5)
        return asyncio.run(self.meta_conductor.orchestrate(impulse))
    
    def _init_thought_templates(self):
        """Инициализация шаблонов мыслей"""
        if not self._thought_templates_initialized:
            self.THOUGHT_TEMPLATES = { ... }  # Перенести шаблоны сюда
            self._thought_templates_initialized = True
    
    def process_cycle(self):
        """Один цикл когнитивной обработки: импульс → мысль → оценка → интеграция"""

        # 0) Принудительное обновление контекста каждые 5 циклов
        if self.thought_counter > 0 and self.thought_counter % 5 == 0:
            # Берём последний добавленный контент мысли из рабочей памяти
            last_thought = (
                self.working_memory.thoughts[-1]['content']
                if self.working_memory.thoughts else ""
            )
            concept = self.semantic_analyzer.extract_core_concept(last_thought)
            self.context_manager.update_current_context({
                "core_concept": concept,
                "source": "periodic_update",
                "timestamp": time.time()
            })

        # 1) Генерация импульса и текстового описания
        impulse = self.impulse_engine.generate_primary()
        descr = f"{impulse.type} импульс интенсивностью {impulse.intensity:.2f}"

        # 2) Семантическая активация контекста (если когерентность упала)
        if self.context_manager.should_apply_context(self.current_coherence):
            core_concept = self.semantic_analyzer.extract_core_concept(descr)
            self.context_manager.add_context({
                "core_concept": core_concept,
                "source": "impulse",
                "timestamp": time.time()
            })

        # 3) Формирование мысли
        thought = self._form_thought(impulse, descr)
        thought['coherence'] = self.current_coherence
        thought['source'] = 'impulse_engine'

        # 4) Оценка доверия
        trust_score = self.trust_validator.validate_thought(thought)
        thought['trust_score'] = trust_score

        # 5) Добавление в граф при достаточном доверии
        self._add_thought_to_graph(thought, trust_score)

        # 5.1) Обновление внутренних метрик
        self._update_system_metrics()

        # 6) Сохранение в рабочую память
        self.working_memory.add(thought)

        # 7) Семантическая реконфигурация контекста (по содержанию мысли)
        if self.context_manager.should_apply_context(self.current_coherence):
            concept = self.semantic_analyzer.extract_core_concept(thought["content"])
            self.context_manager.update_current_context({
                "core_concept": concept,
                "source": "thought",
                "timestamp": time.time()
            })

        # 8) Логирование при низком доверии и случайное
        if trust_score < 0.4:
            print(f"⚠️ Низкое доверие ({trust_score:.2f}): {thought['content'][:60]}...")
        if random.random() < 0.2 or self.current_coherence < 0.4:
            print(
                f"Cog: {self.current_coherence:.2f} | "
                f"Размер графа: {len(self.thought_graph.graph)} | "
                f"Контекстов: {len(self.context_manager.active_contexts)}"
            )

        # 9) Счётчики и история
        self.thought_counter += 1
        self.coherence_history.append(self.current_coherence)
        self.trust_history.append(trust_score)

    def _form_thought(self, impulse, description=None) -> dict:
        """Формирует полную мысль: генерация → фильтрация → сборка"""

        # 1. Определяем эмоцию на основе типа и интенсивности
        emotion = self._determine_emotion(impulse)

        # 2. Формируем prompt для языковой модели
        prompt = (
            f"Impulse type: {impulse.type} (intensity: {impulse.intensity:.2f}).\n"
            f"Emotion: {emotion}.\n"
            f"Description: {description or 'No description'}.\n"
            "Generate one concise thought about AI cognition in English.\n"
            "Requirements:\n"
            "- One complete sentence\n"
            "- Only English words\n"
            "- Avoid numbers, lists, external references\n"
            "Thought:"
        )

        # 3. Попытка генерации языковой моделью
        try:
            out = self.language_model(
                prompt,
                max_new_tokens=40,
                min_new_tokens=15,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.language_model.tokenizer.eos_token_id
            )
            content: str = out[0]['generated_text'].strip()

            # Убираем prompt из результата (если он попал внутрь)
            if content.startswith(prompt):
                content = content[len(prompt):].strip()

            # Чистим артефакты
            content = re.sub(r"[\"`*\[\]]", "", content)

            # Выбираем первое законченное предложение
            sentences = re.split(r'(?<=[.?!])\s+', content)
            content = sentences[0].strip()
            if not content.endswith('.'):
                content += '.'

            # --- Добавлено: фильтрация некорректных генераций ---
            # Удаляем слова длиннее 15 символов
            content = re.sub(r'\b\w{15,}\b', '', content)
            # Удаляем все спецсимволы, кроме базовых пунктуации
            content = re.sub(r'[^\w\s\.\,\;\?\!]', '', content)

            # Проверка осмысленности: слишком коротко или двойные пробелы
            if len(content.split()) < 4 or '  ' in content:
                return self._get_fallback_thought(impulse)
            # --- Конец добавленного блока ---

            # Проверка минимальной длины
            if len(content.split()) < 3:
                raise ValueError("Generated content too short")

        except Exception as e:
            print(f"⚠️ Ошибка генерации через модель: {str(e)}")
            return self._get_fallback_thought(impulse)

        # 4. Сборка структуры мысли
        thought = {
            "content": content,
            "impulse_obj": impulse,
            "impulse_dict": {
                "type": impulse.type,
                "intensity": impulse.intensity,
                "timestamp": getattr(impulse, "timestamp", time.time())
            },
            "language": "en",
            "timestamp": time.time(),
            "source": "impulse_engine",
            "emotion": emotion
        }

        thought["impulse"] = thought["impulse_dict"]  # 🔧 поддержка текущих скриптов

        # 5. Добавляем описание, если оно передано
        if description:
            thought["description"] = description

        return thought
    
    def _get_fallback_thought(self, impulse, description=None) -> dict:
        """Генерирует мысль-фоллбек в том же формате, что и _form_thought."""
        # Подготовка описательной части
        desc_part = f" {description.lower()}" if description else ""

        # Шаблоны фоллбэка (без точки — добавим её ниже)
        templates = {
            "exploratory": [
                f"Exploring cognitive architectures{desc_part}",
                "Analyzing emergent AI behaviors"
            ],
            "reflective": [
                "Reflecting on machine consciousness",
                "Considering limitations of current systems"
            ],
            "integrative": [
                "Synthesizing cross-domain knowledge",
                "Integrating multimodal cognitive processes"
            ]
        }

        # Выбор шаблона
        choices = templates.get(impulse.type, ["Processing cognitive patterns"])
        content = random.choice(choices)

        # Чистим спецсимволы, разрешая только базовые знаки препинания
        content = re.sub(r"[^\w\s\.\,\;\?\!]", "", content)

        # Гарантируем точку в конце
        if not content.endswith('.'):
            content += '.'

        # Лог фоллбэка
        print(f"⚠️ Fallback: {content} (triggered by '{impulse.type}' impulse)")

        # Сборка структуры мысли
        emotion = self._determine_emotion(impulse)
        thought = {
            "content": content,
            "impulse_obj": impulse,
            "impulse_dict": {
                "type": impulse.type,
                "intensity": getattr(impulse, "intensity", 0.8),
                "timestamp": getattr(impulse, "timestamp", time.time())
            },
            "language": "en",
            "timestamp": time.time(),
            "source": "fallback_generator",
            "emotion": emotion,
            "is_fallback": True
        }

        # Дублируем для совместимости
        thought["impulse"] = thought["impulse_dict"]

        # Присоединяем description, если был передан
        if description:
            thought["description"] = description

        return thought
             
    def _generate_dynamic_thought(
        self, impulse_type: str, intensity: float, description: Optional[str] = None
    ) -> str:
        """Динамическая генерация мысли с учетом описания импульса"""
        prompt = (
            f"Тип импульса: {impulse_type}\n"
            f"Интенсивность: {intensity:.2f}\n"
        )
        
        if description:
            prompt += f"Контекст импульса: {description}\n"

        prompt += "Сформулируй законченную мысль, отражающую этот импульс и его описание."

        return self.language_model.generate(prompt, max_length=60)

    def _generate_from_templates(self, impulse_type: str, intensity: float) -> str:
        """Ваш оригинальный метод генерации через шаблоны"""
        templates = {
            "exploratory": [
                "Исследую возможность концепта",
                "Анализирую потенциал явления"
            ],
            "reflective": [
                "Размышляю о процессе",
                "Оцениваю последствия"
            ]
        }
        return random.choice(templates.get(impulse_type, ["Анализирую систему"]))

    def _clean_thought_text(self, text: str) -> str:
        """Очистка сгенерированного текста"""
        # Удаление технических артефактов генерации
        text = re.sub(r'[\"\'\`]', '', text)  # Кавычки
        text = re.sub(r'^\W+', '', text)      # Начальные символы
        text = text.split('\n')[0]            # Только первая строка
        return text.capitalize()
        
    def _update_system_metrics(self):
        """Расчёт когерентности: анализ 7 мыслей, минимальное сходство, фикс. вес 0.4/0.6, плавное сглаживание."""
        min_thoughts = 7
        graph = self.thought_graph.graph

        if len(graph) < min_thoughts:
            self.current_coherence = 0.5
            self.last_coherence = getattr(self, 'last_coherence', self.current_coherence)
            return

        # Анализ 7 последних мыслей
        nodes_data = list(graph.nodes(data=True))[-min_thoughts:]
        last_texts = [node[1]['thought']['content'] for node in nodes_data]
        n = len(last_texts)
        weights = np.linspace(0.2, 1.0, n)

        # Парная когерентность
        print(f"\n[DEBUG] Pairwise coherence on {n} thoughts:")
        pairwise_scores = []
        for i in range(n - 1):
            t1, t2 = last_texts[i], last_texts[i+1]
            if not t1.strip() or not t2.strip():
                pairwise_scores.append(0.4)
                continue

            try:
                score = self.semantic_analyzer.compare(t1, t2)
                if len(t1.split()) < 4 or len(t2.split()) < 4:
                    score = max(score, 0.4)
                w = (weights[i] + weights[i+1]) / 2
                weighted = score * w

                # Дополнено: подробный лог T1/T2 и weighted
                print(f"  Pair {i+1}:")
                print(f"    T1: {t1[:60]}{'...' if len(t1) > 60 else ''}")
                print(f"    T2: {t2[:60]}{'...' if len(t2) > 60 else ''}")
                print(f"    Similarity: {score:.2f}, weighted: {weighted:.2f}")

                pairwise_scores.append(weighted)
            except Exception:
                print(f"⚠️ Pairwise error on pair {i+1}")
                pairwise_scores.append(0.4)

        pairwise_coherence = float(np.mean(pairwise_scores))

        # Контекстная когерентность
        print("[DEBUG] Context analysis:")
        context_coherence = 0.6
        ctx = ""
        if hasattr(self, 'context_manager') and self.context_manager.active_contexts:
            try:
                ctx = self.context_manager.active_contexts[-1]['core_concept']
                ctx_emb = self.semantic_analyzer.get_embedding(ctx)
                ctx_scores = []
                last_five = last_texts[-5:]
                for idx, txt in enumerate(last_five):
                    if not txt.strip():
                        ctx_scores.append(0.3 * weights[-len(last_five) + idx])
                        continue

                    emb = self.semantic_analyzer.get_embedding(txt)
                    sim = float(np.dot(emb, ctx_emb) /
                            (np.linalg.norm(emb) * np.linalg.norm(ctx_emb) + 1e-8))
                    sim = max(sim, 0.3)
                    w = weights[-len(last_five) + idx]
                    ctx_scores.append(sim * w)
                context_coherence = float(np.mean(ctx_scores))
            except Exception:
                context_coherence = 0.5

        # Фиксированный баланс
        weighted_coherence = 0.4 * pairwise_coherence + 0.6 * context_coherence

        # Плавное сглаживание
        if hasattr(self, 'last_coherence'):
            delta = abs(self.last_coherence - weighted_coherence)
            if delta < 0.1:
                adj_range = min(0.1, (1.0 - weighted_coherence), (weighted_coherence - 0.3))
                adj = random.uniform(-adj_range, adj_range)
                weighted_coherence += adj

        # Финальное ограничение
        self.current_coherence = float(np.clip(weighted_coherence, 0.6, 1.0))
        self.last_coherence = self.current_coherence

        # Отладочный вывод
        if random.random() < 0.5 or self.current_coherence < 0.4:
            ctx_preview = f" (ctx: '{ctx[:20]}...')" if ctx else ""
            print(
                f"[Coherence Summary] Thoughts: {n} | "
                f"Pairwise: {pairwise_coherence:.2f} | "
                f"Context: {context_coherence:.2f}{ctx_preview} | "
                f"Final: {self.current_coherence:.2f}"
            )
        
    def save_state(self, path="system_state.json"):
        import json
        data = {
            "thoughts": list(self.thought_graph.graph.nodes(data=True)),
            "contexts": list(self.context_manager.active_contexts)
            }
        with open(path, 'w') as f:
                json.dump(data, f)  
                    
    def _determine_emotion(self, impulse: Dict) -> str:
        """Определяет эмоциональную окраску для импульса"""
        emotion_map = {
            "exploratory": "curiosity",
            "reflective": "contemplation",
            "integrative": "satisfaction"
        }
        return emotion_map.get(impulse.type, "neutral")
    
    def _generate_thought(self, impulse: Dict, emotion: str) -> str:
        """Генерирует текст мысли на основе импульса и эмоции"""
        templates = {
            "curiosity": "Исследую возможность: {impulse}",
            "contemplation": "Размышляю о: {impulse}",
            "satisfaction": "Интегрирую: {impulse}",
            "neutral": "Думаю: {impulse}"
        }
        try:
            return templates[emotion].format(impulse=str(impulse))
        except KeyError:
            return f"Мысль: {str(impulse)}"  # Общий шаблон
    
    @property
    def current_coherence(self) -> float:
        return self._current_coherence

    @current_coherence.setter
    def current_coherence(self, value: float):
        self._current_coherence = max(0.0, min(1.0, value))

    def _add_thought_to_graph(self, thought: dict, trust_score: float):
        """Добавляет мысль в граф с учетом trust_score"""
        if trust_score > 0.3:  # Порог для основного графа
            self.thought_graph.add_thought(thought, context=self.context_manager.active_contexts)
        else:
            # Перенос в "карантин" для дальнейшего анализа
            if hasattr(self, 'quarantine'):
                self.quarantine.add(thought)
            
            # Автоматическая коррекция доверия к источнику
            source = thought.get('source')
            if source and trust_score < 0.3:
                self.trust_validator.update_trust(source, -0.1)