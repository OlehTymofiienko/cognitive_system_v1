#core\orchestra.py

import time
import numpy as np
import sys
import random
import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from itertools import combinations
from typing import Callable, Deque, Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("core.orchestra")  # фиксированное имя
logger.setLevel(logging.ERROR)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def log_error(message: str, exc: Exception = None):
    import sys
    print("LOG_ERROR CALLED")  # диагностический вывод
    print(f"ERROR: {message}", file=sys.stderr)
    if exc:
        print(f"Exception: {str(exc)}", file=sys.stderr)

    # Добавим logging для поддержки caplog в тестах
    if exc:
        logger.error(f"{message} — Exception: {str(exc)}")
    else:
        logger.error(message)


class BaseOrchestra(ABC):
    """Интерфейс музыкальной оркестровки мыслей."""

    @abstractmethod
    def add_thought(self, text: str) -> str:
        """Добавляет новую мысль в оркестр.

        Args:
            text: Текст мысли.

        Returns:
            Название голоса, в который добавилась мысль.
        """
        pass   

    @abstractmethod
    def get_coherence(self) -> float:
        """Возвращает текущую «когнитивную» когерентность оркестра.

        Returns:
            Коэффициент заполнения голосов в диапазоне [0.0, 1.0].
        """
        pass

    @abstractmethod
    def calculate_dissonance_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Вычисляет матрицу диссонансов между голосами: dissonance = 1 − cosine_similarity.
        
        Возвращает:
            - список названий голосов
            - симметричную матрицу диссонансов (np.ndarray размером [n × n])
        """
        # 1. Собираем векторы: name → средневзвешенный эмбеддинг
        voice_names: List[str] = []
        avg_vectors: List[np.ndarray] = []

        for name, bucket in self.voices.items():
            if not bucket:
                continue  # Пропускаем пустые голоса

            weights = np.array([t.weight() for t in bucket], dtype=float)
            embeddings = np.vstack([t.emb for t in bucket])

            avg_emb = np.average(embeddings, axis=0, weights=weights)
            voice_names.append(name)
            avg_vectors.append(avg_emb)

        # 2. Вычисляем матрицу косинусной схожести
        n = len(avg_vectors)
        matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_similarity(
                    avg_vectors[i][np.newaxis],
                    avg_vectors[j][np.newaxis]
                )[0][0]

                # Преобразуем схожесть в диссонанс
                dissonance = 1.0 - sim

                # Защита от NaN
                if math.isnan(dissonance):
                    dissonance = 0.0

                # Симметрично заносим значения
                matrix[i, j] = dissonance
                matrix[j, i] = dissonance

        return voice_names, matrix


class TemporalThought:
    """Помещает мысль с учётом её «возраста» и полураспада."""

    def __init__(self, text: str, emb: np.ndarray, half_life: float = 60.0) -> None:
        """
        Args:
            text: Текст мысли.
            emb: Вектор-эмбеддинг мысли.
            half_life: Время (в сек.) для уменьшения веса вдвое. Должно быть > 0.
        """
        if half_life <= 0:
            raise ValueError(f"half_life must be positive, got {half_life}")
            
        self.text: str = text
        self.emb: np.ndarray = emb
        self.birth_time: float = time.time()
        self.half_life: float = half_life

    def weight(self) -> float:
        """Текущий вес мысли по формуле экспоненциального затухания."""
        age = time.time() - self.birth_time
        if age <= 0:  # Если время не изменилось
            return 1.0  # Полный вес при создании
        return 0.5 ** (age / self.half_life)


class SimpleOrchestra(BaseOrchestra):
    """Минималистичный оркестр из трёх голосов: melody, counterpoint, bass."""

    def __init__(self, embed_fn: Callable[[str], np.ndarray]) -> None:
        """
        Args:
            embed_fn: Функция, возвращающая эмбеддинг текста.
        """
        self.embed_fn = embed_fn
        self.key: np.ndarray = None  # Тональность (эмбеддинг первой мысли)
        self.voices: Dict[str, Deque[TemporalThought]] = {
            'melody': deque(maxlen=4),
            'counterpoint': deque(maxlen=3),
            'bass': deque(maxlen=2)
        }

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        sim = float(np.dot(a, b) / (norm_a * norm_b))
        return sim if not math.isnan(sim) else 0.0  # защита от NaN

    def _calculate_dynamic_threshold(self) -> float:
        """Адаптивный порог по 33-му перцентилю предыдущих сходств."""
        sims: List[float] = []
        for voice in self.voices.values():
            for t in voice:
                sims.append(self._cosine(t.emb, self.key))
        if not sims:
            return 0.7
        return float(np.percentile(sims, 33))

    def add_thought(self, text: str, half_life: float = 60.0) -> str:

        """
        Добавляет мысль в оркестр, распределяя её по голосам на основе семантической близости к ключевой теме.
        
        Голоса распределяются по принципу:
        - melody: наиболее близкие к ключевой теме мысли
        - counterpoint: умеренно близкие мысли
        - bass: наименее близкие мысли

        Args:
            text: Текст мысли. Должен быть непустой строкой.

        Returns:
            Название голоса ('melody', 'counterpoint' или 'bass'), в который была добавлена мысль.

        Raises:
            ValueError: Если text пустой, None или не является строкой
            RuntimeError: Если не удалось получить эмбеддинг текста

        Examples:
            >>> orchestra.add_thought("Важная мысль")
            'melody'
        """
        # Валидация входных данных
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("Text cannot be empty or whitespace")

        try:
            # Получаем эмбеддинг текста
            emb = self.embed_fn(text)
            if emb is None or len(emb) == 0:
                raise RuntimeError("Failed to get text embedding: empty result")
            
            emb = np.array(emb, dtype=np.float32)  # унифицируем тип

            if not np.all(np.isfinite(emb)):
                raise RuntimeError("Embedding contains NaN or Inf values")  
                          
        except Exception as e:
            log_error("Failed to process text", e)  # ← вот здесь
            raise RuntimeError(f"Failed to process text: {str(e)}") from e

        # Инициализация ключевого эмбеддинга при первой мысли
        if self.key is None:
            self.key = emb
            logger.info(f"Initialized orchestra key with first thought: {text[:50]}...")

        # Вычисляем адаптивный порог и схожесть
        try:
            threshold = self._calculate_dynamic_threshold()
            sim = self._cosine(emb, self.key)
                        
            # Распределение по голосам
            if sim >= threshold:
                voice = 'melody'
            elif threshold * 0.6 <= sim < threshold:  # Явно задаем верхнюю границу
                voice = 'counterpoint'
            else:
                voice = 'bass'

            # Отладочная печать после определения voice
            logger.debug(f"Debug: sim={sim:.2f}, threshold={threshold:.2f}, voice={voice}")

            # Создаем временную мысль и добавляем в выбранный голос
            thought = TemporalThought(text, emb, half_life=half_life)
            self.voices[voice].append(thought)
            
            logger.debug(
                f"Added thought to voice '{voice}' (similarity: {sim:.2f}, threshold: {threshold:.2f}): "
                f"{text[:30]}..."
            )
            
            return voice

        except Exception as e:
            print("REACHED EXCEPTION BLOCK")  # временно
            print("DEBUG: Entering log_error()")  # stdout
            print("ERROR: Failed to process text", file=sys.stderr)
            log_error("Failed to process text", e)
            raise RuntimeError(f"Failed to process text: {str(e)}") from e

    def get_coherence(self) -> float:
        """
        Рассчитывает «когнитивную когерентность» 
        как отношение числа занятых слотов к общей емкости.
        """
        filled = sum(len(v) for v in self.voices.values())
        capacity = sum(v.maxlen for v in self.voices.values())
        return filled / capacity if capacity else 0.0

    def calculate_dissonance_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Вычисляет матрицу диссонансов с максимальной защитой от аномалий.
        
        Возвращает:
            Tuple[List[str], np.ndarray]: 
                - Имена голосов, прошедших все проверки
                - Корректную матрицу диссонансов [n x n]
                
        Особенности:
            - Полная проверка входных данных
            - Защита от NaN/Inf на всех этапах
            - Оптимизированные вычисления
            - Детальное логирование проблем
        """
        voice_data = []
        
        for name, thoughts in self.voices.items():
            # Пропускаем пустые голоса
            if not thoughts:
                logger.debug(f"Skipping empty voice: '{name}'")
                continue
                
            try:
                # Проверка и нормализация весов
                weights = np.array([t.weight() for t in thoughts], dtype=np.float64)
                if np.sum(weights) <= 1e-8:
                    logger.debug(f"Skipping voice '{name}' - zero sum weights")
                    continue
                weights /= np.sum(weights)  # Нормализация

                # Безопасное построение матрицы эмбеддингов
                embeddings = []
                for t in thoughts:
                    emb = t.emb.flatten() if t.emb.ndim > 1 else t.emb
                    if not np.isfinite(emb).all():
                        logger.warning(f"Invalid embedding in voice '{name}' - contains Inf/NaN")
                        raise ValueError("Non-finite embedding")
                    embeddings.append(emb)
                    
                embeddings = np.vstack(embeddings)
                
                # Вычисление средневзвешенного с проверками
                avg_emb = np.average(embeddings, axis=0, weights=weights)
                if not np.isfinite(avg_emb).all():
                    logger.warning(f"Skipping voice '{name}' - non-finite average embedding")
                    continue
                    
                # Гарантируем 1D вектор
                avg_emb = avg_emb.flatten()
                voice_data.append((name, avg_emb))
                
            except Exception as e:
                logger.error(f"Error processing voice '{name}': {str(e)}")
                continue

        # Оптимизированное построение матрицы

        # ⏹ Если нет ни одного валидного голоса — возврат пустых данных
        if not voice_data:
            logger.debug("No valid voices to process — returning empty matrix")
            return [], np.zeros((0, 0))

        n = len(voice_data)
        dissonance_matrix = np.zeros((n, n))
        
        # Подготовка данных для vectorized вычислений
        vectors = np.array([v for _, v in voice_data])
        vectors = vectors.reshape(n, -1)  # Гарантируем 2D
        
        try:
            # Векторизованное вычисление схожести
            sim_matrix = cosine_similarity(vectors)
            np.fill_diagonal(sim_matrix, 1.0)  # Исключаем NaN на диагонали
            
            # Преобразование в диссонанс с защитой
            sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
            dissonance_matrix = 1.0 - sim_matrix
            
            # Защита от артефактов вычислений
            dissonance_matrix = np.nan_to_num(dissonance_matrix, nan=1.0, posinf=1.0, neginf=1.0)
            
        except Exception as e:
            logger.error(f"Matrix calculation error: {str(e)}")
            dissonance_matrix = np.ones((n, n))  # Возвращаем max диссонанс при ошибке

        return [name for name, _ in voice_data], dissonance_matrix
