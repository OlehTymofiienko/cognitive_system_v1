# core\semantic_analyzer.py   

import spacy    
import sys
import numpy as np
import re
import subprocess
import langdetect 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List
from spacy.cli import download as spacy_download
from typing import Tuple

import logging
logging.basicConfig(level=logging.DEBUG)

class SemanticAnalyzer:
    def __init__(
        self,
        st_model_name: str = 'paraphrase-MiniLM-L6-v2',
        spacy_model: str = 'en_core_web_sm',
        auto_load: bool = True
    ):
        self.spacy_model = spacy_model
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.model = None
        self.embedding_dim = None
        self.nlp = None

        if auto_load:
            # Инициализация SentenceTransformer
            try:
                self.model = SentenceTransformer(st_model_name)
                test_emb = self.get_embedding("test")
                self.embedding_dim = test_emb.shape[0]
                self.logger.info("SentenceTransformer model loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading SentenceTransformer: {e}", exc_info=True)

            # Инициализация spaCy
            try:
                self.nlp = spacy.load(spacy_model)
                self.logger.info("spaCy model loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading spaCy model '{spacy_model}': {e}", exc_info=True)

        self._fallback_keywords: List[str] = [
            "system", "diagnosis", "analysis",
            "state", "process", "impulse"
        ]

    def try_load_spacy_with_retry(self, retries: int = 2) -> bool:
        download_attempted = False
        subprocess_attempted = False

        for attempt in range(retries + 1):
            try:
                self.nlp = spacy.load(self.spacy_model)
                self.logger.info("spaCy model loaded successfully")
                return True
            except OSError as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < retries:
                    if not download_attempted:
                        download_attempted = True
                        try:
                            spacy.cli.download(self.spacy_model)
                        except Exception:
                            self.logger.warning("CLI download failed")
                    
                    if not subprocess_attempted:
                        subprocess_attempted = True
                        try:
                            subprocess.run(
                                [sys.executable, "-m", "spacy", "download", self.spacy_model],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                        except Exception as e:
                            self.logger.error(f"Subprocess download failed: {e}", exc_info=True)

        return False
    
    def detect_language(self, text: str) -> str:
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0
            text = text.strip()
            if len(text) < 3:
                return "unknown"
            lang = detect(text)
            if lang.startswith("zh"):
                return "zh"
            return lang if lang in {"en", "ru", "es"} else "unknown"
        except Exception:
            return "unknown"

    def is_input_safe(self, text: str) -> bool:
        """Простая проверка на HTML/JS-инъекции"""
        text = text.lower()
        blacklist = ["<script>", "</script>", "javascript:", "onerror=", "onload="]
        return not any(tag in text for tag in blacklist)
    
    def analyze(self, text: str) -> dict:
        return {
            "language": self.detect_language(text),
            "is_secure": self.is_input_safe(text)
        }

    def _setup_nlp(self, model_name: str):
        """
        Пытаемся загрузить spaCy-модель. Если её нет —
        скачиваем через CLI и грузим заново.
        """
        try:
            self.nlp = spacy.load(model_name)
            print(f"✅ Loaded spaCy model '{model_name}'")
        except OSError:
            print(f"⚠️ spaCy model '{model_name}' not found, downloading…")
            # 1) через программу spacy.cli
            try:
                spacy_download(model_name)
            except Exception:
                # 2) если spacy.cli не сработал — fallback на subprocess
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    check=True
                )
            # после загрузки пробуем снова
            self.nlp = spacy.load(model_name)
            print(f"✅ spaCy model '{model_name}' downloaded and loaded")
        except Exception as e:
            print(f"❌ Unrecoverable error loading spaCy '{model_name}': {e}")
            self.nlp = None

    def compare(self, text1: str, text2: str) -> float:
        """Улучшенное сравнение с обработкой ошибок и логгированием"""
        if not text1 or not text2:
            self.logger.warning("⚠️ Один из текстов пуст — возвращаю 0.0")
            return 0.0

        try:
            # Предобработка
            text1 = self._preprocess_text(text1)
            text2 = self._preprocess_text(text2)

            # Обработка коротких текстов
            if max(len(text1.split()), len(text2.split())) < 5:
                try:
                    return self._compare_short_texts(text1, text2)
                except Exception as e:
                    self.logger.error(f"Ошибка в _compare_short_texts: {e}", exc_info=True)
                    return 0.0

            # Получение эмбеддингов
            emb1 = self.get_embedding(text1)
            emb2 = self.get_embedding(text2)

            # Проверка размерности
            if emb1.shape[0] != emb2.shape[0]:
                self.logger.warning("⚠️ Несовпадение размерностей эмбеддингов — возвращаю 0.0")
                return 0.0

            # Вычисление косинусного сходства
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(np.clip(similarity, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Ошибка в compare: {e}", exc_info=True)
            return 0.0

    def extract_core_concept(self, text: str) -> str:
        """Извлекает ключевой концепт из текста."""
        try:
            text = (text or "").strip()
            if not text:
                return "undefined"

            # Очистка текста
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r"[^\w\s\-']", ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            words = text.split()
            lower_text = text.lower()
            stopwords = {"the", "a", "an", "of", "and", "to", "in"}

            # Специальная обработка коротких случаев
            if len(words) == 4 and all(len(w) == 1 for w in words):
                return ' '.join(words[:3])
            if len(words) == 2:
                return max(words, key=len)

            # Шаг 1: spaCy путь
            if self.nlp:
                try:
                    doc = self.nlp(text)
                    candidates = []

                    # noun_chunks
                    for chunk in doc.noun_chunks:
                        chunk_text = chunk.text.lower()
                        chunk_words = [w for w in chunk_text.split() if w not in stopwords]
                        if chunk_words:
                            candidates.append(' '.join(chunk_words))

                    # отдельные существительные
                    nouns = [
                        token.text.lower() for token in doc
                        if token.pos_ in {"NOUN", "PROPN"} and len(token.text) >= 3 and token.text.lower() not in stopwords
                    ]
                    candidates.extend(nouns)

                    if candidates:
                        return max(candidates, key=lambda x: len(x.split()))
                except Exception as e:
                    self.logger.error("spaCy processing error", exc_info=True)

            # Шаг 2: fallback ключевые слова
            for kw in self._fallback_keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', lower_text):
                    return kw

            # Шаг 3: ручная обработка
            cleaned_words = [self._clean_word(w) for w in words]
            valid_words = [
                w for w in cleaned_words if w and len(w) >= 3 and not w.isdigit() and w.lower() not in stopwords
            ]

            if valid_words:
                return max(valid_words, key=len)

            numbers = re.findall(r'\d+', text)
            if numbers:
                return numbers[0]

            if cleaned_words:
                return ' '.join(cleaned_words[:3]) if len(cleaned_words) > 3 else ' '.join(cleaned_words)

            return "undefined"

        except Exception as e:
            self.logger.error("Unexpected error in extract_core_concept", exc_info=True)
            return "undefined"
    
    def _is_valid_concept(self, word: str) -> bool:
        word = word.lower().strip()
        if word.startswith(("http://", "https://", "www.")) or word.endswith((".com", ".net", ".org")):
            return False
        if word.isdigit() or len(word) < 5:
            return False
        return True

    def _clean_word(self, word: str) -> str:
        word = word.strip()
        if not word:
            return ""
        cleaned = re.sub(r'[^\w]', '', word)  # удаляет дефис и спецсимволы
        return cleaned.lower()

    def _clean_phrase(self, phrase: str) -> str:
        parts = re.split(r'(\s+)', phrase)  # сохраняем пробелы
        cleaned_parts = []
        for part in parts:
            if part.isspace():
                cleaned_parts.append(part)
            else:
                cleaned = self._clean_word(part)
                if cleaned:
                    cleaned_parts.append(cleaned)
        return ''.join(cleaned_parts).strip()
   
    def _is_url(self, text: str) -> bool:
        """Проверяет, является ли текст URL"""
        text = text.lower().strip()
        # Проверяем основные URL-префиксы
        if any(text.startswith(prefix) for prefix in 
            ("http://", "https://", "www.", "ftp://")):
            return True
        
        # Проверяем доменные суффиксы
        if any(text.endswith(ext) for ext in (".com", ".org", ".net", ".ru")):
            return True
        
        # Проверяем наличие доменной структуры
        if re.match(r"^[a-z0-9-]+\.[a-z]{2,}", text):
            return True
            
        return False
    
    def _preprocess_text(self, text: str) -> str:
        """Очистка текста перед анализом"""
        text = re.sub(r'[^\w\s]', '', text)  # Удаляем пунктуацию
        text = text.lower().strip()
        return text

    def _compare_short_texts(self, text1: str, text2: str) -> float:
        """Специальная обработка для коротких текстов"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Возвращает numpy массив с эмбеддингом текста"""
        if not text or not hasattr(self, 'model'):
            self.logger.warning("⚠️ Текст пустой или модель не инициализирована — возвращаю нулевой вектор")
            return np.zeros(768)

        try:
            embedding = self.model.encode(text, convert_to_tensor=False)

            # Уточняем тип на случай, если возвращается не numpy
            if isinstance(embedding, np.ndarray):
                return embedding.astype(np.float32)

            # Фолбэк: если `encode` вернул что-то странное
            self.logger.warning(f"⚠️ Unexpected embedding type: {type(embedding)} — возвращаю нулевой вектор")
            return np.zeros(768)

        except Exception as e:
            self.logger.error(f"Ошибка получения эмбеддинга: {str(e)}")
            print(f"⚠️ Ошибка получения эмбеддинга: {str(e)}")
            return np.zeros(768)
