#tests\tests_core\test_semantic_analyzer_ful\test_semantic_analyzer_end.py

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from core.semantic_analyzer import SemanticAnalyzer


class TestSemanticAnalyzerEdgeCases:
    @patch('sentence_transformers.SentenceTransformer')
    @patch('spacy.load')
    def test_model_loading_errors(self, mock_spacy, mock_st):
        """Тестирование обработки ошибок при загрузке моделей (строки 24-26)"""
        # Тест с ошибкой загрузки SentenceTransformer
        mock_st.side_effect = Exception("ST load error")
        with pytest.raises(Exception):
            analyzer = SemanticAnalyzer()
            assert analyzer.model is None

        # Тест с ошибкой загрузки spaCy
        mock_spacy.side_effect = Exception("spaCy load error")
        analyzer = SemanticAnalyzer()
        assert analyzer.nlp is None

    def test_detect_language_edge_cases(self):
        """Тестирование крайних случаев detect_language() (строка 69)"""
        analyzer = SemanticAnalyzer()
        
        # Слишком короткий текст
        assert analyzer.detect_language("a") == "unknown"
        
        # Имитация ошибки langdetect
        with patch('langdetect.detect', side_effect=Exception("Test error")):
            assert analyzer.detect_language("test") == "unknown"

    @patch('core.semantic_analyzer.spacy.load')
    def test_extract_core_concept_fallbacks(self, mock_spacy):
        """Тестирование всех вариантов извлечения концепта (строки 79-98, 118-120)"""
        mock_spacy.side_effect = Exception("spaCy error")
        analyzer = SemanticAnalyzer()
        analyzer.logger = MagicMock()  # Добавляем mock logger
        
        # 1) Пустой текст
        assert analyzer.extract_core_concept("") == "undefined"
        
        # 2) Только с fallback keywords
        assert analyzer.extract_core_concept("system error") == "system"
        
        # 3) Самое длинное слово
        long_word = "verylongword"  # Слово длиннее 5 символов
        assert analyzer.extract_core_concept(f"{long_word} here") == long_word.lower()
        
        # 4) Финальный фолбэк
        assert analyzer.extract_core_concept("!!!") == "undefined"
        
        # 5) Ошибка в spaCy обработке
        analyzer.nlp = MagicMock()
        analyzer.nlp.side_effect = Exception("NLP error")
        # Для короткого слова "test" должен сработать fallback на самое длинное слово
        assert analyzer.extract_core_concept(f"{long_word} here") == long_word.lower()

    def test_is_url_edge_cases(self):
        """Тестирование крайних случаев _is_url() (строка 158)"""
        analyzer = SemanticAnalyzer()
        analyzer.logger = MagicMock()
        
        # Нестандартные URL
        assert analyzer._is_url("example.com") is True
        assert analyzer._is_url("sub.domain.org") is True
        assert analyzer._is_url("not.a.url") is False
        
        # Ошибка в регулярном выражении
        with patch('re.match', side_effect=Exception("Regex error")):
            assert analyzer._is_url("http://test.com") is True

    def test_preprocess_text_empty(self):
        """Тестирование _preprocess_text с пустым вводом (строка 193)"""
        analyzer = SemanticAnalyzer()
        analyzer.logger = MagicMock()
        assert analyzer._preprocess_text("") == ""

    @patch('core.semantic_analyzer.SentenceTransformer')
    @patch('core.semantic_analyzer.logging.getLogger')
    def test_get_embedding_errors(self, mock_get_logger, mock_st):
        """Тестирование обработки ошибок в get_embedding() (строки 229-230)"""
        # Настраиваем моки
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        # Создаем analyzer, обходя конструктор
        analyzer = SemanticAnalyzer.__new__(SemanticAnalyzer)
        analyzer.model = mock_model
        analyzer.logger = mock_logger
        analyzer._fallback_keywords = []  # Инициализируем необходимые атрибуты
        
        # 1) Тест с пустым текстом
        result = analyzer.get_embedding("")
        assert result.shape == (768,)
        mock_logger.warning.assert_called_with(
            "⚠️ Текст пустой или модель не инициализирована — возвращаю нулевой вектор"
        )
        
        # 2) Тест с ошибкой кодирования
        mock_model.encode.side_effect = Exception("Encoding error")
        result = analyzer.get_embedding("test")
        assert result.shape == (768,)
        mock_logger.error.assert_called_with(
            "Ошибка получения эмбеддинга: Encoding error"
        )
        
        # 3) Тест с неожиданным типом возвращаемого значения
        mock_model.encode.side_effect = None
        mock_model.encode.return_value = "not_an_embedding"
        result = analyzer.get_embedding("test")
        assert result.shape == (768,)
        mock_logger.warning.assert_called_with(
            "⚠️ Unexpected embedding type: <class 'str'> — возвращаю нулевой вектор"
        )

    def test_compare_short_texts_special_cases(self):
        """Дополнительные тесты для _compare_short_texts()"""
        analyzer = SemanticAnalyzer()
        analyzer.logger = MagicMock()
        
        # Оба текста пустые
        assert analyzer._compare_short_texts("", "") == 0.0
        
        # Нет пересечений
        assert analyzer._compare_short_texts("a b c", "d e f") == 0.0
        
        # Полное совпадение
        assert analyzer._compare_short_texts("a b c", "a b c") == 1.0

    def test_initialization_errors(self):
        """Тестирование обработки ошибок инициализации"""
        with patch('core.semantic_analyzer.SentenceTransformer', side_effect=Exception("Load error")):
            analyzer = SemanticAnalyzer()
            assert analyzer.model is None

        with patch('core.semantic_analyzer.spacy.load', side_effect=Exception("spaCy error")):
            analyzer = SemanticAnalyzer()
            assert analyzer.nlp is None

    def test_text_cleaning_methods(self):
        """Тестирование вспомогательных методов очистки текста"""
        analyzer = SemanticAnalyzer()
        assert analyzer._clean_word(" Test! ") == "test"
        assert analyzer._clean_phrase("Hello - World") == "hello  world"
        assert analyzer._is_url("https://example.com") is True

    def test_compare_short_texts(self):
        analyzer = SemanticAnalyzer()
        assert analyzer.compare("", "") == 0.0
        assert 0.0 < analyzer.compare("cat", "cat dog") < 1.0

