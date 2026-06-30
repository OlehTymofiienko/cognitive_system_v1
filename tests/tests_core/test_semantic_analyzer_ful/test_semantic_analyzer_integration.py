# tests\tests_core\test_semantic_analyzer_ful\test_semantic_analyzer_integration.py

import pytest
import unittest
import sys
import subprocess
from unittest.mock import patch, MagicMock, call
import numpy as np
from core.semantic_analyzer import SemanticAnalyzer

class TestSemanticAnalyzerErrorHandling(unittest.TestCase):
    """Тесты для обработки ошибок, которые пока не покрыты"""
    
    @patch('core.semantic_analyzer.spacy.load')
    @patch('core.semantic_analyzer.spacy.cli.download')
    @patch('subprocess.run')
    def test_spacy_download_and_subprocess_00(self, mock_run, mock_download, mock_load):
        # Настраиваем моки
        mock_nlp = MagicMock()
        
        # Первый вызов spacy.load - ошибка
        mock_load.side_effect = OSError("Model not found")
        
        # Создаем анализатор - это вызовет конструктор
        with self.assertLogs('core.semantic_analyzer', level='ERROR') as cm:
            analyzer = SemanticAnalyzer()
        
        # Проверяем что модель не была загружена
        self.assertIsNone(analyzer.nlp)
        
        # Проверяем что была одна попытка загрузки
        self.assertEqual(mock_load.call_count, 1)
        
        # Проверяем что не было вызовов download
        mock_download.assert_not_called()
        mock_run.assert_not_called()
        
        # Проверяем сообщение об ошибке в логах
        self.assertTrue(any("Error loading spaCy model" in record.message 
                           for record in cm.records))
        
    @patch('core.semantic_analyzer.spacy.load')
    def test_spacy_initial_load_failure(self, mock_load):
        # Настраиваем моки
        mock_load.side_effect = OSError("Model not found")
        
        # Создаем анализатор
        with self.assertLogs('core.semantic_analyzer', level='ERROR') as cm:
            analyzer = SemanticAnalyzer()
        
        # Проверяем результаты
        self.assertIsNone(analyzer.nlp)
        mock_load.assert_called_once_with('en_core_web_sm')
        self.assertTrue(any("Error loading spaCy model" in record.message 
                         for record in cm.records))

    @patch('core.semantic_analyzer.spacy.load')
    def test_spacy_initial_load_failure(self, mock_load):
        # Имитируем падение при первой загрузке
        mock_load.side_effect = OSError("Model not found")

        # Тестируем с auto_load=True
        with self.assertLogs('core.semantic_analyzer', level='ERROR') as cm:
            analyzer = SemanticAnalyzer(auto_load=True)

        # Проверки
        self.assertIsNone(analyzer.nlp)
        mock_load.assert_called_once_with('en_core_web_sm')
        self.assertTrue(any("Error loading spaCy model" in r.message for r in cm.records))

        @patch('core.semantic_analyzer.spacy.load')
        @patch('core.semantic_analyzer.spacy.cli.download')
        @patch('subprocess.run')
        def test_retry_mechanism_success(self, mock_run, mock_download, mock_load):
            # Отключаем авто-загрузку
            analyzer = SemanticAnalyzer(auto_load=False)
            analyzer.nlp = None  # Состояние: модель не загружена

            # Подготовка моков
            mock_nlp = MagicMock()
            mock_load.side_effect = [
                OSError("Model not found"),
                OSError("Still not found"),
                OSError("Not yet"),
                mock_nlp  # Успешная загрузка на 4-й попытке
            ]
            mock_download.side_effect = Exception("Download failed")  # CLI ломается один раз
            mock_run.return_value = MagicMock(returncode=0)  # subprocess выполняется успешно

            # Вызов метода
            with self.assertLogs('core.semantic_analyzer', level='WARNING') as log_context:
                result = analyzer.try_load_spacy_with_retry(retries=3)

            # Проверки результата
            self.assertTrue(result)
            self.assertEqual(analyzer.nlp, mock_nlp)
            self.assertEqual(mock_load.call_count, 4)  # 3 исключения + успешный вызов

            # CLI и subprocess вызываются по одному разу благодаря флагам
            mock_download.assert_called_once_with('en_core_web_sm')
            mock_run.assert_called_once_with(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Дополнительная проверка логов, если хочешь:
            self.assertTrue(any("CLI download failed" in record.message for record in log_context.records))
            self.assertTrue(any("spaCy model loaded successfully" in record.message for record in log_context.records))

    @patch('core.semantic_analyzer.spacy.load', side_effect=Exception("spaCy error"))
    def test_spacy_load_errors(self, mock_load):
        analyzer = SemanticAnalyzer()
        assert analyzer.nlp is None  # Ожидаем, что модель не загружена
        assert analyzer.extract_core_concept("test") is not None  # Но методы работают
    
    @patch('core.semantic_analyzer.SentenceTransformer')
    def test_sentence_transformer_load_failure(self, mock_st):
        mock_st.side_effect = RuntimeError("Failed to load model")
        analyzer = SemanticAnalyzer()
        assert analyzer.model is None

    @patch.object(SemanticAnalyzer, 'get_embedding')
    def test_compare_embedding_failure(self, mock_get_embedding):
        mock_get_embedding.side_effect = Exception("Embedding failed")
        analyzer = SemanticAnalyzer()
        result = analyzer.compare("text1", "text2")
        assert result == 0.0  # Проверяем fallback-значение при ошибке

    def test_clean_word_exception(self):
        """Тестирование обработки исключений в _clean_word (строки 122-124)"""
        analyzer = SemanticAnalyzer()
        with patch.object(analyzer, '_clean_word', side_effect=Exception("Clean error")):
            result = analyzer.extract_core_concept("test")
            assert result == "test"  # Fallback на исходный текст

    def test_is_url_exception(self):
        """Тестирование обработки исключений в _is_url (строка 188)"""
        analyzer = SemanticAnalyzer()
        with patch('re.match', side_effect=Exception("Regex error")):
            # Проверяем что метод не падает при ошибке
            assert analyzer._is_url("http://test.com") is True

    def test_preprocess_text_exception(self):
        """Тестирование обработки исключений в _preprocess_text (строки 195-197)"""
        analyzer = SemanticAnalyzer()
        with patch.object(analyzer, '_preprocess_text', side_effect=Exception("Preprocess error")):
            with patch.object(analyzer, 'get_embedding') as mock_embed:
                mock_embed.return_value = np.zeros(768)
                result = analyzer.compare("test", "test")
                assert result == 0.0  # Fallback при ошибке

    def test_compare_short_texts_exception(self):
        """Тестирование обработки исключений в _compare_short_texts (строки 201-206)"""
        analyzer = SemanticAnalyzer()
        with patch.object(analyzer, '_compare_short_texts', side_effect=Exception("Compare error")):
            with patch.object(analyzer, 'get_embedding') as mock_embed:
                mock_embed.return_value = np.zeros(768)
                result = analyzer.compare("short", "text")
                assert result == 0.0  # Fallback при ошибке

    def test_unexpected_embedding_type(self):
        """Тестирование обработки неожиданного типа эмбеддинга (строка 241)"""
        analyzer = SemanticAnalyzer()
        with patch.object(analyzer.model, 'encode', return_value="not_an_embedding"):
            emb = analyzer.get_embedding("test")
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (768,)

    