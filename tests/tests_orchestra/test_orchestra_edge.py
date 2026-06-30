# tests\tests_core\tests_orchestra\test_orchestra_edge.py

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from core.orchestra import SimpleOrchestra, TemporalThought, log_error

class TestOrchestraEdgeCases:
    def test_handle_empty_input(self):
        """Проверка обработки пустого ввода."""
        embed_fn = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        orchestra = SimpleOrchestra(embed_fn)
        
        with pytest.raises(ValueError) as excinfo:
            orchestra.add_thought("")
        assert "Text cannot be empty or whitespace" in str(excinfo.value)
        
    def test_invalid_input_type(self):
        """Проверка обработки неверного типа ввода."""
        embed_fn = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        orchestra = SimpleOrchestra(embed_fn)
        
        with pytest.raises(ValueError) as excinfo:
            orchestra.add_thought(None)
        assert "Text must be a string" in str(excinfo.value)
        
    def test_embedding_failure(self):
        """Проверка обработки ошибки при получении эмбеддинга."""
        embed_fn = MagicMock(side_effect=RuntimeError("Embedding failed"))
        orchestra = SimpleOrchestra(embed_fn)
        
        with pytest.raises(RuntimeError) as excinfo:
            orchestra.add_thought("test")
        assert "Failed to process text" in str(excinfo.value)
        
    def test_empty_voices_dissonance(self):
        """Проверка расчета диссонанса при пустых голосах."""
        embed_fn = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        orchestra = SimpleOrchestra(embed_fn)
        
        names, matrix = orchestra.calculate_dissonance_matrix()
        assert len(names) == 0
        assert matrix.shape == (0, 0)
        
    def test_log_error_function(self, caplog):
        """Проверка функции логирования ошибок."""
        test_error = ValueError("Test error")
        log_error("Test message", test_error)
        
        assert "Test message" in caplog.text
        assert "Test error" in caplog.text
        
    def test_invalid_half_life(self):
        """Проверка обработки неверного значения half_life."""
        with pytest.raises(ValueError) as excinfo:
            TemporalThought("test", np.array([1.0]), half_life=0)
        assert "half_life must be positive" in str(excinfo.value)
        
    def test_nan_embedding_handling(self):
        """Проверка обработки NaN в эмбеддингах."""
        embed_fn = MagicMock(return_value=np.array([np.nan, 1.0]))
        orchestra = SimpleOrchestra(embed_fn)
        
        with pytest.raises(RuntimeError) as excinfo:
            orchestra.add_thought("test")
        assert "Embedding contains NaN or Inf values" in str(excinfo.value)