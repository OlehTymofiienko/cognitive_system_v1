#tests\tests_core\test_semantic_analyzer_ful\test_semantic_analyzer_error_handling.py

import pytest
import numpy as np
from unittest.mock import patch
from core.semantic_analyzer import SemanticAnalyzer

def test_embedding_error_handling(caplog):
    analyzer = SemanticAnalyzer()

    # Мокаем model.encode — именно то, что get_embedding вызывает
    with patch.object(analyzer.model, 'encode', side_effect=Exception("Mocked error")):
        emb = analyzer.get_embedding("test")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (768,)  # ← это дефолтный fallback vector
        assert np.all(emb == 0)

        # Проверка логирования
        assert "Mocked error" in caplog.text or "Ошибка получения эмбеддинга" in caplog.text

        print([r.getMessage() for r in caplog.records])
