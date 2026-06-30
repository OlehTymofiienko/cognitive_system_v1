#tests\test_context_manager_additional.py

import pytest
import time
from unittest.mock import MagicMock, patch
from core.context_manager import ContextManager

class TestContextManagerAdditional:
    """Дополнительные тесты для увеличения покрытия context_manager.py"""

    def test_add_context_invalid_thought_type(self):
        """Тест обработки невалидного типа мысли (не dict)"""
        manager = ContextManager()
        with pytest.raises(ValueError, match="Thought must be a dict"):
            manager.add_context("invalid_thought")

    def test_add_context_low_trust_score(self):
        """Тест отбрасывания мысли с низким trust_score"""
        manager = ContextManager()
        result = manager.add_context({"trust_score": 0.2, "content": "test", "language": "en"})
        assert result is None

    def test_add_context_non_english(self):
        """Тест отбрасывания неанглоязычного контента"""
        manager = ContextManager()
        result = manager.add_context({"trust_score": 0.8, "content": "тест", "language": "ru"})
        assert result is None

    def test_add_context_invalid_concept(self, monkeypatch):
        """Тест обработки невалидного концепта"""
        monkeypatch.setattr(
            'core.semantic_analyzer.SemanticAnalyzer.extract_core_concept',
            lambda *args, **kwargs: "undefined"
        )
        manager = ContextManager()
        result = manager.add_context({"trust_score": 0.8, "content": "test", "language": "en"})
        assert result is None

    def test_update_current_context_empty(self):
        """Тест обновления контекста при пустом списке active_contexts"""
        manager = ContextManager()
        
        # Мок для extract_core_concept, чтобы возвращать валидный концепт
        with patch.object(manager.semantic_analyzer, 'extract_core_concept', return_value="valid concept"):
            # Пробуем обновить контекст при пустом списке
            manager.update_current_context({
                "trust_score": 0.8,
                "content": "test content",
                "language": "en",
                "core_concept": "valid concept"  # Добавляем обязательные поля
            })
            
            # Проверяем, что контекст был добавлен
            assert len(manager.active_contexts) == 1
            assert manager.active_contexts[0]['core_concept'] == "valid concept"

    def test_update_current_context_invalid(self):
        """Тест обновления контекста невалидными данными"""
        manager = ContextManager()
        manager.add_context({"trust_score": 0.8, "content": "initial", "language": "en"})
        manager.update_current_context({"core_concept": "x"})  # Слишком короткий концепт
        assert manager.active_contexts[-1]['core_concept'] != "x"

    def test_is_valid_concept_edge_cases(self):
        """Тест валидации концептов с краевыми случаями"""
        manager = ContextManager()
        assert not manager.is_valid_concept("")
        assert not manager.is_valid_concept("undefined")
        assert not manager.is_valid_concept("a")
        assert not manager.is_valid_concept("123")
        assert not manager.is_valid_concept("invalid!")
        assert manager.is_valid_concept("valid concept")

    def test_get_current_context_vector_empty(self):
        """Тест получения вектора для пустого контекста"""
        manager = ContextManager()
        assert manager.get_current_context_vector() is None

    def test_remove_expired_contexts_disabled(self):
        """Тест отключенного TTL (когда _ttl = None)"""
        manager = ContextManager(ttl=None)
        manager.add_context({"trust_score": 0.8, "content": "test", "language": "en"})
        manager._remove_expired_contexts()
        assert len(manager.active_contexts) == 1

    def test_remove_expired_contexts_enabled(self):
        """Тест работы TTL очистки"""
        manager = ContextManager(ttl=0.1)  # Очень короткий TTL для теста
        manager.add_context({"trust_score": 0.8, "content": "test", "language": "en"})
        time.sleep(0.2)
        manager._remove_expired_contexts()
        assert len(manager.active_contexts) == 0

    def test_tick_functionality(self):
        """Тест работы метода tick()"""
        manager = ContextManager()
        manager.add_context({"trust_score": 0.8, "content": "test", "language": "en"})
        manager.tick()
        assert len(manager.history) == 1
        assert manager.history[0]["active_contexts"]

    def test_export_state(self):
        """Тест экспорта состояния"""
        manager = ContextManager()
        manager.add_context({"trust_score": 0.8, "content": "test", "language": "en"})
        state = manager.export_state()
        assert "active_contexts" in state
        assert "history" in state

    def test_get_coherence_empty_list(self):
        """Тест когерентности для пустого списка"""
        manager = ContextManager()
        assert manager.get_coherence([]) == 0.0

    def test_get_dissonance_edge_cases(self):
        """Тест диссонанса для краевых случаев"""
        manager = ContextManager()
        assert manager.get_dissonance([]) == 0.0
        assert manager.get_dissonance([[]]) == 0.0
        assert manager.get_dissonance([[0.5]]) == 0.5

    def test_should_apply_context_empty(self):
        """Тест решения о смене контекста при пустом списке"""
        manager = ContextManager()
        assert manager.should_apply_context(0.5) is True

    @patch.object(ContextManager, '_get_time_factor', return_value=0.5)
    def test_should_apply_context_logic(self, mock_time_factor):
        """Тест логики принятия решения о смене контекста"""
        manager = ContextManager()
        manager.add_context({"trust_score": 0.8, "content": "test", "language": "en"})
        
        # Кейс, когда не нужно менять контекст
        assert manager.should_apply_context(0.9) is False
        
        # Кейс, когда нужно менять контекст
        assert manager.should_apply_context(0.2) is True

    def test_handle_user_message_integration(self):
        """Интеграционный тест обработки пользовательского сообщения"""
        with patch('core.orchestra_context.OrchestraContextManager.add_thought') as mock_add:
            mock_add.return_value = "test_voice"
            manager = ContextManager()
            manager.handle_user_message("test message")
            mock_add.assert_called_once_with("test message")

    def test_update_current_context_partial(self):
        """Тест частичного обновления контекста"""
        manager = ContextManager()
        
        # Добавляем начальный контекст с полным набором полей
        initial_context = {
            "trust_score": 0.8,
            "content": "initial content",
            "language": "en",
            "core_concept": "initial concept",
            "timestamp": time.time(),
            "source": "test"
        }
        manager.active_contexts.append(initial_context.copy())
        
        # Обновляем только core_concept
        manager.update_current_context({
            "core_concept": "updated concept"
        })
        
        # Проверяем обновление
        assert manager.active_contexts[-1]['core_concept'] == "updated concept"
        # Проверяем сохранение других полей
        assert manager.active_contexts[-1]['content'] == "initial content"
        assert manager.active_contexts[-1]['trust_score'] == 0.8

    

    