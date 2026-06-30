#tests\tests_enhanced_system_ful\test_cognition.py

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
from system.enhanced_system import EnhancedAISelfhoodChain
from core.models import Thought, Impulse

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

@pytest.fixture
def mock_system():
    with patch('system.enhanced_system.SemanticAnalyzer') as mock_semantic, \
         patch('system.enhanced_system.TextGenerationPipeline') as mock_lm, \
         patch('system.enhanced_system.MetaConductor') as mock_meta_conductor, \
         patch('system.enhanced_system.ContextManager') as mock_context_manager, \
         patch('system.enhanced_system.ThoughtValidator') as mock_validator:
        
        # SemanticAnalyzer mock
        mock_semantic_instance = MagicMock()
        mock_semantic_instance.compare.return_value = 0.8
        mock_semantic_instance.get_embedding.return_value = np.random.rand(512)
        mock_semantic_instance.extract_core_concept.return_value = "test_concept"
        mock_semantic.return_value = mock_semantic_instance

        # TextGenerationPipeline mock
        mock_lm_instance = MagicMock()
        mock_lm_instance.tokenizer = MagicMock()
        mock_lm_instance.return_value = [{"generated_text": "Generated test thought content."}]
        mock_lm.return_value = mock_lm_instance

        # MetaConductor mock
        mock_meta_conductor_instance = MagicMock()
        mock_meta_conductor_instance.orchestrate.return_value = [
            Thought(content="Test thought", voice="melody", coherence=0.7)
        ]
        mock_meta_conductor.return_value = mock_meta_conductor_instance

        # ContextManager mock
        mock_context_manager_instance = MagicMock()
        mock_context_manager.return_value = mock_context_manager_instance

        # ThoughtValidator mock
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_thought.return_value = 0.8  # Высокое доверие по умолчанию
        mock_validator.return_value = mock_validator_instance

        # Создание и настройка системы
        system = EnhancedAISelfhoodChain(session_topic="Test Topic")
        system.language_model = mock_lm_instance
        system.semantic_analyzer = mock_semantic_instance
        system.impulse_semantic = mock_semantic_instance
        system.thought_semantic = mock_semantic_instance
        system.meta_conductor = mock_meta_conductor_instance
        system.context_manager = mock_context_manager_instance
        system.trust_validator = mock_validator_instance

        yield system

@pytest.fixture
def test_impulse():
    """Фикстура с тестовым импульсом"""
    return Impulse(type="exploratory", intensity=0.9, complexity=0.7)

def test_form_thought_generation(mock_system, test_impulse):
    thought = mock_system._form_thought(test_impulse, "Test context")
    
    assert isinstance(thought, dict)
    assert thought["content"] == "Generated test thought content."  # ✅
    assert thought["emotion"] == "curiosity"
    assert "impulse" in thought
    assert thought["language"] == "en"

def test_fallback_thought_generation(mock_system, test_impulse):
    """Тест генерации фолбэк-мысли"""
    # Эмулируем ошибку генерации
    mock_system.language_model = None
    
    fallback = mock_system._get_fallback_thought(test_impulse)
    
    assert isinstance(fallback, dict)
    assert "content" in fallback
    assert fallback["is_fallback"] is True
    assert fallback["source"] == "fallback_generator"
    assert fallback["content"].endswith('.')

def test_process_cycle(mock_system):
    """Тест полного цикла обработки"""
    # Добавляем тестовые мысли для расчета когерентности
    for i in range(7):
        mock_system.thought_graph.add_thought({
            "content": f"Test thought {i}",
            "source": "test"
        })
    
    initial_coherence = mock_system.current_coherence
    mock_system.process_cycle()
    
    # Проверяем что метрики обновились
    assert mock_system.current_coherence != initial_coherence
    assert len(mock_system.coherence_history) == 1
    assert len(mock_system.trust_history) == 1
    
    # Проверяем что мысль добавилась в граф
    assert len(mock_system.thought_graph.graph) > 0

def test_trust_validation(mock_system):
    """Тест валидации доверия к мыслям"""
    test_thought = {
        "content": "Test thought content",
        "source": "test_source",
        "impulse": {"type": "exploratory", "intensity": 0.8}
    }
    
    # Добавляем мысль с низким trust_score
    mock_system._add_thought_to_graph(test_thought, trust_score=0.2)
    assert len(mock_system.thought_graph.graph) == 0  # Не должна добавиться
    
    # Добавляем мысль с высоким trust_score
    mock_system._add_thought_to_graph(test_thought, trust_score=0.8)
    assert len(mock_system.thought_graph.graph) == 1  # Должна добавиться

def test_system_metrics_update(mock_system):
    """Тест обновления метрик системы"""
    # Добавляем тестовые мысли
    for i in range(10):
        thought = {"content": f"Test thought {i}", "source": "test"}
        mock_system.thought_graph.add_thought(thought)
    
    mock_system._update_system_metrics()
    
    # Проверяем что когерентность в допустимых пределах
    assert 0.6 <= mock_system.current_coherence <= 1.0
    assert mock_system.last_coherence == mock_system.current_coherence

def test_state_persistence(mock_system, tmp_path):
    """Тест сохранения и загрузки состояния"""
    # Добавляем тестовые данные
    test_thought = {"content": "Test state thought", "source": "test"}
    mock_system.thought_graph.add_thought(test_thought)
    
    # Сохраняем состояние
    test_file = tmp_path / "state.json"
    mock_system.save_state(test_file)
    
    # Проверяем что файл создан
    assert test_file.exists()
    assert test_file.stat().st_size > 0

def test_form_thought_generation_error(mock_system, test_impulse):
    """Тест обработки ошибки генерации мысли"""
    mock_system.language_model.side_effect = Exception("Model error")
    thought = mock_system._form_thought(test_impulse, "Test context")
    
    assert thought["is_fallback"] is True
    assert "fallback" in thought["source"]

@pytest.mark.parametrize("impulse_type,expected_emotion", [
    ("exploratory", "curiosity"),
    ("reflective", "contemplation"), 
    ("integrative", "satisfaction"),
    ("unknown", "neutral")
])
def test_determine_emotion(mock_system, impulse_type, expected_emotion):
    """Тест определения эмоции для разных типов импульсов"""
    impulse = Impulse(type=impulse_type, intensity=0.5, complexity=0.5)
    emotion = mock_system._determine_emotion(impulse)
    assert emotion == expected_emotion

def test_periodic_context_update(mock_system):
    """Тест периодического обновления контекста"""
    mock_context_manager = MagicMock()
    mock_system.context_manager = mock_context_manager
    mock_system.thought_counter = 5  # Кратно 5 для срабатывания
    
    mock_system.process_cycle()
    
    # Проверяем конкретный вызов для периодического обновления
    periodic_calls = [
        call for call in mock_context_manager.update_current_context.call_args_list
        if call.args[0].get('source') == 'periodic_update'
    ]
    assert len(periodic_calls) == 1

@patch('builtins.print')
def test_low_trust_logging(mock_print, mock_system):
    """Тест логирования мыслей с низким доверием"""

    # Настройка системы
    mock_system.current_coherence = 0.3  # Низкая когерентность
    mock_system.trust_validator = MagicMock()
    mock_system.trust_validator.validate_thought.return_value = 0.3  # Всегда низкое доверие

    # Подготовка мысли
    low_trust_thought = {
        "content": "Low trust thought",
        "source": "test"
    }
    
    # Внедряем мысль в процесс (в зависимости от реализации)
    mock_system.generate_thoughts = lambda: [low_trust_thought]
    mock_system.process_cycle()  # Запуск цикла

    # Проверяем, что было залогировано сообщение о низком доверии
    printed_messages = [str(call.args[0]) for call in mock_print.call_args_list]
    assert any("Низкое доверие" in msg for msg in printed_messages)

@pytest.mark.parametrize("dirty_text,clean_text", [
    ("\"Quoted\"", "Quoted"),
    ("`Code`", "Code"),
    ("  Trim me  ", "Trim me  "),  # Ожидаем сохранение пробелов в конце
    ("Multi\nline", "Multi")
])
def test_clean_thought_text(mock_system, dirty_text, clean_text):
    """Тест очистки сгенерированного текста"""
    result = mock_system._clean_thought_text(dirty_text)
    assert result == clean_text

def test_coherence_with_few_thoughts(mock_system):
    """Тест расчета когерентности при недостатке мыслей"""
    mock_system.thought_graph.add_thought({"content": "Single thought"})
    mock_system._update_system_metrics()
    assert mock_system.current_coherence == 0.5  # Дефолтное значение

def test_coherence_context_application(mock_system):
    """Тест применения контекста при низкой когерентности"""
    mock_context_manager = MagicMock()
    mock_context_manager.should_apply_context.return_value = True
    mock_system.context_manager = mock_context_manager
    
    # Принудительно устанавливаем низкую когерентность
    mock_system._update_system_metrics = lambda: None
    mock_system.current_coherence = 0.3
    
    mock_system.process_cycle()
    
    # Проверяем что проверка контекста была вызвана с нужным значением
    mock_context_manager.should_apply_context.assert_called_with(0.3)

def test_quarantine_thought(mock_system):
    """Тест добавления мысли в карантин"""
    mock_system.quarantine = MagicMock()
    bad_thought = {"content": "Bad thought", "trust_score": 0.2, "source": "untrusted"}
    mock_system._add_thought_to_graph(bad_thought, trust_score=0.2)
    mock_system.quarantine.add.assert_called_once_with(bad_thought)

def test_quarantine_mechanism(mock_system):
    """Тест работы механизма карантина"""
    mock_system.quarantine = MagicMock()
    mock_system.trust_validator.validate_thought.return_value = 0.2  # Низкое доверие
    
    bad_thought = {"content": "Bad thought", "source": "untrusted"}
    mock_system._add_thought_to_graph(bad_thought, trust_score=0.2)
    
    mock_system.quarantine.add.assert_called_once()
    assert len(mock_system.thought_graph.graph) == 0
