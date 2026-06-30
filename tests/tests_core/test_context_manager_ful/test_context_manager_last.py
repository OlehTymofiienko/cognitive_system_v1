#tests\test_context_manager_last.py

import pytest
import torch
import time
from unittest.mock import patch
from unittest.mock import MagicMock, patch
from core.context_manager import ContextManager

@pytest.fixture
def context_manager():
    return ContextManager(ttl=10.0)  # задаём TTL для тестов


def test_time_factor_normalized(context_manager):
    ctx = {"core_concept": "focus shift", "timestamp": 100.0, "source": "test"}
    context_manager.active_contexts.append(ctx)

    with patch('time.time', return_value=400.0):
        assert context_manager._get_time_factor() == 1.0

    with patch('time.time', return_value=250.0):
        assert pytest.approx(context_manager._get_time_factor(), 0.01) == 150.0 / 300

def test_add_context_skips_undefined(context_manager):
    thought = {"trust_score": 0.9, "language": "en", "content": "??", "impulse": {"type": "test"}}
    with patch.object(context_manager.semantic_analyzer, 'extract_core_concept', return_value="undefined"):
        assert context_manager.add_context(thought) is None
        assert len(context_manager.active_contexts) == 0

def test_update_core_concept(context_manager):
    context_manager.active_contexts.append({"core_concept": "initial", "timestamp": time.time(), "source": "test"})
    context_manager.update_current_context({"core_concept": "updated concept"})
    assert context_manager.active_contexts[-1]["core_concept"] == "updated concept"

def test_update_context_fields(context_manager):
    context_manager.active_contexts.append({"core_concept": "baseline", "timestamp": time.time(), "source": "test"})
    context_manager.update_current_context({"intensity": 0.9, "language": "en"})
    ctx = context_manager.active_contexts[-1]
    assert ctx["intensity"] == 0.9
    assert ctx["language"] == "en"

def test_get_current_context_vector(context_manager):
    context_manager.active_contexts.append({"core_concept": "concept clarity", "timestamp": time.time(), "source": "test"})
    with patch.object(context_manager.semantic_analyzer, 'get_embedding', return_value=torch.tensor([1.0, 2.0])):
        vec = context_manager.get_current_context_vector()
        assert torch.equal(vec, torch.tensor([1.0, 2.0]))

def test_get_dissonance_returns_max_pair(context_manager):
    matrix = [[0.0, 0.3, 0.5], [0.3, 0.0, 0.8], [0.5, 0.8, 0.0]]
    assert context_manager.get_dissonance(matrix) == 0.8

def test_should_apply_context_no_active(context_manager):
    result = context_manager.should_apply_context(current_coherence=0.5)
    assert result is True

def test_should_apply_context_computation(context_manager):
    context_manager.active_contexts.append({
        "core_concept": "focus",
        "timestamp": time.time() - 100  # старый
    })
    result = context_manager.should_apply_context(current_coherence=0.2)
    assert isinstance(result, bool)

def test_add_context_language_filter(context_manager):
    thought = {
        "trust_score": 0.8,
        "language": "de",
        "content": "ausdruck",
        "impulse": {"type": "test"}
    }
    assert context_manager.add_context(thought) is None

def test_add_context_eviction_on_maxlen(context_manager):
    context_manager.active_contexts.extend([
        {"core_concept": f"ctx{i}", "timestamp": time.time(), "source": "test"}
        for i in range(5)
    ])
    thought = {
        "trust_score": 0.9,
        "language": "en",
        "content": "overflow concept",
        "impulse": {"type": "test"}
    }
    with patch.object(context_manager.semantic_analyzer, 'extract_core_concept', return_value="Overflowing"):
        context_manager.add_context(thought)
        assert len(context_manager.active_contexts) == 5
        assert context_manager.active_contexts[-1]["core_concept"] == "overflowing"

def test_add_context_with_impulse_source(context_manager):
    thought = {
        "trust_score": 0.9,
        "language": "en",
        "content": "semantic",
        "impulse": {"type": "external"}
    }
    with patch.object(context_manager.semantic_analyzer, 'extract_core_concept', return_value="Conceptual"):
        context_manager.add_context(thought)
        assert context_manager.active_contexts[-1]["source"] == "external"

def test_update_context_adds_initial(context_manager):
    new_data = {"content": "first one", "language": "en", "trust_score": 0.85}
    with patch.object(context_manager.semantic_analyzer, 'extract_core_concept', return_value="InitialConcept"):
        context_manager.update_current_context(new_data)
        assert context_manager.active_contexts
        assert context_manager.active_contexts[0]["core_concept"] == "initialconcept"

def test_remove_expired_contexts(context_manager):
    ctx = {"core_concept": "temporary", "timestamp": time.time() - 20, "source": "test"}
    context_manager.active_contexts.append(ctx)
    context_manager._remove_expired_contexts()
    assert len(context_manager.active_contexts) == 0

def test_tick_adds_snapshot_and_cleans(context_manager):
    context_manager.active_contexts.append({
        "core_concept": "tick test",
        "timestamp": time.time() - 20,
        "source": "test"
    })
    context_manager.tick()
    assert context_manager.history
    assert all("tick test" not in ctx["core_concept"] for ctx in context_manager.active_contexts)

def test_time_factor_empty_contexts(context_manager):
    context_manager.active_contexts.clear()
    factor = context_manager._get_time_factor()
    assert factor == 0.0

def test_get_dissonance_empty_matrix(context_manager):
    assert context_manager.get_dissonance([]) == 0.0
    assert context_manager.get_dissonance([[]]) == 0.0

def test_get_coherence_empty(context_manager):
    result = context_manager.get_coherence([])
    assert result == 0.0

def test_get_coherence_average(context_manager):
    contexts = [
        {"coherence": 0.6},
        {"coherence": 0.8},
        {"coherence": 1.0}
    ]
    result = context_manager.get_coherence(contexts)
    assert pytest.approx(result, 0.01) == 0.8

def test_get_time_factor_empty_list(context_manager):
    context_manager.active_contexts.clear()
    factor = context_manager._get_time_factor()
    assert factor == 0.0

def test_main_loop_single_tick():
    manager = ContextManager()

    # Мокаем orchestra_mgr
    manager.orchestra_mgr = MagicMock()

    # Прерываем цикл через side_effect
    with patch("time.sleep", side_effect=KeyboardInterrupt) as mock_sleep:
        try:
            manager.main_loop()
        except KeyboardInterrupt:
            pass

    manager.orchestra_mgr.tick.assert_called_once()
    mock_sleep.assert_called_once_with(1.0)





