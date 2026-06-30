# tests/test_trust_validator.py

import pytest
from unittest.mock import MagicMock
from core.trust_validator import ThoughtValidator

@pytest.fixture
def trust_validator():
    return ThoughtValidator()

# ────────────────────────────────────────────────────────────────
# Типы и структура входа
# ────────────────────────────────────────────────────────────────

def test_validate_invalid_type(trust_validator):
    assert trust_validator.validate_thought(123) == 0.3
    assert trust_validator.validate_thought("") == 0.3
    assert trust_validator.validate_thought(None) == 0.3

def test_validate_empty_content(trust_validator):
    assert trust_validator.validate_thought({"content": ""}) == 0.3

def test_validate_without_context_manager(trust_validator):
    trust_validator.context_manager = None
    thought = {"content": "test", "source": "external_input"}
    assert 0.3 <= trust_validator.validate_thought(thought) <= 1.0

# ────────────────────────────────────────────────────────────────
# Источник и динамическое доверие
# ────────────────────────────────────────────────────────────────

def test_validate_unknown_source(trust_validator):
    thought = {"content": "test", "source": "alien"}
    assert 0.3 <= trust_validator.validate_thought(thought) <= 1.0

def test_dynamic_weights_priority(trust_validator):
    trust_validator.dynamic_weights["external_input"] = 0.9
    assert trust_validator.get_trust("external_input") == 0.9

def test_update_trust_clamping(trust_validator):
    trust_validator.update_trust("external_input", -5.0)
    assert trust_validator.get_trust("external_input") == 0.1
    trust_validator.update_trust("external_input", 10.0)
    assert trust_validator.get_trust("external_input") == 1.0

def test_get_trust_logs_unknown_source(trust_validator, caplog):
    with caplog.at_level("WARNING"):
        score = trust_validator.get_trust("unknown")  # Неизвестный источник
    assert score == 0.5
    assert "Unknown source" in caplog.text

# ────────────────────────────────────────────────────────────────
# Импульс: словарь и dataclass
# ────────────────────────────────────────────────────────────────

def test_validate_impulse_dict(trust_validator):
    thought = {
        "content": "intense idea",
        "impulse": {"type": "logic", "intensity": 0.9}
    }
    score = trust_validator.validate_thought(thought)
    assert 0.3 <= score <= 1.0

def test_validate_impulse_dataclass(trust_validator):
    class ImpulseMock:
        type = "intuition"
        intensity = 0.6
        __dataclass_fields__ = {"type": str, "intensity": float}

    thought = {
        "content": "deep insight",
        "impulse": ImpulseMock()
    }
    score = trust_validator.validate_thought(thought)
    assert 0.3 <= score <= 1.0

# ────────────────────────────────────────────────────────────────
# Контекстные противоречия и штрафы
# ────────────────────────────────────────────────────────────────

def test_validate_with_context_contradiction(trust_validator):
    trust_validator.context_manager = MagicMock()
    trust_validator.context_manager.active_contexts = [{"core_concept": "truth"}]

    thought = {"content": "not truth", "source": "context_manager"}
    score = trust_validator.validate_thought(thought)
    assert score < 0.8

def test_validate_context_anomaly_penalty(trust_validator):
    trust_validator.context_manager = MagicMock()
    trust_validator.context_manager.active_contexts = [{"core_concept": "growth"}]

    thought = {
        "content": "not growth",
        "source": "context_manager",
        "coherence": 0.7,
        "impulse": {"intensity": 0.6}
    }
    score = trust_validator.validate_thought(thought)
    assert score < 0.8

# ────────────────────────────────────────────────────────────────
# Диапазон доверия: минимумы и максимумы
# ────────────────────────────────────────────────────────────────

def test_minimum_trust_score(trust_validator):
    thought = {
        "content": "test",
        "source": "external_input",
        "impulse": {"intensity": 0.1},
        "coherence": 0.1
    }
    assert trust_validator.validate_thought(thought) == 0.3

def test_maximum_trust_score(trust_validator):
    trust_validator.dynamic_weights["impulse_engine"] = 0.9  # даже 1.0 не даст >0.8 base_trust
    thought = {
        "content": "perfect",
        "source": "impulse_engine",
        "impulse": {"intensity": 1.0},
        "coherence": 1.0
    }
    assert trust_validator.validate_thought(thought) == 0.8  # логично: clamp на base_trust


# ────────────────────────────────────────────────────────────────
# Параметризация: базовый trust-score
# ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("source,intensity,coherence", [
    ("external_input", 0.2, 0.3),
    ("impulse_engine", 0.9, 0.9),
    ("working_memory", 0.5, 0.7)
])
def test_trust_score_variation(trust_validator, source, intensity, coherence):
    thought = {
        "content": "test",
        "source": source,
        "impulse": {"intensity": intensity},
        "coherence": coherence
    }
    score = trust_validator.validate_thought(thought)
    assert 0.3 <= score <= 1.0

def test_multiple_context_anomalies(trust_validator):
    trust_validator.context_manager = MagicMock()
    trust_validator.context_manager.active_contexts = [
        {"core_concept": "truth"},
        {"core_concept": "growth"}
    ]

    thought = {
        "content": "not truth and not growth",
        "source": "context_manager",
        "coherence": 1.0,
        "impulse": {"intensity": 1.0}
    }

    score = trust_validator.validate_thought(thought)
    assert score < 0.8  # penalty >1 раз

def test_impulse_unrecognized_type(trust_validator):
    class DummyImpulse:
        pass  # нет __dataclass_fields__ и не dict

    thought = {
        "content": "neutral impulse",
        "impulse": DummyImpulse(),
        "coherence": 0.7
    }

    score = trust_validator.validate_thought(thought)
    assert 0.3 <= score <= 1.0


