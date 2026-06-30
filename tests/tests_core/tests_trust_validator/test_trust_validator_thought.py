# tests\tests_core\test_trust_validator_thought.py

import pytest
from core.trust_validator import ThoughtValidator
from core.models import Impulse

class DummyContextManager:
    def __init__(self, active):
        # active — список словарей вида {'core_concept': str}
        self.active_contexts = active

@pytest.fixture
def validator():
    return ThoughtValidator()


def test_update_and_get_trust_within_bounds(validator):
    # исходный вес для impulse_engine == 0.9
    validator.update_trust('impulse_engine', +0.2)
    # новый вес = min(1.0, 0.9+0.2) = 1.0
    assert validator.get_trust('impulse_engine') == pytest.approx(1.0)

    validator.update_trust('impulse_engine', -0.95)
    # теперь = max(0.1, 0.9-0.95) = 0.1
    assert validator.get_trust('impulse_engine') == pytest.approx(0.1)

    # get_trust для несуществующего компонента возвращает базовый вес
    assert validator.get_trust('unknown') == pytest.approx(0.5)  # external_input

def test_validate_thought_invalid_type_and_empty_content(validator):
    # не dict → 0.3
    assert validator.validate_thought(None) == pytest.approx(0.3)

    # dict, но нет content → 0.3
    assert validator.validate_thought({'source': 'impulse_engine'}) == pytest.approx(0.3)

    # пустая строка в content → 0.3
    assert validator.validate_thought({'content': '   '}) == pytest.approx(0.3)

def test_validate_thought_basic_trust_calculation(validator):
    # создаём мысленный импульс
    thought = {
        'content': 'Some valid thought',
        'source': 'impulse_engine',
        'impulse': {'type': 'exploratory', 'intensity': 0.7},
        'coherence': 0.9
    }
    # raw_trust=0.9, base_trust=min(0.8,0.9+0.1)=0.8
    # intensity_factor=0.7, coherence_factor=0.9
    # trust_score=0.8*(0.6*0.7+0.4*0.9)=0.8*(0.42+0.36)=0.8*0.78=0.624
    assert validator.validate_thought(thought) == pytest.approx(0.624, rel=1e-3)

def test_validate_thought_with_anomaly(validator):
    # подвязываем контекстный менеджер
    validator.context_manager = DummyContextManager([
        {'core_concept': 'X'}
    ])
    thought = {
        'content': 'This is NOT X at all',
        'source': 'context_manager',
        'impulse': Impulse(type='reflective', intensity=0.5, complexity=2.0),
        'coherence': 0.8
    }
    # raw_trust=context_manager=0.7, base=min(0.8,0.7+0.1)=0.8
    # intensity_factor=0.5, coherence_factor=0.8
    # base_score=0.8*(0.6*0.5+0.4*0.8)=0.8*(0.30+0.32)=0.8*0.62=0.496
    # anomaly_flags=1 → penalty=max(0.5,1-0.1)=0.9 → final=0.496*0.9=0.4464
    assert validator.validate_thought(thought) == pytest.approx(0.4464, rel=1e-3)



