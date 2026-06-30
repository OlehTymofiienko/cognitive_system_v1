# tests/test_intuition_analyzer.py

import pytest
from core.intuition_analyzer import IntuitionAnalyzer

class DummyGraph:
    def __init__(self, nodes):
        # GraphStub умеет возвращать len() и nodes(data=True)
        class GraphStub:
            def __len__(self_non):
                return len(nodes)
            def nodes(self_non, data=True):
                return nodes
        self.graph = GraphStub()

@pytest.fixture
def analyzer():
    return IntuitionAnalyzer()

def test_analyze_with_too_few_nodes(analyzer):
    # Меньше 3 узлов → всегда False
    nodes = [
        ("n1", {"thought": {"id": 1}}),
        ("n2", {"thought": {"id": 2}})
    ]
    tg = DummyGraph(nodes)
    current = {"id": 3}
    assert analyzer.analyze(tg, current) is False

def test_calculate_deviation_all_same(analyzer):
    thought = {"id": "X"}
    last = [
        ("a", {"thought": thought}),
        ("b", {"thought": thought}),
        ("c", {"thought": thought})
    ]
    dev = analyzer._calculate_deviation(thought, last)
    assert dev == pytest.approx(0.0)

def test_calculate_deviation_all_different(analyzer):
    thought = {"id": "X"}
    last = [
        ("a", {"thought": {"id": 1}}),
        ("b", {"thought": {"id": 2}}),
        ("c", {"thought": {"id": 3}})
    ]
    dev = analyzer._calculate_deviation(thought, last)
    assert dev == pytest.approx(1.0)

def test_analyze_below_threshold(analyzer):
    # 2 расхождения из 3 → 0.666 < 0.7
    nodes = [
        ("n1", {"thought": {"id": 1}}),
        ("n2", {"thought": {"id": 2}}),
        ("n3", {"thought": {"id": 2}})
    ]
    tg = DummyGraph(nodes)
    current = {"id": 2}
    assert analyzer.analyze(tg, current) is False

def test_analyze_above_threshold(analyzer):
    # 3 расхождения из 3 → 1.0 > 0.7
    nodes = [
        ("n1", {"thought": {"id": 1}}),
        ("n2", {"thought": {"id": 2}}),
        ("n3", {"thought": {"id": 3}})
    ]
    tg = DummyGraph(nodes)
    current = {"id": 99}
    assert analyzer.analyze(tg, current) is True
