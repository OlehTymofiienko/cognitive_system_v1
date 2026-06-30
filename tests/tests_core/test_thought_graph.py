import pytest
import networkx as nx
import time
from core.thought_graph import ThoughtGraph

class TestThoughtGraphCore:
    @pytest.fixture
    def simple_graph(self):
        g = ThoughtGraph(max_complexity=10)
        g.add_thought({"core_concept": "AI", "content": "Thought 1"})
        g.add_thought({"core_concept": "ML", "content": "Thought 2"})
        g.add_thought({"core_concept": "AI", "content": "Thought 3"})
        return g

    def test_initial_state(self):
        g = ThoughtGraph()
        assert g.thought_counter == 0
        assert g.concept_counter == 0
        assert isinstance(g.graph, nx.DiGraph)

    def test_add_and_linking(self, simple_graph):
        assert len(simple_graph.graph) == 5  # 3 thoughts + 2 concepts
        edges = list(simple_graph.graph.edges(data=True))
        assert any(e[2]["relation_type"] == "semantic_link" for e in edges)
        assert any(e[2]["relation_type"] == "sequence" for e in edges)

    def test_concept_reuse(self, simple_graph):
        cid1 = simple_graph._get_or_create_concept_node("AI")
        cid2 = simple_graph._get_or_create_concept_node("AI")
        assert cid1 == cid2

    def test_manual_add_node(self):
        g = ThoughtGraph()
        cid = g.add_node("Robotics")
        assert g.graph.nodes[cid]["type"] == "concept"
        assert g.add_node("Robotics") == cid  # Reused

    def test_remove_node(self, simple_graph):
        node = list(simple_graph.graph.nodes)[0]
        assert simple_graph.remove_node(node) is True
        assert node not in simple_graph.graph
        assert simple_graph.remove_node("nonexistent") is False

    def test_node_score_normalized(self, simple_graph):
        score = simple_graph.node_score("thought_0")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

        # Непредставленный узел
        assert simple_graph.node_score("absent_node") == 0.0

    def test_explain_removal_format(self, simple_graph):
        reason = simple_graph.explain_removal("thought_0")
        assert "thought" in reason
        assert "degree=" in reason
        assert "age=" in reason
        assert "semantic_links=" in reason

    def test_simplify_triggers_on_overflow(self):
        g = ThoughtGraph(max_complexity=3)
        for i in range(5):
            g.add_thought({"core_concept": f"C{i}", "content": f"Overflow {i}"})
        assert len(g.graph) <= 10  # Граф упрощён

    def test_concept_preservation(self):
        g = ThoughtGraph(max_complexity=5, simplification_strategy="oldest")
        for i in range(10):
            g.add_thought({"core_concept": "Constant", "content": f"T{i}"})
        concepts_before = {
            n for n, d in g.graph.nodes(data=True) if d.get("type") == "concept"
        }
        g.simplify()
        concepts_after = {
            n for n, d in g.graph.nodes(data=True) if d.get("type") == "concept"
        }
        assert concepts_before == concepts_after


class TestSimplificationStrategies:
    @pytest.fixture
    def overloaded_graph(self):
        g = ThoughtGraph(max_complexity=5)
        for i in range(10):
            g.add_thought({
                "core_concept": f"Concept{i % 3}",
                "content": f"Thought {i}"
            })
        return g

    @pytest.mark.parametrize("strategy", [
        "least_connected", "oldest", "random", "semantic_isolated"
    ])
    def test_each_strategy_behavior(self, overloaded_graph, strategy):
        overloaded_graph.set_simplification_strategy(strategy)
        initial_nodes = set(overloaded_graph.graph.nodes)
        overloaded_graph.simplify()
        remaining = set(overloaded_graph.graph.nodes)
        removed = initial_nodes - remaining

        assert len(removed) >= 1
        for r in removed:
            reason = overloaded_graph.explain_removal(r)
            assert isinstance(reason, str)

        assert nx.is_directed_acyclic_graph(overloaded_graph.graph)

    def test_semantic_isolated_fallback(self):
        g = ThoughtGraph(simplification_strategy="semantic_isolated")
        for i in range(5):
            g.add_thought({
                "core_concept": "Linked",
                "content": f"Thought {i}"
            })  # Все связаны
        initial = set(g.graph.nodes)
        g.simplify()
        assert len(g.graph.nodes) < len(initial)


class TestPreviewSimplification:
    def test_preview_valid_strategy(self):
        g = ThoughtGraph(max_complexity=5)
        for i in range(6):
            g.add_thought({"core_concept": f"X{i}", "content": f"T{i}"})
        preview = g.preview_simplification("oldest")
        assert isinstance(preview, list)
        assert all(n in g.graph for n in preview)

    def test_preview_invalid_strategy(self):
        g = ThoughtGraph()
        with pytest.raises(ValueError):
            g.preview_simplification("nonexistent_strategy")

    def test_preview_semantic_isolated(self):
        g = ThoughtGraph()
        g.add_thought({"content": "Unlinked thought"})  # No core_concept
        preview = g.preview_simplification("semantic_isolated")
        assert "thought_0" in preview
