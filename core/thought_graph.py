import time
import random
import networkx as nx
from typing import Dict, Any, Callable, List


class ThoughtGraph:
    def __init__(self, max_complexity=1000, simplification_strategy="least_connected"):
        self.graph = nx.DiGraph()
        self.max_complexity = max_complexity
        self.thought_counter = 0
        self.concept_counter = 0

        self.simplification_strategies = {
            "least_connected": self._compute_least_connected,
            "oldest": self._compute_oldest,
            "random": self._compute_random,
            "semantic_isolated": self._compute_semantic_isolated
        }
        self.set_simplification_strategy(simplification_strategy)

    def set_simplification_strategy(self, strategy_name: str):
        if strategy_name not in self.simplification_strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(self.simplification_strategies.keys())}")
        self._strategy_func = self.simplification_strategies[strategy_name]

    def simplify(self, verbose=False):
        """Упрощает граф — рассчитывает узлы и удаляет их с пояснением"""
        to_remove = self._strategy_func()
        if not to_remove and self._strategy_func != self._compute_least_connected:
            if verbose:
                print("[simplify] Strategy yielded no removable nodes, falling back to least_connected.")
            to_remove = self._compute_least_connected()
        if not to_remove:
            return

        for node_id in to_remove:
            reason = self.explain_removal(node_id)
            if verbose:
                print(f"[simplify] Removing node: {node_id} → {reason}")
            self.graph.remove_node(node_id)

    def explain_removal(self, node_id: str) -> str:
        """Пояснение, почему узел выбран для удаления"""
        if node_id not in self.graph.nodes:
            return "node not found"
        data = self.graph.nodes[node_id]
        typ = data.get("type", "unknown")
        degree = self.graph.degree(node_id)
        age = time.time() - data.get("timestamp", time.time())
        semantic_links = sum(1 for _, _, d in self.graph.edges(node_id, data=True) if d.get("relation_type") == "semantic_link")

        parts = []
        if typ == "concept":
            parts.append("concept")
        elif typ == "thought":
            parts.append("thought")
        parts.append(f"degree={degree}")
        parts.append(f"age={round(age, 2)}s")
        parts.append(f"semantic_links={semantic_links}")
        return ", ".join(parts)

    def _compute_least_connected(self) -> List[str]:
        nodes = [
            n for n, d in sorted(self.graph.nodes(data=True), key=lambda x: self.graph.degree(x[0]))
            if d.get("type") != "concept"
        ]
        return nodes[:max(1, len(self.graph) // 10)]

    def _compute_oldest(self) -> List[str]:
        nodes = [
            n for n, d in sorted(self.graph.nodes(data=True), key=lambda x: x[1].get("timestamp", 0))
            if d.get("type") != "concept"
        ]
        return nodes[:max(1, len(self.graph) // 10)]

    def _compute_random(self) -> List[str]:
        nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") != "concept"
        ]
        random.shuffle(nodes)
        return nodes[:max(1, len(self.graph) // 10)]

    def _compute_semantic_isolated(self) -> List[str]:
        candidates = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") != "thought":
                continue
            has_link = any(
                edge_data.get("relation_type") == "semantic_link"
                for _, _, edge_data in self.graph.edges(node, data=True)
            )
            if not has_link:
                candidates.append(node)
        return candidates[:max(1, len(candidates))]

    def add_thought(self, thought: Dict[str, Any], context: Dict = None):
        if len(self.graph) >= self.max_complexity:
            self.simplify()

        node_id = f"thought_{self.thought_counter}"
        self.graph.add_node(node_id,
                            type="thought",
                            thought=thought,
                            context=context,
                            timestamp=time.time())

        if self.thought_counter > 0:
            last_node = f"thought_{self.thought_counter - 1}"
            self.graph.add_edge(last_node, node_id, relation_type="sequence")

        concept = thought.get("core_concept")
        if concept and isinstance(concept, str):
            concept_node_id = self._get_or_create_concept_node(concept)
            self.graph.add_edge(node_id, concept_node_id, relation_type="semantic_link")

        self.thought_counter += 1

    def _get_or_create_concept_node(self, concept: str) -> str:
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "concept" and data.get("label") == concept:
                return node_id

        node_id = f"concept_{self.concept_counter}"
        self.graph.add_node(node_id,
                            type="concept",
                            label=concept,
                            timestamp=time.time())
        self.concept_counter += 1
        return node_id

    def add_node(self, label: str) -> str:
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "concept" and data.get("label") == label:
                return node_id

        node_id = f"concept_{self.concept_counter}"
        self.graph.add_node(node_id,
                            type="concept",
                            label=label,
                            timestamp=time.time())
        self.concept_counter += 1
        return node_id

    def remove_node(self, node_id: str) -> bool:
        if node_id in self.graph.nodes:
            self.graph.remove_node(node_id)
            return True
        return False

    def node_score(self, node_id: str) -> float:
        data = self.graph.nodes.get(node_id, {})
        if not data:
            return 0.0
        degree = self.graph.degree(node_id)
        age = time.time() - data.get('timestamp', time.time())
        semantic_links = sum(
            1 for _, _, d in self.graph.edges(node_id, data=True)
            if d.get("relation_type") == "semantic_link"
        )
        return (degree * 0.3 + age * 0.4 + semantic_links * 0.3) / 1000  # нормализация

    def preview_simplification(self, strategy_name: str) -> List[str]:
        if strategy_name not in self.simplification_strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return self.simplification_strategies[strategy_name]()
