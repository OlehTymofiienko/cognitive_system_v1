from typing import List, Dict, Optional
from core.models import Thought

class VoiceLineManager:
    def __init__(self, session_topic: str):
        self.session_topic = session_topic
        self.lines: List[Dict] = []
        self._current_group: Optional[Dict] = None

    def add_thought(self, thought: Thought):
        """
        Вставляет мысль в текущую группу, 
        либо создает новую, если изменился voice/context.
        """
        meta = thought.metadata or {}
        context = meta.get("bridge_of") or thought.voice

        # новая группа при смене контекста или голоса
        if (not self._current_group
            or self._current_group["context"] != context):
            self._current_group = {
                "context": context,
                "thoughts": []
            }
            self.lines.append(self._current_group)

        self._current_group["thoughts"].append({
            "voice": thought.voice,
            "content": thought.content,
            "coherence": thought.coherence,
            "metadata": meta
        })

    def get_script(self) -> str:
        """
        Формирует линейный текст:
        
        [Группа: context]
        voice1: content (coherence)
        voice2: content (coherence)
        """
        parts = [f"=== Topic: {self.session_topic} ==="]
        for group in self.lines:
            parts.append(f"\n--- Group: {group['context']} ---")
            for t in group["thoughts"]:
                parts.append(f"{t['voice']}: {t['content']} ({t['coherence']:.2f})")
        return "\n".join(parts)

    def export_json(self) -> Dict:
        """
        Возвращает весь поток мыслей в виде JSON-совместимой структуры.
        """
        return {
            "session_topic": self.session_topic,
            "groups": self.lines
        }

    def draw(self, filename: str):
        """
        Опционально: визуализация графа, если нужен .png.
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            return

        G = nx.DiGraph()
        for i, group in enumerate(self.lines):
            for j, t in enumerate(group["thoughts"]):
                node_id = f"{i}.{j}"
                G.add_node(node_id, label=t["voice"])
                if j > 0:
                    prev = f"{i}.{j-1}"
                    G.add_edge(prev, node_id)
            # между группами
            if i > 0:
                prev_last = f"{i-1}.{len(self.lines[i-1]['thoughts'])-1}"
                curr_first = f"{i}.0"
                G.add_edge(prev_last, curr_first)

        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=800, font_size=8)
        plt.savefig(filename)
        plt.close()
