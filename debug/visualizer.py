#debug\visualizer.py

import networkx as nx
import matplotlib.pyplot as plt

class GraphVisualizer:
    @staticmethod
    def draw(graph: nx.Graph, filename="thought_graph.png"):
        plt.figure(figsize=(14, 9))
        pos = nx.spring_layout(graph, seed=42)

        # Узлы: тексты мыслей
        labels = {
            node: data['thought'].get('content', '')[:28] + '...'
            for node, data in graph.nodes(data=True)
        }

        # Цвет узлов по trust_score
        colors = [
            data['thought'].get('trust_score', 0.5)
            for _, data in graph.nodes(data=True)
        ]

        # Размер узлов по интенсивности импульса
        sizes = [
            int(data['thought'].get('impulse_dict', {}).get('intensity', 0.5) * 900)
            for _, data in graph.nodes(data=True)
        ]

        # Подписи эмоций
        emotions = {
            node: data['thought'].get('emotion', '')
            for node, data in graph.nodes(data=True)
        }

        nx.draw_networkx_nodes(graph, pos, node_size=sizes, node_color=colors, cmap=plt.cm.coolwarm, alpha=0.85)
        nx.draw_networkx_edges(graph, pos, alpha=0.35)
        nx.draw_networkx_labels(graph, pos, labels, font_size=9)

        # Вторая метка: эмоции (над узлом)
        for node, (x, y) in pos.items():
            plt.text(x, y + 0.08, emotions.get(node, ""), fontsize=8, color='gray', ha='center')

        plt.title("Граф мыслей — когнитивная визуализация", fontsize=15)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"✅ Визуализация сохранена: {filename}")
