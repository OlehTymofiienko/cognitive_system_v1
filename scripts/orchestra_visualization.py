#!/usr/bin/env python3
# scripts/orchestra_visualization.py

import sys
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from core.semantic_analyzer import SemanticAnalyzer
from core.orchestra_context import OrchestraContextManager


def visualize_orchestra(state, coherence_history):
    """
    Рисует три графика:
      1) Bar-чарты весов мыслей по голосам
      2) Heatmap матрицы диссонансов
      3) Линейный график динамики когерентности
    И сохраняет их в PNG-файлы.
    """

    # --- 1) Bar-чарты весов мыслей ---
    voices = state["voices"]
    n_voices = len(voices)
    fig, axes = plt.subplots(nrows=n_voices, figsize=(8, 3 * n_voices), constrained_layout=True)

    if n_voices == 1:
        axes = [axes]

    for ax, (voice, thought_list) in zip(axes, voices.items()):
        weights = [t["weight"] for t in thought_list]
        labels  = [t["text"][:15] + ("…" if len(t["text"]) > 15 else "") for t in thought_list]

        ax.bar(range(len(weights)), weights, color="C0", alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(weights)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"Weights in '{voice}'")
        ax.set_ylabel("Weight")

    fig.suptitle("Thought Weights per Voice", fontsize=16)
    fig.savefig("weights_bar_chart.png")
    plt.close(fig)

    # --- 2) Heatmap диссонансов ---
    names = state["dissonance"]["names"]
    mat   = state["dissonance"]["matrix"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        mat,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        cbar_kws={"label": "Dissonance (1 – cosine)"}
    )
    plt.title("Dissonance Heatmap")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("dissonance_heatmap.png")
    plt.close()

    # --- 3) Динамика когерентности ---
    plt.figure(figsize=(8, 3))
    plt.plot(coherence_history, marker="o", color="C2", linestyle="-")
    plt.title("Coherence Over Time")
    plt.xlabel("Step")
    plt.ylabel("Coherence")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("coherence_over_time.png")
    plt.close()

    print("Visualization complete. Files saved:")
    print("  • weights_bar_chart.png")
    print("  • dissonance_heatmap.png")
    print("  • coherence_over_time.png\n")


if __name__ == "__main__":
    # 1) Инициализация
    print("\n=== ORCHESTRA VISUALIZATION SCRIPT ===\n")
    semantic = SemanticAnalyzer()
    orchestra_mgr = OrchestraContextManager(
        embed_fn=semantic.get_embedding,
        tick_interval=5.0
    )

    # 2) Тестовый набор мыслей
    test_texts = [
        "If you can think of something truly novel.",
        "Analyzing emergent AI behaviors.",
        "What does it mean to understand language?",
        "Exploring cognitive architectures.",
        "How will computers change our society?",
        "Can machines ever feel emotion?",
        "Balancing innovation and ethics in AI.",
        "The nature of consciousness",
        "Building robust neural networks",
        "Future of human-computer symbiosis"
    ]

    # 3) Собираем историю когерентности
    coherence_history = []
    for i, txt in enumerate(test_texts, start=1):
        voice = orchestra_mgr.add_thought(txt)
        coh   = orchestra_mgr.get_coherence()
        coherence_history.append(coh)
        print(f"Step {i}: added to '{voice}', coherence={coh:.2f}")
        # небольшая задержка, чтобы weight_decay сработал
        time.sleep(0.1)

    # 4) Экспортируем финальное состояние оркестра
    state = orchestra_mgr.export_state()

    # 5) Запускаем визуализацию
    visualize_orchestra(state, coherence_history)
