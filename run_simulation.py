#!/usr/bin/env python3
# run_simulation.py

import os
import sys
import time
import torch

# гарантируем, что корень проекта в sys.path
proj_root = os.path.abspath(os.path.dirname(__file__))
if proj_root not in sys.path:
    sys.path.append(proj_root)

from datetime import datetime
from transformers import pipeline

from system.enhanced_system import EnhancedAISelfhoodChain
from core.semantic_analyzer import SemanticAnalyzer
from core.orchestra_context import OrchestraContextManager
from system.orchestra_behavior import OrchestraBehavior
from scripts.orchestra_visualization import visualize_orchestra

from system.voice_manager import VoiceLineManager


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def print_orchestra_state(state):
    print("\n=== Orchestra State ===")
    # 1) Мысли по голосам
    for voice, thought_list in state["voices"].items():
        print(f"{voice.upper()}:")
        for t in thought_list:
            txt = t["text"]
            w = t["weight"]
            print(f"  • {txt[:60]:60s} (weight={w:.2f})")
    # 2) Coherence
    coh = state["coherence"]
    print(f"\nCoherence: {coh:.2f}")
    # 3) Dissonance matrix
    names = state["dissonance"]["names"]
    mat = state["dissonance"]["matrix"]
    print("\nDissonance matrix:")
    for name, row in zip(names, mat):
        row_str = ", ".join(f"{v:.2f}" for v in row)
        print(f"  {name:12s}: [{row_str}]")
    print("========================\n")


def run_thought_simulation(num_thoughts=10):
    print("=== INITIALIZING COGNITIVE SYSTEM ===\n")

    # 1) Языковая модель
    lm = pipeline(
        "text-generation",
        model="gpt2-medium",
        device=0 if torch.cuda.is_available() else -1,
        model_kwargs={"pad_token_id": 50256}
    )
    system = EnhancedAISelfhoodChain(language_model=lm)

    # 2) SemanticAnalyzer и OrchestraContextManager
    semantic_analyzer = system.semantic_analyzer
    orchestra_mgr = OrchestraContextManager(
        embed_fn=semantic_analyzer.get_embedding,
        tick_interval=5.0
    )

    # 2.1) Поведенческая логика
    behavior = OrchestraBehavior(orchestra_mgr, semantic_analyzer)

    print("\n=== STARTING COGNITIVE PROCESS ===")
    all_thoughts = []
    coherence_history = []

    for i in range(1, num_thoughts + 1):
        system.process_cycle()
        node = list(system.thought_graph.graph.nodes)[-1]
        thought = system.thought_graph.graph.nodes[node]["thought"]
        content = thought["content"]
        all_thoughts.append(content)

        # базовый вывод
        print(f"\nTHOUGHT #{i}:")
        print(f" CONTENT:       {content}")
        print(f" IMPULSE TYPE:  {thought['impulse']['type']}")
        print(f" INTENSITY:     {thought['impulse']['intensity']:.2f}")
        print(f" COHERENCE:     {system.current_coherence:.2f}")

        # сравнение с предыдущей
        if i > 1:
            sim = semantic_analyzer.compare(all_thoughts[i-2], all_thoughts[i-1])
            a, b = all_thoughts[i-2][:20], all_thoughts[i-1][:20]
            print(f" SIMILARITY:    '{a}...' vs '{b}...' = {sim:.2f}")

        # embedding
        emb = semantic_analyzer.get_embedding(content)
        print(f" EMBEDDING:     shape={emb.shape}, type={type(emb)}")

        # 3) Оркестровка мысли
        voice = orchestra_mgr.add_thought(content)
        print(f" → assigned to voice: {voice}")

        # 4) Тик оркестра
        orchestra_mgr.tick()

        # 5) Собираем историю когерентности
        state = orchestra_mgr.export_state()
        coh = state["coherence"]
        coherence_history.append(coh)

        # 5.1) Поведенческая логика
        behavior.apply(state, coherence_history)

        # 6) Печать snapshot каждые 5 мыслей
        if i % 5 == 0:
            print_orchestra_state(state)

        time.sleep(0.3)

    # 7) Финальный экспорт и визуализация
    final_state = orchestra_mgr.export_state()
    visualize_orchestra(final_state, coherence_history)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"run_simulation_{timestamp}.txt"
    log_file = open(log_filename, "w", encoding="utf-8")

    tee = Tee(sys.__stdout__, log_file)
    sys.stdout = tee
    sys.stderr = tee

    run_thought_simulation(10)
    print(f"\n[LOG SAVED]: {os.path.abspath(log_file.name)}")
