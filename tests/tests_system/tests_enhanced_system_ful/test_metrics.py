# tests/test_metrics.py

import sys
import os
import time
import itertools

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from system.enhanced_system import EnhancedAISelfhoodChain
from debug.visualizer import GraphVisualizer

def build_chain_with_thoughts(phrases):
    """Создаёт систему с заданными мыслями и возвращает экземпляр."""
    system = EnhancedAISelfhoodChain()
    for phrase in phrases:
        system.thought_graph.add_thought({
            "content": phrase,
            "timestamp": time.time(),
            "language": "ru"
        })
    system._update_system_metrics()
    return system

def test_coherence_calculation():
    # Задаём явно связанный набор мыслей
    related_phrases = [
        "Разработка нейросетевых моделей",
        "Изучение когнитивной архитектуры",
        "Моделирование рассуждений",
        "Понимание машинного мышления",
        "Анализ информационных потоков",
        "Оценка интеллектуальных систем",
        "Разработка моделей обучения"
    ]
    system = build_chain_with_thoughts(related_phrases)
    
    print(f"\n[TEST LOG] Связанные мысли → Когерентность: {system.current_coherence:.2f}")
    assert system.current_coherence >= 0.6

def test_empty_phrases():
    system = build_chain_with_thoughts([""])
    assert 0 <= system.current_coherence <= 1.0

def test_incoherent_phrases():
    incoherent = [
        "Банановое мороженое на закате",
        "Законы квантовой суперпозиции",
        "Фестиваль хип-хоп музыки 2023",
        "Выход на пенсию в Германии",
    ]
    system = build_chain_with_thoughts(incoherent)

    thoughts = [
        data["thought"] for _, data in system.thought_graph.graph.nodes(data=True)
        if "thought" in data
    ]

    for t1, t2 in itertools.combinations(thoughts, 2):
        try:
            sim = system.semantic_analyzer.compare(t1["content"], t2["content"])
            print(f"[DEBUG] '{t1['content']}' ⇄ '{t2['content']}' = similarity: {sim:.2f}")
        except Exception as e:
            print(f"Similarity error: {str(e)}")

    print(f"[TEST LOG] Несвязанные мысли → Когерентность: {system.current_coherence:.2f}")
    assert system.current_coherence < 0.6

def test_visualizer_output():
    phrases = [
        "Нейросети для анализа данных",
        "Когнитивная симуляция мышления",
        "Формальное моделирование рассуждений"
    ]
    system = build_chain_with_thoughts(phrases)

    filename = "coherence_test.png"
    GraphVisualizer.draw(system.thought_graph.graph, filename)
    assert os.path.exists(filename)
    assert os.path.getsize(filename) > 0
