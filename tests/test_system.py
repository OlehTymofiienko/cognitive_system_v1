import sys
import os
import time
from core.models import Thought
from debug.visualizer import GraphVisualizer

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from system.enhanced_system import EnhancedAISelfhoodChain


def test_coherence_calculation(visualize=False):
    """Тест автоматической генерации мыслей с контролем когерентности"""
    system = EnhancedAISelfhoodChain()
    print("\nЗапуск автоматической генерации мыслей...\n")

    for i in range(15):  # Увеличим количество итераций для наглядности
        # Полный цикл обработки
        impulse = system.impulse_engine.generate_primary()
        if impulse:
            emotion = system._determine_emotion(impulse)
            thought = system._generate_thought(impulse, emotion)
            
            # Добавляем контекст при необходимости
            if system.context_manager.should_apply_context(system.current_coherence):
                context = system.impulse_engine.generate_contextual(thought)
                system.context_manager.add_context(context)
            
            system.thought_graph.add_thought({
                "content": thought,
                "context": system.context_manager.active_contexts[-1] if system.context_manager.active_contexts else None
            })
            system._update_system_metrics()
            
            print(f"[Цикл {i+1}] {thought[:60]}... | Когерентность: {system.current_coherence:.2f} | Контекст: {len(system.context_manager.active_contexts)}")
        time.sleep(0.3)  # Регулируем скорость генерации

    # Визуализация при необходимости
    if visualize:
        GraphVisualizer.draw(system.thought_graph.graph, "auto_thoughts_graph.png")
        print("\nГраф мыслей сохранён в auto_thoughts_graph.png")

    assert system.current_coherence > 0.5, "Когерентность упала ниже допустимого уровня"
    print("\nТест автоматической генерации пройден успешно!")
    system = EnhancedAISelfhoodChain(session_topic="AI Cohesion")
    # например, генерируем пару мыслей
    thoughts = system.generate_initial_thoughts()
    # проверяем, что возврат None (pytest не ждёт ничего возвращать)
    assert isinstance(thoughts, list)
    assert all(isinstance(t, Thought) for t in thoughts)

def test_empty_phrases():
    """Тест обработки пустых мыслей"""
    system = EnhancedAISelfhoodChain()
    system.thought_graph.add_thought({"content": ""})
    system._update_system_metrics()
    assert 0 <= system.current_coherence <= 1.0
    print("Тест пустых мыслей пройден")

def test_incoherent_phrases():
    """Тест намеренно несвязных мыслей"""
    system = EnhancedAISelfhoodChain()
    incoherent = [
        "Программирование на Python",
        "Рецепт яблочного пирога",
        "Физика чёрных дыр"
    ]
    
    print("\nТест несвязных мыслей:")
    for phrase in incoherent:
        system.thought_graph.add_thought({"content": phrase})
        system._update_system_metrics()
        print(f"Добавлена мысль: '{phrase}' | Текущая когерентность: {system.current_coherence:.2f}")
    
    assert system.current_coherence < 0.6
    print("Тест несвязных мыслей пройден")

if __name__ == "__main__":
    # Запускаем основные тесты
    tested_system = test_coherence_calculation(visualize=True)
    test_empty_phrases()
    test_incoherent_phrases()
    
    # Дополнительная визуализация
    GraphVisualizer.draw(tested_system.thought_graph.graph, "final_thoughts_graph.png")
    print("\nФинальный граф мыслей сохранён в final_thoughts_graph.png")

