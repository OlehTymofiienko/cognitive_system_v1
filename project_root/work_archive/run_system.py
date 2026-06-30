import os
import time
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from system.enhanced_system import EnhancedAISelfhoodChain

def plot_results(system):
    plt.figure(figsize=(12, 6))

    # График когерентности
    plt.subplot(1, 2, 1)
    plt.plot(system.coherence_history, color='blue')
    plt.title("Динамика когерентности")
    plt.xlabel("Цикл")
    plt.ylabel("Когерентность")

    # Граф мыслей
    plt.subplot(1, 2, 2)
    nx.draw(system.thought_graph.graph, with_labels=True, node_color='lightgreen', node_size=500)
    plt.title("Граф мыслей")

    plt.tight_layout()
    plt.savefig("cognitive_results.png")
    print("Графики сохранены в cognitive_results.png")

def main():
    # Создаём уникальное имя файла с временной меткой
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"execution_log_{timestamp}.txt"
    log_path = os.path.abspath(log_filename)

    system = EnhancedAISelfhoodChain()

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("=== Запуск когнитивной системы ===\n")

        for i in range(10):
            start_time = time.time()
            f.write(f"\nЦикл {i+1}/10:\n")
            system.process_cycle()
            coherence = system.current_coherence
            active_contexts = len(system.context_manager.active_contexts)
            thoughts = len(system.thought_graph.graph.nodes)

            # Извлечение текста последней мысли
            thought_id = f"thought_{i}"
            thought = system.thought_graph.graph.nodes.get(thought_id, {})
            thought_text = thought.get("text", "[текст отсутствует]")

            f.write(f"Когерентность: {coherence:.2f}\n")
            f.write(f"Активных контекстов: {active_contexts}\n")
            f.write(f"Мыслей в графе: {thoughts}\n")

            print(f"\nЦикл {i+1}/10:")
            print(f"Когерентность: {coherence:.2f}")
            print(f"Активных контекстов: {active_contexts}")
            print(f"Мыслей в графе: {thoughts}")

            time.sleep(0.5 - min(0.4, (time.time() - start_time)))

        f.write("\n=== Работа завершена ===\n")
        f.write("Итоговые метрики:\n")
        f.write(f"- Всего мыслей: {len(system.thought_graph.graph.nodes)}\n")
        f.write(f"- Средняя когерентность: {sum(system.coherence_history)/len(system.coherence_history):.2f}\n")

    print("\n=== Работа завершена ===")
    print("Итоговые метрики:")
    print(f"- Всего мыслей: {len(system.thought_graph.graph.nodes)}")
    print(f"- Средняя когерентность: {sum(system.coherence_history)/len(system.coherence_history):.2f}")
    print(f"\nЛог сохранён в: {log_path}")

    # Генерация графиков
    plot_results(system)

if __name__ == "__main__":
    main()
