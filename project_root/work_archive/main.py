import time
from system.enhanced_system import EnhancedAISelfhoodChain
from config import Config

def initialize_system() -> EnhancedAISelfhoodChain:
    """Инициализация когнитивной системы с начальными контекстами и повышенным доверием."""
    system = EnhancedAISelfhoodChain()
    
    # Увеличиваем начальное доверие к внешнему входу
    system.trust_validator.update_trust('external_input', 0.5)  # Было 0.2
    
    # Расширенный список начальных контекстов
    init_thoughts = [
        {
            'content': 'System focusing on cognitive coherence',
            'language': 'en',
            'impulse': {'type': 'exploratory', 'intensity': 0.9},
            'timestamp': time.time()
        },
        {
            'content': 'Prioritizing semantic consistency in thoughts',
            'language': 'en',
            'impulse': {'type': 'reflective', 'intensity': 0.8},
            'timestamp': time.time()
        },
        {
            'content': 'Establishing core identity parameters',
            'language': 'en',
            'impulse': {'type': 'integrative', 'intensity': 1.0},
            'timestamp': time.time()
        }
    ]
    
    for thought in init_thoughts:
        system.context_manager.add_context(thought)
    
    return system

def log_system_state(system: EnhancedAISelfhoodChain, cycle: int):
    """Логирование состояния системы"""
    active_ctx = (system.context_manager.active_contexts[-1]
                  if system.context_manager.active_contexts else None)
    ctx_concept = active_ctx['core_concept'] if active_ctx else "None"
    
    print(f"\nCycle {cycle} | Coherence: {system.current_coherence:.2f}")
    print(f"Active Context: {ctx_concept}")
    print(f"Thoughts in Graph: {len(system.thought_graph.graph)}")
    print(f"Working Memory: {len(system.working_memory.thoughts)} items")

def main():
    try:
        # Инициализация
        print("=== COGNITIVE SYSTEM STARTUP ===")
        system = initialize_system()
        
        # Вывод начального состояния
        print("\nInitial Contexts:")
        for i, ctx in enumerate(system.context_manager.active_contexts, 1):
            age = time.time() - ctx['timestamp']
            print(f"{i}. {ctx['core_concept']} ({ctx['source']}, age: {age:.1f}s)")
        
        # Основной цикл
        print("\n=== MAIN PROCESSING LOOP ===")
        cycle = 0
        
        while True:
            cycle += 1
            start_time = time.time()
            system.process_cycle()
            
            # Логирование каждые 5 циклов или при падении когерентности
            if cycle % 5 == 0 or system.current_coherence < 0.4:
                log_system_state(system, cycle)
            
            # Задержка для стабильного интервала
            elapsed = time.time() - start_time
            sleep_time = max(0, Config.PROCESSING_INTERVAL - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n=== SYSTEM SHUTDOWN ===")
        print("Final Contexts:")
        for ctx in system.context_manager.active_contexts:
            age = time.time() - ctx['timestamp']
            print(f"- {ctx['core_concept']} (age: {age:.1f}s)")
        
        total = cycle
        avg_coh = (sum(system.coherence_history) / len(system.coherence_history)
                   if system.coherence_history else 0.0)
        print(f"\nTotal cycles processed: {total}")
        print(f"Average coherence: {avg_coh:.2f}")

if __name__ == "__main__":
    main()
