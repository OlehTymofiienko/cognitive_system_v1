#memory\short_term_memory.py

import time
from collections import deque

class ShortTermMemory:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self, thought):
        self.memory.append(thought)
        print(f"→ ShortTermMemory stored: {thought}")

    def clear(self):
        """Очищает память"""
        self.memory.clear()

    def get_recent_items(self, count):
        """Возвращает последние N элементов"""
        return list(self.memory)[-count:]

    def search(self, condition):
        """Ищет элементы по условию"""
        return [item for item in self.memory if condition(item)]

    def __str__(self):
        """Строковое представление"""
        return f"ShortTermMemory(capacity={self.capacity}, items={len(self.memory)})"