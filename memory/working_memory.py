# memory\working_memory.py

from collections import deque

class WorkingMemory:
    def __init__(self, capacity=7, short_term_memory=None):
        self.capacity = capacity
        self.thoughts = deque(maxlen=capacity)
        self.short_term_memory = short_term_memory

    def add(self, thought):
        # если переполнилась, сначала переносим старейшую
        if len(self.thoughts) >= self.capacity:
            self._transfer_to_short_term()
        self.thoughts.append(thought)
        
    def _transfer_to_short_term(self):
        oldest = self.thoughts.popleft()
        if self.short_term_memory:
            self.short_term_memory.add(oldest)
        else:
            print(f"Transferring thought to short-term (no handler): {oldest}")