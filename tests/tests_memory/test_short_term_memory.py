#tests\test_short_term_memory.py

import pytest
from unittest.mock import patch, MagicMock
from collections import deque
from memory.short_term_memory import ShortTermMemory

class TestShortTermMemory:
    # Тесты инициализации
    def test_initialization_default_capacity(self):
        """Тест инициализации с дефолтной емкостью"""
        stm = ShortTermMemory()
        assert stm.capacity == 100
        assert isinstance(stm.memory, deque)
        assert len(stm.memory) == 0

    def test_initialization_custom_capacity(self):
        """Тест инициализации с кастомной емкостью"""
        stm = ShortTermMemory(capacity=50)
        assert stm.capacity == 50
        assert stm.memory.maxlen == 50

    # Тесты добавления элементов
    @patch('builtins.print')
    def test_add_single_item(self, mock_print):
        """Тест добавления одного элемента"""
        stm = ShortTermMemory(capacity=3)
        item = "test_item"
        
        stm.add(item)
        
        assert len(stm.memory) == 1
        assert stm.memory[0] == item
        mock_print.assert_called_once_with(f"→ ShortTermMemory stored: {item}")

    @patch('builtins.print')
    def test_add_multiple_items(self, mock_print):
        """Тест добавления нескольких элементов"""
        stm = ShortTermMemory(capacity=2)
        
        stm.add("item1")
        stm.add("item2")
        
        assert len(stm.memory) == 2
        assert stm.memory[0] == "item1"
        assert stm.memory[1] == "item2"
        assert mock_print.call_count == 2

    @patch('builtins.print')
    def test_add_over_capacity(self, mock_print):
        """Тест добавления элементов сверх емкости"""
        stm = ShortTermMemory(capacity=2)
        
        stm.add("item1")
        stm.add("item2")
        stm.add("item3")  # Должен вытеснить item1
        
        assert len(stm.memory) == 2
        assert stm.memory[0] == "item2"
        assert stm.memory[1] == "item3"
        assert mock_print.call_count == 3

    # Тесты очистки памяти
    @patch('builtins.print')
    def test_clear_memory(self, mock_print):
        """Тест очистки памяти"""
        stm = ShortTermMemory(capacity=3)
        stm.add("item1")
        stm.add("item2")
        
        stm.clear()
        
        assert len(stm.memory) == 0
        # Проверяем что после очистки можно добавлять новые элементы
        stm.add("new_item")
        assert len(stm.memory) == 1
        assert stm.memory[0] == "new_item"

    # Тесты получения элементов
    def test_get_recent_items(self):
        """Тест получения последних элементов"""
        stm = ShortTermMemory(capacity=5)
        for i in range(5):
            stm.add(f"item{i}")
        
        recent = stm.get_recent_items(3)
        assert len(recent) == 3
        assert recent == ["item2", "item3", "item4"]

    def test_get_recent_items_more_than_exists(self):
        """Тест получения большего количества элементов чем есть"""
        stm = ShortTermMemory(capacity=5)
        stm.add("item1")
        stm.add("item2")
        
        recent = stm.get_recent_items(5)
        assert len(recent) == 2
        assert recent == ["item1", "item2"]

    def test_get_recent_items_empty(self):
        """Тест получения элементов из пустой памяти"""
        stm = ShortTermMemory()
        recent = stm.get_recent_items(3)
        assert len(recent) == 0
        assert recent == []

    # Тесты поиска в памяти
    def test_search_in_memory(self):
        """Тест поиска элементов в памяти"""
        stm = ShortTermMemory(capacity=5)
        stm.add("apple")
        stm.add("banana")
        stm.add("orange")
        
        results = stm.search(lambda x: "a" in x)
        assert len(results) == 3
        assert set(results) == {"apple", "banana", "orange"}

    def test_search_in_empty_memory(self):
        """Тест поиска в пустой памяти"""
        stm = ShortTermMemory()
        results = stm.search(lambda x: True)
        assert len(results) == 0

    # Тест строкового представления
    def test_str_representation(self):
        """Тест строкового представления памяти"""
        stm = ShortTermMemory(capacity=3)
        stm.add("first")
        stm.add("second")
        
        assert str(stm) == "ShortTermMemory(capacity=3, items=2)"