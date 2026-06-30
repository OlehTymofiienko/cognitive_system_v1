#tests\test_working_memory.py

import pytest
from unittest.mock import MagicMock, patch
from memory.working_memory import WorkingMemory

class TestWorkingMemory:
    def test_initialization_default(self):
        """Тест инициализации с параметрами по умолчанию"""
        wm = WorkingMemory()
        assert wm.capacity == 7
        assert len(wm.thoughts) == 0
        assert wm.short_term_memory is None

    def test_initialization_with_custom_params(self):
        """Тест инициализации с кастомными параметрами"""
        mock_stm = MagicMock()
        wm = WorkingMemory(capacity=5, short_term_memory=mock_stm)
        assert wm.capacity == 5
        assert wm.short_term_memory == mock_stm

    def test_add_without_overflow(self):
        """Тест добавления без переполнения"""
        wm = WorkingMemory(capacity=3)
        wm.add("thought1")
        wm.add("thought2")
        
        assert len(wm.thoughts) == 2
        assert wm.thoughts[0] == "thought1"
        assert wm.thoughts[1] == "thought2"

    @patch('builtins.print')
    def test_add_with_overflow_no_stm(self, mock_print):
        """Тест добавления с переполнением без short_term_memory"""
        wm = WorkingMemory(capacity=2)
        wm.add("thought1")
        wm.add("thought2")
        wm.add("thought3")  # Должен вытеснить thought1
        
        assert len(wm.thoughts) == 2
        assert wm.thoughts[0] == "thought2"
        assert wm.thoughts[1] == "thought3"
        mock_print.assert_called_once_with(
            "Transferring thought to short-term (no handler): thought1"
        )

    def test_add_with_overflow_with_stm(self):
        """Тест добавления с переполнением с short_term_memory"""
        mock_stm = MagicMock()
        wm = WorkingMemory(capacity=2, short_term_memory=mock_stm)
        
        wm.add("thought1")
        wm.add("thought2")
        wm.add("thought3")  # Должен передать thought1 в STM
        
        assert len(wm.thoughts) == 2
        assert wm.thoughts[0] == "thought2"
        assert wm.thoughts[1] == "thought3"
        mock_stm.add.assert_called_once_with("thought1")

    def test_transfer_to_short_term_with_stm(self):
        """Тест переноса в short_term_memory при наличии обработчика"""
        mock_stm = MagicMock()
        wm = WorkingMemory(capacity=2, short_term_memory=mock_stm)
        wm.thoughts.extend(["thought1", "thought2"])
        
        wm._transfer_to_short_term()
        
        assert len(wm.thoughts) == 1
        assert wm.thoughts[0] == "thought2"
        mock_stm.add.assert_called_once_with("thought1")

    @patch('builtins.print')
    def test_transfer_to_short_term_without_stm(self, mock_print):
        """Тест переноса в short_term_memory без обработчика"""
        wm = WorkingMemory(capacity=2)
        wm.thoughts.extend(["thought1", "thought2"])
        
        wm._transfer_to_short_term()
        
        assert len(wm.thoughts) == 1
        assert wm.thoughts[0] == "thought2"
        mock_print.assert_called_once_with(
            "Transferring thought to short-term (no handler): thought1"
        )

    def test_transfer_from_empty_memory(self):
        """Тест переноса из пустой памяти"""
        wm = WorkingMemory()
        with pytest.raises(IndexError):
            wm._transfer_to_short_term()