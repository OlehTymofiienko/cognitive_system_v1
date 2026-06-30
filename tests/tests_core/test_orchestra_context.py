# tests/test_orchestra_context.py

import io
import sys
import time
import numpy as np
import pytest

from unittest.mock import MagicMock, patch, call
from core.orchestra_context import OrchestraContextManager
from core.context_manager import ContextManager

def test_trigger_context_manager():
    cm = ContextManager()
    dummy_contexts = [{"coherence": 0.5}, {"coherence": 0.7}]
    result = cm.get_coherence(dummy_contexts)
    assert result == pytest.approx(0.6, 0.01)


def test_add_and_coherence_and_dissonance():
    # 1) Простая функция-эмбеддер
    def dummy_embed(text: str) -> np.ndarray:
        v = np.zeros(9, dtype=np.float32)
        v[len(text) % 9] = 1.0
        return v

    # 2) Инициализируем менеджер с очень малым интервалом
    mgr = OrchestraContextManager(embed_fn=dummy_embed, tick_interval=0.1)

    # 3) Добавляем три мысли
    voices = [
        mgr.add_thought("A"),
        mgr.add_thought("BB"),
        mgr.add_thought("CCC"),
    ]

    # Убедимся, что у нас только допустимые имена голосов
    for v in voices:
        assert v in ("melody", "counterpoint", "bass")

    # 4) Прочитаем coherence
    coh = mgr.get_coherence()
    assert 0.0 <= coh <= 1.0

    # 5) Прочитаем диссонансы
    names, M = mgr.get_dissonance_matrix()
    assert isinstance(names, list)
    assert isinstance(M, np.ndarray)
    # Размерность матрицы = число голосов
    assert M.shape == (len(names), len(names))
    for name in names:
        assert name in ("melody", "counterpoint", "bass")

def test_tick_and_export_state():
    # 1) Тот же dummy_embed
    def dummy_embed(text: str) -> np.ndarray:
        v = np.zeros(9, dtype=np.float32)
        v[len(text) % 9] = 1.0
        return v

    # 2) Инициализируем
    mgr = OrchestraContextManager(embed_fn=dummy_embed, tick_interval=0.1)

    # 3) Добавляем мысли с паузой, чтобы сработал тик
    mgr.add_thought("X")
    time.sleep(0.11)
    mgr.add_thought("YY")

    # 4) Перехватываем stdout вручную
    buffer = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buffer
        mgr.tick()
    finally:
        sys.stdout = old_stdout

    output = buffer.getvalue()
    # Убедимся, что внутри tick() был print
    assert "Orchestra Tick" in output

    # 5) Проверяем export_state()
    state = mgr.export_state()
    # Должны быть ключи: voices, coherence, dissonance
    assert set(state.keys()) == {"voices", "coherence", "dissonance"}

    # voices — словарь с тремя ключами (или менее, если какой-то голос пуст)
    assert isinstance(state["voices"], dict)
    for name, thought_list in state["voices"].items():
        assert name in ("melody", "counterpoint", "bass")
        # каждая мысль — словарь с text и weight
        for item in thought_list:
            assert "text" in item and "weight" in item

    # coherence — число
    assert isinstance(state["coherence"], float)

    # dissonance — словарь с names и matrix
    diss = state["dissonance"]
    assert set(diss.keys()) == {"names", "matrix"}
    assert isinstance(diss["names"], list)
    assert isinstance(diss["matrix"], list)

class TestOrchestraContextManager:
    def test_tick_skips_when_interval_not_passed(self):
        """Проверка что tick пропускает выполнение если интервал не пройден"""
        # Создаем менеджер с моком orchestra
        manager = OrchestraContextManager(embed_fn=MagicMock(), tick_interval=3600)
        manager.orchestra = MagicMock()
        
        # Устанавливаем текущее время как последний тик
        manager._last_tick = time.time()
        
        # Вызываем tick
        manager.tick()
        
        # Проверяем что методы не вызывались
        manager.orchestra.get_coherence.assert_not_called()
        manager.orchestra.calculate_dissonance_matrix.assert_not_called()

    def test_tick_executes_when_interval_passed(self):
        """Проверка что tick выполняется когда интервал пройден"""
        # Создаем менеджер с моком orchestra
        manager = OrchestraContextManager(embed_fn=MagicMock(), tick_interval=0.1)
        manager.orchestra = MagicMock()
        manager.orchestra.get_coherence.return_value = 0.8
        manager.orchestra.calculate_dissonance_matrix.return_value = (
            ['melody'], 
            np.array([[0]])
        )
        
        # Устанавливаем что прошлый тик был давно
        manager._last_tick = time.time() - 1
        
        # Вызываем tick
        manager.tick()
        
        # Проверяем что методы были вызваны
        manager.orchestra.get_coherence.assert_called_once()
        manager.orchestra.calculate_dissonance_matrix.assert_called_once()

    @patch('builtins.print')
    def test_tick_output(self, mock_print):
        """Проверка вывода в консоль при выполнении tick"""
        # Создаем менеджер с моком orchestra
        manager = OrchestraContextManager(embed_fn=MagicMock(), tick_interval=0.1)
        manager.orchestra = MagicMock()
        manager.orchestra.get_coherence.return_value = 0.75
        manager.orchestra.calculate_dissonance_matrix.return_value = (
            ['melody', 'counterpoint'],
            np.array([[0, 0.5], [0.5, 0]])
        )
        
        # Устанавливаем что прошлый тик был давно
        manager._last_tick = time.time() - 1
        
        # Вызываем tick
        manager.tick()
        
        # Проверяем вывод
        expected_calls = [
            call("[Orchestra Tick] coherence=0.750"),
            call("  melody: [0.00, 0.50]"),
            call("  counterpoint: [0.50, 0.00]")
        ]
        mock_print.assert_has_calls(expected_calls, any_order=True)

