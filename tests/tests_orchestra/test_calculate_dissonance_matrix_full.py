#tests\tests_orchestra_ful\test_calculate_dissonance_matrix_full.py

import numpy as np
import pytest
import math
import logging
from core.orchestra import SimpleOrchestra

class DummyThought:
    def __init__(self, emb, weight_val):
        self.emb = emb
        self._w = weight_val

    def weight(self):
        return self._w

@pytest.fixture
def orchestra_full():
    orchestra = SimpleOrchestra(embed_fn=lambda x: np.array([1.0, 0.0]))  # unused

    orchestra.voices = {
        "main": [
            DummyThought(np.array([1.0, 0.0]), 1.0)
        ],
        "counterpoint": [
            DummyThought(np.array([0.0, 1.0]), 1.0)
        ],
        "bass": [],  # пустой голос → activates "if not bucket"
        "noise": [
            DummyThought(np.array([np.nan, np.nan]), 1.0)  # activates math.isnan()
        ]
    }

    return orchestra

def test_calculate_dissonance_matrix_comprehensive():
    """Объединённый тест матрицы диссонансов со всеми случаями."""
    orchestra = SimpleOrchestra(embed_fn=lambda x: np.array([1.0, 0.0]))

    orchestra.voices = {
        "main": [DummyThought(np.array([1.0, 0.0]), 1.0)],        # Базовый голос
        "counter": [DummyThought(np.array([0.0, 1.0]), 1.0)],     # Ортогональный
        "identical": [DummyThought(np.array([1.0, 0.0]), 1.0)],   # Совпадает с main
        "empty": [],                                              # Пустой → исключается
        "zero_weight": [DummyThought(np.array([0.5, 0.5]), 0.0)], # Вес 0 → исключается
        "nan_emb": [DummyThought(np.array([np.nan, np.nan]), 1.0)] # NaN → исключается
    }

    # Ожидаем только голоса с валидными данными
    expected_names = ["main", "counter", "identical"]
    names, matrix = orchestra.calculate_dissonance_matrix()

    # Проверка структуры
    assert set(names) == set(expected_names)  # Используем set для независимости от порядка
    assert matrix.shape == (len(expected_names), len(expected_names))

    # Получаем индексы для проверок
    idx_main = names.index("main")
    idx_counter = names.index("counter")
    idx_identical = names.index("identical")

    # Проверка диагонали
    for i in range(len(names)):
        assert matrix[i, i] == pytest.approx(0.0)

    # main vs counter → ортогональны → диссонанс = 1.0
    assert matrix[idx_main, idx_counter] == pytest.approx(1.0)
    assert matrix[idx_counter, idx_main] == pytest.approx(1.0)

    # main vs identical → совпадают → диссонанс ≈ 0.0
    assert matrix[idx_main, idx_identical] == pytest.approx(0.0, abs=1e-8)
    assert matrix[idx_identical, idx_main] == pytest.approx(0.0, abs=1e-8)

    # Проверка симметричности
    for i in range(len(names)):
        for j in range(len(names)):
            assert matrix[i, j] == pytest.approx(matrix[j, i], abs=1e-8), (
                f"Асимметрия матрицы на позициях ({i},{j}) и ({j},{i})"
            )

def test_calculate_dissonance_matrix_weight_variation():
    """Проверка расчёта при неравных весах мыслей внутри голосов."""
    orchestra = SimpleOrchestra(embed_fn=lambda x: np.array([1.0, 0.0]))  # не используется напрямую

    orchestra.voices = {
        "lead": [
            DummyThought(np.array([1.0, 0.0]), 3.0),
            DummyThought(np.array([0.9, 0.1]), 1.0)
        ],
        "echo": [
            DummyThought(np.array([0.0, 1.0]), 1.0),
            DummyThought(np.array([0.1, 0.9]), 3.0)
        ]
    }

    names, matrix = orchestra.calculate_dissonance_matrix()

    # Проверка голосов
    assert names == ["lead", "echo"]
    assert matrix.shape == (2, 2)

    # Диагональ
    assert matrix[0, 0] == pytest.approx(0.0)
    assert matrix[1, 1] == pytest.approx(0.0)

    # Ожидаемый высокий диссонанс между lead и echo
    d_val = matrix[0, 1]
    assert 0.5 <= d_val <= 1.0

    # Симметричность
    assert matrix[0, 1] == pytest.approx(matrix[1, 0])

def test_nan_voice_handling():
    orchestra = SimpleOrchestra(embed_fn=lambda x: np.array([1.0]))
    orchestra.voices = {
        "valid": [DummyThought(np.array([1.0]), 1.0)],
        "nan_voice": [DummyThought(np.array([np.nan]), 1.0)]
    }
    names, _ = orchestra.calculate_dissonance_matrix()
    assert "nan_voice" not in names
    assert len(names) == 1

def test_zero_weight_logging(caplog):
    orchestra = SimpleOrchestra(embed_fn=lambda x: np.array([1.0]))
    orchestra.voices = {"zero": [DummyThought(np.array([1.0]), 0.0)]}
    
    with caplog.at_level(logging.DEBUG):
        names, matrix = orchestra.calculate_dissonance_matrix()
        
        # Проверяем логи
        assert "Skipping voice 'zero' - zero sum weights" in caplog.text
        
        # Проверяем возвращаемые значения
        assert names == []
        assert matrix.shape == (0, 0)