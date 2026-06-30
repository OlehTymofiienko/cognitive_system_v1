import pytest
import logging
import numpy as np
from core.orchestra import SimpleOrchestra, TemporalThought, BaseOrchestra

@pytest.fixture(autouse=True)
def setup_logging():
    logging.basicConfig(level=logging.WARNING)


@pytest.fixture
def embed_fn_mock():
    """Мок функции для создания эмбеддингов."""
    def mock_embed(text: str) -> np.ndarray:
        # Простой детерминированный эмбеддинг для тестов
        if "error" in text.lower():
            raise RuntimeError("Simulated embedding error")
        return np.array([1.0 if c.isalpha() else 0.0 for c in text[:10]])
    return mock_embed

@pytest.fixture
def orchestra_instance(embed_fn_mock):
    """Фикстура создает экземпляр SimpleOrchestra для тестов."""
    return SimpleOrchestra(embed_fn=embed_fn_mock)

@pytest.fixture
def populated_orchestra(orchestra_instance):
    """Оркестр с предзаполненными данными."""
    # Добавляем тестовые мысли
    orchestra_instance.add_thought("test thought 1")
    orchestra_instance.add_thought("test thought 2")
    return orchestra_instance

@pytest.fixture(autouse=True)
def setup_logging():
    """Настраиваем логгер для тестов."""
    logger = logging.getLogger("core.orchestra")
    logger.setLevel(logging.DEBUG)
    yield
    logger.setLevel(logging.ERROR)