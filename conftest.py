# conftest.py

import logging
import pytest

@pytest.fixture(autouse=True)
def configure_logging():
    """
    Автоматическая настройка логгера для всех тестов:
    - уровень логирования: DEBUG
    - вывод в консоль (stdout)
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Проверяем, есть ли уже хендлеры (чтобы не дублировать)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
