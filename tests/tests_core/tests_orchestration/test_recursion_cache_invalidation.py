# tests/test_recursion_cache_invalidation.py

import pytest
from unittest.mock import MagicMock
from unittest.mock import AsyncMock
from core.orchestration.conductor_optimizer import ConductorOptimizer
from core.orchestration.meta_conductor import MetaConductor
from core.orchestration.recursion_cache import RecursionCache

class TestRecursionCacheInvalidation:
    @pytest.fixture
    def mock_meta(self):
        meta = MetaConductor(session_topic="test")
        # Подменяем атрибут _parallel_orchestration и его метод invalidate
        meta._parallel_orchestration = MagicMock()
        meta._parallel_orchestration.invalidate = MagicMock()
        return meta

    def test_cache_invalidation_on_optimize(self, mock_meta):
        optimizer = ConductorOptimizer(mock_meta)
        # Три любых «импульса» для теста
        impulses = [MagicMock(), MagicMock(), MagicMock()]

        # Синхронно запускаем optimize_sync
        result = optimizer.optimize_sync(impulses)

        # Метод invalidate() должен был вызваться ровно один раз
        mock_meta._parallel_orchestration.invalidate.assert_called_once()
        print(dir(mock_meta._parallel_orchestration.invalidate))

        # Проверяем, что результат — словарь с базовыми полями
        assert isinstance(result, dict)
        assert "avg_coherence" in result
        assert "history" in result

    def test_cache_invalidation_without_parallel_orchestration(self, mock_meta):
        # Убираем возможность invalidate — проверяем, что не упадёт
        del mock_meta._parallel_orchestration

        optimizer = ConductorOptimizer(mock_meta)
        result = optimizer.optimize_sync([])

        # Прошли без исключений, получили словарь-результат
        assert isinstance(result, dict)
        assert "history" in result


class TestRecursionCache:
    @pytest.mark.asyncio
    async def test_cache_hit_returns_copy(self):
        """Тест что при попадании в кэш возвращается копия списка (строка 26)"""
        # Создаем тестовые данные
        mock_impulse = MagicMock()
        mock_impulse.type = "test_type"
        mock_impulse.intensity = 0.5
        mock_impulse.complexity = 0.7
        
        mock_instance = MagicMock()
        mock_instance.bridge_threshold = 0.8
        mock_instance.tournament_depth = 3
        
        # Ожидаемый результат в кэше
        cached_result = ["result1", "result2"]
        
        # Создаем декоратор с предзаполненным кэшем
        cache = RecursionCache()
        key = (
            "test_type",
            0.5,
            0.7,
            0.8,
            3
        )
        cache._cache[key] = cached_result.copy()  # заполняем кэш
        
        # Мокируем оригинальную функцию
        async def mock_fn(instance, impulse, *args, **kwargs):
            return ["new_result"]
        
        decorated_fn = cache(mock_fn)
        
        # Вызываем декорированную функцию
        result = await decorated_fn(mock_instance, mock_impulse)
        
        # Проверяем что вернулся кэшированный результат
        assert result == cached_result
        # Проверяем что это копия (не тот же объект)
        assert result is not cached_result
        # Проверяем что оригинальная функция не вызывалась
        mock_fn.assert_not_called()
        
        # Проверяем что модификация результата не влияет на кэш
        result.append("modified")
        assert len(cache._cache[key]) == len(cached_result)  # кэш не изменился

    @pytest.mark.asyncio
    async def test_cache_hit_returns_copy(self):
        """Тест что при попадании в кэш возвращается копия списка (строка 26)"""
        # Создаем тестовые данные
        mock_impulse = MagicMock()
        mock_impulse.type = "test_type"
        mock_impulse.intensity = 0.5
        mock_impulse.complexity = 0.7
        
        mock_instance = MagicMock()
        mock_instance.bridge_threshold = 0.8
        mock_instance.tournament_depth = 3
        
        # Ожидаемый результат в кэше
        cached_result = ["result1", "result2"]
        
        # Создаем декоратор с предзаполненным кэшем
        cache = RecursionCache()
        key = (
            "test_type",
            0.5,
            0.7,
            0.8,
            3
        )
        cache._cache[key] = cached_result.copy()  # заполняем кэш
        
        # Создаем mock-функцию вместо обычной функции
        mock_fn = AsyncMock(return_value=["new_result"])
        
        decorated_fn = cache(mock_fn)
        
        # Вызываем декорированную функцию
        result = await decorated_fn(mock_instance, mock_impulse)
        
        # Проверяем что вернулся кэшированный результат
        assert result == cached_result
        # Проверяем что это копия (не тот же объект)
        assert result is not cached_result
        # Проверяем что оригинальная функция не вызывалась
        mock_fn.assert_not_called()
        
        # Проверяем что модификация результата не влияет на кэш
        result.append("modified")
        assert len(cache._cache[key]) == len(cached_result)  # кэш не изменился
