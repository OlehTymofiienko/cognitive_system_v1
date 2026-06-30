#core\orchestration\recursion_cache.py

import asyncio
from functools import wraps

class RecursionCache:
    """
    Кэширует результаты _parallel_orchestration по ключам:
    (impulse.type, intensity, complexity, bridge_threshold, tournament_depth).
    """
    def __init__(self):
        self._cache = {}

    def __call__(self, fn):
        cache = self._cache

        @wraps(fn)
        async def wrapper(instance, impulse, *args, **kwargs):
            key = (
                impulse.type,
                round(impulse.intensity, 3),
                round(impulse.complexity, 3),
                round(instance.bridge_threshold, 3),
                instance.tournament_depth
            )
            if key in cache:
                # возвращаем копию, чтобы не было внешних мутаций
                return list(cache[key])    # строка 26

            result = await fn(instance, impulse, *args, **kwargs)
            cache[key] = list(result)
            return result

        def invalidate():
            cache.clear()

        wrapper.invalidate = invalidate
        wrapper.cache = cache
        return wrapper
