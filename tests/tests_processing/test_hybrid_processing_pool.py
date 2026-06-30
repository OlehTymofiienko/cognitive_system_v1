#tests\test_hybrid_processing_pool.py

import pytest
from core.models import Impulse, Thought
from core.processing.hybrid_processing_pool import HybridProcessingPool

@pytest.mark.asyncio
async def test_hybrid_routing_exploratory():
    pool = HybridProcessingPool("AI Test")
    impulse = Impulse(type="exploratory", intensity=0.3, complexity=3.0)
    result = await pool.process(impulse)
    assert isinstance(result, Thought)
    assert "Locally processed" in result.content

@pytest.mark.asyncio
async def test_hybrid_routing_async_parallel():
    pool = HybridProcessingPool("AI Test")
    impulse = Impulse(type="reflective", intensity=0.9, complexity=6.0)
    result = await pool.process(impulse)
    assert isinstance(result, list)
    assert len(result) == 3

@pytest.mark.asyncio
async def test_hybrid_routing_delegate():
    pool = HybridProcessingPool("AI Test")
    impulse = Impulse(type="conflict", intensity=0.5, complexity=4.0, priority="external")
    result = await pool.process(impulse)
    assert "Delegated" in result.content

@pytest.mark.asyncio
async def test_hybrid_routing_dreamlike():
    pool = HybridProcessingPool("Fantasy")
    impulse = Impulse(type="dreamlike", intensity=0.1, complexity=2.0)
    result = await pool.process(impulse)
    assert isinstance(result, Thought)
    assert "wondrous dream emerges" in result.content

@pytest.mark.asyncio
async def test_hybrid_routing_system_priority_simple():
    pool = HybridProcessingPool("SystemTest")
    impulse = Impulse(type="exploratory", intensity=0.1, complexity=3.5, priority="system")
    result = await pool.process(impulse)
    assert isinstance(result, Thought)
    # попадает в _local_cpu_processing via MetaConductor
    assert "Locally processed" in result.content

@pytest.mark.asyncio
async def test_hybrid_routing_system_priority_parallel():
    pool = HybridProcessingPool("SystemTest")
    impulse = Impulse(type="reflective", intensity=0.9, complexity=6.0, priority="system")
    result = await pool.process(impulse)
    assert isinstance(result, list)
    assert len(result) == 3

@pytest.mark.asyncio
async def test_hybrid_routing_default_fallback():
    """Тест фоллбэк-обработки (строка 38)"""
    pool = HybridProcessingPool("FallbackTest")
    # Импульс, который не подходит ни под одно условие
    impulse = Impulse(type="generic", intensity=0.5, complexity=4.0)
    result = await pool.process(impulse)
    
    assert isinstance(result, Thought)
    assert "Default processing" in result.content
    assert result.voice == "melody"
    assert result.coherence == 0.5

@pytest.mark.asyncio
async def test_default_method_thought_creation_01():
    """Тест создания Thought в методе _default (строка 62)"""
    pool = HybridProcessingPool("DefaultTest")
    impulse = Impulse(type="test_type", intensity=0.3, complexity=2.0)
    result = await pool._default(impulse)
    
    # Проверяем все атрибуты Thought
    assert isinstance(result, Thought)
    assert result.content == f"Default processing of test_type"
    assert result.voice == "melody"
    assert result.coherence == 0.5
    # Проверяем что metadata существует, но пустой
    assert hasattr(result, 'metadata')
    assert result.metadata == {}

@pytest.mark.asyncio
async def test_default_method_thought_creation_02():
    """Тест создания Thought в методе _default (строка 62)"""
    pool = HybridProcessingPool("DefaultTest")
    impulse = Impulse(type="test_type", intensity=0.3, complexity=2.0)
    result = await pool._default(impulse)
    
    # Проверяем что объект создан с минимальными required полями
    assert isinstance(result, Thought)
    assert result.content.startswith("Default processing of")
    assert result.voice == "melody"
    assert result.coherence == 0.5

@pytest.mark.asyncio
async def test_hybrid_routing_edge_case():
    """Тест обработки импульса с пограничными параметрами"""
    pool = HybridProcessingPool("EdgeCaseTest")
    # complexity ровно 5.0 - граничное значение для local_cpu_processing
    impulse = Impulse(type="exploratory", intensity=0.3, complexity=5.0)
    result = await pool.process(impulse)
    
    # Должен попасть в _default, а не в _local_cpu_processing
    assert "Default processing" in result.content
