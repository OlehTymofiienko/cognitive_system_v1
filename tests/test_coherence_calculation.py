# tests/test_coherence_calculation.py

import pytest
import numpy as np
from sentence_transformers import SentenceTransformer
from core.models import Thought

@pytest.mark.asyncio
async def test_coherence_with_embeddings():
    # Инициализация модели
    model = SentenceTransformer("all-MiniLM-L6-v2")

    thoughts = [
        Thought(content="Изучение искусственного интеллекта", voice='melody', coherence=0.0),
        Thought(content="Разработка когнитивных архитектур", voice='counterpoint', coherence=0.0),
        Thought(content="Моделирование процессов мышления", voice='bass', coherence=0.0)
    ]

    # Векторизация
    embeddings = model.encode([t.content for t in thoughts], normalize_embeddings=True)

    # Расчёт pairwise similarity
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1])
        print(f"Similarity T{i} vs T{i+1}: {sim:.3f}")
        similarities.append(sim)

    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    avg_coherence = round(sum(similarities) / len(similarities), 3)
    print(f"Средняя когерентность между мыслями: {avg_coherence:.3f}")

    # Временный порог, пока не внедрён BridgeSynthesizer и динамическая фильтрация
    assert avg_coherence >= 0.50, f"Когерентность слишком низкая: {avg_coherence:.3f}"