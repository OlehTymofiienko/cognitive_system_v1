# core/orchestration/utils.py

from typing import List, Tuple
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

from core.models import Thought
from core.semantic_analyzer import SemanticAnalyzer

def calculate_dissonance_matrix(thoughts: List[Thought]) -> Tuple[List[str], np.ndarray]:
    """
    Для списка Thought:
      - собираем voices = [t.voice …]
      - получаем эмбеддинги: если у t есть .emb, используем его,
        иначе вызываем SemanticAnalyzer.get_embedding(t.content)
      - диссонанс(i,j) = 1 − cosine_similarity(emb_i, emb_j)
      - заменяем NaN на 0.0, возвращаем симметричную матрицу [n×n]
    """
    names = [t.voice for t in thoughts]
    n = len(thoughts)
    M = np.zeros((n, n), dtype=float)
    if n < 2:
        return names, M

    analyzer = SemanticAnalyzer()
    embeddings: List[np.ndarray] = []
    for t in thoughts:
        if hasattr(t, 'emb'):
            embeddings.append(t.emb)
        else:
            embeddings.append(analyzer.get_embedding(t.content))

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(
                embeddings[i][np.newaxis],
                embeddings[j][np.newaxis]
            )[0][0]
            d = 1.0 - sim
            if math.isnan(d):
                d = 0.0
            M[i, j] = M[j, i] = d

    return names, M
