class Config:
    # Параметры памяти
    WORKING_MEMORY_CAPACITY = 7
    SHORT_TERM_CAPACITY = 100
    
    # Пороговые значения
    COHERENCE_THRESHOLD = 0.4
    CREATIVITY_THRESHOLD = 0.3
    
    # Графовые параметры
    MAX_GRAPH_COMPLEXITY = 1000

    # Настройки логирования
    LOGGING_LEVEL = "detailed"  # "minimal" | "detailed" | "debug"
    
    # Шаблоны мыслей
    THOUGHT_TEMPLATES = {
        "exploratory": [...],
        "reflective": [...],
        "integrative": [...] 
    }

    # Временные параметры
    PROCESSING_INTERVAL = 0.5  # секунд между циклами
    
    # Настройки контекста
    MAX_CONTEXTS = 5
    
    # Параметры когерентности
    COHERENCE = {
        'MIN_VALUE': 0.1,
        'MAX_VALUE': 1.0,
        'PAIRWISE_WEIGHT': 0.7,  # Исправлено с PAIRWIRE на PAIRWISE
        'CONTEXT_WEIGHT': 0.3,
        'MIN_ADJUSTMENT': -0.1,
        'MAX_ADJUSTMENT': 0.1,
        'HISTORY_SIZE': 5  # Количество учитываемых предыдущих мыслей
    }
    
    # Доверительные пороги
    TRUST_THRESHOLDS = {
        'LOW': 0.4,
        'MEDIUM': 0.7,
        'HIGH': 0.9
    }
