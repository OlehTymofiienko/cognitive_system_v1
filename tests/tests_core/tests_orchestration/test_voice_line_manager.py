import os
import sys
import pytest
import types

from core.orchestration.voice_line_manager import VoiceLineManager
from core.models import Thought


# =========================
# Fixtures & Helpers
# =========================

@pytest.fixture
def manager():
    return VoiceLineManager("TestSession")


@pytest.fixture
def dummy_thoughts():
    # Используется в старом тесте
    return [
        Thought("Hello", voice="melody", coherence=0.5),
        Thought("World", voice="melody", coherence=0.6),
        Thought("Opposite", voice="bass", coherence=0.4,
                metadata={"bridge_of": [0, 1]})
    ]


# =========================
# Объединённые тесты
# =========================

def test_voice_line_manager_basic(manager, dummy_thoughts):
    # Старый базовый тест
    for t in dummy_thoughts:
        manager.add_thought(t)

    script = manager.get_script()
    assert "Group: melody" in script
    assert "melody: Hello" in script
    assert "Group: [0, 1]" in script


def test_export_json_and_internal_lines(manager, dummy_thoughts):
    for t in dummy_thoughts:
        manager.add_thought(t)

    data = manager.export_json()
    assert data["session_topic"] == "TestSession"
    # группы совпадают
    assert data["groups"] == manager.lines
    # минимальные проверки по структуре
    assert isinstance(data["groups"], list)
    assert all("thoughts" in g for g in data["groups"])


def test_draw_without_dependencies(manager, tmp_path, monkeypatch):
    # эмулируем отсутствие networkx/plt
    monkeypatch.setitem(sys.modules, "networkx", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    out = manager.draw(str(tmp_path/"out.png"))
    assert out is None
    assert not (tmp_path/"out.png").exists()


def test_draw_with_dependencies(manager, dummy_thoughts, tmp_path, monkeypatch):
    import types

    # заглушка графа с необходимыми методами
    class DummyGraph:
        def add_node(self, *args, **kwargs): pass
        def add_edge(self, *args, **kwargs): pass

    NX = types.SimpleNamespace(
        DiGraph=lambda: DummyGraph(),
        spring_layout=lambda g: {},
        get_node_attributes=lambda g, k: {},
        draw=lambda *a, **k: None
    )
    PLT = types.SimpleNamespace(
        savefig=lambda fn: open(fn, "w").close(),
        close=lambda: None
    )

    monkeypatch.setitem(sys.modules, "networkx", NX)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", PLT)

    # заполняем хотя бы одну группу
    for t in dummy_thoughts:
        manager.add_thought(t)

    out_file = tmp_path/"graph.png"
    manager.draw(str(out_file))

    # файл должен быть создан
    assert out_file.exists()

