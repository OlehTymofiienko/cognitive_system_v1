#tests\test_voice_manager.py

import pytest
from system.voice_manager import VoiceLineManager

class TestVoiceLineManager:
    def test_init_with_default_voices(self):
        """Тест инициализации с голосами по умолчанию"""
        manager = VoiceLineManager()
        assert set(manager.voices.keys()) == {"melody", "harmony", "bass", "counterpoint"}
        for voice in manager.voices.values():
            assert voice == []

    def test_init_with_custom_voices(self):
        """Тест инициализации с пользовательскими голосами"""
        custom_voices = ["voice1", "voice2"]
        manager = VoiceLineManager(custom_voices)
        assert set(manager.voices.keys()) == set(custom_voices)
        for voice in manager.voices.values():
            assert voice == []

    def test_register_voice_new(self):
        """Тест регистрации нового голоса"""
        manager = VoiceLineManager()
        manager.register_voice("new_voice")
        assert "new_voice" in manager.voices
        assert manager.voices["new_voice"] == []

    def test_register_voice_existing(self):
        """Тест регистрации существующего голоса (не должен изменяться)"""
        manager = VoiceLineManager(["existing_voice"])
        manager.register_voice("existing_voice")
        assert "existing_voice" in manager.voices
        assert manager.voices["existing_voice"] == []

    def test_add_event_new_voice(self):
        """Тест добавления события в новый голос"""
        manager = VoiceLineManager()
        manager.add_event("new_voice", "text1")
        assert "new_voice" in manager.voices
        assert len(manager.voices["new_voice"]) == 1
        assert manager.voices["new_voice"][0]["text"] == "text1"
        assert "weight" not in manager.voices["new_voice"][0]

    def test_add_event_with_weight(self):
        """Тест добавления события с весом"""
        manager = VoiceLineManager()
        manager.add_event("voice", "text1", 0.5)
        assert manager.voices["voice"][0]["text"] == "text1"
        assert manager.voices["voice"][0]["weight"] == 0.5

    def test_add_event_multiple(self):
        """Тест добавления нескольких событий"""
        manager = VoiceLineManager()
        manager.add_event("voice", "text1")
        manager.add_event("voice", "text2", 0.7)
        manager.add_event("voice", "text3")
        
        assert len(manager.voices["voice"]) == 3
        assert manager.voices["voice"][0]["text"] == "text1"
        assert manager.voices["voice"][1]["text"] == "text2"
        assert manager.voices["voice"][1]["weight"] == 0.7
        assert manager.voices["voice"][2]["text"] == "text3"

    def test_get_latest_text_with_entries(self):
        """Тест получения последнего текста для голоса с событиями"""
        manager = VoiceLineManager()
        manager.add_event("voice", "text1")
        manager.add_event("voice", "text2")
        assert manager.get_latest_text("voice") == "text2"

    def test_get_latest_text_empty_voice(self):
        """Тест получения последнего текста для пустого голоса"""
        manager = VoiceLineManager()
        assert manager.get_latest_text("nonexistent") == ""
        assert manager.get_latest_text("nonexistent", "fallback") == "fallback"

    def test_get_weights(self):
        """Тест получения весов событий"""
        manager = VoiceLineManager()
        manager.add_event("voice", "text1", 0.1)
        manager.add_event("voice", "text2")
        manager.add_event("voice", "text3", 0.3)
        
        weights = manager.get_weights("voice")
        assert weights == [0.1, 0, 0.3]

    def test_get_weights_empty_voice(self):
        """Тест получения весов для пустого голоса"""
        manager = VoiceLineManager()
        assert manager.get_weights("nonexistent") == []

    def test_get_all(self):
        """Тест получения всех событий голоса"""
        manager = VoiceLineManager()
        manager.add_event("voice", "text1", 0.1)
        manager.add_event("voice", "text2")
        
        all_events = manager.get_all("voice")
        assert len(all_events) == 2
        assert all_events[0]["text"] == "text1"
        assert all_events[0]["weight"] == 0.1
        assert all_events[1]["text"] == "text2"
        assert "weight" not in all_events[1]

    def test_get_all_empty_voice(self):
        """Тест получения всех событий для пустого голоса"""
        manager = VoiceLineManager()
        assert manager.get_all("nonexistent") == []