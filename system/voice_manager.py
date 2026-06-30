# system/voice_manager.py

class VoiceLineManager:
    """
    Управляет списками событий (тексты, веса) для каждого голоса оркестра.
    """

    def __init__(self, initial_voices=None):
        if initial_voices is None:
            initial_voices = ["melody", "harmony", "bass", "counterpoint"]
        self.voices = {name: [] for name in initial_voices}

    def register_voice(self, name: str):
        if name not in self.voices:
            self.voices[name] = []

    def add_event(self, name: str, text: str, weight: float = None):
        self.register_voice(name)
        entry = {"text": text}
        if weight is not None:
            entry["weight"] = weight
        self.voices[name].append(entry)

    def get_latest_text(self, name: str, fallback: str = "") -> str:
        lst = self.voices.get(name, [])
        return lst[-1]["text"] if lst else fallback

    def get_weights(self, name: str) -> list[float]:
        return [e.get("weight", 0) for e in self.voices.get(name, [])]

    def get_all(self, name: str) -> list[dict]:
        return list(self.voices.get(name, []))
