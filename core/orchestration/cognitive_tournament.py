from typing import List
from transformers import TextGenerationPipeline
from core.models import Thought

class CognitiveTournament:
    def __init__(
        self,
        session_topic: str,
        depth_threshold: int = 3,
        language_model: TextGenerationPipeline | None = None
    ):
        self.session_topic = session_topic
        self.depth_threshold = depth_threshold
        self.language_model = language_model

    def run(self, thoughts: List[Thought], depth: int) -> List[Thought]:
        """
        Запускает турнир, если depth > self.depth_threshold.
        Возвращает тройку: [pro_winner, con_winner, final_synthesis]
        """
        if depth <= self.depth_threshold or len(thoughts) < 2:
            return thoughts

        # 1. Классификация
        pros, cons = self._split_sides(thoughts)

        # 2. Элиминация внутри групп
        pro_champ = self._eliminate(pros)
        con_champ = self._eliminate(cons)

        # 3. Синтез
        synthesis = self._synthesize(pro_champ, con_champ)

        return [pro_champ, con_champ, synthesis]

    def _split_sides(self, thoughts: List[Thought]) -> tuple[List[Thought], List[Thought]]:
        pros, cons = [], []
        for t in thoughts:
            if "not" in t.content.lower() or "no" in t.content.lower():
                cons.append(t)
            else:
                pros.append(t)
        return pros or thoughts, cons or thoughts

    def _eliminate(self, side: List[Thought]) -> Thought:
        # простая рулетка: выбираем max по coherence
        return max(side, key=lambda t: t.coherence)

    def _synthesize(self, pro: Thought, con: Thought) -> Thought:
        prompt = (f"On topic «{self.session_topic}», "
                  f"reconcile these views:\nPro: {pro.content}\nCon: {con.content}")
        if self.language_model:
            out = self.language_model(prompt)[0]["generated_text"]
        else:
            out = f"Bridge: {pro.content} & {con.content}"
        return Thought(
            content=out,
            voice="melody",
            coherence=(pro.coherence + con.coherence) / 2,
            metadata={"tournament_of": [pro.content, con.content]}
        )
