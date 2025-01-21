from conversation_generator import Conversation

class QuestionGenerator:
    def __init__(self, conversation: Conversation):
        self.conversation = conversation

    def get_question(self) -> list[str]:
        return [
            c.comment.replace(self.conversation.phrase, "<?>")
            for c in self.conversation.comments
        ]

    def get_hint(self) -> str:
        return self.conversation.nuance.replace(self.conversation.phrase.rstrip(".!"), "<?>")

    def get_japanese(self) -> str:
        return self.conversation.japanese_explanation

    def get_correct_answer(self) -> str:
        return self.conversation.phrase

    def confirm(self, input: str) -> bool:
        return self.conversation.phrase.lower().rstrip(".!?") == input.lower().rstrip(".!?")