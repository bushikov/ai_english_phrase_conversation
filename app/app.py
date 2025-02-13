import os
import random
import csv
from pydantic import ValidationError

from langchain_google_genai import ChatGoogleGenerativeAI

from conversation_generator import Conversation, ConversationGenerator
from question_generator import QuestionGenerator
from judge import Judge
from speech import Speech

REPO_PATH = "/speeches"

def read_csv(file_path: str) -> list[str]:
    phrases = []
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            phrases.append(row["english"])

    return phrases

def select_phrase(phrases: list[str]) -> str:
    random_index = random.randint(0, len(phrases) - 1)
    return phrases[random_index]

def main():
    phrases = read_csv(os.environ["PHRASE_FILE"])
    generator = ConversationGenerator(llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1.0))

    user_input = None
    retry_count = 0
    speech = None
    speech_choice = "s: speech, " if os.getenv("ELEVEN_LABS_API_KEY") else ''

    while True:
        try:
            if user_input is None:
                phrase = select_phrase(phrases)
                conversation = generator.run(phrase)

                retry_count = 0

                question_generator = QuestionGenerator(conversation)
                judge = Judge(
                    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0),
                    question_generator=question_generator
                )

                print("### CONVERSATION ##################################")
                for c in question_generator.get_question():
                    print(c)

            print(f"\n(q: quit, a: answer, h: hint, j: janapese, {speech_choice}n: next question) => ", end="")
            user_input = input()

            match user_input.lower():
                case "q":
                    print("Good bye")
                    break
                case "a":
                    print("### Answer")
                    print(question_generator.get_correct_answer())
                    print(question_generator.get_japanese())
                case "h":
                    print("### Hint")
                    print(question_generator.get_hint())
                case "j":
                    print("### Japanese")
                    print(question_generator.get_japanese())
                case "s":
                    if not speech:
                        speech = Speech(api_key=os.environ["ELEVEN_LABS_API_KEY"], repo_path=REPO_PATH)
                    speech.run(conversation.phrase)
                case "n":
                    user_input = None
                case "":
                    continue
                case _:
                    print("Correct" if question_generator.confirm(user_input) else "Wrong")

                    feedback = judge.run(user_input)

                    print("### FEEDBACK")
                    print(feedback.correction_result)
                    print()
                    for i, example in enumerate(feedback.examples):
                        print(f"--- Example {i}: {example}")
        except ValidationError as e:
            if retry_count > 3:
                print("FAILURE")
                print(e)
                break
            retry_count += 1

if __name__ == "__main__":
    main()