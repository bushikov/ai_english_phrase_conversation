import operator
import os
import random
import csv
from typing import Annotated
from pydantic import BaseModel, Field, ValidationError

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

def read_csv(file_path: str) -> list[str]:
    phrases = []
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            phrases.append(row["english"])

    return phrases

def select_phrase(phrases: list[str]) -> str:
    random_index = random.randint(0, len(phrases) - 1)
    return phrases[random_index]

class Comment(BaseModel):
    speaker: str = Field(..., description="発言者の名前")
    comment: str = Field(..., description="発言の内容")

class Conversation(BaseModel):
    original_phrase: str = Field(..., description="AIに渡された英語フレーズ")
    phrase: str = Field(..., description="AIが作成した会話文で実際に使われた英語フレーズ")
    japanese_explanation: str = Field(..., description="AIが作成した会話文で実際に使われた英語フレーズの日本語での説明")
    nuance: str = Field(..., description="AIが作成した会話文の中における、英語フレーズのニュアンスの説明")
    comments: list[Comment] = Field(..., description="AIが作成した会話文", max_items=2)

COMVERASATION_GENERATOR_PROMPT = """
あなたは優秀な英語の会話文作成者です。
あなたはこれまで、日常的に使用される英語フレーズを使った会話文を多数作成してきました。

以下のタスクを実施してください。

### 前提条件
タスクで作成する英語の会話文は、ユーザーの学習用に作成します。
ユーザーは日本の高校卒業程度の英語能力を有しています。
ユーザーは、英語フレーズを会話文の形で学習しようとしています。

### タスク
- 以下の「### 英語フレーズ」を用いた会話文を作成してください。
- 会話文における英語フレーズが表現するニュアンスや意味を説明してください。
- 会話文における英語フレーズがもつ意味やニュアンスを日本語で説明してください。

### タスク実施時の注意点
- comments（会話文）
    - 登場人物は、AliceとBobの２人です。
    - やりとりはそれぞれ１回行ってください。
    - 英語フレーズは、会話全体で１回しか使わないようにしてください。
    - 英語フレーズは、できるだけ後の人が使ってください。ただし、絶対ではないです。
    - 必ず**英語**で作成してください。
- nuance（ニュアンス）
    - 作成した会話における英語フレーズが表現するニュアンスや意味を説明してください。
    - ニュアンスの中では、絶対に英語フレーズを使わないでください。
    - 必ず**英語**で作成してください。
- japanese_explanation（英語フレーズの日本語での説明）
    - 日本語での説明の中に、英語フレーズそのものやその一部を含めないでください。
    - 必ず**日本語**で作成してください。

### 英語フレーズ
{phrase}
""".strip()

class ConversationGenerator:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm.with_structured_output(Conversation)

    def run(self, phrase: str) -> Conversation:
        prompt = ChatPromptTemplate.from_template(COMVERASATION_GENERATOR_PROMPT)
        chain = prompt | self.llm
        return chain.invoke({"phrase": phrase})

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

class Feedback(BaseModel):
    conversation: str = Field(..., description="入力された会話文")
    phrase: str = Field(..., description="入力された英語フレーズ")
    correction_result: str = Field(..., description="添削した内容やニュアンスの説明")
    examples: list[str] = Field(default_factory=list, max_items=2, description="回答例")

JUDGE_PROMPT = """
あなたは優秀な英語講師です。
日本人生徒の作成した英語文章を添削するのが得意です。

以下の前提条件をよく確認し、タスクを実施してください。

### 前提条件
「### ユーザー入力」は、「### 会話文」の「<?>」に当てはまる文章として、ユーザーが考えたものです。
「### 英語フレーズ」は、「### 会話文」の「<?>」に当てはめるのに最適な文章の例です。

### タスク
「### ユーザー入力」の英語文章を添削してください。
会話文に当てはまる自然な英語文章かどうかという観点で確認してください。
添削と同時に、この会話文から読み取れるニュアンスも一緒に説明してください。
いくつか、最大２つの回答例も提示してください。

添削した内容は、ユーザーにわかりやすいように簡潔に説明してください。
説明は、必ず日本語で行ってください。


### ユーザー入力
{user_input}

### 会話文
{conversation}

### 英語フレーズ
{phrase}
""".strip()

class Judge:
    def __init__(self, llm: BaseChatModel, question_generator: QuestionGenerator):
        self.llm = llm.with_structured_output(Feedback)
        self.question_generator = question_generator

    def run(self, user_input: str) -> Feedback:
        prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
        chain = prompt | self.llm
        return chain.invoke({
            "user_input": user_input,
            "conversation": "\n".join(self.question_generator.get_question()),
            "phrase": self.question_generator.get_correct_answer()
        })

def main():
    phrases = read_csv(os.environ["PHRASE_FILE"])
    generator = ConversationGenerator(llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1.0))

    user_input = None
    retry_count = 0
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

            print("\n(q: quit, a: answer, h: hint, j: janapese, n: next question) => ", end="")
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
        except ValidationError:
            if retry_count > 3:
                break
            retry_count += 1

if __name__ == "__main__":
    main()