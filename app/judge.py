from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from question_generator import QuestionGenerator

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