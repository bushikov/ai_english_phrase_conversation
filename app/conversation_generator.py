from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

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