# @title ReAct XML Perser
import re
from typing import Union

from langchain import hub
from langchain.tools.render import render_text_description
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.agents.agent import AgentOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from typing import List, Tuple

from libs.services.tools.search_tools import ddg_search,get_internal_links_with_text, extract_body_content,analyze_content_from_url
from libs.services.tools.htmltextextractor import HTMLTextExtractor


FINAL_ANSWER_ACTION = "<Final Answer>"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action' after 'Thought"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input' after 'Action'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)

def format_log_to_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "<Observation>",
    observation_suffix: str = "</Observation>",
    llm_prefix: str = "<Thought>",
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}{observation_suffix}\n{llm_prefix}"
    return thoughts


class ClaudeReActSingleInputOutputParser(AgentOutputParser):

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        includes_summary = "<Summary>" in text

        regex = (
            r"<Action>[\s]*(.*?)[\s]*</Action>[\s]*<Action Input>[\s]*(.*)[\s]*</Action Input>"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            summary = ""
            if includes_summary:
                summary_match = re.search(r"<Summary>(.*?)</Summary>", text, re.DOTALL)
                summary = summary_match.group(1).strip() if summary_match else ""
            final_answer = text.split(FINAL_ANSWER_ACTION)[-1].replace("</Final Answer>", "").strip()

            return AgentFinish(
                {"output": final_answer, "summary": summary}, text
            )

        if not re.search(r"<Action>[\s]*(.*?)[\s]*</Action>", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"[\s]<Action Input>[\s]*(.*)[\s]*</Action Input>", text, re.DOTALL
        ):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    @property
    def _type(self) -> str:
        return "react-single-input"



def format_log_to_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "<Observation>",
    observation_suffix: str = "</Observation>",
    llm_prefix: str = "<Thought>",
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}{observation_suffix}\n{llm_prefix}"
    return thoughts

def get_references(
    intermediate_steps: List[Tuple[AgentAction, ChatPromptTemplate]],
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    refs = []
    for action, ref in intermediate_steps:
        refs.append(ref)
    return refs

def web_search_agent_executor():
    # Define Prompt
    # デフォルトのプロンプトを取得
    prompt = hub.pull("hwchase17/react")
    # ツール情報を LLM に渡すために変換
    tools = [ddg_search, analyze_content_from_url,get_internal_links_with_text]
    tool_names = ", ".join([t.name for t in tools])
    # ツール情報をプロンプトに渡す
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=tool_names,
    )

    # カスタムプロンプトテンプレートの定義
    prompt.template = """
以下の instruction で与えられた指示に従ってください。以下のツールを利用することが可能です。

<ツール>
{tools}
</ツール>

以下のフォーマットを使用してください:

<output-format>
<Thought>指示に答えるために何をするべきか検討</Thought>
<Action>実行するアクション。必ず {tool_names} から選択する。</Action>
<Action Input>Action への入力</Action Input>
<Observation>アクションの結果</Observation>
... (Thought/Action/Action Input/Observation を N 回繰り返す)
<Thought>答えるのに必要な情報が揃いました</Thought>
<Final Answer>最終回答</Final Answer>
</output-format>

以下のinstructionを熟読して指示に従ってください。
<instruction>
{instruction}
</instruction>

それでは以下のinputをから開始してください。
<input>
{input}
</input>

Assistant:
<Thought>{agent_scratchpad}
"""


    llm = ChatOpenAI(model="gpt-4o")
    # 生成 AI が生成を止めるためのルールを定義。Bedrock に渡され、このトークンが生成されたら生成が止まる
    llm_with_stop = llm.bind(stop=["\n<Observation>"])

    # Agent の作成
    agent = {
        "input": lambda x: x["input"],
        "instruction": lambda x: x["instruction"],
        "agent_scratchpad": lambda x: format_log_to_str(x['intermediate_steps'])
    } | prompt | llm_with_stop | ClaudeReActSingleInputOutputParser()

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        return_intermediate_steps=True,
        # verbose=True,
        max_iterations=50,  # max_iterationsを増やす
        early_stopping_method="generate",  # early_stopping_methodを変更
        max_execution_time=300,  # max_execution_timeを増やす
    )
    return agent_executor


from langchain_community.callbacks.manager import get_bedrock_anthropic_callback, get_openai_callback

web_search_agent_executor = web_search_agent_executor()
instruction = """
Case1 -- urlがユーザーの入力に含まれる場合:
1. analyze_content_from_urlでタスクに必要な情報があるのか確認する
2. 必要な情報があれば、タスクに従って出力する
3. さらなる情報が必要な場合:
   - get_internal_links_with_textで元のurlから次の検索候補のurl一覧を抽出する
   - ddg_searchで検索結果のurl一覧を抽出する
4. 検索するべきurlをテキストとurlから判断し、analyze_content_from_urlを実行して情報があるか確認する
5. 3回は別のurlのコンテンツを調べて情報の精度を上げる

Case2 -- urlがユーザーの入力に含まれない場合:
1. ddg_searchで5つのページを検索する
2. 検索結果からホームページのurlを判断する
3. Case1の手順と同様に情報を抽出する

最後に:
- 収集した情報を整理し、求人票に必要な情報が揃っているか確認する
- 足りない情報がある場合は、調べきれてないことを前提で再検索してください
"""

web_search_task = """
あなたには求人票を作成するための情報を収集してもらいます。
以下の情報をホームページからすべて集めてきてください。特に、会社概要、事業内容、所在地、財務ページに注目してください。情報は詳しく出力すること。
また、不明な情報は不明として出力してください。

-業界: 〇〇業界のように具体的な業界名を記載してください。複数の業界にまたがる場合は、主要な業界を全て記載してください。
-従業員数: 検索結果の最新の従業員数を記載してください。
-売上高: 検索結果の最新の売上高を記載してください。ホームページ内に財務やIRに関わるページがあればそちらを参照してください。
-事業内容: 当社の主要な事業領域について、具体的なサービスや製品、それらが市場や顧客にどのように価値を提供しているかを記述してください。強みや独自性、技術革新の事例を交えて、情熱的かつインスパイアするような形で記載し、求職者が当社でのキャリアを通じて達成できる影響を感じられるようにしてください。
-会社設立年月: 何年何月に設立されたかを記載してください
-会社所在地: 複数の事業所がある場合は、本社と主要な支社・営業所の所在地を記載してください。ホームページ内に所在地に関わるurlがあった場合はそちらの情報を参照してください。
-理念: 当社のミッションやビジョンを記載してください。ミッションやビジョンが明確に示されていない場合は、代わりに経営理念や企業理念を記載してください。
-福利厚生: 当社の福利厚生について、社員の声や福利厚生制度、ワークライフバランス、社員の働き方などを記載してください。福利厚生に関する情報が明確に示されていない場合は、代わりに社員の声や福利厚生制度を記載してください。
-休暇: 年間休日数、有給休暇取得率、特別休暇、リフレッシュ休暇など、当社の休暇制度について記載してください。休暇制度に関する情報が明確に示されていない場合は、代わりに年間休日数や有給休暇取得率を記載してください。
-その他魅力的な情報: その他、求職者にとって魅力的な情報があれば記載してください。例えば、社内イベントや社内制度、社員の声、社内風土、社内環境、社内風景、社内制度
"""


def execute_web_search(url: str, company_name: str, web_search_task: str, instruction: str):
    from langchain_community.callbacks.manager import get_openai_callback
    
    web_search_task = f"url: {url}\ntask: {web_search_task}\n今回解析する会社は{company_name}です。"

    with get_openai_callback() as cb:
        web_search_result = web_search_agent_executor.invoke({
            "input": web_search_task,
            "instruction": instruction,
        })
        
        token_usage = {
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost_usd": cb.total_cost
        }
        
        return web_search_result, token_usage

# 使用例:
# result, usage = execute_web_search(url, company_name, web_search_task, instruction)
# print(f"Total Tokens: {usage['total_tokens']}")
# print(f"Prompt Tokens: {usage['prompt_tokens']}")
# print(f"Completion Tokens: {usage['completion_tokens']}")
# print(f"Total Cost (USD): ${usage['total_cost_usd']}")

# 使用例:
# url = "https://www.google.com/"
# company_name = "Google"
# instruction = "Googleに関する情報を収集してください。"

# result, usage = execute_web_search(url, company_name, web_search_task, instruction)

# print("検索結果:")
# print(result)
# print("\nトークン使用状況:")
# print(f"合計トークン数: {usage['total_tokens']}")
# print(f"プロンプトトークン数: {usage['prompt_tokens']}")
# print(f"完了トークン数: {usage['completion_tokens']}")
# print(f"総コスト (USD): ${usage['total_cost_usd']:.4f}")


