from datetime import datetime, timedelta
from langchain.agents import tool
from duckduckgo_search import DDGS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

@tool
def extract_additional_info(query: str, transcript: str) -> str:
  """
  It further searches the text for missing information.

  Args:
    query(str): The information to search for.

  Returns:
    str: A list of potentially relevant information
  """
#   chat = ChatAnthropic(temperature=0, max_tokens_to_sample=4096, model="claude-3-haiku-20240307")
  chat = ChatOpenAI(model="gpt-4o")
  system = "あなたはテキストの中から、必用な情報を抽出するプロです。"
  human = """
  <task>以下のinterviewから{query}に関する情報を出力してください。</task>
  <instruct>情報の記載がなければ、"不明"で出力してください。{query}に関する情報は、複数抽出してください。直接的に関係がなくても、使えそうな情報であれば出力すること。</instruct>
  <interview>interview: {interview}</interview>
  """
  prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
  chain = prompt | chat | StrOutputParser()
  ans = chain.invoke(
    {
        "query": query,
        "interview": transcript
    }
  )

  return ans + "\n"
