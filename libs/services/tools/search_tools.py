from typing import Dict, Any
from langchain.agents import tool
from duckduckgo_search import DDGS
from datetime import datetime, timedelta
from libs.services.tools.htmltextextractor import HTMLTextExtractor
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field





def create_timelimit(years):
    today = datetime.today().date()
    past_date = today - timedelta(days=years*365)
    timelimit = f"timelimit='{past_date}..{today}'"

    return timelimit

@tool
def ddg_search(query: str, region: str = 'jp-jp', max_results: int = 5) -> str:
    """
    Performs a DuckDuckGo search using the provided query and returns the search results as a formatted string.

    Args:
        query (str): The search query.
        region (str, optional): The region for the search. Defaults to 'jp-jp'.
        max_results (int, optional): The maximum number of results to return. Defaults to 5.

    Returns:
        str: A formatted string containing the search results.
    """
    search = DDGS()
    results = search.text(
        keywords=query,
        region=region,
        safesearch='off',
        timelimit=create_timelimit(3),
        max_results=max_results
    )
    results_str = "\n\n".join(
        [f"title: {result['title']}\nsummary: {result['body']}\nurl: {result['href']}" for result in results]
    ) + "\n\n"
    return results_str


@tool
def extract_links_from_url(url: str) -> str:
    """
    Extracts links from the specified URL and returns them as a comma-separated string.

    Args:
        url (str): The URL of the webpage to extract links from.

    Returns:
        str: A comma-separated string containing the extracted links, or None if an error occurred.
    """
    extractor = HTMLTextExtractor()
    links = extractor(url, extract_type='links')
    return links

@tool
def extract_body_content(url: str) -> str:
    """
    Extracts the body content from the specified URL and returns it as a string.

    Args:
        url (str): The URL of the webpage to extract the body content from.

    Returns:
        str: The extracted body content as a string.
    """
    extractor = HTMLTextExtractor()
    body_content = extractor(url, extract_type='text')
    return body_content

@tool
def analyze_content_from_url(url: str, task: str, model_type: str = "bedrock", model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0") -> str:
    """
    Extracts the body content from the specified URL, analyzes it using the specified language model,
    and returns a formatted report in Japanese based on the given task.

    Args:
        url (str): The URL to analyze.
        task (str): The task to perform on the content.
        model_type (str, optional): The type of model to use ("openai" or "bedrock"). Defaults to "bedrock".
        model_id (str, optional): The model ID to use for analysis. Defaults to "anthropic.claude-3-5-sonnet-20240620-v1:0" for OpenAI.

    Returns:
        str: A formatted string containing the generated report in Japanese.
    """
    body_content = extract_body_content(url)
    
    if model_type.lower() == "openai":
        chat = ChatOpenAI(model=model_id)
    elif model_type.lower() == "bedrock":
        chat = ChatBedrock(model_id=model_id, region_name="ap-northeast-1")
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'openai' or 'bedrock'.")

    system = "あなたは求人情報を収集するアシスタントです。"
    human2 = f"""
    今回あなたにお願いしたいのは提供されたbody情報の整理です。body情報は以下です。
    {body_content}
    ここからは具体的なタスクの内容です。
    - 適切な項目を設定して、情報を落とすことなく箇条書きにしてください
    - 情報のソースがわかるようにurlを記載してください。
    - 特に、"{task}"に関連する情報に注目してください。
    具体的な出力の形式は以下のようにお願いします。
    <Report>
    <Title>タイトル</Title>
    <Content>
    - 箇条書きのコンテンツ
    </Content>
    <Url>ソースのurl</Url>
    </Report>
    それでは情報収集のお手伝いをお願いします。
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system),
         ("human", human2)]
    )
    chain = prompt | chat | StrOutputParser()

    report = chain.invoke({"body_content": body_content, "task": task})

    return report + "\n"

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

@tool
def get_internal_links_with_text(url: str) -> str:
    """
    Extracts internal links from the specified URL and returns a string representation
    of each link's URL and text.

    Args:
        url (str): The base URL from which to extract internal links.

    Returns:
        str: A string containing the URLs and text of internal links, separated by newlines.
             Each link is represented in the format "text : URL".

    Raises:
        requests.exceptions.RequestException: If an HTTP error occurs.
    """
    # URLからHTMLコンテンツを取得
    response = requests.get(url)
    # response.raise_for_status()  # HTTPエラーがある場合は例外を投げる

    # BeautifulSoupを使用してHTMLを解析し、すべてのリンクを抽出
    soup = BeautifulSoup(response.text, 'html.parser')
    all_links = soup.find_all('a', href=True)  # href属性がある<a>タグをすべて取得

    # ベースURLのホストを取得
    base_host = urlparse(url).netloc

    # 内部リンクのみを抽出し、重複を削除
    unique_internal_links = {}
    for link in all_links:
        href = link['href']
        parsed_href = urlparse(href)

        # ドメインが一致するか、リンクが相対URLの場合を確認
        if parsed_href.netloc == base_host or not parsed_href.netloc:
            # 相対リンクを絶対URLに変換
            absolute_link = requests.compat.urljoin(url, href)

            # テキストの取得
            text = link.get_text(strip=True)
            if not text:
                # テキストが空の場合、alt属性を探す
                img = link.find('img')
                text = img['alt'] if img and 'alt' in img.attrs else 'No text'

            # リンクとテキストを辞書に追加して重複を防ぐ
            if absolute_link not in unique_internal_links:
                unique_internal_links[absolute_link] = text

    # リンクとリンクテキストを文字列で返す
    return '\n'.join(f"'{text}' : {href}" for href, text in unique_internal_links.items()) + '\n'