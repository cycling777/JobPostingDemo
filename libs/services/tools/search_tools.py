from langchain.agents import tool
from duckduckgo_search import DDGS
from datetime import datetime, timedelta
from libs.services.tools.htmltextextractor import HTMLTextExtractor
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate




def create_timelimit(years):
    today = datetime.today().date()
    past_date = today - timedelta(days=years*365)
    timelimit = f"timelimit='{past_date}..{today}'"

    return timelimit

@tool
def ddg_search(query: str) -> str:
    """
    Performs a DuckDuckGo search using the provided query and returns the search results as a formatted string.

    Use this function to search for information on the web using DuckDuckGo and retrieve the top search results.
    The function takes a search query as input and returns a formatted string containing the titles, summaries,
    and URLs of the top search results.
    """
    search = DDGS()
    results = search.text(
        keywords=query,
        region='jp-jp',
        safesearch='off',
        timelimit=create_timelimit(3),
        max_results=5
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
def analyze_content_from_url(input_str: str) -> str:
    """
    Extracts the body content from the specified URL, analyzes it using the chatgpt language model,
    and returns a formatted report in Japanese based on the given task.

    Args:
        input_str (str): A string containing the URL and task, separated by a delimiter (e.g., "url|task").

    Returns:
        str: A formatted string containing the generated report in Japanese, including the title,
             content (in bullet points), and the source URL.
    """
    # Extract the URL and task from the input string
    url, task = input_str.split("|", 1)

    # Extract the body content from the URL
    body_content = extract_body_content(url)
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"


    chat = ChatOpenAI(model="gpt-4o")
    system = "あなたは求人情報を収集するアシスタントです。"
    # human1 = "はじめまして！よろしくお願いします！"
    # assistant1 = "はじめまして！私は、あなたの情報収集のお手伝いをして、日本語のレポートを提出します。何についての情報を収集したいのか、レポートのトピックは何ですか？詳しく教えてください、私はお手伝いします！"
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