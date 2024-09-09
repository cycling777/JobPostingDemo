from langchain_community.callbacks.manager import get_openai_callback, get_bedrock_anthropic_callback
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from libs.services.web_search import execute_web_search, web_search_task, instruction

if __name__ == "__main__":
    url = "https://www.google.com/"
    company_name = "Google"
    instruction = "Googleに関する情報を収集してください。"

    bedrock_client = ChatBedrock(
    model_id='anthropic.claude-3-5-sonnet-20240620-v1:0',
    region_name='ap-northeast-1'
    )
    result, usage = execute_web_search(url, company_name, web_search_task, instruction, bedrock_client)

    print(f"Total Tokens: {usage['total_tokens']}")
    print(f"Prompt Tokens: {usage['prompt_tokens']}")
    print(f"Completion Tokens: {usage['completion_tokens']}")
    print(result)