from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from libs.schemas.job_posting import JobPosting
import pandas as pd
import os

def create_data_validation_prompt():
    data_validation_task = """
    Task:
    求人票作成の仕事をしていただきます。

    Detail:
    ユーザーから2つのデータが与えられます。
    1. クライアントとの商談の議事録
    2. クライアントについてのweb検索の結果

    これらを用いて、求人票を作成します。
    schemaに従って、情報を落とすことなくデータを整理してください。
    ポイントは、
    1. 与えられたデータの内容をもとに求人票を作成すること。できるだけ内容は埋めること。
    2. 改行をうまく使って、読みやすいフォーマットにすること。
    3. 求職者にとってわかりやすく説明すること。
    4. 事業内容は、具体的なサービスや製品、それらが市場や顧客にどのように価値を提供しているかを記述してください。強みや独自性、技術革新の事例を交えて、情熱的かつインスパイアするような形で記載し、求職者が当社でのキャリアを通じて達成できる影響を感じられるようにすること。
    5. 業務内容は、求職者にわかるように、具体的な業務内容を記載して魅力的に感じるように記述すること。
    6. 給与詳細は、月給、手当、賞与、昇給を記述すること。
    """

    return ChatPromptTemplate.from_messages([
        ("system", data_validation_task),
        ("user", "クライアントとの商談の議事録: {minute}"),
        ("user", "クライアントについてのweb検索の結果: {web_search_result}")
    ])

def generate_job_posting(minute, web_search_result, llm: ChatOpenAI | ChatBedrock):
    data_validation_prompt = create_data_validation_prompt()
    data_validation_chain = data_validation_prompt | llm.with_structured_output(JobPosting)

    with get_openai_callback() as cb:
        job_posting = data_validation_chain.invoke({
            "minute": minute,
            "web_search_result": web_search_result
        })
        
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    return job_posting

def save_job_posting_to_csv(job_posting, output_dir, output_filename):
    reference_columns = ['☆企業名', '☆従業員数', '☆売上高', '☆会社設立日',
                         '☆事業内容', '業界', '会社所在地', '☆募集ポジション名',
                         '☆雇用形態（期間）', '☆試用期間の有無', '☆業務内容',
                         '☆勤務地(詳細)', '☆応募資格', '☆選考プロセス',
                         '☆年収上限 [万円]', '☆年収下限 [万円]', '☆給与(詳細)',
                         '☆休日休暇', '☆勤務時間', '☆待遇・福利厚生',
                         '☆受動喫煙防止措置', '転職回数', '学歴', '年齢',
                         'フィー（紹介手数料）', '選考対策', '求める人物像', 'その他魅力',
                         '採用担当者ID']

    df = pd.DataFrame([job_posting.dict()])
    df.columns = reference_columns

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, output_filename)
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")

def main(minute, web_search_result, output_dir, output_filename):
    job_posting = generate_job_posting(minute, web_search_result)
    save_job_posting_to_csv(job_posting, output_dir, output_filename)

if __name__ == "__main__":
    # ここで実際の minute, web_search_result, folder_path を指定して main 関数を呼び出す
    main("サンプル議事録", "サンプルWeb検索結果", "./output", "job_postings.csv")