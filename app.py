import streamlit as st
import os
import ffmpeg
import pathlib
import time
from threading import Thread
from queue import Queue
# from deepgram import DeepgramClient, PrerecordedOptions
import os
import json
# from pydub import AudioSegment
# from langchain_openai import ChatOpenAI
# from langchain_community.callbacks import get_openai_callback
import shutil

import vertexai
import vertexai.generative_models as genai
from google.cloud import storage
# from libs.core.config import GEMINI_MODEL_NAME, GOOGLE_APPLICATION_CREDENTIALS, GCP_BUCKET_NAME
from libs.services.analyze_sounds import analyze_sounds
from libs.services.prompts.business_meeting import generate_business_meeting_prompt
from libs.services.web_search import execute_web_search, web_search_task, instruction
from libs.services.generate_posting import generate_job_posting, save_job_posting_to_csv
from libs.services.media_converter import FFmpegM4AConverter
import mimetypes

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GEMINI_MODEL_NAME = st.secrets["GEMINI_MODEL_NAME"]
GCP_BUCKET_NAME = st.secrets["GCP_BUCKET_NAME"]
GOOGLE_APPLICATION_CREDENTIALS = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
BEDROCK_ACCESS_KEY_ID = st.secrets["BEDROCK_ACCESS_KEY_ID"]
BEDROCK_SECRET_ACCESS_KEY = st.secrets["BEDROCK_SECRET_ACCESS_KEY"]

# 環境変数に定数を追加
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GEMINI_MODEL_NAME"] = GEMINI_MODEL_NAME
os.environ["GCP_BUCKET_NAME"] = GCP_BUCKET_NAME
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["BEDROCK_ACCESS_KEY_ID"] = BEDROCK_ACCESS_KEY_ID
os.environ["BEDROCK_SECRET_ACCESS_KEY"] = BEDROCK_SECRET_ACCESS_KEY

google_application_credentials = st.secrets["json_key"]["key"]
pathlib.Path(GOOGLE_APPLICATION_CREDENTIALS).write_text(google_application_credentials)

storage_client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
gcp_bucket = storage_client.bucket(GCP_BUCKET_NAME)
vertexai.init(project='dev-rororo')


def is_valid_media_file(file):
    mime_type, _ = mimetypes.guess_type(file.name)
    return mime_type and mime_type.startswith(('audio/', 'video/'))

def handle_uploaded_file(uploaded_file):
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1]
    save_dir = "uploaded_files"
    output_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"ファイル '{file_name}' がアップロードされました。")

    if file_extension.lower() not in ['.m4a', '.wav']:
        with st.spinner('音声ファイルに変換中...'):
            converter = FFmpegM4AConverter()
            m4a_path = converter.convert(file_path, save_dir)
        st.success(f"ファイルが変換されました: {m4a_path}")
    else:
        m4a_path = file_path
        st.success(f"ファイルの変換は不要です: {m4a_path}")

    return m4a_path

def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            st.error(f"ファイル {file_path} の削除中にエラーが発生しました: {e}")

def delete_all_files():
    directories = ['uploaded_files', 'output']
    for directory in directories:
        if os.path.exists(directory):
            delete_files_in_directory(directory)
    st.success("すべてのファイルが削除されました。")

def main():
    st.title("求人票デモアプリ")
    # ファイル削除ボタン
    if st.button("すべてのファイルを削除"):
        delete_all_files()

    # ユーザー入力
    target_company = st.text_input("対象企業名を入力してください", "株式会社マエノメリ")
    url = st.text_input("企業のホームページURLを入力してください", "https://maenomery.jp/")
    client_name = st.text_input("企業担当者の名前を入力してください", "山本")
    interviewee_name = st.text_input("面接者の名前を入力してください", "岩淵")

    # インタビュープロンプトの生成
    interview_prompt = generate_business_meeting_prompt(
        interviewee_name=interviewee_name,
        client_name=client_name
    )

    uploaded_file = st.file_uploader("音声または動画ファイルをアップロードしてください", type=None)

    if uploaded_file is not None:
        if not is_valid_media_file(uploaded_file):
            st.error("サポートされていないファイル形式です。音声または動画ファイルをアップロードしてください。")
            return

        m4a_path = handle_uploaded_file(uploaded_file)

        # 分析実行ボタン
        if st.button("分析開始"):
            with st.spinner("分析中..."):
                # 音声分析
                response = analyze_sounds(
                    gcp_bucket=gcp_bucket,
                    gcp_bucket_name=GCP_BUCKET_NAME,
                    audio_file=m4a_path,
                    model_name=GEMINI_MODEL_NAME,
                    prompt=interview_prompt
                )
                minutes = response.candidates[0].content.parts[0].text

                # 議事録の保存
                with open('./output/minutes.txt', 'w', encoding='utf-8') as f:
                    f.write(minutes)
                st.success("議事録が生成されました")

                # Web検索（リトライ機能付き）
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        web_result, usage = execute_web_search(url, target_company, web_search_task, instruction)
                        with open('./output/web_search_result.txt', 'w', encoding='utf-8') as f:
                            f.write(web_result["output"])
                        st.success("Web検索が完了しました")
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            st.error(f"Web検索に失敗しました。エラー: {str(e)}")
                        else:
                            st.warning(f"Web検索に失敗しました。リトライ中... ({retry_count}/{max_retries})")

                # 求人情報生成
                job_posting = generate_job_posting(minutes, web_result["output"])
                save_job_posting_to_csv(job_posting, "./output")
                st.success("求人情報が生成されました")

        # ファイルダウンロードボタン
        if os.path.exists('./output/minutes.txt'):
            with open('./output/minutes.txt', 'r', encoding='utf-8') as f:
                st.download_button(
                    label="議事録をダウンロード",
                    data=f.read(),
                    file_name=uploaded_file.name + "_minutes.txt",
                    mime="text/plain"
                )

        if os.path.exists('./output/web_search_result.txt'):
            with open('./output/web_search_result.txt', 'r', encoding='utf-8') as f:
                st.download_button(
                    label="Web検索結果をダウンロード",
                    data=f.read(),
                    file_name=uploaded_file.name + "_web_search_result.txt",
                    mime="text/plain"
                )
        
        if os.path.exists('./output/job_postings.csv'):
            with open('./output/job_postings.csv', 'r', encoding='utf-8') as f:
                st.download_button(
                    label="求人情報をダウンロード (CSV)",
                    data=f.read(),
                    file_name=uploaded_file.name + "_job_postings.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()


