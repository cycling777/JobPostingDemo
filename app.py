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
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from libs.services.analyze_sounds import analyze_sounds
from libs.services.prompts.business_meeting import generate_business_meeting_prompt
from libs.services.web_search import execute_web_search, web_search_task, instruction
from libs.services.generate_posting import generate_job_posting, save_job_posting_to_csv
from libs.services.media_converter import FFmpegM4AConverter
import mimetypes
import uuid
import re
import unicodedata
import csv

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

def sanitize_filename(filename):
    # Unicode正規化
    filename = unicodedata.normalize('NFKC', filename)
    # 特殊文字と空白を置換
    filename = re.sub(r'[^\w\-_\. ]', '_', filename)
    # 連続する空白とアンダースコアを単一のアンダースコアに置換
    filename = re.sub(r'[\s_]+', '_', filename)
    return filename.strip('_')

def is_valid_media_file(file):
    mime_type, _ = mimetypes.guess_type(file.name)
    return mime_type and mime_type.startswith(('audio/', 'video/'))

def handle_uploaded_file(uploaded_file, session_id, output_filename=None):
    original_filename = uploaded_file.name
    sanitized_filename = sanitize_filename(original_filename)
    file_extension = os.path.splitext(sanitized_filename)[1]
    save_dir = f"uploaded_files/{session_id}"
    output_dir = f"output/{session_id}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(save_dir, sanitized_filename)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"ファイル '{sanitized_filename}' がアップロードされました。")

    if file_extension.lower() not in ['.m4a', '.wav']:
        with st.spinner('音声ファイルに変換中...'):
            converter = FFmpegM4AConverter()
            m4a_path = converter.convert(
                input_file=file_path,
                output_dir=save_dir,
                output_filename=output_filename
            )
        st.success(f"ファイルが変換されました: {m4a_path}")
    else:
        m4a_path = file_path
        st.success(f"ファイルの変換は不要です: {m4a_path}")

    return m4a_path

def delete_files_in_directory(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.unlink(file_path)
            except Exception as e:
                st.error(f"ファイル {file_path} の削除中にエラーが発生しました: {e}")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
            except Exception as e:
                st.error(f"ディレクトリ {dir_path} の削除中にエラーが発生しました: {e}")
    try:
        os.rmdir(directory)
    except Exception as e:
        st.error(f"ディレクトリ {directory} の削除中にエラーが発生しました: {e}")

def delete_files_by_id(session_id):
    directories = [f'uploaded_files/{session_id}', f'output/{session_id}']
    for directory in directories:
        if os.path.exists(directory):
            delete_files_in_directory(directory)
            try:
                os.rmdir(directory)
            except Exception as e:
                st.error(f"ディレクトリ {directory} の削除中にエラーが発生しました: {e}")
    st.success(f"セッションID {session_id} に関連するすべてのファイルとディレクトリが削除されました。")

def delete_all_files():
    directories_to_delete = ['uploaded_files', 'output']
    for directory in directories_to_delete:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory, topdown=False):
                for file in files:
                    os.unlink(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(directory)
    st.success("uploaded_filesとoutputフォルダ以下のすべてのファイルとディレクトリが削除されました。")


def main():
    # セッションIDの生成（ユーザーごとに一意）
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    st.title("求人票デモアプリ")


    # ユーザー入力
    target_company = st.text_input("対象企業名を入力してください", "株式会社マエノメリ")
    url = st.text_input("企業のホームページURLを入力してください", "https://maenomery.jp/")
    client_name = st.text_input("企業担当者の名前を入力してください", "山本")
    interviewee_name = st.text_input("面接者の名前を入力してください", "岩淵")

    # モデル選択のオプション
    model_options = [
        {"provider": "OpenAI", "name": "GPT-4o", "model_id": "gpt-4o-2024-08-06"},
        {"provider": "OpenAI", "name": "GPT-4o-mini", "model_id": "gpt-4o-mini"},
        {"provider": "BedRock", "name": "Claude 3.5 Sonnet", "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"},
        {"provider": "BedRock", "name": "Claude 3 Haiku", "model_id": "anthropic.claude-3-haiku-20240307-v1:0"}
    ]

    # セレクトボックスの表示用関数
    def format_model_option(option):
        return f"{option['name']} ({option['provider']})"

    # モデル選択のドロップダウン
    selected_model = st.selectbox(
        "使用するモデルを選択してください",
        options=model_options,
        format_func=format_model_option
    )

    # 選択されたモデルに基づいてLLMインスタンスを作成
    if selected_model["provider"] == "OpenAI":
        llm = ChatOpenAI(model=selected_model["model_id"])
    elif selected_model["provider"] == "BedRock":
        llm = ChatBedrock(model_id=selected_model["model_id"], region_name="ap-northeast-1")
    else:
        st.error(f"未サポートのプロバイダー: {selected_model['provider']}")

    # インタビュープロンプトの生成
    interview_prompt = generate_business_meeting_prompt(
        interviewee_name=interviewee_name,
        client_name=client_name
    )

    uploaded_file = st.file_uploader("音声または動画ファイルをアップロードしてください", type=["mp3", "wav", "ogg", "m4a", "mp4", "mov", "avi"])
    
        # ファイル削除ボタン
    col1, col2 = st.columns(2)
    with col1:
        if st.button("すべてのファイルを削除", key="delete_all_files_button"):
            delete_all_files()
            # セッションステートをクリア
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("現在のセッションのファイルを削除", key="delete_session_files_button"):
            delete_files_by_id(st.session_state.session_id)
            st.success("現在のセッションのファイルが削除されました。")
            # セッションステートをクリア（session_idは保持）
            session_id = st.session_state.session_id
            st.session_state.clear()
            st.session_state.session_id = session_id
            st.rerun()

    if uploaded_file is not None:
        if not is_valid_media_file(uploaded_file):
            st.error("サポートされていないファイル形式です。音声または動画ファイルをアップロードしてください。")
            return

        safe_filename = sanitize_filename(os.path.splitext(uploaded_file.name)[0])
        m4a_path = handle_uploaded_file(uploaded_file, st.session_state.session_id, output_filename=safe_filename)

        # 全体実行ボタン
        if st.button("全体実行"):
            with st.spinner("全ての処理を実行中..."):
                # 議事録作成
                response = analyze_sounds(
                    gcp_bucket=gcp_bucket,
                    gcp_bucket_name=GCP_BUCKET_NAME,
                    audio_file=m4a_path,
                    model_name=GEMINI_MODEL_NAME,
                    prompt=interview_prompt
                )
                minutes = response.candidates[0].content.parts[0].text
                minutes_filename = f'./output/{st.session_state.session_id}/{safe_filename}_minutes.txt'
                with open(minutes_filename, 'w', encoding='utf-8') as f:
                    f.write(minutes)
                st.success("議事録が生成されました")

                # Web検索
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        web_result, usage = execute_web_search(url, target_company, web_search_task, instruction, llm)
                        web_search_filename = f'./output/{st.session_state.session_id}/{safe_filename}_web_search_result.txt'
                        with open(web_search_filename, 'w', encoding='utf-8') as f:
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
                job_posting = generate_job_posting(minutes, web_result["output"], llm)
                save_job_posting_to_csv(job_posting, f'./output/{st.session_state.session_id}', f'{safe_filename}_job_postings.csv')
                st.success("求人情報が生成されました")

        st.markdown("---")
        st.subheader("個別処理")

        col1, col2, col3 = st.columns(3)

        with col1:
            # 議事録作成ボタン
            if st.button("議事録作成"):
                with st.spinner("議事録作成中..."):
                    response = analyze_sounds(
                        gcp_bucket=gcp_bucket,
                        gcp_bucket_name=GCP_BUCKET_NAME,
                        audio_file=m4a_path,
                        model_name=GEMINI_MODEL_NAME,
                        prompt=interview_prompt
                    )
                    minutes = response.candidates[0].content.parts[0].text
                    minutes_filename = f'./output/{st.session_state.session_id}/{safe_filename}_minutes.txt'
                    with open(minutes_filename, 'w', encoding='utf-8') as f:
                        f.write(minutes)
                    st.success("議事録が生成されました")

        with col2:
            # Web検索ボタン
            if st.button("Web検索実行"):
                with st.spinner("Web検索中..."):
                    max_retries = 3
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            web_result, usage = execute_web_search(url, target_company, web_search_task, instruction, llm)
                            web_search_filename = f'./output/{st.session_state.session_id}/{safe_filename}_web_search_result.txt'
                            with open(web_search_filename, 'w', encoding='utf-8') as f:
                                f.write(web_result["output"])
                            st.success("Web検索が完了しました")
                            break
                        except Exception as e:
                            retry_count += 1
                            if retry_count == max_retries:
                                st.error(f"Web検索に失敗しました。エラー: {str(e)}")
                            else:
                                st.warning(f"Web検索に失敗しました。リトライ中... ({retry_count}/{max_retries})")

        with col3:
            # CSV出力ボタン
            if st.button("求人情報CSV出力"):
                minutes_filename = f'./output/{st.session_state.session_id}/{safe_filename}_minutes.txt'
                web_search_filename = f'./output/{st.session_state.session_id}/{safe_filename}_web_search_result.txt'
                if not os.path.exists(minutes_filename) or not os.path.exists(web_search_filename):
                    st.error("議事録とWeb検索結果が必要です。先に議事録作成とWeb検索を実行してください。")
                else:
                    with st.spinner("求人情報生成中..."):
                        with open(minutes_filename, 'r', encoding='utf-8') as f:
                            minutes = f.read()
                        with open(web_search_filename, 'r', encoding='utf-8') as f:
                            web_result_output = f.read()
                        job_posting = generate_job_posting(minutes, web_result_output, llm)
                        save_job_posting_to_csv(job_posting, f'./output/{st.session_state.session_id}', f'{safe_filename}_job_postings.csv')
                        st.success("求人情報が生成されました")

        st.markdown("---")
        st.subheader("ファイルダウンロード")

        col1, col2, col3 = st.columns(3)

        with col1:
            minutes_filename = f'./output/{st.session_state.session_id}/{safe_filename}_minutes.txt'
            if os.path.exists(minutes_filename):
                with open(minutes_filename, 'r', encoding='utf-8') as f:
                    st.download_button(
                        label="議事録をダウンロード",
                        data=f.read(),
                        file_name=f"{safe_filename}_minutes.txt",
                        mime="text/plain"
                    )

        with col2:
            web_search_filename = f'./output/{st.session_state.session_id}/{safe_filename}_web_search_result.txt'
            if os.path.exists(web_search_filename):
                with open(web_search_filename, 'r', encoding='utf-8') as f:
                    st.download_button(
                        label="Web検索結果をダウンロード",
                        data=f.read(),
                        file_name=f"{safe_filename}_web_search_result.txt",
                        mime="text/plain"
                    )
        
        with col3:
            csv_filename = f'./output/{st.session_state.session_id}/{safe_filename}_job_postings.csv'
            if os.path.exists(csv_filename):
                with open(csv_filename, 'r', encoding='utf-8') as f:
                    st.download_button(
                        label="求人情報をダウンロード (CSV)",
                        data=f.read(),
                        file_name=f"{safe_filename}_job_postings.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()


