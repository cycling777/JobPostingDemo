import vertexai.generative_models as genai

def analyze_sounds(gcp_bucket, gcp_bucket_name, audio_file, model_name, prompt):
    global price_text 
    blob = gcp_bucket.blob(audio_file)
    with open(audio_file, 'rb') as audio_file_path:
        blob.upload_from_file(audio_file_path)
    print("success")
    gcs_uri = f'gs://{gcp_bucket_name}/{audio_file}'
    print(gcs_uri)

    audio = genai.Part.from_uri(
        mime_type="audio/wav",
        uri = gcs_uri
    )

    model = genai.GenerativeModel(
        model_name
    )
    price_text = "1"

    response = model.generate_content(
        [audio, prompt], stream=False
    )


    print(response)
    return response