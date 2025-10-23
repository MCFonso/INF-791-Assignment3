# scripts/00_verify_translate.py
import os
try:
    from dotenv import load_dotenv, find_dotenv
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or not os.getenv("GOOGLE_CLOUD_PROJECT"):
        load_dotenv(find_dotenv())
except Exception:
    pass

from google.cloud import translate

def main():
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set (check your .env).")
    parent = f"projects/{project_id}/locations/global"

    client = translate.TranslationServiceClient()
    resp = client.translate_text(
        request={
            "parent": parent,
            "contents": ["bonjour le monde"],
            "mime_type": "text/plain",
            "source_language_code": "fr",
            "target_language_code": "en",
        }
    )
    print("OK:", resp.translations[0].translated_text)

if __name__ == "__main__":
    main()
