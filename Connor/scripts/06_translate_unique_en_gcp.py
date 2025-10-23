from __future__ import annotations
from pathlib import Path
import os
import json
import time
from typing import List
import pandas as pd
from tqdm import tqdm

# --- Load .env if needed ---
try:
    from dotenv import load_dotenv, find_dotenv
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or not os.getenv("GOOGLE_CLOUD_PROJECT"):
        load_dotenv(find_dotenv())  # looks upward for .env
except Exception:
    pass

# --- Google Cloud Translation v3 ---
# pip install google-cloud-translate==3.*
from google.cloud import translate

# --- Paths (script-relative) ---
PROJECT_DIR = Path(__file__).resolve().parents[1]
UNIQUE_PATH = PROJECT_DIR / "data" / "processed" / "lexicon_french_unique.csv"
CACHE_PATH  = PROJECT_DIR / "data" / "processed" / "cache_translations_en.json"

# --- Config ---
TARGET_LANG = "en"
SOURCE_LANG = "fr"
BATCH_SIZE  = 100  # drop to 50/25 if throttled
EMPTY_LIKE  = {"", "nan", "none", "null", "na", "n/a"}

# Idiom/override map (expand as needed)
POST_OVERRIDE = {
    "pas mal": "pretty good",
    "pas mauvais": "not bad (pretty good)",
    "pas terrible": "not great",
    "pas fameux": "not great",
    "au top": "top-notch",
    "trop bien": "awesome",
    "ça passe": "it’ll do",
    "ca passe": "it’ll do",
    "bof": "meh",
    "nickel": "spotless",
}

# ---------------- Helpers ----------------
def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def normalize_english_series(s: pd.Series) -> pd.Series:
    """
    Normalize 'english' column to detect empty-like values reliably.
    Returns a stripped string series (never NaN), but original df will be updated later.
    """
    s = s.astype(str)
    # Replace non-breaking spaces etc. then strip
    s = s.str.replace("\u00A0", " ", regex=False)  # NBSP -> space
    s_trim = s.str.strip()
    return s_trim

def needs_translation_mask(eng_trim: pd.Series) -> pd.Series:
    lower = eng_trim.str.lower()
    return eng_trim.eq("") | lower.isin(EMPTY_LIKE)

def make_key(french_norm: str, nature_std: str) -> str:
    return f"{(french_norm or '').lower()}||{(nature_std or '').lower()}"

def tag_notes_idiom(notes: str) -> str:
    n = (notes or "").strip()
    if "idiom" in n:
        return n
    return (n + "; idiom").strip("; ").strip()

# ---------------- GCP ----------------
def get_parent_location() -> str:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set (check your .env).")
    # Glossaries require us-central1; for plain translation we can use 'global'
    return f"projects/{project_id}/locations/global"

def translate_batch(client: translate.TranslationServiceClient, parent: str, texts: List[str]) -> List[str]:
    resp = client.translate_text(
        request={
            "parent": parent,
            "contents": texts,
            "mime_type": "text/plain",
            "source_language_code": SOURCE_LANG,
            "target_language_code": TARGET_LANG,
        }
    )
    return [t.translated_text for t in resp.translations]

# ---------------- Main ----------------
def main():
    if not UNIQUE_PATH.exists():
        raise FileNotFoundError(f"Unique sheet not found: {UNIQUE_PATH}")

    df = pd.read_csv(UNIQUE_PATH)

    # Ensure required columns exist
    for col in ["english", "notes", "source", "qa_status_translate", "French_norm", "Nature_std"]:
        if col not in df.columns:
            df[col] = ""

    eng_trim = normalize_english_series(df["english"])
    need_mask = needs_translation_mask(eng_trim)

    # Work only on rows that truly need translation
    pending = df.loc[need_mask, ["French_norm", "Nature_std"]].copy()
    if pending.empty:
        print("[i] Nothing to translate.")
        return

    # Build unique key per lemma + POS to avoid duplicate API calls
    pending["key"] = pending.apply(lambda r: make_key(r["French_norm"], r["Nature_std"]), axis=1)
    keys_df = pending.drop_duplicates("key").copy()

    cache = load_cache()
    client = translate.TranslationServiceClient()
    parent = get_parent_location()

    # Prepare batches, honoring overrides
    to_translate, order_keys = [], []
    for _, row in keys_df.iterrows():
        fr = (row["French_norm"] or "").strip()
        k  = row["key"]
        if k in cache:
            continue
        if fr.lower() in POST_OVERRIDE:
            cache[k] = POST_OVERRIDE[fr.lower()]
            continue
        to_translate.append(fr)
        order_keys.append(k)

    print(f"[i] Unique items pending translation: {len(order_keys)}")
    # Batch with retries
    for i in tqdm(range(0, len(to_translate), BATCH_SIZE), desc="Translating"):
        batch = to_translate[i : i + BATCH_SIZE]
        tries = 0
        while True:
            try:
                out = translate_batch(client, parent, batch)
                break
            except Exception as e:
                tries += 1
                wait = min(60, 2 ** tries)
                print(f"[warn] API error: {e} -> retrying in {wait}s (try {tries})")
                time.sleep(wait)
        # write to cache
        for j, eng in enumerate(out):
            cache[order_keys[i + j]] = eng
        save_cache(cache)

    # Fill english back into df (only for rows needing translation)
    def fill_english(row) -> str:
        # Keep any existing non-empty/non-placeholder
        curr = str(row["english"] or "").strip()
        if curr and curr.lower() not in EMPTY_LIKE:
            return curr
        k = make_key(row["French_norm"], row["Nature_std"])
        fr_low = str(row["French_norm"] or "").lower()
        if fr_low in POST_OVERRIDE:
            return POST_OVERRIDE[fr_low]
        return cache.get(k, "")

    df.loc[need_mask, "english"] = df.loc[need_mask].apply(fill_english, axis=1)

    # Defaults for notes/source/qa_status_translate
    # - Only set where 'english' is now non-empty and field is blank
    filled = normalize_english_series(df["english"]).ne("")
    df["notes"]  = df["notes"].fillna("").astype(str)
    df["source"] = df["source"].fillna("").astype(str)
    df["qa_status_translate"] = df["qa_status_translate"].fillna("").astype(str)

    # Mark auto fills
    df.loc[filled & (df["notes"].str.strip() == ""), "notes"] = "auto:gcp"
    df.loc[filled & (df["source"].str.strip() == ""), "source"] = "gcp"
    df.loc[filled & (df["qa_status_translate"].str.strip() == ""), "qa_status_translate"] = "ok"

    # Tag idioms explicitly in notes if override applied
    is_idiom = df["French_norm"].str.lower().isin(POST_OVERRIDE.keys()) & filled
    df.loc[is_idiom, "notes"] = df.loc[is_idiom, "notes"].apply(tag_notes_idiom)

    # Save back
    df.to_csv(UNIQUE_PATH, index=False, encoding="utf-8")
    print(f"[✓] Updated translations -> {UNIQUE_PATH}")

if __name__ == "__main__":
    main()
