# scripts/09_backfill_enriched_english.py
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
ENRICHED = PROJECT_DIR / "data" / "processed" / "lexicon_french_enriched.csv"
UNIQUE   = PROJECT_DIR / "data" / "processed" / "lexicon_french_unique.csv"

# Toggle this to True if you want to auto-translate any still-empty rows afterwards
USE_GCP_TRANSLATE = False   # set True if you want it to translate remaining empties

def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("\u00A0", " ", regex=False)  # NBSP->space
    return s.str.strip()

def main():
    df = pd.read_csv(ENRICHED)
    u  = pd.read_csv(UNIQUE)

    # Ensure columns exist & normalize keys
    for c in ["French_norm","Nature_std","english","notes","source","qa_status_translate"]:
        if c not in df.columns: df[c] = ""
        if c not in u.columns:  u[c]  = ""

    for frame in (df, u):
        frame["French_norm"] = norm(frame["French_norm"])
        frame["Nature_std"]  = norm(frame["Nature_std"])
        frame["english"]     = frame["english"].astype(str).fillna("").str.replace("\u00A0"," ",regex=False).str.strip()

    # Build a lookup from UNIQUE
    look = (
        u[["French_norm","Nature_std","english","notes","source","qa_status_translate"]]
        .drop_duplicates(subset=["French_norm","Nature_std"])
        .set_index(["French_norm","Nature_std"])
    )

    # Identify empties in enriched
    empty = df["english"].astype(str).str.strip().str.lower().isin({"", "nan", "none", "null", "na", "n/a"})
    before = int(empty.sum())

    # Backfill by exact key
    exact_idx = list(zip(df.loc[empty, "French_norm"], df.loc[empty, "Nature_std"]))
    matched = look.reindex(exact_idx)
    fill_mask = matched["english"].notna() & (matched["english"].astype(str).str.strip() != "")
    df.loc[empty[empty].index[fill_mask], ["english","notes","source","qa_status_translate"]] = matched.loc[fill_mask, ["english","notes","source","qa_status_translate"]].values

    # If still empty, try fallback: French_norm only (ignore POS)
    empty2 = df["english"].astype(str).str.strip().eq("")
    if empty2.any():
        first_by_fr = (
            u[u["english"].astype(str).str.strip() != ""]
            .groupby("French_norm", as_index=False)
            .first()[["French_norm","english","notes","source","qa_status_translate"]]
            .set_index("French_norm")
        )
        fr_vals = df.loc[empty2, "French_norm"].map(first_by_fr["english"])
        fill2 = fr_vals.notna() & (fr_vals.astype(str).str.strip() != "")
        df.loc[empty2[empty2].index[fill2], "english"] = fr_vals[fill2].values

        # carry the meta tags where possible
        for col in ["notes","source","qa_status_translate"]:
            meta_map = df.loc[empty2, "French_norm"].map(first_by_fr[col])
            df.loc[empty2[empty2].index[meta_map.notna()], col] = meta_map[meta_map.notna()].values

    after_exact = int(df["english"].astype(str).str.strip().eq("").sum())

    # Optional: auto-translate remaining empties via GCP
    if USE_GCP_TRANSLATE:
        import os, time, json
        from google.cloud import translate
        from dotenv import load_dotenv, find_dotenv

        if not os.getenv("GOOGLE_CLOUD_PROJECT"):
            load_dotenv(find_dotenv())

        client = translate.TranslationServiceClient()
        parent = f"projects/{os.getenv('GOOGLE_CLOUD_PROJECT')}/locations/global"

        remaining_idx = df.index[df["english"].astype(str).str.strip().eq("")]
        batch = df.loc[remaining_idx, "French_norm"].fillna("").astype(str).tolist()

        def translate_batch(texts):
            resp = client.translate_text(
                request={
                    "parent": parent,
                    "contents": texts,
                    "mime_type": "text/plain",
                    "source_language_code": "fr",
                    "target_language_code": "en",
                }
            )
            return [t.translated_text for t in resp.translations]

        B = 100
        out = []
        for i in range(0, len(batch), B):
            tries = 0
            while True:
                try:
                    out.extend(translate_batch(batch[i:i+B]))
                    break
                except Exception as e:
                    tries += 1
                    wait = min(60, 2**tries)
                    print(f"[warn] GCP error {e}; retry {tries} in {wait}s")
                    time.sleep(wait)

        df.loc[remaining_idx, "english"] = out[:len(remaining_idx)]
        # tag metadata
        meta_mask = df.index.isin(remaining_idx)
        df.loc[meta_mask, "notes"]  = df.loc[meta_mask, "notes"].replace("", "auto:gcp")
        df.loc[meta_mask, "source"] = df.loc[meta_mask, "source"].replace("", "gcp")
        df.loc[meta_mask, "qa_status_translate"] = df.loc[meta_mask, "qa_status_translate"].replace("", "ok")

    # Save
    df.to_csv(ENRICHED, index=False, encoding="utf-8")
    final_empty = int(df["english"].astype(str).str.strip().eq("").sum())

    print(f"[i] Initially empty: {before}")
    print(f"[i] After backfill (exact + fallback): {final_empty}")
    if USE_GCP_TRANSLATE:
        print("[i] GCP translation enabled; any remaining empties were translated as well.")

if __name__ == "__main__":
    main()
