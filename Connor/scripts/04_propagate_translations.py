from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEANED = PROJECT_DIR / "data" / "processed" / "lexicon_french_cleaned.csv"
UNIQUE  = PROJECT_DIR / "data" / "processed" / "lexicon_french_unique.csv"
OUT_ALL = PROJECT_DIR / "data" / "processed" / "lexicon_french_enriched.csv"

def main():
    full = pd.read_csv(CLEANED)
    uniq = pd.read_csv(UNIQUE)

    # Ensure translation columns exist and are strings
    for c in ["French_norm","Nature_std","english","notes","source","qa_status_translate"]:
        if c not in uniq.columns:
            uniq[c] = ""
        uniq[c] = uniq[c].astype(str)

    # Keep only the columns we need to merge back
    trans_cols = ["French_norm","Nature_std","english","notes","source","qa_status_translate"]

    # Merge translations onto the full dataset
    merged = full.merge(uniq[trans_cols], on=["French_norm","Nature_std"], how="left")

    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_ALL, index=False, encoding="utf-8")
    print(f"[✓] Saved enriched file -> {OUT_ALL}")
    print(f"[i] Rows: {len(merged):,}")

if __name__ == "__main__":
    main()
