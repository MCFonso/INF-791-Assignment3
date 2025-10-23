# scripts/03_make_unique_sheet.py
from pathlib import Path
import pandas as pd
import numpy as np
import re

PROJECT_DIR = Path(__file__).resolve().parents[1]
IN_PATH  = PROJECT_DIR / "data" / "processed" / "lexicon_french_cleaned.csv"
OUT_PATH = PROJECT_DIR / "data" / "processed" / "lexicon_french_unique.csv"

# --- heuristics to clean leftovers ------------------------------------------
def fix_sentiment(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().lower()
    s = s.replace("neuitre", "neutral")  # typo
    return s

def guess_expr(french_norm):
    # expressions if contain space, apostrophe with space pattern, or common litotes/idioms
    f = french_norm
    if pd.isna(f): return False
    f = str(f)
    if " " in f: return True
    # common idioms without space (rare) can be added here if needed
    return False

def fix_pos(nature_std, french_norm):
    ns = str(nature_std).strip().lower() if pd.notna(nature_std) else ""
    if ns in {"positif","mm","missing",""}:
        ns = "other"
    # Upgrade to expr if it looks like a multi-token phrase
    if guess_expr(french_norm):
        return "expr"
    return ns

def priority_from_score(score):
    if pd.isna(score): return "medium"
    s = int(score)
    if abs(s) >= 7: return "high"
    if abs(s) >= 3: return "medium"
    return "low"

def pick_representative(group: pd.DataFrame) -> pd.Series:
    # choose representative score by mode; tie-breaker: highest |score|
    s = group["Score_int"].dropna().astype(int) if "Score_int" in group else pd.Series(dtype=int)
    if s.empty:
        rep_score = np.nan
    else:
        mode = s.mode()
        rep_score = int(mode.iloc[0]) if not mode.empty else int(s.iloc[0])
    row = group.iloc[0].copy()
    row["Score_int"] = rep_score

    # majority sentiment after fix
    sent = group["Sentiment_std"].dropna()
    row["Sentiment_std"] = sent.mode().iloc[0] if not sent.empty else np.nan
    return row

def main():
    df = pd.read_csv(IN_PATH)
    # tidy up sentiment leftovers
    df["Sentiment_std"] = df["Sentiment_std"].apply(fix_sentiment)
    # infer/clean POS
    df["Nature_std"] = [fix_pos(n, f) for n, f in zip(df["Nature_std"], df["French_norm"])]

    # build unique set by lemma + POS
    base = df.dropna(subset=["French_norm"]).copy()
    uniq = (
        base.groupby(["French_norm","Nature_std"], as_index=False)
            .apply(pick_representative)
            .reset_index(drop=True)
    )

    # add translator columns
    for c in ["english","notes","source","qa_status_translate"]:
        if c not in uniq.columns:
            uniq[c] = ""

    # priority bucket
    uniq["priority"] = uniq["Score_int"].apply(priority_from_score)

    # nice column order
    cols = [
        "French_norm","Nature_std","Score_int","Sentiment_std","priority",
        "english","notes","source","qa_status_translate"
    ]
    uniq = uniq[cols].sort_values(["priority","French_norm"], ascending=[False, True])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    uniq.to_csv(OUT_PATH, index=False, encoding="utf-8")

    # Console summary
    print(f"[✓] Saved unique translation sheet -> {OUT_PATH}")
    print(f"[i] Unique items: {len(uniq):,}")
    print("\nNature_std distribution (unique):")
    print(uniq["Nature_std"].value_counts())

if __name__ == "__main__":
    main()
