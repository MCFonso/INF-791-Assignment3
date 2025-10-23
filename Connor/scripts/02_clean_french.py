# scripts/02_clean_french.py
from __future__ import annotations
import argparse
from pathlib import Path
import re, sys, math
import numpy as np
import pandas as pd

# ---------- Paths (script-relative) ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUT_DIR = PROJECT_DIR / "data" / "processed"

# ---------- Helpers ----------
def clamp_score(x):
    """Return an int in [-9,9] or NaN. Robust to blanks/strings/NaN."""
    if x is None:
        return np.nan
    try:
        v = float(x)
    except Exception:
        return np.nan
    if math.isnan(v):
        return np.nan
    v = int(round(v))
    return max(-9, min(9, v))

def strip_accents_for_match(s: str) -> str:
    return (
        s.replace("é","e").replace("è","e").replace("ê","e")
         .replace("ï","i").replace("î","i")
         .replace("à","a").replace("â","a")
         .replace("ù","u").replace("û","u")
         .replace("ô","o").replace("ç","c")
    )

SENT_MAP = {
    "positif": "positive",
    "tres positif": "very_positive",
    "tres-positif": "very_positive",
    "negatif": "negative",
    "tres negatif": "very_negative",
    "tres-negatif": "very_negative",
    "neutre": "neutral",
    "positif ou neutre": "mixed",
    "ambivalent": "mixed",
}
POS_MAP = {
    "verbe": "verb",
    "nom": "noun",
    "article": "article",
    "nombre": "number",
    "adjectif": "adj",
    "adjetif": "adj",
    "qualite": "adj",
    "numeral": "number",
    "mot": "other",
    "temps": "other",
    "mois": "other",
    "symbole": "other",
}

def norm_sent(s):
    if pd.isna(s):
        return np.nan
    s0 = strip_accents_for_match(str(s).strip().lower())
    s0 = re.sub(r"\s+", " ", s0)
    if s0 in SENT_MAP:
        return SENT_MAP[s0]
    if "tres positif" in s0:
        return "very_positive"
    if "tres negatif" in s0:
        return "very_negative"
    if "posit" in s0:
        return "positive"
    if "negat" in s0:
        return "negative"
    if "neutr" in s0:
        return "neutral"
    return s0  # leave as-is for manual review

def norm_pos(s):
    if pd.isna(s):
        return np.nan
    s0 = strip_accents_for_match(str(s).strip().lower())
    s0 = re.sub(r"\s+", " ", s0)
    s0 = POS_MAP.get(s0, s0)
    if s0 in {"adj","adjective"}: return "adj"
    if s0 in {"verb","verbe"}:    return "verb"
    if s0 in {"noun","nom"}:      return "noun"
    if s0 in {"expr","expression"}: return "expr"
    if s0 in {"interj","interjection"}: return "interj"
    if s0 in {"num","number","numeral"}: return "number"
    if s0 in {"art","article"}:   return "article"
    return s0 if s0 else "other"

def try_read_csv(path: Path):
    encodings = ["utf-8", "utf-16", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def load_any(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith(".xlsx"):
        return pd.read_excel(path, engine="openpyxl")
    if str(path).lower().endswith(".csv"):
        return try_read_csv(path)
    try:
        return try_read_csv(path)
    except Exception:
        return pd.read_excel(path, engine="openpyxl")

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_norm = {re.sub(r"\s+", "", c.strip().lower()): c for c in df.columns}
    for name in candidates:
        key = re.sub(r"\s+", "", name.strip().lower())
        if key in cols_norm:
            return cols_norm[key]
    return None

def resolve_input_path(arg_infile: str | None) -> Path:
    if arg_infile:
        p = Path(arg_infile)
        if p.exists():
            return p
        p2 = RAW_DIR / arg_infile
        if p2.exists():
            return p2
        raise FileNotFoundError(f"Input not found: {arg_infile} (checked as given and in {RAW_DIR})")
    for base in ["lexicon_original.csv", "lexicon_original.xlsx"]:
        p = RAW_DIR / base
        if p.exists():
            return p
    raise FileNotFoundError("No input file provided and no default found in data/raw/.")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Clean the French lexicon.")
    parser.add_argument("infile", nargs="?", help="Input CSV/XLSX filename or path (default: search data/raw)")
    parser.add_argument("--outfile", default="lexicon_french_cleaned.csv", help="Output CSV name (in data/processed)")
    args = parser.parse_args()

    src = resolve_input_path(args.infile)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / args.outfile

    print(f"[i] Loading: {src}")
    df = load_any(src)
    df.columns = [str(c).strip() for c in df.columns]

    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed") and df[c].isna().all()]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    col_ciluba = find_col(df, ["CILUBA","Ciluba","ciluba"])
    col_french = find_col(df, ["French","FR","french"])
    col_score  = find_col(df, ["Score","score","Intensity"])
    col_sent   = find_col(df, ["Sentiment","sentiment","Polarity"])
    col_nature = find_col(df, ["Nature","POS","PartOfSpeech"])

    missing = []
    if col_french is None: missing.append("French")
    if col_score  is None: missing.append("Score")
    if col_sent   is None: missing.append("Sentiment")
    if col_nature is None: missing.append("Nature")
    if col_french is None:
        raise ValueError("Required column 'French' not found. Please check your headers.")
    if missing:
        print("[!] Warning: missing expected columns:", ", ".join(missing))

    df["French_norm"] = df[col_french].astype(str).str.strip().str.lower()
    df["Score_int"] = df[col_score].apply(clamp_score) if col_score else np.nan
    df["Sentiment_std"] = df[col_sent].apply(norm_sent) if col_sent else np.nan
    df["Nature_std"] = df[col_nature].apply(norm_pos) if col_nature else np.nan

    needs = pd.isna(df["Sentiment_std"]) | pd.isna(df["Nature_std"])
    df["qa_status"] = np.where(needs, "needs_spotcheck", "ok")

    preferred_order = []
    if col_ciluba: preferred_order.append(col_ciluba)
    for c in [col_french, "French_norm", col_score, "Score_int", col_sent, "Sentiment_std", col_nature, "Nature_std", "qa_status"]:
        if c and c not in preferred_order:
            preferred_order.append(c)
    rest = [c for c in df.columns if c not in preferred_order]
    df = df[preferred_order + rest]

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[✓] Saved cleaned file -> {out_path}")
    print(f"[i] Rows: {len(df):,} | Columns: {len(df.columns)}")

    try:
        sent_counts = df["Sentiment_std"].fillna("MISSING").value_counts()
        pos_counts  = df["Nature_std"].fillna("MISSING").value_counts()
        print("\nSentiment_std distribution:")
        for k, v in sent_counts.items():
            print(f"  {k}: {v}")
        print("\nNature_std distribution:")
        for k, v in pos_counts.items():
            print(f"  {k}: {v}")
    except Exception:
        pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
