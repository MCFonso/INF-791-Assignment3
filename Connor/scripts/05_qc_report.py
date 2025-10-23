from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_DIR / "data" / "processed" / "lexicon_french_enriched.csv"
OUT_TXT = PROJECT_DIR / "data" / "outputs" / "qc_summary.txt"

def add_dist(lines, df, col):
    if col not in df.columns:
        lines.append(f"\n{col}: (missing)")
        return
    s = df[col].astype(str).fillna("").str.strip()
    vc = s.replace({"": "MISSING"}).value_counts()
    lines.append(f"\n{col} distribution:")
    for k, v in vc.items():
        lines.append(f"  {k}: {v}")

def main():
    df = pd.read_csv(IN_PATH)
    lines = [f"Rows: {len(df):,}"]
    for col in ["Sentiment_std","Nature_std","english","qa_status","qa_status_translate"]:
        add_dist(lines, df, col)

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[✓] Wrote QC summary -> {OUT_TXT}")

if __name__ == "__main__":
    main()
