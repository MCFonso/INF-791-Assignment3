#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse, re, json
import pandas as pd
from collections import Counter

ROOT = Path(__file__).resolve().parent
LEXICON_XLSX = ROOT / "lexicon_6000 words.xlsx"
CORPUS_STD_CSV = ROOT / "corpus.csv"
COVERAGE_TXT = ROOT / "corpus_coverage.txt"
OOV_CSV = ROOT / "oov_candidates.csv"
CAND_CSV = ROOT / "candidate_enrichment.csv"
QC_TXT = ROOT / "corpus_qc_summary.txt"

ALLOWED_LABELS = {"positive", "negative", "neutral"}
TOKEN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ’'-]+")

# ------------------ utilities ------------------
def normalize_text(x):
    s = "" if pd.isna(x) else str(x)
    s = s.replace("\u00A0"," ").strip()
    return re.sub(r"\s+"," ", s)

def pick_col(df, candidates):
    lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lc: return lc[cand]
    for c in df.columns:
        low = c.lower()
        if any(cand in low for cand in candidates): return c
    return None

def parse_label_map(s):
    if not s: return {}
    out={}
    for pair in s.split(","):
        if "=" in pair:
            k,v = pair.split("=",1)
            out[k.strip().lower()] = v.strip().lower()
    return out

def read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf==".csv":
        for enc in ("utf-8","utf-8-sig","latin1"):
            try: return pd.read_csv(path, encoding=enc)
            except Exception: pass
        raise ValueError(f"CSV read failed for {path}")
    if suf in (".tsv",".tab"): return pd.read_csv(path, sep="\t", encoding="utf-8")
    if suf in (".xlsx",".xls"): return pd.read_excel(path)
    if suf in (".jsonl",".ndjson"):
        return pd.DataFrame([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])
    raise ValueError(f"Unsupported extension: {suf}")

# ------------------ lexicon import ------------------
def _pick_col(cols, candidates):
    lc = {c.lower(): c for c in cols}
    for c in candidates:
        if c in lc: return lc[c]
    for c in cols:
        low = c.lower()
        if any(cand in low for cand in candidates): return c
    return None

def import_lexicon_xlsx(xlsx_path: Path) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Lexicon Excel not found: {xlsx_path}")
    xls = pd.ExcelFile(xlsx_path)
    sheet = xls.sheet_names[0]
    df = xls.parse(sheet_name=sheet)

    cols = list(df.columns)
    fr_col  = _pick_col(cols, ["french_norm","french","term","mot","mot_fr","fr"])
    en_col  = _pick_col(cols, ["english","en","translation","gloss","english_norm"])
    pos_col = _pick_col(cols, ["nature_std","nature","pos","partofspeech"])
    sc_col  = _pick_col(cols, ["score_int","score","intensity","valence","polarity_score"])
    se_col  = _pick_col(cols, ["sentiment_std","sentiment","polarity"])

    if fr_col is None or en_col is None:
        raise KeyError(f"Could not find French/English columns in {xlsx_path}. Columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["French_norm"] = (df[fr_col].astype(str).str.replace("\u00A0"," ", regex=False)
                          .str.strip().str.lower())
    out["english"]     = (df[en_col].astype(str).str.replace("\u00A0"," ", regex=False)
                          .str.strip())
    out["Nature_std"]  = (df[pos_col].astype(str).str.strip().str.lower()
                          if pos_col else "other")
    if sc_col:
        out["Score_int"] = pd.to_numeric(df[sc_col], errors="coerce").round().fillna(0).astype(int)
    else:
        out["Score_int"] = 0
    if se_col:
        se = df[se_col].astype(str).str.strip().str.lower()
        map_se = {"positif":"positive","positive":"positive","pos":"positive",
                  "negatif":"negative","négatif":"negative","negative":"negative","neg":"negative",
                  "neutral":"neutral","neutre":"neutral","neu":"neutral"}
        out["Sentiment_std"] = se.map(lambda x: map_se.get(x,x))
        out.loc[~out["Sentiment_std"].isin(ALLOWED_LABELS), "Sentiment_std"] = "neutral"
    else:
        out["Sentiment_std"] = "neutral"

    out = out[(out["French_norm"]!="") & (out["english"]!="")]
    out = out.drop_duplicates(subset=["French_norm","Nature_std","english"], keep="first").reset_index(drop=True)
    return out

def vocab_from_lexicon_df(lex_df: pd.DataFrame) -> set:
    df = lex_df.copy()
    df["French_norm"] = df["French_norm"].astype(str).str.replace("\u00A0"," ", regex=False).str.strip().str.lower()
    eng = df["english"].astype(str).str.strip().str.lower()
    df = df[eng.ne("") & eng.ne("nan") & eng.ne("none")]
    dedup = df.groupby(["French_norm","Nature_std"], as_index=False).agg(Score_int=("Score_int","median"))
    return set(dedup["French_norm"])

# ------------------ corpus formatting ------------------
def format_corpus(raw_path: Path, out_csv: Path, label_map_str: str, default_lang: str) -> pd.DataFrame:
    df_raw = read_any(raw_path)

    # 1) try normal corpus headings
    text_col  = pick_col(df_raw, ["text","tweet","review","content","message","body"])
    label_col = pick_col(df_raw, ["label","sentiment","polarity","target"]) if any(
        k in [c.lower() for c in df_raw.columns] for k in ["label","sentiment","polarity","target"]) else None
    lang_col  = pick_col(df_raw, ["lang","language","iso"]) if any(
        k in [c.lower() for c in df_raw.columns] for k in ["lang","language","iso"]) else None

    # 2) if missing, FALL BACK to **lexicon-style** headings (your case)
    if text_col is None:
        text_col = pick_col(df_raw, ["french_norm","french","term","mot","mot_fr","fr"])
    if label_col is None:
        label_col = pick_col(df_raw, ["sentiment_std","sentiment","polarity"])  # map to pos/neg/neutral
    # lang can remain None → default to 'fr'

    if text_col is None or label_col is None:
        raise KeyError(f"Could not determine text/label columns. Got columns: {list(df_raw.columns)}")

    # build standardized df
    df = pd.DataFrame()
    df["text"] = df_raw[text_col].apply(normalize_text)

    # label mapping
    mapping = {"pos":"positive","positive":"positive","positif":"positive",
               "neg":"negative","negative":"negative","negatif":"negative","négatif":"negative",
               "neu":"neutral","neutral":"neutral","neutre":"neutral",
               "1":"positive","-1":"negative","0":"neutral"}
    mapping.update(parse_label_map(label_map_str))
    raw_lab = df_raw[label_col].astype(str).str.strip().str.lower()
    df["label"] = raw_lab.map(lambda x: mapping.get(x,x))
    df = df[df["label"].isin(ALLOWED_LABELS)]

    if lang_col is None:
        df["lang"] = default_lang.strip().lower()
    else:
        df["lang"] = df_raw[lang_col].astype(str).str.strip().str.lower()

    df = df[df["text"]!=""].copy().reset_index(drop=True)
    df.insert(0, "id", df.index + 1)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df

# ------------------ coverage & OOV ------------------
def tokenize(s: str):
    return [t.lower().strip("’-") for t in TOKEN.findall(s or "") if t.strip("’-")]

def coverage_and_oov(corpus_df: pd.DataFrame, vocab: set):
    total = in_lex = 0
    oov = Counter()
    for t in corpus_df["text"]:
        ts = tokenize(t)
        total += len(ts)
        in_lex += sum(x in vocab for x in ts)
        for x in ts:
            if x not in vocab: oov[x]+=1
    cov = (in_lex/total*100.0) if total else 0.0
    return total, in_lex, cov, oov

# ------------------ candidate enrichment ------------------
def candidate_enrichment(corpus_df: pd.DataFrame, vocab: set, min_df: int = 3) -> pd.DataFrame:
    counts = {k: Counter() for k in ALLOWED_LABELS}
    for _, r in corpus_df.iterrows():
        tset = set(tokenize(r["text"]))
        for t in tset:
            counts[r["label"]][t]+=1

    cand=[]
    all_tokens = set().union(*[set(c.keys()) for c in counts.values()])
    for t in all_tokens:
        if t in vocab: continue
        pos,neg,neu = counts["positive"][t],counts["negative"][t],counts["neutral"][t]
        tot = pos+neg+neu
        if tot < min_df: continue
        direction = max((("positive",pos),("negative",neg),("neutral",neu)), key=lambda x:x[1])[0]
        score = round(10*abs(pos-neg)/max(1,tot),2) if direction!="neutral" else 0
        cand.append({"French_norm":t,"Nature_std":"other","suggested_sentiment":direction,
                     "suggested_score":score,"df_pos":pos,"df_neg":neg,"df_neu":neu})
    return pd.DataFrame(cand).sort_values(["suggested_sentiment","suggested_score"], ascending=[True,False])

# ------------------ QC summary ------------------
def write_qc_summary(coverage_txt: Path, cand_csv: Path, out_txt: Path):
    lines = ["=== Corpus QC Summary ===\n"]
    if coverage_txt.exists():
        lines.append(coverage_txt.read_text(encoding="utf-8"))
    else:
        lines.append("(no coverage yet)\n")
    if cand_csv.exists():
        df = pd.read_csv(cand_csv)
        lines += ["\nTop 10 candidates by suggested_score:\n",
                  df.sort_values("suggested_score", ascending=False).head(10).to_string(index=False)]
    out_txt.write_text("\n".join(lines), encoding="utf-8")

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser(description="Single-file corpus pipeline (root-based)")
    ap.add_argument("--in", dest="raw_corpus", required=True,
                    help="Path to your raw corpus file in ROOT (csv/tsv/xlsx/jsonl).")
    ap.add_argument("--label-map", default="", help="Custom mapping like 'positif=positive,negatif=negative,neutre=neutral'")
    ap.add_argument("--lang", default="fr", help="Fallback language if none in file (default: fr)")
    args = ap.parse_args()

    # 1) load lexicon to build vocab
    lex_df = import_lexicon_xlsx(LEXICON_XLSX)
    vocab = vocab_from_lexicon_df(lex_df)

    # 2) standardize corpus (supports BOTH corpus-style and lexicon-style headings)
    raw_path = ROOT / args.raw_corpus if not Path(args.raw_corpus).is_absolute() else Path(args.raw_corpus)
    df_corpus = format_corpus(raw_path, CORPUS_STD_CSV, args.label_map, args.lang)

    # 3) coverage & OOV
    total, in_lex, cov, oov = coverage_and_oov(df_corpus, vocab)
    COVERAGE_TXT.write_text(f"Tokens: {total:,}\nIn-lex: {in_lex:,}\nCoverage: {cov:.2f}%\nUnique OOV: {len(oov):,}\n", encoding="utf-8")
    pd.DataFrame(oov.most_common(), columns=["token","count"]).to_csv(OOV_CSV, index=False, encoding="utf-8-sig")

    # 4) candidate enrichment
    cand = candidate_enrichment(df_corpus, vocab, min_df=3)
    cand.to_csv(CAND_CSV, index=False, encoding="utf-8-sig")

    # 5) QC one-pager
    write_qc_summary(COVERAGE_TXT, CAND_CSV, QC_TXT)

    print(f"[✓] corpus.csv                -> {CORPUS_STD_CSV}")
    print(f"[✓] coverage summary         -> {COVERAGE_TXT}")
    print(f"[✓] OOV candidates           -> {OOV_CSV}")
    print(f"[✓] candidate enrichments    -> {CAND_CSV}")
    print(f"[✓] QC summary               -> {QC_TXT}")

if __name__ == "__main__":
    main()
