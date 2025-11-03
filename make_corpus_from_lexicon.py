#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import random

ROOT = Path(__file__).resolve().parent

# Support both names (you said your file has an underscore)
LEX1 = ROOT / "lexicon 6000 words.xlsx"
LEX2 = ROOT / "lexicon_6000 words.xlsx"
LEXICON_XLSX = LEX1 if LEX1.exists() else LEX2

OUT = ROOT / "my_corpus.csv"

# Column picking helper
def pick(cols, candidates):
    m = {c.lower(): c for c in cols}
    for k in candidates:
        if k in m: return m[k]
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in candidates): return c
    raise KeyError(f"Could not find from candidates {candidates} in columns {list(cols)}")

def main():
    if not LEXICON_XLSX.exists():
        raise FileNotFoundError(f"Lexicon not found: {LEX1.name} or {LEX2.name}")

    xls = pd.ExcelFile(LEXICON_XLSX)
    df = xls.parse(xls.sheet_names[0])

    fr_col = pick(df.columns, ["french_norm","french","term","mot","mot_fr","fr"])
    se_col = pick(df.columns, ["sentiment_std","sentiment","polarity"])
    pos_col = pick(df.columns, ["nature_std","nature","pos","partofspeech"])

    # Normalize sentiment to 3 classes
    map_se = {
        "positif":"positive","positive":"positive","pos":"positive",
        "negatif":"negative","négatif":"negative","negative":"negative","neg":"negative",
        "neutral":"neutral","neutre":"neutral","neu":"neutral",
    }

    tmp = pd.DataFrame({
        "term": df[fr_col].astype(str).str.replace("\u00A0"," ", regex=False).str.strip().str.lower(),
        "label": df[se_col].astype(str).str.strip().str.lower().map(map_se).fillna("neutral"),
        "nature": df[pos_col].astype(str).str.strip().str.lower()
    })

    # Keep only non-empty terms
    tmp = tmp[tmp["term"] != ""].drop_duplicates().reset_index(drop=True)

    # Simple French sentence templates
    # We vary a bit so the corpus isn't totally uniform.
    TEMPLATES = {
        "positive": [
            "C'est {t}, j'adore.",
            "{t} — c'est super !",
            "Franchement, {t}, c'est bien."
        ],
        "negative": [
            "C'est {t}, je déteste.",
            "{t} — c'est nul.",
            "Honnêtement, {t}, c'est mauvais."
        ],
        "neutral": [
            "À propos de {t}.",
            "On parle de {t}.",
            "{t} est mentionné."
        ],
    }

    # Light tweak if it looks like a verb (very rough heuristic)
    def as_sentence(term, label, nature):
        cand = TEMPLATES.get(label, TEMPLATES["neutral"])
        t = term
        if nature in {"verb", "verbe"} or t.endswith(("er","ir","re")):
            # infinitive-ish framing
            pos_tpl = [
                "Il faut {t}, c'est utile.",
                "{t}, c'est une bonne idée.",
                "On aime {t}."
            ]
            neg_tpl = [
                "Éviter de {t}, ce n'est pas bien.",
                "{t}, ce n'est pas recommandé.",
                "On n'aime pas {t}."
            ]
            neu_tpl = [
                "On discute de {t}.",
                "{t} est à considérer.",
                "{t} est cité."
            ]
            if label == "positive": cand = pos_tpl
            elif label == "negative": cand = neg_tpl
            else: cand = neu_tpl
        return random.choice(cand).format(t=term)

    # Build rows (cap to a manageable size, e.g., 5000)
    rows = []
    for i, r in tmp.iterrows():
        s = as_sentence(r["term"], r["label"], r["nature"])
        rows.append({"id": i+1, "text": s, "label": r["label"], "lang": "fr"})
        if len(rows) >= 5000:  # adjust up/down if you want
            break

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[✓] Wrote {OUT.name} with {len(out)} rows")

if __name__ == "__main__":
    main()
