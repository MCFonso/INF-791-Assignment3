from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DIRS = [
    PROJECT_DIR / "data" / "raw",
    PROJECT_DIR / "data" / "processed",
    PROJECT_DIR / "data" / "outputs",
    PROJECT_DIR / "docs",
    PROJECT_DIR / "gcp",
    PROJECT_DIR / "scripts",
]

GITIGNORE = PROJECT_DIR / ".gitignore"
ENV_EXAMPLE = PROJECT_DIR / ".env.example"
REQ = PROJECT_DIR / "requirements.txt"
README = PROJECT_DIR / "docs" / "README_lexicon.md"

def main():
    # 1) Folders
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
    print("[✓] Ensured project folders exist.")

    # 2) .gitignore
    if not GITIGNORE.exists():
        GITIGNORE.write_text(
            "\n".join([
                "# Python",
                "__pycache__/",
                "*.pyc",
                ".venv/",
                "",
                "# Secrets",
                ".env",
                "gcp/sa-key.json",
                "*.json",
                "",
                "# OS",
                ".DS_Store",
                "Thumbs.db",
            ]) + "\n",
            encoding="utf-8"
        )
        print(f"[✓] Wrote {GITIGNORE}")
    else:
        print(f"[i] Exists: {GITIGNORE}")

    # 3) .env.example
    if not ENV_EXAMPLE.exists():
        ENV_EXAMPLE.write_text(
            "\n".join([
                "# Copy this file to .env and fill the values",
                "GOOGLE_CLOUD_PROJECT=your-gcp-project-id-or-number",
                "GOOGLE_APPLICATION_CREDENTIALS=C:\\\\path\\\\to\\\\your\\\\project\\\\gcp\\\\sa-key.json",
            ]) + "\n",
            encoding="utf-8"
        )
        print(f"[✓] Wrote {ENV_EXAMPLE}")
    else:
        print(f"[i] Exists: {ENV_EXAMPLE}")

    # 4) requirements.txt (minimal)
    if not REQ.exists():
        REQ.write_text(
            "\n".join([
                "pandas",
                "numpy",
                "openpyxl",
                "tqdm",
                "python-dotenv",
                "google-cloud-translate==3.*",
            ]) + "\n",
            encoding="utf-8"
        )
        print(f"[✓] Wrote {REQ}")
    else:
        print(f"[i] Exists: {REQ}")

    # 5) README for the lexicon workflow
    if not README.exists():
        README.write_text(
            "# Lexicon Workflow (French stream)\n\n"
            "1. Place raw files in `data/raw/` (CSV or XLSX).\n"
            "2. Run `02_clean_french.py` → outputs `data/processed/lexicon_french_cleaned.csv`.\n"
            "3. Run `03_make_unique_sheet.py` → outputs `data/processed/lexicon_french_unique.csv`.\n"
            "4. (Optional) Set up `.env` from `.env.example` and run `06_translate_unique_en_gcp.py` to auto-fill English.\n"
            "5. Run `04_propagate_translations.py` to produce `lexicon_french_enriched.csv`.\n"
            "6. Run `05_qc_report.py` and review `data/outputs/qc_summary.txt`.\n",
            encoding="utf-8"
        )
        print(f"[✓] Wrote {README}")
    else:
        print(f"[i] Exists: {README}")

if __name__ == "__main__":
    main()
