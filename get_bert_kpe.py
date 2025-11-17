#!/usr/bin/env python3
"""
Clone the BERT-KPE repository and download the checkpoint from Google Drive,
then extract the checkpoint into the repository directory.

Config (env or CLI overrides):
  KPE_REPO_URL      default: https://github.com/thunlp/BERT-KPE.git
  KPE_DIR           default: ./BERT-KPE
  KPE_GDRIVE_ID     default: 13FvONBTM4NZZCR-I7LVypkFa0xihxWnM
  KPE_GDRIVE_URL    optional alternative to ID; full Drive URL
  KPE_OUTPUT_ZIP    default: {KPE_DIR}/checkpoint.zip
  KPE_EXTRACT_DIR   default: {KPE_DIR}
"""

import os
import sys
import argparse
import subprocess
import zipfile
from pathlib import Path

def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except Exception:
        print("gdown not found; installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-url", default=os.environ.get("KPE_REPO_URL", "https://github.com/thunlp/BERT-KPE.git"))
    p.add_argument("--repo-dir", default=os.environ.get("KPE_DIR", "BERT-KPE"))
    p.add_argument("--gdrive-id", default=os.environ.get("KPE_GDRIVE_ID", "13FvONBTM4NZZCR-I7LVypkFa0xihxWnM"))
    p.add_argument("--gdrive-url", default=os.environ.get("KPE_GDRIVE_URL"))  # optional
    p.add_argument("--output-zip", default=os.environ.get("KPE_OUTPUT_ZIP"))
    p.add_argument("--extract-dir", default=os.environ.get("KPE_EXTRACT_DIR"))
    return p.parse_args()

def run(cmd, **kw):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)

def main():
    args = parse_args()
    ensure_gdown()
    import gdown

    repo_dir = Path(args.repo_dir).resolve()
    extract_dir = Path(args.extract_dir or repo_dir).resolve()
    output_zip = Path(args.output_zip or (repo_dir / "checkpoint.zip")).resolve()

    # 1) Clone repo if needed
    if not (repo_dir / ".git").exists():
        print(f"Cloning {args.repo_url} into {repo_dir} ...")
        run(["git", "clone", "--depth", "1", args.repo_url, str(repo_dir)])
    else:
        print(f"Repo already present: {repo_dir}")

    # 2) Download checkpoint zip if missing
    if not output_zip.exists():
        print(f"Downloading checkpoint to: {output_zip}")
        output_zip.parent.mkdir(parents=True, exist_ok=True)

        if args.gdrive_url:
            # Accept full Drive URL
            gdown.download(args.gdrive_url, str(output_zip), quiet=False, fuzzy=True)
        elif args.gdrive_id:
            # Download by file ID
            gdown.download(id=args.gdrive_id, output=str(output_zip), quiet=False)
        else:
            print("❌ No KPE_GDRIVE_ID or KPE_GDRIVE_URL provided; cannot download.")
            sys.exit(1)
    else:
        print(f"Checkpoint zip already exists: {output_zip}")

    # 3) Extract (idempotent)
    print(f"Extracting {output_zip} into {extract_dir} ...")
    with zipfile.ZipFile(output_zip, "r") as zf:
        zf.extractall(extract_dir)
    print("✅ Extraction complete.")

    # 4) Quick sanity: list key files if present
    kpe_model_dirs = [p for p in extract_dir.glob("**/checkpoint") if p.is_dir()]
    if kpe_model_dirs:
        print("Found KPE checkpoint folders:")
        for p in kpe_model_dirs[:5]:
            print(" -", p)
    else:
        print("ℹ️ No 'checkpoint' folders found; content may be organized differently.")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
