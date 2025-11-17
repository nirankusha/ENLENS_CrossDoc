"""Download legacy model artifacts from Google Drive.

This helper consolidates the Google Drive downloads that were previously
performed manually in notebooks/Colab scripts.  It downloads the
PreSumm checkpoints and the reverse-attribution PDF bundle into a local
`models/` and `data/` directory structure so the Streamlit apps can run on
self-hosted machines.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import zipfile
import tarfile
from typing import Optional

try:
    import gdown  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "gdown is required to download Google Drive assets. Install it with 'pip install gdown'."
    ) from exc


@dataclass(frozen=True)
class LegacyAsset:
    """Metadata for a legacy artifact."""

    name: str
    gdrive_id: str
    target_dir: Path
    filename: Optional[str] = None
    description: str = ""

    @property
    def download_path(self) -> Path:
        fname = self.filename or f"{self.name}.bin"
        return self.target_dir / fname


LEGACY_ASSETS = (
    LegacyAsset(
        name="bert_presumm",
        gdrive_id="1hBVQcpEMgvzpmUpbgBbUPcOFjmusp7Cu",
        target_dir=Path("models/presumm"),
        filename="model_step_30000.pt",
        description="PreSumm extractive BERT checkpoint",
    ),
    LegacyAsset(
        name="bert_abs_ext",
        gdrive_id="1bjceqRy2NTrtQHkODL2BVOf4_l3u2y7Y",
        target_dir=Path("models/presumm"),
        description="PreSumm abstractive checkpoint bundle",
    ),
    LegacyAsset(
        name="rev_attr_summ_pdfs",
        gdrive_id="1I9j8QaOYKYzlDnoPsSnpV0-yBLwZes5j",
        target_dir=Path("data/reverse_attribution"),
        description="Reverse attribution PDF corpus",
    ),
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_asset(asset: LegacyAsset, *, overwrite: bool = False) -> None:
    _ensure_dir(asset.target_dir)
    destination = asset.download_path

    if destination.exists() and not overwrite:
        print(f"✓ {asset.name}: already present at {destination}")
        return

    tmp_path = destination.with_suffix(destination.suffix + ".download")
    if tmp_path.exists():
        tmp_path.unlink()

    url = f"https://drive.google.com/uc?id={asset.gdrive_id}"
    print(f"⬇️  {asset.name}: downloading {asset.description or 'asset'}")
    gdown.download(url, str(tmp_path), quiet=False)

    if zipfile.is_zipfile(tmp_path):
        print(f"📦 {asset.name}: extracting ZIP archive")
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(asset.target_dir)
        tmp_path.unlink(missing_ok=True)
        return

    if tarfile.is_tarfile(tmp_path):
        print(f"📦 {asset.name}: extracting TAR archive")
        with tarfile.open(tmp_path, "r:*") as tf:
            tf.extractall(asset.target_dir)
        tmp_path.unlink(missing_ok=True)
        return

    tmp_path.rename(destination)
    print(f"✓ {asset.name}: saved to {destination}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download legacy ENLENS models")
    parser.add_argument(
        "--base-dir",
        default=Path.cwd(),
        type=Path,
        help="Root directory where models/ and data/ will be created",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    base_dir = args.base_dir.expanduser().resolve()
    print(f"Using base directory: {base_dir}")
    for asset in LEGACY_ASSETS:
        adjusted = LegacyAsset(
            name=asset.name,
            gdrive_id=asset.gdrive_id,
            target_dir=(base_dir / asset.target_dir),
            filename=asset.filename,
            description=asset.description,
        )
        _download_asset(adjusted, overwrite=args.overwrite)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])