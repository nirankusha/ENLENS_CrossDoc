"""Prime the Hugging Face cache used by the Cross-Doc apps."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

from huggingface_hub import snapshot_download

HF_REPOS: tuple[str, ...] = (
    "sadickam/sdg-classification-bert",
    "sentence-transformers/paraphrase-mpnet-base-v2",
    "allenai/scibert_scivocab_uncased",
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Hugging Face models")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory to use for the Hugging Face cache (defaults to $HF_CACHE_DIR or ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--repos",
        nargs="*",
        default=HF_REPOS,
        help="Optional explicit list of repo IDs to cache",
    )
    return parser.parse_args(argv)


def _resolve_cache_dir(cache_dir: Optional[Path]) -> Path:
    if cache_dir is not None:
        return cache_dir.expanduser().resolve()
    env = os.environ.get("HF_CACHE_DIR") or os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".cache" / "huggingface"


def _cache_repo(repo_id: str, cache_dir: Path) -> None:
    print(f"Caching {repo_id} -> {cache_dir}")
    snapshot_download(repo_id=repo_id, cache_dir=cache_dir, resume_download=True)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    cache_dir = _resolve_cache_dir(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))

    repos: Iterable[str] = args.repos or HF_REPOS
    for repo_id in repos:
        _cache_repo(repo_id, cache_dir)


if __name__ == "__main__":  # pragma: no cover
    main()