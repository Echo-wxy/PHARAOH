"""JSONL / text / hashing helpers (gzip-aware)."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def open_gz_text_write(path: str | Path, newline: str | None = None):
    """Text writer for .gz files with a fixed header (mtime 0, no file name).

    Fixed headers make byte-identical outputs for identical content, so the
    SHA-256 values in manifests are reproducible across runs.
    """
    raw = open(path, "wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    stream = io.TextIOWrapper(gz, encoding="utf-8", newline=newline)

    orig_close = stream.close

    def close() -> None:
        orig_close()
        raw.close()

    stream.close = close  # type: ignore[method-assign]
    return stream


def _open(path: Path, mode: str):
    path = Path(path)
    if path.suffix == ".gz":
        if mode == "w":
            return open_gz_text_write(path)
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as one JSON object per line; returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a (optionally gzipped) JSONL file."""
    with _open(Path(path), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_lines(path: str | Path, lines: Iterable[str]) -> int:
    """Write one identifier per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")
            n += 1
    return n


def read_lines(path: str | Path) -> list[str]:
    """Read non-empty lines."""
    with _open(Path(path), "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def write_json(path: str | Path, obj: Any) -> None:
    """Pretty-print a JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of the file bytes (for split manifests)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_hash(obj: Any, length: int = 16) -> str:
    """Order-independent hash of a JSON-serializable object."""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]
