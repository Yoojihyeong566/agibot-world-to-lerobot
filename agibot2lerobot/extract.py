"""Streaming extraction of (possibly multi-part) gzip tar archives.

AgiBot archives come in two shapes:
  * single file  : ``553049_553079.tar.gz``                 (IL / RI)
  * multi-part   : ``videos.tar.gz.000``, ``.001``, ...     (simulation)

A multi-part archive is simply one ``.tar.gz`` byte stream split across files in
numeric order. We concatenate the parts on the fly and feed them to ``tarfile``
in streaming mode, so we never materialise the joined archive on disk.
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Iterable


class _ConcatReader:
    """A read-only binary file object spanning several files, in order."""

    def __init__(self, paths: Iterable[Path]):
        self._paths = [Path(p) for p in paths]
        if not self._paths:
            raise ValueError("no archive parts given")
        self._i = 0
        self._f = open(self._paths[0], "rb")

    def read(self, size: int = -1) -> bytes:
        # streaming tar (mode 'r|gz') only ever calls read() sequentially
        if size is None or size < 0:
            chunks = [self._f.read()]
            while self._advance():
                chunks.append(self._f.read())
            return b"".join(chunks)

        out = bytearray()
        while len(out) < size:
            chunk = self._f.read(size - len(out))
            if chunk:
                out.extend(chunk)
                continue
            if not self._advance():
                break
        return bytes(out)

    def _advance(self) -> bool:
        """Move to the next part. Returns False when exhausted."""
        self._f.close()
        self._i += 1
        if self._i >= len(self._paths):
            return False
        self._f = open(self._paths[self._i], "rb")
        return True

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def extract_archive(parts: Iterable[Path], dst: Path) -> Path:
    """Extract one (multi-part) ``.tar.gz`` into ``dst``.

    Args:
        parts: ordered list of archive part paths (length 1 for a single file).
        dst:   destination directory (created if missing).

    Returns:
        ``dst``.
    """
    parts = sorted(Path(p) for p in parts)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    with _ConcatReader(parts) as fileobj:
        with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
            # filter='data' avoids extracting unsafe members (py>=3.12 default-safe)
            try:
                tar.extractall(dst, filter="data")
            except TypeError:  # older python without the filter kwarg
                tar.extractall(dst)
    return dst
