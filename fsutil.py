"""Small local-file helpers shared by every script.

All data lives under the `decore/` folder next to where you run the scripts (run them from the repo root).
`fs` offers the four operations the scripts use (ls, head, put, rm); `show_html` saves a review page
(a grid of photos to check by eye) as an HTML file under `pages/`.
"""
import itertools
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PAGES = Path("pages")


@dataclass
class FileInfo:
    path: str
    name: str   # folders end with "/"
    size: int


class _FS:
    def ls(self, path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        if p.is_file():
            return [FileInfo(str(p), p.name, p.stat().st_size)]
        return [FileInfo(str(c) + ("/" if c.is_dir() else ""), c.name + ("/" if c.is_dir() else ""),
                         0 if c.is_dir() else c.stat().st_size) for c in sorted(p.iterdir())]

    def head(self, path, max_bytes=None):
        data = Path(path).read_bytes()
        return (data if max_bytes is None else data[:max_bytes]).decode("utf-8")

    def put(self, path, text, overwrite=False):
        p = Path(path)
        if p.exists() and not overwrite:
            raise FileExistsError(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"Wrote {len(text.encode('utf-8'))} bytes.")
        return True

    def rm(self, path, recurse=False):
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(p) if recurse else p.rmdir()
        elif p.exists():
            p.unlink()
        return True


fs = _FS()
_page_counter = itertools.count(1)


def show_html(html, name=None):
    """Save a review page as an HTML file and print where it is (open it in a browser)."""
    PAGES.mkdir(parents=True, exist_ok=True)
    path = PAGES / f"{name or Path(sys.argv[0]).stem or 'page'}_{next(_page_counter):02d}.html"
    path.write_text(f"<html><body>{html}</body></html>", encoding="utf-8")
    print("page saved:", path.resolve())
