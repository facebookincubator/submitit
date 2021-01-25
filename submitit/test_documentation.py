# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import re
from pathlib import Path
from typing import List


class _MarkdownLink:
    """Handle to a markdown link, for easy existence test and printing
    (external links are not tested)
    """

    def __init__(self, root: Path, file: Path, string: str, link: str) -> None:
        self._root = root
        self._file = file
        self._string = string
        self._link = link

    def exists(self) -> bool:
        if self._link.startswith("http"):
            # We don't check external urls.
            return True
        link = self._link.split("#")[0]
        if not link:
            return False
        fullpath = self._root / self._file.parent / link
        return fullpath.exists()

    def __repr__(self) -> str:
        return f"[{self._link}]({self._string}) in file {self._file}"


def _get_root() -> Path:
    root = Path(__file__).parent.parent.absolute()
    assert (root / "setup.py").exists(), f"Wrong root folder: {root}"
    return root


def _get_markdown_files(root: Path) -> List[Path]:
    return [md for pattern in ("*.md", "submitit/**/*.md", "docs/**/*.md") for md in root.glob(pattern)]


def _get_all_markdown_links(root: Path, files: List[Path]) -> List[_MarkdownLink]:
    """Returns a list of all existing markdown links"""
    pattern = re.compile(r"\[(?P<string>.+?)\]\((?P<link>\S+?)\)")
    links = []
    for file in files:
        for match in pattern.finditer(file.read_text()):
            links.append(_MarkdownLink(root, file, match.group("string"), match.group("link")))
    return links


def test_assert_markdown_links_not_broken() -> None:
    root = _get_root()
    files = _get_markdown_files(root)
    assert len(files) > 3

    links = _get_all_markdown_links(root, files)
    assert len(links) > 5, "There should be several hyperlinks!"
    broken_links = [l for l in links if not l.exists()]
    assert not broken_links
