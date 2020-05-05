# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import re
from pathlib import Path
from typing import List, Union


def assert_markdown_links_not_broken(folder: Union[str, Path]) -> None:
    """Asserts that all relative hyperlinks are valid in markdown files of the folder
    and its subfolders.

    Note
    ----
    http hyperlinks are not tested.
    """
    links = _get_all_markdown_links(folder)
    broken = [l for l in links if not l.exists()]
    if broken:
        text = "\n - ".join([str(l) for l in broken])
        raise AssertionError(f"Broken markdown links:\n - {text}")


class _MarkdownLink:
    """Handle to a markdown link, for easy existence test and printing
    (external links are not tested)
    """

    def __init__(self, folder: Path, filepath: Path, string: str, link: str) -> None:
        self._folder = folder
        self._filepath = filepath
        self._string = string
        self._link = link

    def exists(self) -> bool:
        if self._link.startswith("http"):  # consider it exists
            return True
        link = self._link.split("#")[0]
        if not link:
            return True
        fullpath = self._folder / self._filepath.parent / link
        return fullpath.exists()

    def __repr__(self) -> str:
        return f"{self._link} ({self._string}) from file {self._filepath}"


def _get_all_markdown_links(folder: Union[str, Path]) -> List[_MarkdownLink]:
    """Returns a list of all existing markdown links
    """
    pattern = re.compile(r"\[(?P<string>.+?)\]\((?P<link>\S+?)\)")
    folder = Path(folder).expanduser().absolute()
    links = []
    for rfilepath in folder.glob("**/*.md"):
        filepath = folder / rfilepath
        with filepath.open("r") as f:
            text = f.read()
        for match in pattern.finditer(text):
            links.append(_MarkdownLink(folder, rfilepath, match.group("string"), match.group("link")))
    return links


def test_assert_markdown_links_not_broken() -> None:
    folder = Path(__file__).parent.parent.absolute()
    assert (folder / "setup.py").exists(), f"Wrong root folder: {folder}"
    assert _get_all_markdown_links(folder), "There should be at least one hyperlink!"
    assert_markdown_links_not_broken(folder)
