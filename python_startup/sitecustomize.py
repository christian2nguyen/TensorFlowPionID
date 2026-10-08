"""Safely combine ANNIE's shared packages with the active environment."""

from __future__ import annotations

import os
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder
from pathlib import Path
import site
import sys


shared_site = os.environ.get("PIONID_SHARED_SITE_PACKAGES")
if shared_site:
    shared_path = Path(shared_site)
    if shared_path.is_dir():
        shared_path_text = str(shared_path)

        class _SharedAnniePackageFinder(MetaPathFinder):
            """Prefer compatible shared ROOT packages without replacing NumPy."""

            preferred_roots = {"awkward", "awkward_cpp", "uproot"}

            def find_spec(self, fullname, path=None, target=None):
                if "." not in fullname and fullname in self.preferred_roots:
                    return PathFinder.find_spec(fullname, [shared_path_text])
                return None

        sys.meta_path.insert(0, _SharedAnniePackageFinder())
        if shared_path_text not in sys.path:
            site.addsitedir(shared_path_text)
