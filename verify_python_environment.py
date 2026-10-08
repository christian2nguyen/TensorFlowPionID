#!/usr/bin/env python3
"""Report where PionID dependencies resolve and verify the ANNIE stack."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

from packaging.specifiers import SpecifierSet


DEPENDENCIES = (
    ("numpy", "numpy", "==1.26.4"),
    ("pandas", "pandas", "==2.3.3"),
    ("scikit-learn", "sklearn", "==1.6.1"),
    ("scipy", "scipy", "==1.13.1"),
    ("uproot", "uproot", "==5.6.9"),
    ("awkward", "awkward", "==2.8.12"),
    ("awkward-cpp", "awkward_cpp", "==51"),
    ("matplotlib", "matplotlib", "==3.9.4"),
    ("gast", "gast", "==0.7.0"),
    ("tensorflow", "tensorflow", "==2.20.0"),
    ("keras", "keras", ">=3.10,<4"),
)


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    failures = []
    for distribution, module_name, requirement in DEPENDENCIES:
        try:
            module = import_module(module_name)
            installed = str(
                getattr(module, "__version__", None) or version(distribution)
            )
            location = Path(module.__file__).resolve()
        except (PackageNotFoundError, ImportError) as error:
            failures.append(
                f"{distribution}: not importable ({type(error).__name__}: {error})"
            )
            continue
        compatible = installed in SpecifierSet(requirement)
        state = "OK" if compatible else "OUTSIDE REQUESTED RANGE"
        print(
            f"{distribution}: {installed} [{state}]\n"
            f"  requested: {requirement}\n"
            f"  loaded from: {location}"
        )
        if not compatible:
            failures.append(
                f"{distribution} {installed} does not satisfy {requirement}"
            )

    if failures:
        print("Errors:", file=sys.stderr)
        for message in failures:
            print(f"  - {message}", file=sys.stderr)
        print(
            "The project expects the tested TensorFlow 2.20 stack from "
            "/exp/annie/app/users/dajana/myboy/lib/python3.9/site-packages.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
