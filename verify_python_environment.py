#!/usr/bin/env python3
"""Report where PionID dependencies resolve and enforce critical versions."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

from packaging.specifiers import SpecifierSet


DEPENDENCIES = (
    ("numpy", "numpy", ">=1.22,<=1.24.3", True),
    ("pandas", "pandas", ">=1.5,<2.1", False),
    ("scikit-learn", "sklearn", ">=1.1,<1.4", False),
    ("uproot", "uproot", ">=5.0,<6", True),
    ("awkward", "awkward", ">=2.0,<3", True),
    ("matplotlib", "matplotlib", ">=3.5,<3.8", False),
    ("gast", "gast", ">=0.2.1,<=0.4.0", True),
    ("tensorflow", "tensorflow", "==2.13.1", True),
)


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    failures = []
    warnings = []
    for distribution, module_name, requirement, critical in DEPENDENCIES:
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
            message = f"{distribution} {installed} does not satisfy {requirement}"
            (failures if critical else warnings).append(message)

    if warnings:
        print("Warnings:", file=sys.stderr)
        for message in warnings:
            print(f"  - {message}", file=sys.stderr)
    if failures:
        print("Errors:", file=sys.stderr)
        for message in failures:
            print(f"  - {message}", file=sys.stderr)
        print(
            "Keep a TensorFlow-2.13-compatible NumPy in the active environment; "
            "the shared NumPy 1.26.4 must not take priority.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
