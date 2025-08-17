#!/usr/bin/env python3
"""
Reset the active Python environment by uninstalling all third-party packages
and reinstalling a curated set without version constraints.

Intended use: run INSIDE the target environment (e.g., your "commands" env).

Safety: excludes pip, setuptools, wheel from uninstall.

Usage:
  python -m src.reset_commands_env --reinstall

Optional:
  --dry-run  Only print actions without executing
  --extra    Additional packages to install (comma-separated)
"""
from __future__ import annotations

import subprocess
import sys
import shlex
from typing import List


BASE_PACKAGES: List[str] = [
    # Core
    "orjson",
    "numpy",
    "pandas",
    "tqdm",
    "python-dateutil",
    "rich",
    # Modeling / NLP
    "torch",
    "transformers",
    "datasets",
    "scikit-learn",
    "evaluate",
    "sentencepiece",  # MT and some seq2seq models
    # Augmentation
    "nlpaug",
    "faker",
]

PROTECTED = {"pip", "setuptools", "wheel"}


def run(cmd: str, dry_run: bool = False) -> int:
    print(f"$ {cmd}")
    if dry_run:
        return 0
    return subprocess.call(shlex.split(cmd))


def list_installed() -> List[str]:
    out = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=freeze"]).decode()
    pkgs: List[str] = []
    for line in out.splitlines():
        if "@" in line or "==" in line:
            name = line.split("==")[0].split("@")[0].strip()
        else:
            name = line.strip()
        if not name:
            continue
        pkgs.append(name)
    return pkgs


def uninstall_all(dry_run: bool = False) -> None:
    pkgs = [p for p in list_installed() if p.lower() not in PROTECTED]
    if not pkgs:
        print("No third-party packages to uninstall.")
        return
    cmd = f"{sys.executable} -m pip uninstall -y " + " ".join(shlex.quote(p) for p in pkgs)
    run(cmd, dry_run=dry_run)


def install_set(packages: List[str], dry_run: bool = False) -> None:
    if not packages:
        return
    cmd = f"{sys.executable} -m pip install " + " ".join(shlex.quote(p) for p in packages)
    run(cmd, dry_run=dry_run)


def main(argv: List[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Reset and reinstall packages in the active environment")
    parser.add_argument("--reinstall", action="store_true", help="Perform uninstall and reinstall")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--extra", type=str, default="", help="Comma-separated extra packages to install")
    args = parser.parse_args(argv)

    extras: List[str] = [p.strip() for p in args.extra.split(",") if p.strip()]
    if not args.reinstall:
        print("Nothing to do. Pass --reinstall to proceed. Use --dry-run to preview.")
        return 0

    print("Uninstalling third-party packages (excluding pip/setuptools/wheel)...")
    uninstall_all(dry_run=args.dry_run)
    print("Reinstalling base packages (no version pins)...")
    install_set(BASE_PACKAGES + extras, dry_run=args.dry_run)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


