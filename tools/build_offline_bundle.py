"""Build the offline distribution ZIP for a Windows 11 workstation.

The customer this bundle targets cannot reach GitHub, and may not be able to
reach PyPI either, so the ZIP has to carry every Python package it needs as a
prebuilt wheel. Wheels are ABI-specific, which is why the whole bundle is
pinned to CPython 3.14 on win_amd64 -- the exact interpreter the pipeline was
verified against.

Run from a checkout with network access:

    python tools/build_offline_bundle.py

Output: dist/voice-classifier-offline-win64-py314.zip
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "tools" / "offline-lock.txt"
VENDOR = REPO / "tools" / "vendor"
DIST = REPO / "dist"

# The bundle is built for exactly one interpreter. Both values feed pip's
# wheel resolution and are restated in the setup guide the customer follows.
PYTHON_VERSION = "3.14"
PLATFORM = "win_amd64"

# Extracting the ZIP into C:\Users\<name>\ must produce C:\Users\<name>\voice-classifier\,
# because the setup guide's later steps use that absolute path.
PREFIX = "voice-classifier"
ZIP_NAME = f"voice-classifier-offline-win64-py{PYTHON_VERSION.replace('.', '')}.zip"

# Tracked files that would mislead or clutter an end user working offline.
# The online guide is the dangerous one: its first step is a GitHub download
# that cannot succeed in the target environment.
EXCLUDE_FROM_BUNDLE = [
    "セットアップ手順書.md",
    "セットアップ手順書.docx",
    "CLAUDE.md",
    ".github",
]


def export_source(staging: Path) -> Path:
    """Export the tracked tree at HEAD into `staging`.

    Using `git archive` rather than copying the working directory is a safety
    property, not a convenience: .env, data/input, data/output and cache/ are
    all gitignored, so a tracked-files-only export cannot leak credentials or
    customer PII into a bundle we hand to a third party.
    """
    tarball = staging / "source.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", f"--prefix={PREFIX}/", "HEAD", "-o", str(tarball)],
        cwd=REPO,
        check=True,
    )
    with tarfile.open(tarball) as tf:
        tf.extractall(staging, filter="data")
    tarball.unlink()

    root = staging / PREFIX
    for relative in EXCLUDE_FROM_BUNDLE:
        target = root / relative
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    return root


def canonical(name: str) -> str:
    """Normalise a distribution name the way PEP 503 does."""
    return re.sub(r"[-_.]+", "-", name).lower()


def vendored_names() -> set[str]:
    return {canonical(w.name.split("-")[0]) for w in VENDOR.glob("*.whl")}


def collect_wheels(wheel_dir: Path, staging: Path) -> None:
    """Download every pinned wheel, then add the ones PyPI cannot serve.

    `--only-binary=:all:` is mandatory when cross-downloading for another
    platform, and it is also the check we want: if a pinned package has no
    win_amd64 wheel, the build fails here rather than on the customer's PC.

    Vendored packages are filtered out of the download list precisely because
    they would trip that check -- hnswlib has no wheel to fetch. They stay in
    the lock, though, since the lock is what the customer installs from.
    """
    wheel_dir.mkdir(parents=True, exist_ok=True)

    vendored = vendored_names()
    downloadable = [
        line for line in LOCK.read_text(encoding="utf-8").split()
        if canonical(line.split("==")[0]) not in vendored
    ]
    download_list = staging / "download-list.txt"
    download_list.write_text("\n".join(downloadable) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable, "-m", "pip", "download",
            "--only-binary=:all:",
            "--platform", PLATFORM,
            "--python-version", PYTHON_VERSION,
            "--implementation", "cp",
            "--dest", str(wheel_dir),
            "-r", str(download_list),
        ],
        check=True,
    )
    download_list.unlink()

    # hnswlib publishes no wheels at all, so tools/vendor holds one built from
    # source on the Windows workstation. Without it the tuner's Leiden sweep
    # silently disables itself (src/tuner.py imports the three graph packages
    # as a single unit).
    for wheel in VENDOR.glob("*.whl"):
        shutil.copy2(wheel, wheel_dir / wheel.name)


def verify(wheel_dir: Path) -> None:
    """Fail the build unless the wheel set exactly matches the lock.

    The lock is what the customer runs `pip install -r` against, so every entry
    in it needs a matching wheel in the bundle -- including the vendored ones.
    """
    present = {canonical(w.name.split("-")[0]): w.name.split("-")[1] for w in wheel_dir.glob("*.whl")}
    expected = {}
    for line in LOCK.read_text(encoding="utf-8").split():
        name, version = line.split("==")
        expected[canonical(name)] = version

    problems = [f"missing: {n}" for n in expected if n not in present]
    problems += [
        f"version drift: {n} expected {v}, got {present[n]}"
        for n, v in expected.items()
        if n in present and present[n] != v
    ]
    if problems:
        raise SystemExit("Offline bundle is incomplete:\n  " + "\n  ".join(problems))
    print(f"Verified {len(present)} wheels against {LOCK.name}")


def main() -> None:
    staging = DIST / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    root = export_source(staging)
    collect_wheels(root / "wheels", staging)
    verify(root / "wheels")

    # The offline guide replaces steps 1-3 of the online one, so it must be
    # present or the customer has no entry point into the bundle.
    guide = root / "セットアップ手順書_オフライン版.md"
    if not guide.exists():
        raise SystemExit(f"Missing offline setup guide: {guide.name}")

    DIST.mkdir(exist_ok=True)
    archive = DIST / ZIP_NAME
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging))
    shutil.rmtree(staging)

    size_mb = archive.stat().st_size / 1024 / 1024
    print(f"Built {archive.relative_to(REPO)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
