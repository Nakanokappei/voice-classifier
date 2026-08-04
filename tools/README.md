# tools/

Build tooling for the offline Windows distribution.

## Why this exists

Some customers run the pipeline on a Windows 11 workstation that cannot reach
GitHub, and may not be able to reach PyPI either. For them we ship a single ZIP
that carries the source *and* every Python package as a prebuilt wheel.

```bash
python tools/build_offline_bundle.py     # -> dist/voice-classifier-offline-win64-py314.zip
```

The customer-facing instructions for that ZIP are
[`../セットアップ手順書_オフライン版.md`](../セットアップ手順書_オフライン版.md).

## The Python version is not negotiable

Wheels are built per interpreter ABI, so the bundle only works on **CPython
3.14 / win_amd64** — the interpreter the pipeline was verified against. The
setup guide links the exact installer and tells the user to check
`python --version`; a 3.13 or 3.15 install fails at the install step.

To move to a different Python version, change `PYTHON_VERSION` in
`build_offline_bundle.py`, rebuild `vendor/` (see below), and update the
python.org link in the setup guide.

## `offline-lock.txt`

Exact versions, generated from a Windows venv where the pipeline was confirmed
working — not the newest resolvable set. `requirements.txt` uses `>=` ranges,
which would let the customer receive an untested combination (pandas and numpy
in particular move fast). The bundle installs from this lock instead.

To refresh it, run `pip freeze` in a working Windows venv, drop `pip` and
`hnswlib`, and write the result here.

## `vendor/`

`hnswlib` publishes **no wheels at all** on PyPI — only an sdist, which needs a
C++ compiler to install. `vendor/hnswlib-0.8.0-cp314-cp314-win_amd64.whl` was
repackaged from a copy built from source on the Windows workstation, verified
against the `RECORD` hashes pip wrote at install time. It is Apache-2.0
licensed and the LICENSE travels inside the wheel.

Dropping it is not a neutral choice: [`../src/tuner.py`](../src/tuner.py)
imports `hnswlib`, `igraph` and `leidenalg` in one `try` block, so a missing
hnswlib disables the entire Leiden sweep and leaves the tuner with KMeans and
HDBSCAN only.
