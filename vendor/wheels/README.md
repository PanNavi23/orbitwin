# Offline wheel bundle

The internet must not be a competition-day dependency. `scripts/bootstrap_linux.sh` installs
from PyPI, so it cannot be the competition-day path. This directory is.

## Before competition day, on a machine with internet

```bash
bash scripts/fetch_offline_wheels.sh
```

Downloads every pinned dependency in `requirements-dev.txt` for Python 3.11
through 3.14 on x86-64 Linux -- one bundle covering every interpreter
`pyproject.toml` declares, because the organizer's Python version is not known
in advance. 39 wheels and 278 MB, pip's own among them for an interpreter
without ensurepip. Writes `SHA256SUMS.txt`.

Every version comes from `constraints.txt`, dependencies of dependencies
included, so a refetch brings back the same versions (PyPI could still add a
file to a release; the manifest's diff would show it) and replaces every wheel
in the directory. A bundle for another target (`WHEEL_PYTHON_VERSIONS`,
`WHEEL_PLATFORMS`) therefore goes in a directory of its own, `WHEELS_DIR`.
`requirements-dev.txt` names `constraints.txt`, and the fetch, the offline
bootstrap and CI read no other constraints file. To move a version, change it
there, refetch, and commit both files.

## On the competition machine

```bash
bash scripts/bootstrap_offline.sh
. .venv/bin/activate
```

Refuses a bundle without `SHA256SUMS.txt`, with a wheel it does not list or
without one it lists, or whose hashes differ. Then creates `.venv` and installs
with `--no-index` and pip's `--require-hashes`, from the hashes `SHA256SUMS.txt`
lists for the versions `constraints.txt` pins (`scripts/wheel_requirements.py`),
so pip itself refuses any other file it finds here, and a missing wheel fails
loudly instead of silently reaching for a network that is not supposed to be
used. The runbook's later steps call `python` and
`orbitwin` by name, so every terminal used for them activates `.venv` first
(docs/RUNBOOK.md step 0).

## What is committed

`README.md`, `SHA256SUMS.txt` and `constraints.txt` only. The `.whl` files are gitignored: they are
third-party binaries, and a bundle carried on a drive can be checked against the
committed manifest to prove it is the set that was validated.

`make package` adds the wheels `SHA256SUMS.txt` lists to the release archive,
after checking each one against it, so the archive installs with no network.
The root `SHA256SUMS.txt` lists them too. A listed wheel that is missing, a
link or different, or a manifest that is missing, empty or not in the form
`sha256sum ./*.whl` (or `sha256sum -b`) writes, stops the build. Put back the
wheels that were validated, or refetch; if the manifest the refetch writes
differs, it is a different set, to be committed and tested again. `scripts/build_release.py --without-wheels` builds
`tasa_orbitwin-<version>-<commit>-without-wheels.zip`, which needs the network.
