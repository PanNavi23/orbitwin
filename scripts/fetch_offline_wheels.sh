#!/usr/bin/env bash
# Fill vendor/wheels/ so the toolchain can be installed with no network.
#
# Run this at home, with internet, before competition day. The internet must
# not be a competition-day dependency, so `pip install` reaching PyPI on the day is not
# an option. Carry vendor/wheels/ with the repository.
#
# Wheels are Python-version and platform specific, so one bundle per supported
# interpreter is fetched: 3.11 to 3.14, see VERSIONS below.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

cd "$(dirname "$0")/.."
WHEELS="${WHEELS_DIR:-vendor/wheels}"
# pip matches --platform as an exact wheel tag, so every tag the pinned set
# actually publishes has to be listed. numpy and scipy ship manylinux_2_28;
# ruff and mypy still ship manylinux_2_17/2014.
PLATFORMS="${WHEEL_PLATFORMS:-manylinux_2_28_x86_64 manylinux_2_17_x86_64 manylinux2014_x86_64}"
# 3.14 is here because pyproject.toml declares `>=3.11` with no upper bound and
# Ubuntu 26.04 already ships 3.14 as `python3`. Leaving it out made
# bootstrap_offline.sh fail on the default interpreter of a current machine --
# correctly and loudly, but with no remedy on a day that has no network. Every
# pin in requirements-dev.txt publishes a cp314 wheel, so covering it is free.
VERSIONS="${WHEEL_PYTHON_VERSIONS:-3.11 3.12 3.13 3.14}"
# Every version the bundle holds, dependencies included, so a refetch brings
# back the same wheels instead of whatever was newest that day.
CONSTRAINTS=vendor/wheels/constraints.txt

mkdir -p "$WHEELS"
# Into an empty directory first, then in place of the old wheels: a refetch
# into the old directory kept every superseded wheel beside the new one and
# listed both, and pip installs the newest it finds. A fetch replaces every
# wheel in $WHEELS, so a bundle for another target goes in a WHEELS_DIR of its
# own. The staging directory sits beside $WHEELS, on its file system, so the
# move below is a rename: a copy from a tmpfs could run out of space half-way
# with the old wheels already gone.
staging="$(mktemp -d "$(dirname "$WHEELS")/.wheels-staging.XXXXXX")"
trap 'rm -rf "$staging"' EXIT

for version in $VERSIONS; do
  echo "==> ${version}"
  platform_args=()
  for tag in $PLATFORMS; do
    platform_args+=(--platform "$tag")
  done
  # requirements-dev.txt names the constraints too; naming them here does not depend on it.
  "$PYTHON" -m pip download \
    --dest "$staging" \
    --only-binary=:all: \
    "${platform_args[@]}" \
    --python-version "$version" \
    --requirement requirements-dev.txt \
    --constraint "$CONSTRAINTS"
done

# Pure-Python build backend, needed for `pip install -e .` without an index;
# and pip itself, for an interpreter without ensurepip (Debian's and Ubuntu's
# python3.X-venv package), which bootstrap_offline.sh then installs from here.
"$PYTHON" -m pip download --dest "$staging" --only-binary=:all: \
  --constraint "$CONSTRAINTS" setuptools wheel pip

rm -f "$WHEELS"/*.whl
mv "$staging"/*.whl "$WHEELS"/
( cd "$WHEELS" && sha256sum ./*.whl > SHA256SUMS.txt )

count=$(find "$WHEELS" -name '*.whl' | wc -l)
size=$(du -sh "$WHEELS" | cut -f1)
echo
echo "${count} wheels, ${size}, in ${WHEELS}"
echo "Verify on the target machine with: cd ${WHEELS} && sha256sum -c SHA256SUMS.txt"
