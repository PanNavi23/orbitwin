.PHONY: test test-isolated test-slow test-gmat fuzz coverage lint typecheck validate package clean

PYTHON ?= python
# Where `make package` writes the release ZIP. Repo-relative and gitignored;
# override with `make package PACKAGE_DIR=/somewhere/else`.
PACKAGE_DIR ?= dist
# Where test-isolated and coverage write. Gitignored, like output/validation/,
# where `make validate` writes its evidence.
REPORT_DIR ?= output/make
export PYTHONPATH := src

# Fast deterministic test suite. CI's Fast suite step passes pytest-xdist and
# --durations options in PYTEST_ARGS (.github/workflows/ci.yml).
PYTEST_ARGS ?=
test:
	$(PYTHON) -m pytest -m "not slow and not gmat" $(PYTEST_ARGS)

# Every test file in its own process, so nothing one leaves behind can reach
# the next. Same selection as `test`; 7% slower on 2026-08-21 (234 s to 250 s); the
# default stays plain pytest because that is the target people add -k and -x to.
test-isolated:
	$(PYTHON) scripts/run_isolated_test_files.py --log $(REPORT_DIR)/pytest_isolated.log

# Computationally expensive OOD cases.
test-slow:
	$(PYTHON) -m pytest -m slow

# Against stock GmatConsole. Needs GMAT_CONSOLE and GMAT_STARTUP; without them
# every test fails rather than skips, so a wrong path cannot pass unnoticed.
test-gmat:
	$(PYTHON) -m pytest -o addopts='' -m gmat $(PYTEST_ARGS)

fuzz:
	$(PYTHON) scripts/ood_fuzz.py --cases 1000 --seed 20260818

# The `test` selection, as scripts/release_validate.sh measures it. What is
# measured, workers and subprocesses included, is [tool.coverage.run].
# Its data and pytest's cache go there too, not into the checkout.
coverage:
	mkdir -p $(REPORT_DIR)
	COVERAGE_FILE=$(REPORT_DIR)/.coverage $(PYTHON) -m pytest -m "not slow and not gmat" \
		-o cache_dir=$(REPORT_DIR)/.pytest_cache --cov --cov-report=term-missing \
		--cov-report=xml:$(REPORT_DIR)/coverage.xml $(PYTEST_ARGS)

# Only pyproject.toml configures ruff: a ruff.toml or pyproject.toml below it is not read.
lint:
	$(PYTHON) -m ruff check --config pyproject.toml src tests scripts

typecheck:
	$(PYTHON) -m mypy src/orbitwin

# PYTEST_ARGS reaches its coverage run only, e.g. "-n 16" with pytest-xdist.
validate:
	PYTHON="$(PYTHON)" PYTEST_ARGS="$(PYTEST_ARGS)" bash scripts/release_validate.sh

package:
	$(PYTHON) scripts/build_release.py --project-root . --output-dir $(PACKAGE_DIR)

# What `make package` cleans (build_release.clean), in shell so it needs no
# Python: caches, coverage data, src/*.egg-info and bytecode. dist/, build/,
# output/ and the environments stay.
clean:
	sh scripts/clean.sh "$(CURDIR)"
