"""The CI workflow is the one reviewed, byte for byte.

Six review rounds each found a spelling that a YAML reader in this file let
through and GitHub's parser did not, so the file is pinned by hash and a
person reviews every change to it against REVIEW.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

# Update these only after the review in REVIEW.
REVIEWED_CI_SHA256 = "5d1be91968c61509de1629dca64ba2d2e20e35ad0a6a60459e65feef7b04eedc"
# ci.yml's timeout-minutes: the test job on the repository's own runners and on
# GitHub's, and the checks job. test_make_targets keeps faulthandler_timeout below each.
OWN_RUNNERS_MIN = 45
GITHUB_RUNNERS_MIN = 90
CHECKS_MIN = 30

VARIABLES = ("ORBITWIN_RUNS_ON_TEST", "ORBITWIN_RUNS_ON_CHECKS")

REVIEW = f"""\
.github/workflows changed. Review it, then update REVIEWED_CI_SHA256 in
tests/unit/test_ci_runner.py:
- runs-on of each job is fromJSON(github.event.repository.private &&
  vars.ORBITWIN_RUNS_ON_<JOB> || '"ubuntu-latest"'), and the test job's
  timeout-minutes has the same guard: {OWN_RUNNERS_MIN} with the variable, else
  {GITHUB_RUNNERS_MIN}; checks has {CHECKS_MIN}.
- on: push to main, pull_request, workflow_dispatch with the one `ref` input;
  no pull_request_target or workflow_run. permissions: contents: read at the
  top, none on a job. No secrets.
- every `uses:` is a full commit SHA of a reviewed action, and its `# vX.Y.Z`
  comment matches `gh api repos/<action>/git/ref/tags/<tag>`.
- in the test job, no `if:`, `continue-on-error:`, `defaults:` or wrapper
  (such as `timeout` in `run:`) that skips the Fast suite or ends it early.
- no `container:` or `services:`.
- hung tests: faulthandler_timeout (pyproject.toml) is below every limit that
  can end a test step; if a timeout changes, change OWN_RUNNERS_MIN,
  GITHUB_RUNNERS_MIN or CHECKS_MIN with it.
- inputs.ref and other text a pull request controls reach no `run:` script.
- the only file under .github/workflows is ci.yml."""


def workflow_files(directory: Path) -> dict[str, bytes]:
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def check(files: dict[str, bytes]) -> None:
    assert set(files) == {"ci.yml"}, f"{sorted(files)}\n{REVIEW}"
    assert hashlib.sha256(files["ci.yml"]).hexdigest() == REVIEWED_CI_SHA256, REVIEW


def test_the_workflow_is_the_reviewed_one() -> None:
    check(workflow_files(WORKFLOWS))


def test_any_one_byte_change_to_the_workflow_is_refused() -> None:
    reviewed = (WORKFLOWS / "ci.yml").read_bytes()
    for offset in range(len(reviewed)):
        changed = bytearray(reviewed)
        changed[offset] ^= 0x01
        with pytest.raises(AssertionError, match="update REVIEWED_CI_SHA256"):
            check({"ci.yml": bytes(changed)})
    for changed in (reviewed[:-1], reviewed + b"\n", b" " + reviewed):
        with pytest.raises(AssertionError, match="update REVIEWED_CI_SHA256"):
            check({"ci.yml": changed})


@pytest.mark.parametrize("names", [
    pytest.param(["ci.yml", "release.yml"], id="second-workflow"),
    pytest.param(["ci.yaml"], id="renamed"),
    pytest.param(["ci.yml", "sub/other.yml"], id="nested"),
    pytest.param(["ci.yml", "notes.txt"], id="other-file"),
])
def test_another_file_under_workflows_is_refused(tmp_path: Path, names: list[str]) -> None:
    reviewed = (WORKFLOWS / "ci.yml").read_bytes()
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(reviewed)
    with pytest.raises(AssertionError, match="update REVIEWED_CI_SHA256"):
        check(workflow_files(tmp_path))


def described(variable: str, readme: str) -> bool:
    """Whether the README names the variable whole, in backticks."""
    return re.search(rf"`{re.escape(variable)}`", readme) is not None


@pytest.mark.parametrize("readme", ["README.md", "README.zh-TW.md"])
@pytest.mark.parametrize("variable", VARIABLES)
def test_the_readme_says_what_each_variable_holds(readme: str, variable: str) -> None:
    assert variable.encode() in (WORKFLOWS / "ci.yml").read_bytes()
    assert described(variable, (ROOT / readme).read_text(encoding="utf-8")), (
        f"{readme} does not say what {variable} holds")


@pytest.mark.parametrize("readme", [
    pytest.param("ORBITWIN_RUNS_ON_TEST holds the labels.", id="no-backticks"),
    pytest.param("`ORBITWIN_RUNS_ON_TESTS` holds the labels.", id="longer-name"),
])
def test_an_undescribed_variable_is_found(readme: str) -> None:
    assert not described("ORBITWIN_RUNS_ON_TEST", readme)
