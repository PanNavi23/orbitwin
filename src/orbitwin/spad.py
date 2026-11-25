"""Which SPAD files the template's drag and SRP read, and whether they are GMAT's stock one.

GMAT's stock data/vehicle/spad/SphericalModel.spo states its azimuth and
elevation in the opposite convention to the one GMAT reads them in (GMAT Jira
GMT-6588, closed in R2019a by documenting it). With SPADFile drag that file
pushes along the relative wind and raises orbital energy, measured on R2026a in
output/review/r2/dragsign. Every build since R2019a does this with the stock
file, and our verification flies the same file; another .spo would fly
differently on the organizer's machine and nothing else here would notice.

The fake console (scripts/fake_gmat_console.py) sits where no data/ folder is,
so a template naming '../data/vehicle/spad/...' is refused under it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .gmat_parser import GmatScriptDocument, propagator_calls
from .gmat_runner import force_model_file_problems, resolve_spad_file, stageable_name
from .models import Scenario, file_sha256

#: SHA-256 of GMAT R2026a's data/vehicle/spad/SphericalModel.spo (200655 bytes).
STOCK_SPHERICAL_MODEL_SHA256 = "5168dea005333255ed8bfff02e04f3aa0f915eef44e45bb65e6d1ca5901e531c"
GMT_6588 = "https://gmat.atlassian.net/browse/GMT-6588"

_OFF = frozenset({"", "off", "false", "none"})


@dataclass(frozen=True, slots=True)
class SpadFile:
    """One craft's SPAD file as a force it is flown with reads it."""

    craft: str
    #: SPADDragFile or SPADSRPFile.
    field: str
    force_models: tuple[str, ...]
    #: None when the script never sets the field.
    named: str | None
    #: What GMAT reads from our run folders: the script and what it names beside the template.
    path: str
    sha256: str
    stock: bool
    #: What GMAT reads running the template in its own folder, as the organizer may.
    template_path: str
    template_sha256: str

    def to_dict(self) -> dict[str, Any]:
        # A list, as JSON reads it back.
        return {**asdict(self), "force_models": list(self.force_models)}


def _plain(value: str | None) -> str:
    return "" if value is None else value.strip().strip("'\"").strip()


def _spad_fields(fields: dict[str, str], model: str) -> list[str]:
    """The craft fields a ForceModel reads a SPAD file from."""
    def field(key: str) -> str:
        return _plain(fields.get(f"{model}.{key}")).lower()

    used = []
    # The deprecated `Drag = <model>` form still switches drag on.
    atmosphere = field("Drag.AtmosphereModel") or field("Drag")
    if atmosphere not in _OFF and field("Drag.DragModel") == "spadfile":
        used.append("SPADDragFile")
    if field("SRP") not in _OFF and field("SRP.SRPModel") == "spadfile":
        used.append("SPADSRPFile")
    return used


def _flown_calls(
    scenario: Scenario, document: GmatScriptDocument
) -> list[tuple[str, tuple[str, ...]]]:
    """(propagator, craft) for every Propagate the mission runs, alignment included."""
    calls: list[tuple[str, tuple[str, ...]]] = []
    for segment in scenario.propagate_segments:
        if not segment.implicit:
            calls.extend(propagator_calls(document.lines[segment.line_index]))
    if scenario.alignment is not None:
        calls.append((scenario.alignment.propagator, (scenario.alignment.early,)))
    return calls


def flown_force_models(scenario: Scenario, document: GmatScriptDocument) -> frozenset[str]:
    """The ForceModels a Propagate the mission runs flies with."""
    models = (
        _plain(scenario.propagator_fields.get(f"{propagator}.FM"))
        for propagator, _ in _flown_calls(scenario, document)
    )
    return frozenset(model for model in models if model)


def spad_uses(scenario: Scenario, document: GmatScriptDocument) -> dict[tuple[str, str], list[str]]:
    """(craft, SPAD field) -> the ForceModels that read it, over every Propagate that flies it."""
    uses: dict[tuple[str, str], list[str]] = {}
    for propagator, craft in _flown_calls(scenario, document):
        model = _plain(scenario.propagator_fields.get(f"{propagator}.FM"))
        if not model:
            continue  # GMAT's own default force model is point mass.
        for field in _spad_fields(scenario.force_model_fields, model):
            for name in craft:
                models = uses.setdefault((name, field), [])
                if model not in models:
                    models.append(model)
    return uses


def spad_files(
    scenario: Scenario,
    document: GmatScriptDocument,
    startup_file: str | Path,
    gmat_console: str | Path,
) -> list[SpadFile]:
    """Every SPAD file the template's forces read, hashed; refuses one GMAT cannot find."""
    craft = {scenario.pursuer.name: scenario.pursuer, scenario.target.name: scenario.target}
    found: list[SpadFile] = []
    missing: list[str] = []
    for (name, field), models in sorted(spad_uses(scenario, document).items()):
        spec = craft.get(name)
        raw = spec.fields.get(field) if spec is not None else None
        named = None if raw is None else _plain(raw)
        template_dir = document.path.resolve().parent
        in_place = resolve_spad_file(named, field, template_dir, startup_file, gmat_console)
        # run_gmat copies a file found beside the template into each run, unless
        # its name climbs out of the folder.
        staged = named is not None and stageable_name(named)
        flown = (
            in_place if staged
            else resolve_spad_file(named, field, None, startup_file, gmat_console)
        )
        if flown is None or in_place is None:
            reads = ", ".join(models)
            where = (
                "" if named is None or in_place is None
                else " (found beside the template only, by a name with '..' that our run "
                "folders cannot hold; place it in this install where GMAT looks from "
                "GmatConsole's folder)"
            )
            missing.append(
                f"{name}.{field} = {named!r}, read by {reads}{where}"
                if named is not None
                else f"{name} sets no {field}, which {reads} reads"
            )
            continue
        digest = file_sha256(flown)
        found.append(SpadFile(
            craft=name, field=field, force_models=tuple(models), named=named,
            path=str(flown), sha256=digest, stock=digest == STOCK_SPHERICAL_MODEL_SHA256,
            template_path=str(in_place), template_sha256=file_sha256(in_place),
        ))
    if missing:
        raise ConfigurationError(
            "GMAT would find no SPAD file for "
            + "; ".join(missing)
            + ". It looks in the script's folder, then beside GmatConsole for ./ and ../ "
            "names and for bare drag names, and in SPAD_SRP_FILE's folder for bare SRP names. "
            "Copy every file that came with the template next to it (RUNBOOK step 1)."
        )
    return found


def spad_warnings(files: list[SpadFile]) -> list[str]:
    """Loud lines for a SPAD file other than R2026a's stock one, or one that depends on the folder."""
    notes = [
        f"SPAD FILE IS NOT GMAT'S STOCK SphericalModel.spo: {item.craft}.{item.field} reads "
        f"{item.path} (sha256 {item.sha256}). The stock file's az/el convention is the "
        f"reverse of GMAT's (GMT-6588, {GMT_6588}), and every drag and SRP force this "
        "project has verified was flown with it; ask the organizer which file their "
        "template reads before trusting a verified result."
        for item in files
        if not item.stock
    ]
    notes += [
        f"SPAD FILE DEPENDS ON THE SCRIPT'S FOLDER: {item.craft}.{item.field} reads "
        f"{item.template_path} (sha256 {item.template_sha256}) when the template runs in its "
        f"own folder, but {item.path} from the run folders our verification uses."
        for item in files
        if item.template_sha256 != item.sha256
    ]
    return notes


def spad_check(
    scenario: Scenario,
    document: GmatScriptDocument,
    gmat_console: str | Path | None,
    startup_file: str | Path | None,
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    """The SPAD files for a run's record, and the warnings its caller must show once.

    The record is None when nothing could be resolved. Raises ConfigurationError
    for a SPAD file, or any file a ForceModel names, that GMAT would not find.
    """
    if gmat_console is not None and startup_file is not None:
        problems = force_model_file_problems(
            document.path, startup_file, gmat_console, flown_force_models(scenario, document)
        )
        if problems:
            raise ConfigurationError(
                "GMAT would stop on a file the template's force model names: "
                + "; ".join(problems)
                + ". Copy every file that came with the template next to it (RUNBOOK step 1); "
                "one GMAT does not look for there belongs in its data folders."
            )
    if not spad_uses(scenario, document):
        return [], []
    if gmat_console is None or startup_file is None:
        return None, [
            "The template's drag or SRP reads a SPAD file, which was not resolved or "
            "hashed without --gmat-console and --startup-file"
        ]
    files = spad_files(scenario, document, startup_file, gmat_console)
    return [item.to_dict() for item in files], spad_warnings(files)
