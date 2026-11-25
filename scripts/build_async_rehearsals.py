#!/usr/bin/env python3
"""Build the asynchronous-start rehearsals R6-R9 from R1 and R2 on real GMAT.

    python scripts/build_async_rehearsals.py --workdir output/async-build [--only R9]

R6-R8 each move one craft's epoch 1800 s earlier and put it where GMAT's own
force model flies it back to in that time, so the geometry at the later epoch
is R1's or R2's. R9 is R8 with SC_Alien also moved 1800 s later, flown on by
GMAT, so SC_Human has a 3600 s head start on R1's physics. R6 also writes SC_Human's state as the EarthFixed
SphericalAZFPA insertion GMAT reads at R1's epoch. A joint Propagate's goals
count from that Propagate, so each keeps R1's or R2's T_max as its length. Needs $GMAT_CONSOLE and $GMAT_STARTUP; writes the scripts
into examples/ranking/.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitwin.gmat_runner import run_gmat
from orbitwin.io import prepare_empty_output
from orbitwin.report_parser import parse_numeric_report

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "ranking"
LEAD_S = 1800.0
_CARTESIAN = ("X", "Y", "Z", "VX", "VY", "VZ")
_SPHERICAL = ("RMAG", "RA", "DEC", "VMAG", "AZI", "FPA")


def _resources(text: str) -> str:
    return text.split("BeginMissionSequence;")[0]


def _gmat(text: str, mission: str, columns: list[str], workdir: Path) -> list[float]:
    """Run ``mission`` on ``text``'s resources and return the last row of ``columns``."""
    workdir.mkdir(parents=True, exist_ok=False)
    script = workdir / "build.script"
    script.write_text(
        _resources(text)
        + "Create ReportFile Build;\nBuild.Filename = 'build.txt';\nBuild.Precision = 17;\n"
        + "Build.WriteHeaders = true;\nBuild.Add = {" + ", ".join(columns) + "};\n\n"
        + "BeginMissionSequence;\n" + mission,
        encoding="utf-8",
    )
    result = run_gmat(
        os.environ["GMAT_CONSOLE"], script, workdir / "run", ["build.txt"],
        startup_file=os.environ["GMAT_STARTUP"], timeout_s=600.0,
    )
    if not result.success:
        raise SystemExit(f"{workdir}: {result.error}")
    return [float(value) for value in parse_numeric_report(result.reports["build.txt"]).data[-1]]


def _back(text: str, craft: str, workdir: Path, seconds: float = -LEAD_S) -> list[float]:
    """``craft``'s EarthMJ2000Eq state ``seconds`` from its epoch, flown by GMAT."""
    columns = [f"{craft}.EarthMJ2000Eq.{axis}" for axis in _CARTESIAN]
    back = "BackProp " if seconds < 0.0 else ""
    mission = (
        f"Propagate {back}DefaultProp({craft}) {{{craft}.ElapsedSecs = {seconds!r}}};\n"
    )
    return _gmat(text, mission, columns, workdir)


def _set_state(text: str, craft: str, system: str, kind: str, names: tuple[str, ...],
               values: list[float], epoch: str | None) -> str:
    """Replace ``craft``'s Epoch, frame and state lines, keeping the rest of its block."""
    lines = text.splitlines()
    state_fields = {"SMA", "ECC", "INC", "RAAN", "AOP", "TA", *_CARTESIAN, *_SPHERICAL}
    out: list[str] = []
    for line in lines:
        match = re.match(rf"^{craft}\.(\w+) = ", line)
        field = match.group(1) if match else None
        if field in state_fields:
            continue
        if field == "Epoch" and epoch is not None:
            line = f"{craft}.Epoch = '{epoch}';"
        elif field == "CoordinateSystem":
            line = f"{craft}.CoordinateSystem = {system};"
        elif field == "DisplayStateType":
            line = f"{craft}.DisplayStateType = {kind};"
            out.append(line)
            out.extend(f"{craft}.{name} = {value!r};" for name, value in zip(names, values, strict=True))
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def _mission(text: str, mission: str, header: str, report_add: str) -> str:
    head = _resources(text).split("\n", 1)[1]
    head = re.sub(r"^% Ranking rehearsal R\d.*?\n(?=\n|%-)", "", head, flags=re.S)
    head = head.replace(
        "ReportFile1.Add = {", "ReportFile1.Add = {" + report_add, 1
    )
    return "%General Mission Analysis Tool(GMAT) Script\n" + header + head + "BeginMissionSequence;\n" + mission


def _build_r9(workdir: Path) -> None:
    """R8 with SC_Alien 1800 s later, where GMAT flies R1's on to: a 3600 s head start."""
    r1 = (EXAMPLES / "R1_NearPlane_Reachable.script").read_text(encoding="utf-8")
    r8 = (EXAMPLES / "R8_PursuerFirst_Aligned.script").read_text(encoding="utf-8")
    alien_on = _back(r1, "SC_Alien", workdir / "r9_alien", LEAD_S)
    r9 = _set_state(r8, "SC_Alien", "EarthMJ2000Eq", "Cartesian", _CARTESIAN, alien_on,
                    "09 Nov 2026 02:30:00.000")
    (EXAMPLES / "R9_PursuerFirst_LongHeadStart.script").write_text(_mission(
        r9,
        "Propagate DefaultProp(SC_Human) {SC_Human.UTCModJulian = SC_Alien.UTCModJulian};\n"
        "Propagate DefaultProp(SC_Alien, SC_Human) {SC_Alien.ElapsedSecs = 4911.667, SC_Human.ElapsedSecs = 4911.667};\n",
        "% Ranking rehearsal R9 (synthetic, not an organizer file).\n"
        "% R8 with SC_Alien starting 1800 s later, at 02:30:00, where GMAT flies R1's\n"
        "% on to: SC_Human has a 3600 s head start, long enough for burns in it to\n"
        "% matter. T_max from SC_Human's epoch is R8's.\n",
        "",
    ), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--only", choices=["R9"], help="build this one and leave R6-R8 alone")
    args = parser.parse_args()
    prepare_empty_output(args.workdir, "rehearsal build")
    if args.only == "R9":
        _build_r9(args.workdir)
        return 0
    r1 = (EXAMPLES / "R1_NearPlane_Reachable.script").read_text(encoding="utf-8")
    r2 = (EXAMPLES / "R2_SteepPlane_NearMiss.script").read_text(encoding="utf-8")
    early_epoch = "09 Nov 2026 01:30:00.000"
    altitude = "SC_Human.Earth.Altitude, "

    # R6: SC_Alien 1800 s first; SC_Human inserted from EarthFixed SphericalAZFPA.
    # RMAG hangs off a body, the angles and speed off a coordinate system.
    insertion = [
        f"SC_Human.{'Earth' if name == 'RMAG' else 'EarthFixed'}.{name}" for name in _SPHERICAL
    ]
    spherical = _gmat(
        r1, "Report Build " + " ".join(insertion) + ";\n", insertion, args.workdir / "r6_insertion"
    )
    alien_back = _back(r1, "SC_Alien", args.workdir / "r6_alien")
    r6 = _set_state(r1, "SC_Human", "EarthFixed", "SphericalAZFPA", _SPHERICAL, spherical, None)
    r6 = _set_state(r6, "SC_Alien", "EarthMJ2000Eq", "Cartesian", _CARTESIAN, alien_back, early_epoch)
    (EXAMPLES / "R6_InsertedLate_Aligned.script").write_text(_mission(
        r6,
        "Propagate DefaultProp(SC_Alien) {SC_Alien.UTCModJulian = SC_Human.UTCModJulian};\n"
        "Propagate DefaultProp(SC_Alien, SC_Human) {SC_Alien.ElapsedSecs = 6711.667, SC_Human.ElapsedSecs = 6711.667};\n",
        "% Ranking rehearsal R6 (synthetic, not an organizer file).\n"
        "% R1's geometry at 02:00:00, with SC_Alien starting 1800 s earlier where GMAT\n"
        "% flies it back to, and SC_Human inserted from an EarthFixed SphericalAZFPA\n"
        "% state. The tutorial's alignment brings SC_Alien to SC_Human's epoch.\n",
        altitude,
    ), encoding="utf-8")

    # R7: R2 with SC_Alien 1800 s first and no alignment: a team adds it (Q4).
    alien_back = _back(r2, "SC_Alien", args.workdir / "r7_alien")
    r7 = _set_state(r2, "SC_Alien", "EarthMJ2000Eq", "Cartesian", _CARTESIAN, alien_back, early_epoch)
    (EXAMPLES / "R7_NoAlignment_NearMiss.script").write_text(_mission(
        r7,
        "Propagate DefaultProp(SC_Alien, SC_Human) {SC_Alien.ElapsedSecs = 5791.381, SC_Human.ElapsedSecs = 5791.381};\n",
        "% Ranking rehearsal R7 (synthetic, not an organizer file).\n"
        "% R2's geometry at 02:00:00, SC_Alien starting 1800 s earlier, and no\n"
        "% alignment: refused unless policy.add_epoch_alignment adds one.\n",
        altitude,
    ), encoding="utf-8")

    # R8: SC_Human 1800 s first, flown back on its own circular orbit.
    human_back = _back(r1, "SC_Human", args.workdir / "r8_human")
    r8 = _set_state(r1, "SC_Human", "EarthMJ2000Eq", "Cartesian", _CARTESIAN, human_back, early_epoch)
    (EXAMPLES / "R8_PursuerFirst_Aligned.script").write_text(_mission(
        r8,
        "Propagate DefaultProp(SC_Human) {SC_Human.UTCModJulian = SC_Alien.UTCModJulian};\n"
        "Propagate DefaultProp(SC_Alien, SC_Human) {SC_Alien.ElapsedSecs = 6711.667, SC_Human.ElapsedSecs = 6711.667};\n",
        "% Ranking rehearsal R8 (synthetic, not an organizer file).\n"
        "% R1's geometry at 02:00:00, with SC_Human starting 1800 s earlier on its\n"
        "% own orbit; it flies the alignment alone, 421 km up throughout.\n",
        altitude,
    ), encoding="utf-8")
    _build_r9(args.workdir)
    # For async_scenarios.yaml's built_with: what the geometry rests on.
    hashes = run_gmat(
        os.environ["GMAT_CONSOLE"], EXAMPLES / "R8_PursuerFirst_Aligned.script",
        args.workdir / "hashes", [], startup_file=os.environ["GMAT_STARTUP"], timeout_s=600.0,
    ).data_file_sha256
    for key in (
        "EOP_FILE", "LEAP_SECS_FILE", "LSK_FILE", "NUTATION_COEFF_FILE",
        "EARTH_LATEST_PCK_FILE", "EARTH_PCK_CURRENT_FILE", "EARTH_PCK_PREDICTED_FILE",
        "EGM96_FILE", "PLANETARY_SPK_FILE", "CSSI_FLUX_FILE", "SPAD_SRP_FILE",
        "script:SC_Alien.SPADDragFile", "script:SC_Human.SPADDragFile",
    ):
        print(f"  {key}: {hashes.get(key, {}).get('sha256')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
