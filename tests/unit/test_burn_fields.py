"""ImpulsiveBurn fields the parser cannot model are refused (E2E audit A4, B17).

R2026a reads `Burn.V`, `.N` and `.B` as Element1-3 and flies the later of the
two; measured with `V = 0` after the Element lines, GMAT flew 70.69 m/s where
the render, verify-policy and the certificate all said 91.15.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from conftest import quick_config

from orbitwin.config import load_config
from orbitwin.errors import ParseError
from orbitwin.gmat_parser import BURN_ALIASES, BURN_FIELDS, parse_gmat_script

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "burn_alias"
FIRST = "ImpulsiveBurn1.Element1 = 0.5983;\n"
LAST = "ImpulsiveBurn1.Element3 = 0;\n"


def _with(workshop: Path, tmp_path: Path, line: str, *, before: bool) -> Path:
    text = workshop.read_text(encoding="utf-8")
    assert FIRST in text and LAST in text
    text = text.replace(FIRST, line + FIRST) if before else text.replace(LAST, LAST + line)
    path = tmp_path / "burn.script"
    path.write_text(text, encoding="utf-8")
    return path


def _line_of(path: Path, line: str) -> int:
    return path.read_text(encoding="utf-8").splitlines().index(line.rstrip("\n")) + 1


@pytest.mark.parametrize("before", [True, False], ids=["before", "after"])
@pytest.mark.parametrize("alias", sorted(BURN_ALIASES))
def test_an_element_alias_is_refused_wherever_it_sits(
    workshop_script: Path, tmp_path: Path, alias: str, before: bool
) -> None:
    line = f"ImpulsiveBurn1.{alias} = 0.1;\n"
    path = _with(workshop_script, tmp_path, line, before=before)
    with pytest.raises(ParseError, match=rf"Line {_line_of(path, line)} sets ImpulsiveBurn1\.{alias}\. "
                       rf"GMAT reads {alias} as {BURN_ALIASES[alias]}"):
        parse_gmat_script(path, quick_config())


@pytest.mark.parametrize(
    ("field", "gmat"),
    [
        ("X", "GmatConsole R2026a refuses the script \\(the field name is not permitted\\)"),
        ("Element1Label", "GmatConsole R2026a refuses the script \\(the field name is not permitted\\)"),
        ("v", "GmatConsole R2026a refuses the script \\(the field name is not permitted\\)"),
        ("element1", "GmatConsole R2026a refuses the script \\(the field name is not permitted\\)"),
        ("SpacecraftName", "GmatConsole R2026a refuses the script\\. Modelled"),
        ("Covariance", "R2026a defines it and nothing here models it\\. Modelled"),
    ],
)
def test_a_field_not_modelled_is_refused_with_what_gmat_would_do(
    workshop_script: Path, tmp_path: Path, field: str, gmat: str
) -> None:
    line = f"ImpulsiveBurn1.{field} = 0;\n"
    path = _with(workshop_script, tmp_path, line, before=False)
    with pytest.raises(
        ParseError,
        match=rf"Line {_line_of(path, line)} sets ImpulsiveBurn1\.{field}, which is not an "
        rf"ImpulsiveBurn field modelled here; {gmat}",
    ):
        parse_gmat_script(path, quick_config())


@pytest.mark.parametrize(
    "lines",
    [
        "ImpulsiveBurn1.VectorFormat = Spherical;\n",
        "ImpulsiveBurn1.DeltaTankMass = 5;\n",
    ],
    ids=["VectorFormat", "DeltaTankMass"],
)
def test_a_field_inert_for_the_impulse_is_read_past(
    workshop_script: Path, tmp_path: Path, lines: str
) -> None:
    """Measured on R2026a: each left the impulse flown as written, to 2e-16 km/s.

    The workshop sets DecrementMass, Isp and GravitationalAccel already.
    """
    path = _with(workshop_script, tmp_path, lines, before=False)
    scenario, _ = parse_gmat_script(path, quick_config())
    assert np.array_equal(scenario.burn_slots[0].vector, [0.5983, 0.0, 0.0])


def test_every_field_the_bundled_templates_set_is_modelled(project_root: Path) -> None:
    scenario, _ = parse_gmat_script(project_root / "examples" / "two_burn_rendezvous.script",
                                    quick_config())
    assert {"Element1", "Element2", "Element3", "CoordinateSystem", "Origin", "Axes",
            "DecrementMass", "Isp", "GravitationalAccel"} <= BURN_FIELDS
    assert len(scenario.burn_slots) == 2


def test_an_unmaneuvered_burn_with_an_alias_is_refused(
    unmaneuvered_burns_script: Path, tmp_path: Path
) -> None:
    """An open template's declared burn is the one the render uses."""
    text = unmaneuvered_burns_script.read_text(encoding="utf-8")
    anchor = next(line for line in text.splitlines(keepends=True) if ".Element3 = " in line)
    path = tmp_path / "open.script"
    burn = anchor.split(".", 1)[0]
    path.write_text(text.replace(anchor, anchor + f"{burn}.B = 0.2;\n"), encoding="utf-8")
    with pytest.raises(ParseError, match=rf"sets {burn}\.B\. GMAT reads B as Element3"):
        parse_gmat_script(path, dataclasses.replace(quick_config(), pursuer_name="Pursuer"))


def test_the_audits_b17_is_refused() -> None:
    """A4's variant: `ImpulsiveBurn1.V = 0;` after the Element lines of a ranking template."""
    config = load_config(Path(__file__).resolve().parents[2] / "config" / "ranking.yaml")
    with pytest.raises(ParseError, match=r"Line 126 sets ImpulsiveBurn1\.V\."):
        parse_gmat_script(FIXTURES / "B17_burn_alias_after.script", config)


SPARE = (
    "Create ImpulsiveBurn SpareBurn;\nSpareBurn.CoordinateSystem = Local;\n"
    "SpareBurn.Origin = Earth;\nSpareBurn.Axes = VNB;\n"
)


@pytest.mark.parametrize(
    "fields",
    ["SpareBurn.V = 0.1;\n", "SpareBurn.Element1 = 0;\nSpareBurn.Element2 = 0;\n"
     "SpareBurn.Element3 = 0;\nSpareBurn.N = 0.1;\n"],
    ids=["alias-only", "elements-then-alias"],
)
def test_an_alias_on_a_burn_never_maneuvered_is_left_to_gmat(
    workshop_script: Path, tmp_path: Path, fields: str
) -> None:
    """Review V3: GMAT runs a spare burn's V/N/B, and nothing here flies or rewrites it."""
    anchor = "Create ImpulsiveBurn ImpulsiveBurn1;\n"
    path = tmp_path / "spare.script"
    path.write_text(
        workshop_script.read_text(encoding="utf-8").replace(anchor, SPARE + fields + anchor),
        encoding="utf-8",
    )
    scenario, _ = parse_gmat_script(path, quick_config())
    assert [slot.name for slot in scenario.burn_slots] == ["ImpulsiveBurn1", "ImpulsiveBurn2"]


def test_an_unknown_field_on_a_spare_burn_is_still_refused(
    workshop_script: Path, tmp_path: Path
) -> None:
    anchor = "Create ImpulsiveBurn ImpulsiveBurn1;\n"
    path = tmp_path / "spare.script"
    path.write_text(
        workshop_script.read_text(encoding="utf-8").replace(
            anchor, SPARE + "SpareBurn.X = 0.1;\n" + anchor
        ),
        encoding="utf-8",
    )
    with pytest.raises(ParseError, match=r"sets SpareBurn\.X, which is not an ImpulsiveBurn field"):
        parse_gmat_script(path, quick_config())


def _old_style(unmaneuvered_burns_script: Path) -> str:
    text = unmaneuvered_burns_script.read_text(encoding="utf-8")
    for index, alias in zip((1, 2, 3), ("V", "N", "B"), strict=True):
        assert text.count(f"Burn2.Element{index} = 0;\n") == 1
        text = text.replace(f"Burn2.Element{index} = 0;\n", f"Burn2.{alias} = 0;\n")
    return text


def test_an_open_templates_old_style_burn_is_left_alone(
    unmaneuvered_burns_script: Path, tmp_path: Path
) -> None:
    """Review V3: V/N/B and no Element lines; as before, a warning and no slot."""
    path = tmp_path / "open.script"
    path.write_text(_old_style(unmaneuvered_burns_script), encoding="utf-8")
    config = dataclasses.replace(quick_config(), pursuer_name="Pursuer", target_name="AlienSC")
    scenario, _ = parse_gmat_script(path, config)
    assert [slot.name for slot in scenario.burn_slots] == ["Burn1"]
    assert any("Burn2 is declared without explicit" in warning for warning in scenario.warnings)


def test_a_final_script_that_flies_the_old_style_burn_is_refused(
    unmaneuvered_burns_script: Path, tmp_path: Path
) -> None:
    """Review V3's forgery: Element lines added before the V/N/B ones and Burn2 maneuvered."""
    from orbitwin.script_renderer import assert_final_script_policy

    text = _old_style(unmaneuvered_burns_script)
    original = tmp_path / "original.script"
    original.write_text(text, encoding="utf-8")
    prop = "Propagate Prop(Pursuer, AlienSC) {{Pursuer.ElapsedSecs = {0}, AlienSC.ElapsedSecs = {0}}};\n"
    assert text.count(prop.format("33988.71424199411")) == 1
    forged = tmp_path / "forged.script"
    forged.write_text(
        text.replace(
            "Burn2.V = 0;\n",
            "Burn2.Element1 = 0.3;\nBurn2.Element2 = 0;\nBurn2.Element3 = 0;\nBurn2.V = 0;\n",
        ).replace("Burn1.Element1 = 0;\n", "Burn1.Element1 = 0.2;\n").replace(
            prop.format("33988.71424199411"),
            prop.format("1000") + "Maneuver Burn1(Pursuer);\n" + prop.format("500")
            + "Maneuver Burn2(Pursuer);\n" + prop.format("32488.71424199411"),
        ),
        encoding="utf-8",
    )
    config = dataclasses.replace(quick_config(), pursuer_name="Pursuer", target_name="AlienSC")
    with pytest.raises(ParseError, match=r"sets Burn2\.V\. GMAT reads V as Element1"):
        assert_final_script_policy(original, forged, config)
