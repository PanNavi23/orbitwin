"""What GMAT R2026a does under each propagator setting the organizer might use.

Measured by scripts/duel_gmat_fit.py on GmatConsole (2026-11-02, run
output/duel/gmat-fit/20261102T020420Z): point-mass Earth, one craft from
periapsis 7000 km, e in {0, 0.3, 0.5, 0.7, 0.78, 0.9, 0.95, 0.99, 1.2, 2.0},
true anomaly 0, 120 and 240 deg, each flown for MEASURED_SPAN_S with no cap
on the step (MaxStep 1e6 s), so c is the integrator's own. Every value
is 1.5 times the largest measured, rounded up (DUEL_ENGINE.md 2.5, 2.6). The
match laptop needs no GMAT; `Choices.row_gap_setting` picks a row.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

#: How long every orbit of the fit was flown.
MEASURED_SPAN_S = 2.0 * 86400.0


@dataclass(frozen=True, slots=True)
class PropagatorSetting:
    #: A row gap is at most row_gap_c * r_max^1.5 / sqrt(mu).
    row_gap_c: float
    #: One craft's position error against our coast after t seconds is at
    #: most position_floor_km + position_rate_km_s * t over the measured span.
    position_floor_km: float
    position_rate_km_s: float
    #: GMAT's error in a over the measured span, its SMA against eq. (2) included.
    a_km: float

    def position_error_km(self, t_s: float) -> float:
        """The line over the measured span; past it, grown as t^2.

        An error in a makes the along-track error grow quadratically, so the
        line is not extrapolated as a line.
        """
        t = max(t_s, 0.0)
        growth = self.position_rate_km_s * t * max(1.0, t / MEASURED_SPAN_S)
        return self.position_floor_km + growth

    def a_error_km(self, duration_s: float) -> float:
        """The a error over a round, scaled up linearly past the measured span."""
        return self.a_km * max(1.0, duration_s / MEASURED_SPAN_S)


SETTINGS: dict[str, PropagatorSetting] = {
    # RK89 at Accuracy 1e-11, GMAT's default.
    "rk89-default": PropagatorSetting(0.1911, 2.3530e-06, 3.8150e-10, 7.4060e-05),
    "rk89-1e-10": PropagatorSetting(0.2366, 2.7500e-06, 5.1460e-09, 8.7930e-04),
    "rk89-1e-9": PropagatorSetting(0.3154, 2.8660e-06, 6.8850e-08, 7.4090e-03),
    "pd78": PropagatorSetting(0.2095, 2.6470e-06, 1.3890e-10, 8.3120e-06),
    "rk56": PropagatorSetting(0.03738, 2.9870e-06, 1.0120e-09, 4.7630e-04),
    "pd45": PropagatorSetting(0.03698, 3.0180e-06, 2.2900e-09, 5.1790e-04),
}
#: The default: the worst of every column, so no single setting is assumed.
SETTINGS["cautious"] = PropagatorSetting(*(
    max(getattr(row, f.name) for row in SETTINGS.values())
    for f in fields(PropagatorSetting)
))


def setting(name: str) -> PropagatorSetting:
    try:
        return SETTINGS[name]
    except KeyError as error:
        raise KeyError(f"unknown propagator setting {name!r}; known: {sorted(SETTINGS)}") \
            from error

