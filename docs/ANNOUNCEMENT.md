# Entering the announcement

On the ranking day of 2026-11-09 two announced values were typed wrong: kt as
0.00095073 for 5.9073e-4, and T_max as 96200 s for 10 Nov 2026 07:43:24 UTC,
which is 96204 s. Two people read the printed readback and passed both. The
day's script also gave SC_Alien an ECC, RAAN and TA other than the problem
slide's, and nothing compared the two.

So two people each type the announcement into a file of their own, from the
announcement itself, as it is printed. `scripts/ranking_day_config.py` takes the
two files in place of `--horizon-s`, `--mission-boundary-km` and the score
flags:

```bash
python scripts/ranking_day_config.py --output run/day.yaml --team-number 54 \
  --announcement run/announced-alice.toml --announcement run/announced-bob.toml
```

## The file

```toml
format = "orbitwin-announcement/1"
entered_by = "Alice"

[craft.SC_Alien]
Epoch = "09 Nov 2026 02:00:00.000 UTC"
CoordinateSystem = "EarthMJ2000Eq"
DisplayStateType = "Keplerian"
SMA = "−99,650.11045 km"
ECC = "1.06852111"
INC = "35.336 deg"
RAAN = "57.046 deg"
AOP = "202.341 deg"
TA = "241.212 deg"

[craft.SC_Human]
Epoch = "09 Nov 2026 02:00:00.000 UTC"

[mission]
T_max = "09 Nov 2026 03:51:51.667 UTC"
T_max_s = "6711.667 s"        # if the announcement prints it too
# boundary = "50000 km"       # the mission boundary, if announced

[score]                       # all five, or none until announced
C_r = "300 km"
kt = "3e-3 1/s"
Ct = "5369.334 s"
kv = "0.0012 1/(m/s)"
Cv = "2000 m/s"
```

Every value is text in quotes, with its unit, so the file keeps the digits the
announcement printed. A slide's minus sign (U+2212), a degree sign and
thousands grouped with commas are read as written. Times are
`DD Mon YYYY HH:MM:SS[.mmm] UTC`.

A craft's fields take GMAT's names: SMA, ECC, INC, RAAN, AOP and TA;
RMAG, RA, DEC, VMAG, AZI and FPA; X, Y, Z, VX, VY and VZ. Lengths are in km,
speeds in km/s and angles in deg; ECC has no unit. `CoordinateSystem` and
`DisplayStateType` are names. The score's units are the ones the 2026/11/04
revision fixes, and C_r, the boundary and `T_max_s` must be positive. A value
in another unit, an unknown field, a number outside quotes and an `entered_by`
that is not one line of printable text are refused.

`[mission]` may give `T_max_s` without `T_max` when only the seconds are
announced. An entry may give `[score]` alone, for a score announced after
T_max (step 5).

## What the tool checks

- The two files are entered by two people (`entered_by` differs) and agree on
  every value and unit. They are compared as numbers, so `3e-3` and `0.003`
  agree. A difference is refused with the field and both texts.
- Each craft value is held to the field the script assigns, within half a unit
  of the last digit the announcement printed: `1.06852111` holds the script's
  `1.0685211112076598`. No value is held closer than a part in 10^12 of itself,
  since a script's double can differ from a printing in its last digit
  (`7.175430940911812` against `7.175430940911815` on 2026-11-09). Angles are
  compared as directions, so -110 deg holds 250. This runs first, before T_max
  is checked against SC_Alien's coast. A value the script contradicts is
  refused, and nothing is written.
- A `WARNING:` asks a person to compare by hand what the tool does not read the
  way GMAT does. That is a field the script does not set, or writes as an
  expression or past a double's range; an epoch not in UTCGregorian, set before
  its `DateFormat`, or overridden by an `Epoch.<scale>` field; a
  `CoordinateSystem` or `DisplayStateType` other than the announced one, or left
  out of an entry that gives the craft's state; and every state field of a craft whose script also sets a field
  GMAT reads into the state (`Element1` to `Element6`, `CartesianX` and the
  like, the `*State` fields, an `AnomalyType` other than TA), sets its state
  before its `CoordinateSystem`, or is in another `CoordinateSystem` than the
  announced one. A field the script leaves unset is read as GMAT's default:
  EarthMJ2000Eq, Cartesian, TAIModJulian, TA.
- Under `ranking-onsite` an announced SC_Human epoch is the earliest one the
  script may set.
- T_max in seconds is T_max less the earlier epoch, or SC_Alien's when the rules
  let a team set SC_Human's (`ranking-onsite`), taken from the script's own
  epoch when the script writes it in UTCGregorian. Leap seconds are not counted.
  `T_max_s`, when given, must be that number within the rounding of the three
  printed values (against times printed to the second, `96,204 s` takes
  96203 s to 96205 s), and is the one taken when it is printed finer than T_max.
- The boundary, when given, goes into `policy.mission_boundary_km`, and SC_Alien's
  coast is held to it as before.
- The score is written only by step 5's config, the one given `--day-config`.
  When step 2 took T_max from two entries, step 5 must take the score from two
  as well; the score flags are refused there. Step 5's entries may give the
  score alone. A T_max or boundary they give must be step 2's, and craft values
  they give are held to the script again. The written config's first lines name
  both files and their SHA-256.

The comparison assumes the announcement rounds; one that truncates can be
refused for a value it printed correctly, and then the organizer settles it.

When the two entries agree, they agree only with each other. They can still both
be wrong; the readback is still read against the announcement.

`orbitwin day` takes the two entries too (docs/RUNBOOK.md, "One command"). It
holds them to the official script before it makes a day folder, and holds a
later score's entries to it before it records them.
Not yet done: the workspace (`announce.toml`) still takes T_max and the
coefficients as typed numbers.
