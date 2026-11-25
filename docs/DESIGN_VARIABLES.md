# Design variables

A rule set may let a team set a value in the template itself. The only one so
far is the pursuer's epoch.

## The pursuer's epoch

The onsite ranking rules of 2026/11/08 (`ranking-onsite`, §4 入軌時刻) let a
team choose SC_Human's epoch, not before SC_Alien's. For an insertion given in
EarthFixed, the epoch also turns the orbit with the Earth. On the 2026 ranking
problem with the slide's numbers, the template's own epoch left no hit within
3 km/s in a two-body sweep; an epoch 77,700 s after SC_Alien's gave a hit that
certified at 87.95.

```bash
orbitwin set-epoch official.script --config config/ranking-onsite.yaml \
  --rules ranking-onsite --after-target-s 77700
```

`--epoch '10 Nov 2026 02:35:00.000'` gives the same epoch as text. The command
writes two files beside the official script:

- `official.epoch-20261110T023500000Z.script`: the official script with the
  text between the quotes of the pursuer's `Epoch` line replaced, every other
  byte kept;
- `official.epoch-20261110T023500000Z.script.design.json`: what was changed,
  under which rule set, and the SHA-256 of both scripts.

They go beside the original because GMAT reads a data file named by a relative
path from the script's folder.

It refuses when the rule set fixes the epoch (every rule set but
`ranking-onsite`), when the epoch is before the target's, when it is at or past
T_max and the config gives T_max, when the offset is not a whole number of
milliseconds, and when either craft does not set `Epoch` once in UTCGregorian,
after its `DateFormat` line.

`inspect`, `solve`, `baseline`, `rank`, `certify` and `verify-policy` take the
copy as the official script. When it has a record beside it they check it first:
the original unchanged since set-epoch, the copy as recorded, the two differing
only in the pursuer's `Epoch` line, the record's pursuer and target the
config's, the pursuer not before the target, and the config's rules allowing
it. A failure is `orbitwin: ERROR [policy]:`. `certify` adds the record to `certification.json`
as `design`, or to `diagnostic/verification.json` when it stops for the GMAT GUI.
Nothing checks the original against the organizer's file: compare
`original_sha256` by hand. A copy moved away from its record is checked by
nothing; keep the two together.

Under `ranking-onsite` the parser also refuses any template, with a record or
without, whose pursuer starts before the target.

With the pursuer starting later, `policy.closest_approach_window: from_origin`
counts the alignment's rows, where SC_Alien moves and SC_Human waits at its
insertion state. Whether those rows count is still the organizer's to say
(docs/ASYNC_START_AND_INSERTION.md, Q1 and Q5).

Not yet done: nothing sweeps the epoch, and `orbitwin day` refuses a script
with a record beside it. The step-7 tools (`check_report_rows`,
`audit_burn_table`) do not read the record.
