# Rule packs

A contest's rulebooks live in rule packs, not in code. Each rulebook is one
TOML file; a pack is a Python package holding those files. OrbitWin ships one
pack, `orbitwin.rulepacks.tasa_odc_2026`, with TASA's 2026 preliminary, the
ranking round (`ranking`, the 2026/11/04 revision, and `ranking-onsite`, its
2026/11/08 venue version) and the duel.

```bash
orbitwin rules                          # every rule set, built-in and installed
orbitwin rules tasa-odc-2026/ranking    # one, in full, as JSON with the file's keys
orbitwin rules --json                   # every one, as JSON
```

A config names its rulebook in `policy.rules`, by id (`tasa-odc-2026/ranking`)
or by name (`ranking`). The `--rules` flag of the day-path tools takes either.

## A rule set file

```toml
format = 1                     # the file format this OrbitWin reads
kind = "interception"          # or "duel"
id = "tasa-odc-2026/ranking"   # "<pack>/<name>", unique across packs
name = "ranking"               # what a config may call it; see below
...
[source]                       # the organizer's document the numbers come from
document = "..."
sha256 = "..."
header_date = "..."
received = "..."
```

The keys are the fields of `InterceptionRules` or `DuelRules` in
`src/orbitwin/rules.py`, where each is documented. The reader refuses an
unknown key, a missing one, a number where a string belongs (or the reverse),
a non-finite number and a value outside a field's allowed set; for an
interception it also refuses an impulse limit, sphere or total cap that is not
positive; a negative ε, spacing, penalty or advance; `miss_ranked` without
`miss_scored`; and a total cap, an Earth crossing rule or a `pursuer_epoch`
other than `fixed` without its clause, or a clause without it. An id needs a
pack name before its `/`. A field that may be absent (`total_dv_limit_km_s`,
`score_units`, ...) is left out, not written empty.

`orbitwin rules <id>` prints a rule set as JSON with the file's keys, so a new
rulebook can start from the closest existing file in
`src/orbitwin/rulepacks/tasa_odc_2026/` and be checked against it.

`pursuer_epoch` is `fixed`, or `at-or-after-target` where the rules let a team
set the pursuer's epoch (docs/DESIGN_VARIABLES.md).

## A pack of your own

Put the TOML files in a package and register it under the `orbitwin.rulepacks`
entry point group:

```toml
# pyproject.toml of your package
[project.entry-points."orbitwin.rulepacks"]
acme = "acme_contest_rules"
```

Every `*.toml` directly in `acme_contest_rules` is then a rule set; the entry
point must name a package, not a module. Installed packs are read the first
time a config is checked or a rule set is looked up; a pack that cannot be
read stops that command with the file and the reason, rather than being
skipped. A pack may not use a pack name OrbitWin ships (`tasa-odc-2026`).

Ids are unique: two rule sets with one id are refused. Names need not be. A
shipped name such as `ranking` always means the shipped rule set, and an
installed rule set whose name another set holds answers to its id only. A name
that only installed rule sets share names none of them: a config or `--rules`
giving it is refused with the ids that claim it.

What a rule set cannot express needs code: a new distance term, a time origin
that is neither the earlier epoch nor left unruled, a horizon that is neither
`4*T_A` nor announced.
Add the variant to the field's allowed values in `rules.py`, then teach the
code that reads the field.

The day-path tools are still written for TASA's 2026 ranking round.
`scripts/ranking_day_config.py` refuses a rule set without its shape: an
announced T_max, ranked misses and time counted from the earlier epoch.
`orbitwin day` runs only `ranking` itself: it runs `orbitwin inspect` with no
`--rules`, which refuses any other rule set (`config.rulebook_refusal`), and
`orbitwin init` and `orbitwin run` take only `tasa-odc-2026/ranking`
(`workspace.DAY_RULES`). `scripts/competition_day.sh` runs only the ranking
round, or the preliminary with `--preliminary`. `orbitwin inspect`, `solve`, `rank`, `certify` and
`check_report_rows.py` run under whichever rule set the config names; for any
but the ranking round's, `--rules` must name it too, so a config under the
wrong rules is never run by accident.
