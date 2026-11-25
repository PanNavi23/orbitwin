# Error codes and exit statuses

Every `orbitwin` command that refuses says why on one line of stderr:

```text
orbitwin: ERROR [<code>]: <what is wrong, and where>
```

The code names the kind of failure, so a script or an agent can branch on it
without reading the sentence. `orbitwin init`, `run` and `status` give the same
codes on stdout as `{"error": {"code": ..., "message": ...}}` under `--json`
(docs/WORKSPACE.md lists theirs). `orbitwin day` reports its own stops as
`orbitwin day: STOP:` and quotes the error line of the step that refused.

## Codes

| code | raised as | means |
|---|---|---|
| `config` | `ConfigurationError` | the config is missing, inconsistent, or gives a value without its unit |
| `rules` | `RulePackError` | a rule pack cannot be read, or `orbitwin rules` names a rule set that does not exist |
| `parse` | `ParseError` | the GMAT script cannot be read one way only; nothing was rewritten |
| `policy` | `PolicyError` | a script would change something the rules or the config keep fixed |
| `dynamics` | `DynamicsError`, `LambertError` | an orbital calculation failed |
| `gmat-run` | `GmatRunError` | a GmatConsole run did not give what was needed: it refused the script, crashed, timed out, or wrote no, a stale or an unreadable report. Many of these are answers (GMAT refuses that script every time); whether running again can help is exit status 75, not this code |
| `gmat-verdict` | `GmatVerdictError` and its kinds | GMAT flew the candidate and its flight refuses it: a verdict |
| `search` | `ParallelSearchError` | a search worker failed |
| `duel` | the duel engine's errors | a duel submission, state or parameter is refused |
| `environment` | `UnpinnedError` | numpy, SciPy or PyYAML is not the version requirements.txt pins |
| `io` | `OSError` | a file could not be read or written, or the disk is full |
| `value` | `ValueError` | a value on the command line or in a file is out of range, or a file's text cannot be decoded (not UTF-8, not JSON) |
| `error` | `OrbitWinError` | any other refusal |

## Exit statuses

| status | means |
|---|---|
| 0 | done |
| 2 | refused or stopped; the error line says why |
| 3 | `orbitwin day` and `run`: paused for a person (a readback, the coefficients) |
| 75 | `orbitwin certify`: no verdict on the candidate, because GmatConsole crashed, timed out or wrote no report, or this machine failed (a full disk); run it again |

`scripts/check_report_rows.py` has its own statuses for step 7 (docs/RUNBOOK.md
step 7).
