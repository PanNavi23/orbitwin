# GMAT GUI parse oracle for the console copy

`gui_oracle_cases.json` holds what GMAT R2026a's GUI said of each edit to our
two-burn template (`examples/two_burn_rendezvous.script`) or, for the `G..`
cases, to R1's OpenFrames template
(`tests/fixtures/report_bind/R1_openframes.script`), measured with
`gui_oracle.sh` under xvfb on 2026-11-19. The GUI loads the OpenFrames plugin
GmatConsole skips, so it is the only reader of the lines the console copy
leaves out. It is a test oracle only: nothing at runtime runs it.

Each case's script is its base with `base_edits` and then the case's `edits`
applied in order, each replacing the first occurrence of its old text. The
top-level `base_edits` apply to every case without its own. They set
`DefaultOrbitView.ShowPlot = false`, whose window segfaults under xvfb, and
shorten the flight so a run takes seconds: the two-burn template's three
Propagates to 20 s each. Each `G..` case's own `base_edits` set the same ShowPlot
in R1's OpenFrames template and shorten its one Propagate to 60 s. So the five edits to
a ShowPlot are made on `DefaultGroundTrackPlot.ShowPlot`. Every case runs to the
end of its mission.
Outcomes, from the run's GmatLog.txt:

- "refused at build": "Failed to build the script";
- "refused when it ran": an ERROR after the build, such as the Sandbox's;
- "ran to the end": "Mission run completed" and no error. Only this passes.

The checker does not judge whether the GUI takes a plot line. The plot lines
of an upload are the official template's, copied verbatim and proven identical
(`console_copy.assert_plot_lines_unchanged`), so the GUI refuses one exactly when
it refuses the organizer's template. `tests/unit/test_console_copy.py` therefore
requires the copy to be proven of every edit in `cases`, and each one the GUI
refused is marked `checker_differs`. The eight `G..` cases are the review of
#189's grammar findings, on R1's OpenFrames template.

`command_cases` hold the GUI's verdicts on Toggle, PenUp, PenDown, MarkPoint and
ClearPlot lines; `tests/unit/test_plot_commands.py` requires the parser to read
one exactly when the GUI ran it to the end, except the three marked
`checker_differs` (a Toggle of a ReportFile, and a command without its `;`),
which it refuses on purpose.

The oracle was first measured on 2026-11-06 on other bases; on 2026-11-19 every
case was moved to the bases above, with the same intent, and run again. Each
base has its `base_sha256` beside it, and the tests fail when the base's bytes
are no longer the ones the GUI was run on. Then measure the oracle again and
record the base's new sha256 there. To run one case (one GMAT run per hold of
the shared lock), write its script as above and:

    GMAT_ROOT=<R2026a> ORACLE_DIR=<scratch> COMPAT_LIBS=<dir with libtiff.so.5> \
      flock <gmat.lock> bash gui_oracle.sh <case.script> <run dir>

It prints the outcome and the GUI's exit status; the run dir keeps GmatLog.txt.
