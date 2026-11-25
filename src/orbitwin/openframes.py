"""The OpenFrames types, which GmatConsole cannot create.

GmatConsole skips GUI plugins ("GUI plugins are skipped in console mode"), so
every OpenFrames object is an unknown type to it (R2026a: the workshop
template, 58 errors, exit 1). The console copy leaves their lines out; whether
the GMAT GUI takes those lines is not judged here. They are the official
template's own, byte for byte (console_copy.assert_plot_lines_unchanged), so
the GUI runs the submitted script exactly when it runs the organizer's.
"""

from __future__ import annotations

OPENFRAMES_TYPES = frozenset(
    {"OpenFramesInterface", "OpenFramesView", "OpenFramesVector", "OpenFramesSensorMask"}
)
