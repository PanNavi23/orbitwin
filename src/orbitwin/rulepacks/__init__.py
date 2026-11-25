"""Rule packs that ship with OrbitWin, one package per contest.

Each ``*.toml`` directly in a pack's package is one rule set; ``orbitwin.rules``
reads and validates them. A pack outside OrbitWin registers its package under
the ``orbitwin.rulepacks`` entry point group.
"""
