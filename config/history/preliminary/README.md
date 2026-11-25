# Preliminary profiles (historical)

These configs ran the ODC 2026 preliminary of 2026-09-05 under the rules of
2026/05/20 (`orbitwin.rules.PRELIMINARY`). The upload they produced scored 95.25
after the organizer's validator first returned 73.02
(docs/LESSONS_2026-09-05.md).

They are kept to reproduce that run and the tests built on it. They are not a
default for anything ahead. The 1.5 km/s impulse limit, 4·T_A horizon and
penalty for an oversized burn are the preliminary's. The ranking round uses
`config/ranking.yaml` (`orbitwin.rules.RANKING`). The gold/bronze duel
(`orbitwin.rules.DUEL`) has no mission config.
The CLI says so on stderr whenever it runs under these rules.
