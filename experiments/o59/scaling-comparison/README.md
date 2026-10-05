# O59 comparison with the O39 scaling fit

These files preserve the evidence used by `notebooks/039_scaling_viz.py`.
`evidence.json` records the run identities, source identity, compute convention,
and hashes of the three source metric snapshots. The snapshots were copied
without changes from `results/039_capacity_scaling/` on October 5, 2026.

O59 is excluded from the O39 fit. Its frame +4 teacher-forced loss proxy is
descriptive: the objectives, data, controller codec, and return conditioning
differ. Do not interpret the comparison as an isolated architecture effect.

The notebook still needs the original O39 input CSVs in
`results/039_capacity_scaling/`. `local-inputs.json` records their SHA-256 hashes.
The 64 MB outcome table is not included here. The original local inputs remain
in place. Generated figures remain outside Git.

Run from the repository root with the analysis dependencies installed:

```sh
MPLCONFIGDIR=/tmp/hal-matplotlib MPLBACKEND=Agg uv run notebooks/039_scaling_viz.py
```

This evidence records observed metrics. It does not certify successful training
completion or qualify an O60/O61 production run.
