# September 2026 evaluation evidence

These files were moved from the untracked working tree on October 5, 2026.
`sha256.json` maps each original repository path to its file hash. The files
retain their original bytes. They were not rerun or updated for the current
HAL API during archival.

## Contents

- `outputs/o58-confirmation-eval`: launcher, test, and launch record. The record
  says `launched`; it does not establish completion.
- `outputs/o59-fixed120-eval`: completed fixed-120 versus p90 comparison,
  match rows, launch record, and the evaluation source and tests.
- `outputs/o59-cody-fox-eval`: completed Cody Fox comparison, match rows,
  figures, launch record, and the evaluation source and tests.
- `outputs/o59-rank1-*`: return-calibration scripts, sampled frame rows,
  analysis records, and figures.
- `outputs/o59-value-correlation*`: value-head analysis records and report.
- `outputs/latency-explainer.html`: explanatory artifact.
- `notebooks/hybrid_73.png`: earlier controller-vocabulary figure. The selected
  pruned81 treatment and its notebook remain in `notebooks/`.

## Reproduction

Use each launch or provenance record's Git SHA, checkpoint hash, environment,
and data identity. The evaluation source copies contain changes that are not
present as identical files in reachable Git history. Keep those copies with
their results. Some scripts use old module paths or temporary directories.
Relative paths assume their original locations; reconstruct that directory
layout in the historical checkout when reproducing a result.

This directory is historical evidence, not a maintained runtime or test suite.
Do not import these scripts into maintained code. Archival does not validate
the recorded findings or establish that remote jobs are still running.

Local bytecode and logs were not included. `outputs/o60-plain-bc/state.json`
remains local: it records operational state for the separate `o60-plain-bc`
branch, not the compute-optimal action-sequence experiment.
