# Compute-optimal action-sequence experiment

This directory belongs to `060_compute_optimal_action_sequence_v2` in
`experiments/060_compute_optimal_action_sequence.py`.

The working copy continued the v1 program saved in commit `bb67dbdc` on
`codex/experiment-060`. V2 omits return conditioning, uses six trunk layers and
eight decoder layers, and counts the 32 sampled policy prefixes per window in
the scaling-law data budget. It contains fixed control and treatment proxy
arms. Select the depth allocation by gameplay evaluation. Offline validation
is diagnostic only. See `qualification.md` for the hardware checks; this
record does not establish that qualification passed.

The older branch `o60-plain-bc` uses the same experiment number for a separate
O52 plain-BC control (`060_plain_bc.py`, commit `c186022e`). That program and
its artifacts are not the control for this experiment. Use the full experiment
identity and Git commit when referring to either study. Do not combine their
run records or rename persisted identities to resolve the numbering conflict.
