# Fitting a player identity (experiment 062)

`experiments/062_player_identity_finetune.py` makes HAL imitate one specific
player. It starts from the released `hal-250m.hal`, adds (or refits) that
player's 32-number identity row, and trains only that row on the player's own
games. The other 247M parameters are frozen, so the base policy and every
other identity behave exactly as before.

The output is a normal `.hal` bundle:

```bash
uv run hal-policy play player.hal --imitate 'ABCD#123' --character FOX
```

## What it trains

| | 059 base training | 062 identity fit |
|---|---|---|
| Starts from | random init | released `hal-250m.hal` |
| Data | ~1.3M games | one player's games |
| Trainable | all 247M parameters | one 32-wide `player_embedding` row |
| Objective | AWR-weighted offset NLL + value | same offset NLL, all weights 1 |
| Return conditioning | 60-frame return, 20% dropout | identical |

The objective is plain behavior cloning. A clone should reproduce the
player's habits, mistakes included, so actions are not reweighted by outcome.

The base vocabulary holds 21,177 connect codes from the professional
manifests. Ranked games are anonymized to Platinum/Diamond/Master, so most
netplay players are not in it. `scan` reports whether an identity is already
present; if it is, `train` refits that row starting from its current value.

## Naming the player in each game

Netplay games carry connect codes:

```bash
uv run python experiments/062_player_identity_finetune.py scan \
  --base hal-250m.hal --source.replays ~/Slippi --source.code 'ABCD#123'
```

Offline (console/tournament) games have no codes, and in-game tags are not
identities. Pass a sides file instead, one JSON object per game, with the
player's libmelee port (1-4). Relative paths resolve against the file's
directory:

```json
{"path": "genesis-7/Game_001.slp", "port": 3}
```

Use `--identity` to name the vocabulary key, for example a stable label from
whatever system matched the player to the port:

```bash
uv run python experiments/062_player_identity_finetune.py scan \
  --base hal-250m.hal --source.sides sides.jsonl --identity 'LS:283275'
```

Games are kept only if they are human 1v1s, name the player on exactly one
port, and have at least 284 frames (256 context + 28 chunk). `scan` lists
kept games by character and skipped games by reason.

## Training

```bash
uv run python experiments/062_player_identity_finetune.py train \
  --base hal-250m.hal --output player.hal \
  --source.replays ~/Slippi --source.code 'ABCD#123'
```

Defaults: 1,500 steps, batch 16, Adam at 1e-2, 10% of games held out by replay
hash. The row starts at zero, which is exactly the unconditioned base policy,
unless `--init-from` names a rank (`MASTER`) or another code.

Each validation pass reports the 059 objective and the next-frame joint NLL in
bits on held-out games for three settings: unconditioned, the starting row, and
the fitted row. The difference between unconditioned and fitted is how much
more predictable the player becomes once the model knows who they are.
Everything lands in `player.json` next to the bundle and in the checkpoint
provenance under `player_finetune`.

Gameplay quality is still judged in play; held-out NLL measures imitation,
not strength.

## Hardware

Training runs the full trunk forward and backward (only the row's gradient
is kept), so it needs a GPU for real runs. On CUDA it uses BF16 autocast.
The CPU path works and is how the tests and smoke run were done: two steps at
batch 2 take under a minute on an Apple Silicon laptop. The CUDA path uses the
same eager model calls as 059's uncompiled path but has not been run yet.

## Export guarantees

The bundle differs from the base only in:

- `player_embedding.weight` (one row inserted or updated; every existing code
  keeps its exact vector, at its possibly shifted ID)
- `player_code_bytes` (the sorted vocabulary)
- `cfg.player_vocab_size`, `cfg.player_vocab_sha256`, and
  `provenance.player_vocab_sha256`, plus `provenance.player_finetune`
- `backend.json`'s checkpoint hash and the bundle name

`stats.json`, the return calibration, and every other tensor are byte-identical
to the base. `read_action_sequence_artifact` validates the result unchanged.
Refitting from a bundle that is already a player fine-tune is refused; start
from the released policy.

## Tests

```bash
uv run pytest -q tests/experiments/test_062_player_identity_finetune.py
```
