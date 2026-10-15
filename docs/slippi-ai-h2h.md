# slippi-ai versus 059

The local runner loads `all_d21_imitation_v3` with upstream `slippi-ai` main at
`977b7a67d52d77f6183641ce7d86b462f29e18f2`. It uses HAL's Dolphin `Session` and
`PolicyBatchAdapter` for the 059 cooldown checkpoint. Existing HAL code and
dependencies are unchanged.

The downloaded model is
`outputs/slippi-ai-h2h/all_d21_imitation_v3` (42,338,678 bytes), SHA-256
`4cb632296a5387f4a1a05f0a18cba5528ad50f3e8a343b7ed3d7f192b0ece4b7`.
It came from the user-supplied Dropbox folder. The complete folder archive is
`outputs/slippi-ai-h2h/dropbox.zip`, SHA-256
`605eb621155ce2533fae0bbf56216bda86b8738e2204f0393c5fba120a02b612`.

The 059 bundle is `runs/netplay/o59-vywk3cih.hal`. Its source checkpoint hash is
`52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b`.

The same Dropbox archive contains `gm-v1` (42,340,455 bytes), SHA-256
`194d95f774c8e902fcb7ba4249d1a54207d884cdee0b1a27e25a1fd803120870`.
The worker accepts only these two verified upstream artifacts. Both use a
21-frame delay and support all characters. `gm-v1` records `Master Player` as
its RL identity; the runner uses that identity for both models. Each manifest
records the selected artifact name, hash, configuration, and loaded identity.

## Protocol

- Treatment: 059 cooldown, anonymous player identity, checkpoint p90 return,
  temperature 1, and KV history. Compiled inference uses the RTX 3060.
- Control: `all_d21_imitation_v3`, `Master Player` identity, temperature 1,
  and upstream's TensorFlow delayed agent on two CPU threads.
- Fixed inputs: Fox dittos, Final Destination, four stocks, and the same policy
  seed within each pair. Each pair swaps controller ports. Melee RNG is not
  seeded, so port swaps are not exact mirrored emulator trajectories.
- Native timing: 059 has a two-frame transport delay and replans every two
  frames. Upstream has its trained 21-frame delay and samples every frame.
  This compares deployed policies; it does not isolate architecture at equal delay.
- Each policy receives every frame through its own observation parser.
  The emulator blocks for inference; wall-clock inference time does not add
  game-frame delay.

The default limit is 28,800 steps per game. A frame-limited game has no assigned
winner. The runner saves each completed game's result before starting the next.
Two games establish that the matchup works; they do not estimate a reliable win rate.

## Recorded games

All four full games used policy seed `20261007`. 059 won all four.

| Output directory | 059 port | Final stocks, 059 / slippi-ai | Replay status |
| --- | --- | --- | --- |
| `matchup/game-00` | 1 | 2 / 0 | Missing the final eight frames |
| `matchup/game-01` | 2 | 4 / 0 | Missing terminal frames |
| `complete-matchup/game-00` | 1 | 3 / 0 | Terminal stock loss verified in replay |
| `complete-matchup/game-01` | 2 | 2 / 0 | Terminal stock loss verified in replay |

These paths are under `outputs/slippi-ai-h2h/`. The first pair exposed a shutdown
bug in the new runner: it stopped Dolphin before its replay writer saved the last
frames. Its live terminal results remain recorded, but use the second pair for
complete replay review. The runner now advances through the Game End event before
closing Dolphin. The second pair measured 89.8 and 90.3 complete-path FPS.

Generate viewer links with:

```sh
uv run slp-link outputs/slippi-ai-h2h/complete-matchup
```

## Run

From the HAL repository:

```sh
TMPDIR="$PWD/outputs/slippi-ai-h2h" uv run python -m scripts.slippi_ai_h2h \
  --output outputs/slippi-ai-h2h/new-matchup --games 2
```

The output directory must be new. Use `--max-frames 360` for a short smoke test.
Use `--no-compiled` for eager 059 inference. The manifest records resolved options,
both Git revisions, package versions, policy settings, source hashes, and artifact
hashes. The exact runner sources are copied into each run. Each game saves a
Slippi replay and a numeric trace. Trace player order is
059 first and slippi-ai second, regardless of controller port.

For the standard 96-game evaluation:

```sh
uv run python -m scripts.slippi_ai_h2h \
  --schedule standard --games 96 --output outputs/slippi-ai-h2h/eval96
```

For the second evaluation, add
`--slippi-checkpoint outputs/slippi-ai-h2h/gm-v1` and use a separate output
directory. Each model plays 96 games against the same 059 checkpoint; the two
evaluations use the same schedule and pair seeds.

This uses `hal.eval.matchups.mirrored_configs(48)`: 48 paired configurations,
16 games on each of the six stages, and 11 characters. Characters stay on their
assigned ports while the models swap ports. Each pair gets its own policy seed.
Both models retain the identities, temperature, return setting, and native
timing described above. `--game-offset` must be even and permits independent
shards without changing game IDs, pair seeds, characters, or stages. Eight
12-game shards exactly cover the 96-game schedule. `--schedule-only` writes the
resolved schedule without loading either model.

Cloud preparation is recorded under `outputs/slippi-ai-h2h/eval96-preparation/`.
`launch.json` records the VM identity, inputs, hashes, and fixed settings;
`standard/schedule.json` contains all 96 assignments. The dedicated VM is
`hal-slippi-h2h96-20261007`, instance ID `7667133916266526410`, in project
`centering-star-502613-k3`, zone `us-west4-c`. The initial `us-west1-a` VM
remains stopped after repeated GPU stockouts. Its `g4-standard-48` machine has
one RTX PRO 6000 Blackwell. A three-hour shutdown timer limits an unattended run.
The retained netplay host is unchanged.

The TensorFlow environment is separate from HAL:

```sh
uv venv --python 3.12 outputs/slippi-ai-h2h/venv
uv pip install --python outputs/slippi-ai-h2h/venv/bin/python \
  -r outputs/slippi-ai-h2h/requirements.txt
```

The resolved environment is recorded in `outputs/slippi-ai-h2h/requirements.txt`.
The runner also sets the worker's Python path to the verified upstream checkout.
To check model loading without Dolphin:

```sh
outputs/slippi-ai-h2h/venv/bin/python scripts/slippi_ai_worker.py \
  --checkpoint outputs/slippi-ai-h2h/all_d21_imitation_v3
```

## Compatibility and checks

HAL uses libmelee `0.47.0+hal.realtime.1`; upstream uses `0.47.3`. Both versions
are checked at startup. HAL adds dataclass types that upstream cannot unpickle.
The bridge copies dataclasses to attribute records, preserving values and enums,
and omits private fields that upstream's parser does not use. It does not convert
the policy's feature representation. Remove this bridge when both environments
can use the same libmelee objects.

`ObservedSession` overrides HAL's frame-read hook to retain the exact raw state
that produced each canonical observation. A regression test checks alignment
through a live Dolphin session. The manifest records the Session source hash.
Replace the override if HAL gains a public observation callback.

The tests check controller wire conversion, invalid actions, raw observation
serialization, live frame alignment, and waiting for the Game End event. A
separate replay check compared every upstream Parser field from HAL's reader
against upstream libmelee's reader: all 12,700 frames from the two complete games
matched exactly, including the terminal stock losses. The check is recorded in
`outputs/slippi-ai-h2h/complete-observation-parity.txt`.

The downloaded checkpoint's MD5 is `56a6e6757e5d7d2540b80187c79ea8ab`, which
matches the checkpoint identity embedded in upstream's saved-output reference.
The reference's numerical arrays have not yet been compared: automatic approval
review blocked unpickling that upstream file because pickle can execute code.
Its hash field was inspected without executing pickle instructions.

Short games measured 25.9–28.3 complete-path FPS with eager 059 inference and
36.8–43.4 FPS with compiled inference. These include startup and ran while other
validation was active; they are not controlled performance estimates.

## Validation results

Final checks on 2026-10-07:

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Passed; 324 files |
| `uv run ruff check .` | Passed |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Passed; zero diagnostics |
| `uv run pytest -q -rs -m "not integration" --basetemp outputs/slippi-ai-h2h/pytest-final` | 1,793 passed; 2 skipped; 25 deselected |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -rs tests/test_roundtrip.py tests/test_session_cleanup.py tests/test_slippi_ai_h2h.py -m integration --basetemp outputs/slippi-ai-h2h/integration-final` | 9 passed; 16 deselected; no skips |

Both skipped tests are opt-in production GPU qualification checks in
`tests/test_netplay_hardware.py`: `test_declared_profiles_meet_complete_path_budget`
and `test_prepared_profiles_do_not_compile_or_capture_during_play`. They require
`HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1` and a configured production policy.
No required integration fixture was missing.

The first full unit-suite attempt in the filesystem sandbox failed with
`ValueError: I/O operation on closed file`. The successful runs used project-local
temporary storage (`TMPDIR=$PWD/outputs/slippi-ai-h2h`) outside that sandbox.
The initial focused test collection failed to import `scripts`; it was corrected
to use the repository's existing script-loading test pattern. Subsequent focused
and full checks passed.

Logs are `outputs/slippi-ai-h2h/unit-tests-final.log` and
`outputs/slippi-ai-h2h/integration-tests-final.log`. They include the existing
multi-threaded `fork()` deprecation warnings. Replay URLs were checked with HTTP
200 responses. Archived runner source hashes match both run manifests.

After adding the standard schedule and shard offsets, the same required checks
passed again: 1,801 unit tests, the same two opt-in hardware skips, and 9
integration tests with no skips. The cloud-preparation directory contains the
new test logs. A short Captain Falcon/Falco pair on Battlefield also completed
with both port assignments.

After adding `gm-v1`, validation passed again:

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Passed; 324 files |
| `uv run ruff check .` | Passed |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Passed |
| `uv run pytest -q -rs -m "not integration" --basetemp outputs/slippi-ai-h2h/eval96-preparation/gm-unit-tmp` | 1,804 passed; the same two opt-in hardware skips; 25 deselected |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -rs tests/test_roundtrip.py tests/test_session_cleanup.py tests/test_slippi_ai_h2h.py -m integration --basetemp outputs/slippi-ai-h2h/eval96-preparation/gm-integration-tmp` | 9 passed; 27 deselected; no skips |

The unit and integration runs set `TMPDIR` to `outputs/slippi-ai-h2h`. Logs are
`eval96-preparation/gm-unit-tests.log` and
`eval96-preparation/gm-integration-tests.log`. Two 360-frame Fox/Sheik games on
Yoshi's Story passed for `gm-v1`, with swapped ports, at 44.4 and 44.0 complete-path
FPS. These smoke games do not count toward the 96-game evaluation.

## Full cloud evaluation

The 2026-10-08 evaluations completed all 192 games on the RTX PRO 6000:

| Opponent | 059 wins | Opponent wins | 059 win rate |
| --- | ---: | ---: | ---: |
| `all_d21_imitation_v3` | 91 | 5 | 94.8% |
| `gm-v1` | 24 | 72 | 25.0% |

Every evaluation game ended by stock loss. There were no failed, drawn, or
frame-limited evaluation games. Four short cloud smoke games are excluded.
Both evaluations used the same standard schedule, pair seeds, identities,
return setting, temperatures, and native delays documented above.

All 192 downloaded replays passed artifact-hash, schedule, frame-continuity,
stock-trace, terminal-stock, and Game End checks. The cloud and downloaded
archive SHA-256 hashes matched. All 16 workers and the container exited zero.
The runs recorded 694,113 and 697,656 policy-observation frames, respectively,
at weighted per-worker game throughput of 86.3 and 87.1 FPS.

The final report is `outputs/slippi-ai-h2h/cloud-eval96/report.md`. Each model's
`results/<model>/replay-links.md` contains all 96 viewer links;
`validated-results.json` records the individual results and links. The complete
archive, runtime logs, validation log, and manifests are in that directory.
`eval96-preparation/launch.json` records the actual successful VM identity,
source revisions, configuration, seeds, and input hashes.
