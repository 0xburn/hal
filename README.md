# HAL

HAL trains a transformer to play *Super Smash Bros. Melee* from Slippi replays of human games, then runs it in Dolphin. The [Hillclimbing Melee](https://ericyuegu.com/hillclimbing-melee/) blog post explains the approach and the results.

| Release | Link |
| --- | --- |
| Model: HAL 250M, a 247M-parameter policy | [ericyuegu/hal-base-250m-rc](https://huggingface.co/ericyuegu/hal-base-250m-rc) |
| Dataset: 1.3 million games as MosaicML Streaming shards | [ericyuegu/hal-mds-dataset](https://huggingface.co/datasets/ericyuegu/hal-mds-dataset) |

## Setup

You need Linux, an NVIDIA GPU, [uv](https://docs.astral.sh/uv/), and a Melee 1.02 NTSC ISO.

```bash
git clone https://github.com/ericyuegu/hal && cd hal
uv sync
uv run fetch --name dolphin-exiai                  # headless Dolphin for CPU matches
uv run fetch --name Slippi_Online-x86_64.AppImage  # Slippi Dolphin with a window
export HAL_ISO_PATH=/path/to/ssbm.iso
```

## Download the model

```bash
uv run hf download ericyuegu/hal-base-250m-rc hal-250m.hal --local-dir .
```

`hal-250m.hal` holds the weights, the feature statistics, and the return calibration. To load it in Python:

```python
from hal.inference.action_sequence_artifact import load_action_sequence_policy

policy = load_action_sequence_policy("hal-250m.hal", device="cuda")
```

## Play against HAL

Plug your controller into port 1 of a GameCube controller adapter. On Linux, Dolphin needs the adapter's [udev rule](https://dolphin-emu.org/docs/guides/how-use-official-gc-controller-adapter-wii-u/).

```bash
uv run hal-policy play hal-250m.hal --character FOX --stage FINAL_DESTINATION
```

Dolphin opens with HAL on port 2. Pick your character, and HAL starts the match. Dolphin waits for each of HAL's actions, so HAL plays with no input delay. Replays go to `replays/local-play/`.

`--imitate` selects a play style from the training vocabulary: a player's connect code, or `PLATINUM`, `DIAMOND`, or `MASTER`.

## Evaluate against the CPU

```bash
uv run hal-policy eval hal-250m.hal --n-matches 4
```

HAL plays a level-9 CPU on Final Destination in headless Dolphin and writes a replay of each match.

## Download the dataset

The [dataset card](https://huggingface.co/datasets/ericyuegu/hal-mds-dataset) describes the sources, the filter, and the columns. The full dataset is large, so start with one split of one source:

```bash
uv run hf download ericyuegu/hal-mds-dataset --repo-type dataset \
  --include "ranked-anonymized-1/val/*" --local-dir data/hf
```

```python
from streaming import StreamingDataset

ds = StreamingDataset(local="data/hf/ranked-anonymized-1/val", shuffle=False)
game = ds[0]
print(game["replay_id"], game["num_frames"], game["p1_position_x"][:10])
```

## Train with the 059 recipe

[`experiments/059_muon_action_sequence.py`](experiments/059_muon_action_sequence.py) trained HAL 250M. It runs on a Modal B200 and depends on the maintainer's private Cloudflare R2 bucket:

- It streams the corpus from `s3://hal/processed/`. The Hugging Face dataset holds the same files, but the loader reads only from R2.
- It uploads checkpoints to `s3://hal/runs/`, and `--resume` downloads them.
- Every 8,192 updates it starts a Modal evaluator, which reads the checkpoint from R2.
- It logs to W&B.

With access to an R2 or S3 bucket that mirrors `s3://hal/processed/`, copy `.env.example` to `.env`, fill in the keys, and launch:

```bash
source .env
uv run modal setup
uv run modal secret create hal \
  AWS_ENDPOINT_URL="$AWS_ENDPOINT_URL" AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" AWS_BUCKET="$AWS_BUCKET" \
  WANDB_API_KEY="$WANDB_API_KEY"
uv run scripts/launch_modal.py -- uv run experiments/059_muon_action_sequence.py train
```

The launcher builds its image from your pushed commit and resumes the run after a preemption. Add `--proxy` after `train` for the 17M-parameter proxy. To turn a checkpoint into a `.hal` file:

```bash
uv run hal-policy export runs/<run>/final.pt policy.hal
```

The same bucket holds the ISO and test datasets that `uv run fetch` downloads when you omit `--name`. The integration tests and the netplay service in [deploy/netplay](deploy/netplay/README.md) need those files too.

## Layout

| Path | Contents |
| --- | --- |
| `hal/` | Library code for data, models, inference, emulation, evaluation, and the netplay service |
| `experiments/` | Training scripts, with 059 as the maintained recipe |
| `scripts/` | Cloud launchers and benchmarks |
| `docs/` | Inference and evaluation details, starting at [inference.md](docs/inference.md) |
| `archive/` | Earlier experiments and notes |

Before you send a change, run the checks in [AGENTS.md](AGENTS.md).

## Raw replays

The `ranked-anonymized` sources come from these Slippi ranked dumps, shared on the Slippi Discord server:

- [ranked-anonymized-1-116248](https://drive.google.com/file/d/1pFjgh1dapX34s0T-Q1TC7JUO-qjYbQZf/view)
- [ranked-anonymized-2-151807](https://drive.google.com/file/d/1jEIzvhpV3778J2s2-Np9vCVqSLf9lZnk/view)
- [ranked-anonymized-3-128787](https://drive.google.com/file/d/1glzlkAPxHC58oXZljJXQV8dsTBKmlhkE/view)
- [ranked-anonymized-4-148358](https://drive.google.com/file/d/1qdIZUW4Er_Vu6rD3-VUvyak3lKa1KxVk/view)
- [ranked-anonymized-5-133261](https://drive.google.com/file/d/1Hqmj6C8g1BzuRAIqOrQcMDL0MX4GtffE/view)
- [ranked-anonymized-6-171694](https://drive.google.com/file/d/1g8yZ-Q4ldyhDEmXLSPBoWxywJRMRVGc3/view)

To build a dataset from a replay archive:

```bash
uv run hal/scripts/build_index.py --archive replays.7z --output data/processed/mine/index.jsonl
uv run hal/scripts/filter.py --index data/processed/mine/index.jsonl --output data/processed/mine/paths.txt
uv run hal/scripts/materialize.py --replay-format policy-world --paths-file data/processed/mine/paths.txt \
  --index data/processed/mine/index.jsonl --output data/processed/mine/mds
```
