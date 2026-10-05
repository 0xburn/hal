import hashlib
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from hal.data.extract import extract_replay
from hal.data.feature_stats import FeatureStats
from hal.training.dataloader import collate_train_batch, make_window
from hal.training.features import BASE_ITEMS_PROJECTION, ITEM_PLAYER_COLUMNS, ITEM_PLAYER_PROJECTION
from hal.training.player_identity import PlayerVocabulary, decode_player_codes


ROOT = Path('/tmp/hal-rank1-o59-calibration')
CHECKPOINT = Path('runs/netplay/tmp/hal-policy-f7z3pbp7/checkpoint.pt')
STATS = Path('runs/netplay/tmp/hal-policy-f7z3pbp7/stats.json')


def selected_frames(values: np.ndarray, valid: np.ndarray, frame_count: int, seed: int, context: int, chunk: int):
    rng = np.random.default_rng(seed)
    eligible = np.arange(context - 1, frame_count - chunk)
    eligible = eligible[valid[eligible]]
    if len(eligible) < 16:
        raise ValueError('too few eligible frames')
    bins = np.array_split(eligible, 16)
    base = set(int(rng.choice(group)) for group in bins)
    high = np.zeros(frame_count, dtype=bool)
    high[eligible] = values[eligible] > 100
    starts = np.flatnonzero(high & ~np.r_[False, high[:-1]])
    stops = np.flatnonzero(high & ~np.r_[high[1:], False]) + 1
    clusters = [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]
    tail = set()
    for start, stop in clusters:
        tail.update(int(index) for index in np.linspace(start, stop - 1, min(5, stop - start)).round())
    if len(tail) > 20:
        tail = set(int(index) for index in rng.choice(sorted(tail), 20, replace=False))
    return base, tail, clusters


def score_game(record, model, cfg, stats, exp):
    replay = ROOT / 'replays' / f"{record['id']:03d}.slp"
    sample = extract_replay(str(replay))
    if sample is None:
        raise ValueError(f'failed to parse {replay}')
    frames = sample['frame']
    if int(frames[-1]) != record['frame_count_index']:
        raise ValueError(f"last frame differs from index for {record['id']}: {int(frames[-1])} != {record['frame_count_index']}")
    if not np.all(np.diff(frames.astype(np.int64)) == 1):
        raise ValueError(f"nonconsecutive frames for {record['id']}")
    labels = exp.future_return_labels(sample)
    port = record['perspective']
    target = labels[f'{port}_return60']
    valid = labels[f'{port}_return60_valid']
    base, tail, clusters = selected_frames(
        target, valid, len(frames), 20260925 ^ int(record['id']), cfg.arch.L_ctx, cfg.arch.sample_chunk_length
    )
    indices = sorted(base | tail)
    rows = []
    for first in range(0, len(indices), 16):
        selected = indices[first:first + 16]
        windows = []
        for index in selected:
            window = make_window(
                sample, ego_prefix=port, start=index + 1 - cfg.arch.L_ctx,
                pad=0, length=cfg.arch.L_ctx, projection=BASE_ITEMS_PROJECTION
            )
            window['ego_player_id'] = np.zeros(cfg.arch.L_ctx, dtype=np.int64)
            window['ctx_pad'] = np.int64(0)
            windows.append(window)
        batch = collate_train_batch(
            windows, stats=stats, L_ctx=cfg.arch.L_ctx,
            extra=ITEM_PLAYER_COLUMNS, projection=ITEM_PLAYER_PROJECTION
        )
        with torch.inference_mode():
            hidden = model.forward_dense(batch.context.features, batch.context.ctx_pad)
            prediction = model.value_head(exp.decoder_rmsnorm(hidden[:, -1]).float()).squeeze(-1)
        predictions = prediction.detach().cpu().numpy()
        if not np.isfinite(predictions).all():
            raise ValueError(f'nonfinite values in game {record["id"]}')
        for index, value in zip(selected, predictions, strict=True):
            cluster = next((number for number, (start, stop) in enumerate(clusters) if start <= index < stop), None)
            rows.append({
                'index': index,
                'frame': int(frames[index]),
                'value': float(value),
                'return60': float(target[index]),
                'base': index in base,
                'tail': index in tail,
                'cluster': cluster,
            })
    eligible_frames = int(valid[cfg.arch.L_ctx - 1:len(frames) - cfg.arch.sample_chunk_length].sum())
    return {
        'id': record['id'], 'path': record['path'], 'sha1': record['sha1'],
        'perspective': port, 'set': record['set'], 'frame_count': len(frames),
        'eligible_frames': eligible_frames, 'high_clusters': clusters, 'rows': rows,
    }


def main() -> None:
    torch.set_num_threads(8)
    selection = json.loads((ROOT / 'selection.json').read_text())
    exp = importlib.import_module('experiments.059_muon_history_decoder')
    state = torch.load(CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    cfg = exp.config_from_state(state['cfg'])
    exp.validate_conditioning_state(state, cfg)
    if state['wandb_id'] != 'vywk3cih' or state['step'] != 131071:
        raise ValueError('checkpoint identity changed')
    if state['provenance']['git_sha'] != 'f05d41502a429b399fe9b64d534c442d1a9a6379':
        raise ValueError('checkpoint source changed')
    encoded = state['model']['player_code_bytes']
    vocabulary = PlayerVocabulary(decode_player_codes(encoded.numpy().tobytes()))
    with torch.device('meta'):
        model = exp.GPT(cfg, vocabulary)
    model.load_state_dict(state['model'], strict=True, assign=True)
    model.eval()
    payload = json.loads(STATS.read_text())
    stats = {name: FeatureStats(**value) for name, value in payload.items()}
    out = ROOT / 'rows'
    out.mkdir(exist_ok=True)
    started = time.monotonic()
    for position, record in enumerate(selection['records'], 1):
        destination = out / f"{record['id']:03d}.json"
        if destination.exists():
            continue
        result = score_game(record, model, cfg, stats, exp)
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, allow_nan=False, separators=(',', ':')))
        temporary.replace(destination)
        if position % 10 == 0:
            print(f'scored {position}/200 in {time.monotonic()-started:.1f}s', flush=True)
    print(f'all games scored in {time.monotonic()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
