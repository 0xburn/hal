import hashlib
import json
from pathlib import Path

import numpy as np

from hal.data.archive import parse_archive_member_path
from hal.data.index import ReplayIndexEntry, replay_uuid_from_path
from hal.scripts.materialize import bucket_fraction
from hal.scripts.rematerialize_policy_world_v8 import _v7_predicates, _v8_predicates


INDEX = Path('data/processed/ranked-anonymized-1/index.jsonl')
ARCHIVE = Path('data/raw/ranked-anonymized-1-116248.7z')
OUT = Path('/tmp/hal-rank1-o59-calibration')
SEED = 20260925


def main() -> None:
    candidates = []
    with INDEX.open() as source:
        for line in source:
            entry = ReplayIndexEntry.from_dict(json.loads(line))
            fraction = bucket_fraction(replay_uuid_from_path(entry.path))
            if not 0.98 <= fraction < 0.99:
                continue
            if not all(predicate(entry) for _, predicate in _v7_predicates()):
                continue
            if not all(predicate(entry) for _, predicate in _v8_predicates()):
                continue
            parsed = parse_archive_member_path(entry.path)
            if parsed is None or parsed[0] != ARCHIVE:
                raise ValueError(entry.path)
            candidates.append(entry)
    if len(candidates) < 200:
        raise ValueError(f'only {len(candidates)} eligible held-out games')
    rng = np.random.default_rng(SEED)
    chosen = [candidates[int(i)] for i in rng.choice(len(candidates), size=200, replace=False)]
    records = []
    for i, entry in enumerate(chosen):
        port = ('p1', 'p2')[int(rng.integers(2))]
        records.append({
            'id': i,
            'path': entry.path,
            'sha1': entry.sha1,
            'frame_count_index': entry.frame_count,
            'perspective': port,
            'set': 'fit' if i < 150 else 'test',
        })
    OUT.mkdir(parents=True, exist_ok=False)
    index_sha = hashlib.sha256(INDEX.read_bytes()).hexdigest()
    payload = {'seed': SEED, 'index_sha256': index_sha, 'archive': str(ARCHIVE), 'candidate_count': len(candidates), 'records': records}
    (OUT / 'selection.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(f'{len(candidates)} eligible held-out games; selected 200, fit 150/test 50', flush=True)
    print(f'selection: {OUT / "selection.json"}', flush=True)


if __name__ == '__main__':
    main()
