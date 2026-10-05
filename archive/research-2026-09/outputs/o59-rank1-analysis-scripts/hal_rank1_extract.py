import hashlib
import json
import shutil
import time
from pathlib import Path

from hal.data.archive import iter_archive_members, parse_archive_member_path


ROOT = Path('/tmp/hal-rank1-o59-calibration')


def main() -> None:
    selection = json.loads((ROOT / 'selection.json').read_text())
    records = {record['path']: record for record in selection['records']}
    members = set()
    for path in records:
        parsed = parse_archive_member_path(path)
        if parsed is None or parsed[0] != Path(selection['archive']):
            raise ValueError(path)
        members.add(parsed[1])
    destination = ROOT / 'replays'
    destination.mkdir(exist_ok=False)
    start = time.monotonic()
    count = 0
    for synthetic, temporary in iter_archive_members(
        Path(selection['archive']), tmpfs_root=Path('/tmp/hal-rank1-archive-tmp'), filter_paths=members, queue_size=8
    ):
        try:
            record = records[synthetic]
            replay = destination / f"{record['id']:03d}.slp"
            shutil.copyfile(temporary, replay)
            digest = hashlib.sha1(replay.read_bytes()).hexdigest()
            if digest != record['sha1']:
                replay.unlink()
                raise ValueError(f"SHA-1 mismatch for {synthetic}")
            count += 1
            if count % 20 == 0:
                print(f'extracted {count}/200 in {time.monotonic()-start:.1f}s', flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    if count != len(records):
        raise ValueError(f'extracted {count}, expected {len(records)}')
    print(f'extracted all {count} in {time.monotonic()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
