"""CLI: download fixtures into `<repo>/data/`.

Idempotent: skips fixtures whose local sha256 already matches. The emulators
come from public GitHub releases. The ISO and test datasets live in the
maintainer's private R2 bucket and need its `AWS_*` keys.

Usage:
    uv run fetch                         # public fixtures, then R2 fixtures
    uv run fetch --name dolphin-exiai    # one fixture
"""

import tyro

from hal import r2
from hal.fixtures import ALL
from hal.fixtures import BY_NAME
from hal.fixtures import ensure


def fetch(name: str | None = None) -> None:
    """Fetch one fixture by name, or all if --name omitted."""
    if name is not None:
        if name not in BY_NAME:
            raise SystemExit(f"unknown fixture {name!r}; known: {sorted(f.name for f in ALL)}")
        ensure(BY_NAME[name])
        return
    public = [fixture for fixture in ALL if fixture.r2_key is None]
    private = [fixture for fixture in ALL if fixture.r2_key is not None]
    for fixture in public:
        ensure(fixture)
    missing = r2.missing_credentials()
    if missing:
        raise SystemExit(
            f"fetched {', '.join(f.name for f in public)}. "
            f"{', '.join(f.name for f in private)} are in the maintainer's private R2 bucket, "
            f"which needs {', '.join(missing)} (see .env.example). "
            "Set HAL_ISO_PATH to use your own Melee 1.02 NTSC ISO."
        )
    for fixture in private:
        ensure(fixture)


def main() -> None:
    tyro.cli(fetch)


if __name__ == "__main__":
    main()
