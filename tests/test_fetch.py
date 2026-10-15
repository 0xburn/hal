import pytest

from hal import r2
from hal.fixtures import ALL
from hal.fixtures import Fixture
from hal.scripts import fetch as fetch_cli


def test_fetch_downloads_public_fixtures_before_rejecting_missing_r2_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[str] = []

    def ensure(fixture: Fixture) -> None:
        fetched.append(fixture.name)

    monkeypatch.setattr(fetch_cli, "ensure", ensure)
    monkeypatch.setattr(r2, "missing_credentials", lambda: ["AWS_ACCESS_KEY_ID"])

    with pytest.raises(SystemExit, match="private R2 bucket.*HAL_ISO_PATH"):
        fetch_cli.fetch()

    assert fetched == [fixture.name for fixture in ALL if fixture.r2_key is None]
    assert fetched == ["Slippi_Online-x86_64.AppImage", "dolphin-exiai"]


def test_fetch_downloads_every_fixture_with_r2_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[str] = []

    def ensure(fixture: Fixture) -> None:
        fetched.append(fixture.name)

    monkeypatch.setattr(fetch_cli, "ensure", ensure)
    monkeypatch.setattr(r2, "missing_credentials", lambda: [])

    fetch_cli.fetch()

    assert sorted(fetched) == sorted(fixture.name for fixture in ALL)
