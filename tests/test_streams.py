from pathlib import Path

import pytest

from hal import streams
from hal.data.feature_stats import FeatureStatsSufficient
from hal.data.feature_stats import dump_sufficient_stats
from hal.data.feature_stats import load_consolidated_mixture_stats


def test_policy_world_manifest_has_the_verified_natural_mix() -> None:
    names = {source.name for source in streams.POLICY_WORLD_V7_SOURCES}

    assert len(streams.POLICY_WORLD_V7_SOURCES) == len(names) == 44
    assert names == set(streams.POLICY_WORLD_V7_TRAIN_REPLAYS)
    assert names == set(streams.POLICY_WORLD_V7_TRAIN_FRAMES)
    assert sum(streams.POLICY_WORLD_V7_TRAIN_REPLAYS.values()) == 1_300_640
    assert sum(streams.POLICY_WORLD_V7_TRAIN_FRAMES.values()) == 13_292_693_140


def test_policy_world_v8_manifest_has_the_verified_natural_mix() -> None:
    names = {source.name for source in streams.POLICY_WORLD_V8_SOURCES}

    assert len(streams.POLICY_WORLD_V8_SOURCES) == len(names) == 44
    assert names == set(streams.POLICY_WORLD_V8_TRAIN_REPLAYS)
    assert names == set(streams.POLICY_WORLD_V8_TRAIN_FRAMES)
    assert names == set(streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256)
    assert all(
        len(digest) == 64 and not set(digest) - set("0123456789abcdef")
        for digest in streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256.values()
    )
    assert sum(streams.POLICY_WORLD_V8_TRAIN_REPLAYS.values()) == 1_295_370
    assert sum(streams.POLICY_WORLD_V8_TRAIN_FRAMES.values()) == 13_266_364_175

    ranked_one = streams.RANKED_ANONYMIZED_POLICY_WORLD_V8[0]
    assert ranked_one.name == "ranked-anonymized-1-policy-world-v8"
    assert ranked_one.remote == "s3://hal/processed/ranked-anonymized-1/mds-policy-world-v8"
    assert ranked_one.local == Path("data/processed/ranked-anonymized-1/mds-policy-world-v8")
    assert ranked_one in streams.ALL


def test_stats_loader_pulls_only_selected_registered_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = streams.RANKED_ANONYMIZED_1_POLICY_V7
    calls: list[streams.StreamSource] = []
    monkeypatch.setattr(streams, "REPO_DIR", str(tmp_path))

    def fake_pull_stats(src: streams.StreamSource) -> Path:
        calls.append(src)
        destination = src.local_root / "stats.json"
        destination.parent.mkdir(parents=True)
        sufficient = {
            "p1_x": FeatureStatsSufficient(count=2, mean=1.0, m2=2.0, min=0.0, max=2.0),
        }
        dump_sufficient_stats(destination, sufficient, split="train", mds_schema_version=7)
        return destination

    monkeypatch.setattr(streams, "pull_stats", fake_pull_stats)

    result = load_consolidated_mixture_stats(
        [tmp_path / selected.local / "stats.json"], [1.0], expected_mds_schema_version=7
    )

    assert result["x"].mean == 1.0
    assert calls == [selected]
    assert streams.remote_for_local(selected.local) == selected.remote
    assert streams.remote_for_local(tmp_path / selected.local) == selected.remote


def test_ensure_stats_leaves_existing_and_unregistered_paths_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "local" / "stats.json"
    existing.parent.mkdir()
    existing.write_text("{}")

    def fail_pull_stats(_src: streams.StreamSource) -> Path:
        raise AssertionError("unexpected download")

    monkeypatch.setattr(streams, "pull_stats", fail_pull_stats)

    assert streams.ensure_stats(existing) == existing
    assert streams.ensure_stats(tmp_path / "missing" / "stats.json") == tmp_path / "missing" / "stats.json"


def test_consolidated_mixture_stats_use_source_proportions(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    dump_sufficient_stats(
        first,
        {
            "p1_x": FeatureStatsSufficient(count=2, mean=0.0, m2=2.0, min=-1.0, max=1.0),
            "p2_x": FeatureStatsSufficient(count=2, mean=0.0, m2=2.0, min=-1.0, max=1.0),
        },
        split="train",
        mds_schema_version=7,
    )
    dump_sufficient_stats(
        second,
        {
            "p1_x": FeatureStatsSufficient(count=2, mean=4.0, m2=8.0, min=2.0, max=6.0),
            "p2_x": FeatureStatsSufficient(count=2, mean=4.0, m2=8.0, min=2.0, max=6.0),
        },
        split="train",
        mds_schema_version=7,
    )

    mixture = load_consolidated_mixture_stats([first, second], [0.25, 0.75], expected_mds_schema_version=7)
    stats = mixture["x"]

    assert stats.mean == pytest.approx(3.0)
    assert stats.std == pytest.approx(2.5)
    assert stats.min == -1.0
    assert stats.max == 6.0
    for name in ("direction", "nana_direction"):
        assert mixture[name].mean == 0.0
        assert mixture[name].std == 1.0
        assert mixture[name].min == -1.0
        assert mixture[name].max == 1.0


def test_consolidated_stats_add_packed_direction_bounds(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.json"
    dump_sufficient_stats(
        stats_path,
        {"p1_x": FeatureStatsSufficient(count=1, mean=0.0, m2=0.0, min=0.0, max=0.0)},
        split="train",
        mds_schema_version=7,
    )

    stats = load_consolidated_mixture_stats([stats_path], [1.0], expected_mds_schema_version=7)

    for name in ("direction", "nana_direction"):
        assert stats[name].mean == 0.0
        assert stats[name].std == 1.0
        assert stats[name].min == -1.0
        assert stats[name].max == 1.0


def test_consolidated_mixture_stats_check_every_schema(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    sufficient = {"p1_x": FeatureStatsSufficient(count=1, mean=0.0, m2=0.0, min=0.0, max=0.0)}
    dump_sufficient_stats(first, sufficient, split="train", mds_schema_version=7)
    dump_sufficient_stats(second, sufficient, split="train", mds_schema_version=6)

    with pytest.raises(ValueError, match="mds_schema_version"):
        load_consolidated_mixture_stats([first, second], [1.0, 1.0], expected_mds_schema_version=7)


def test_huggingface_sources_mirror_the_v8_registry(tmp_path: Path) -> None:
    sources = streams.huggingface_policy_world_v8_sources(tmp_path)

    assert [source.name for source in sources] == [source.name for source in streams.POLICY_WORLD_V8_SOURCES]
    ranked_one, pro_aklo = sources[0], sources[6]
    assert ranked_one.remote == "hf://datasets/ericyuegu/hal-mds-dataset/ranked-anonymized-1"
    assert ranked_one.for_split("train") == (
        "hf://datasets/ericyuegu/hal-mds-dataset/ranked-anonymized-1/train",
        tmp_path / "ranked-anonymized-1" / "train",
    )
    assert pro_aklo.name == "professional-aklo-policy-world-v8"
    assert pro_aklo.local_root == tmp_path / "pro" / "aklo"


def test_pull_stats_reads_huggingface_sources_from_the_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_pull(path: str, dest: Path) -> Path:
        calls.append((path, dest))
        return dest

    monkeypatch.setattr(streams, "pull_huggingface_file", fake_pull)
    source = streams.huggingface_policy_world_v8_sources(tmp_path)[6]

    assert streams.pull_stats(source) == tmp_path / "pro" / "aklo" / "stats.json"
    assert calls == [("pro/aklo/stats.json", tmp_path / "pro" / "aklo" / "stats.json")]


def test_export_statistics_come_from_the_selected_mirror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hal.inference.action_sequence_artifact import source_stats_paths

    names = tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
    cached = tmp_path / "ranked-anonymized-1" / "stats.json"
    cached.parent.mkdir()
    cached.write_text("{}")
    pulled: list[str] = []

    def fake_pull_stats(src: streams.StreamSource) -> Path:
        pulled.append(src.name)
        return src.local_root / "stats.json"

    monkeypatch.setattr(streams, "pull_stats", fake_pull_stats)

    paths = source_stats_paths(names, tmp_path)

    assert paths[0] == cached
    assert paths[6] == tmp_path / "pro" / "aklo" / "stats.json"
    assert pulled == list(names[1:])
    with pytest.raises(ValueError, match="registry order"):
        source_stats_paths(names[:1], tmp_path)
