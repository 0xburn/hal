"""Compare Cody/Fox p90 and fixed-120 evals across 24 CPU opponents."""

import json
import os
from collections import Counter
from collections import defaultdict
from pathlib import Path

import melee
import modal
import numpy as np

from hal.eval.cross_stage import MatchRow

RUN_NAME = "260921-092548_059_muon_history_decoder_o59v5-cooldown32768"
CHECKPOINT_SHA256 = "52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b"
ROOT = Path(__file__).resolve().parent
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_RESAMPLES = 10000
SCHEDULE_SHA256 = "8049dc6414c032049df716f17cc8fbee6ca7658459ef4fb84873d69a940bf0c4"

app = modal.App("hal-o59-cody-fox-compare")
image = modal.Image.debian_slim(python_version="3.14").pip_install("boto3")
secret = modal.Secret.from_name("hal")


@app.function(image=image, secrets=[secret], cpu=1, timeout=300, serialized=True, include_source=False)
def fetch_rows() -> dict[str, bytes]:
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    result = {}
    for arm in ("p90", "fixed120"):
        key = f"runs/{RUN_NAME}/checkpoints/eval96-step-0131072-cody-fox-all-cpus-{arm}/match_rows.json"
        result[arm] = client.get_object(Bucket=os.environ["AWS_BUCKET"], Key=key)["Body"].read()
    return result


def boot_totals(rows: list[MatchRow]) -> dict[int, tuple[float, float, float]]:
    totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for row in rows:
        if row.active_frames <= 0:
            continue
        values = totals[row.boot_index]
        values[0] += row.stocks_taken
        values[1] += row.stocks_lost
        values[2] += row.active_frames
    return {boot: (values[0], values[1], values[2]) for boot, values in totals.items()}


def stratified_bootstrap_indices(groups: np.ndarray, rng: np.random.Generator, resamples: int) -> np.ndarray:
    if groups.shape != (24, 4):
        raise ValueError("expected four boots for each of 24 CPU opponents")
    draws = rng.integers(0, 4, size=(resamples, 24, 4))
    return groups[np.arange(24)[None, :, None], draws].reshape(resamples, 96)


def main() -> None:
    client = modal.Client.from_env()
    client.hello()
    with modal.enable_output(), app.run(client=client):
        content = fetch_rows.remote()

    payloads = {arm: json.loads(raw) for arm, raw in content.items()}
    for arm, raw in content.items():
        (ROOT / f"{arm}-match_rows.json").write_bytes(raw)
    for arm, payload in payloads.items():
        protocol = payload["protocol"]
        if payload["schema_version"] != 7 or protocol["checkpoint_sha256"] != CHECKPOINT_SHA256:
            raise ValueError(f"{arm} has a different schema or checkpoint")
        if protocol["n_matchups"] != 96 or protocol["return_target"] != ("p90" if arm == "p90" else "fixed"):
            raise ValueError(f"{arm} has an unexpected matchup count or target")
        if protocol["ego_player_code"] != "IBDW#0" or protocol["fixed_ego_character"] != int(
            melee.Character.FOX.value
        ):
            raise ValueError(f"{arm} is not conditioned on Cody/Fox")
        if protocol["matchup_schedule_sha256"] != SCHEDULE_SHA256 or protocol["cpu_characters"] != 24:
            raise ValueError(f"{arm} has an unexpected CPU opponent schedule")
        if protocol["cpu_level"] != 9 or protocol["max_frames"] != 7200:
            raise ValueError(f"{arm} has an unexpected CPU level or frame budget")
        if arm == "fixed120" and protocol["desired_return"] != 120.0:
            raise ValueError("the treatment did not use fixed return 120")

    control_protocol = payloads["p90"]["protocol"]
    treatment_protocol = payloads["fixed120"]["protocol"]
    differing = {key for key in control_protocol if control_protocol[key] != treatment_protocol[key]}
    if differing != {"return_target", "desired_return"}:
        raise ValueError(f"unexpected protocol differences: {sorted(differing)}")

    rows = {arm: [MatchRow.from_dict(item) for item in payload["rows"]] for arm, payload in payloads.items()}
    boots = {arm: boot_totals(items) for arm, items in rows.items()}
    if set(boots["p90"]) != set(range(96)) or set(boots["fixed120"]) != set(range(96)):
        raise ValueError("one or more matchup boots have no active frames")
    opponent_counts = Counter()
    for boot in range(96):
        control_pairs = {(row.ego_character, row.opp_character) for row in rows["p90"] if row.boot_index == boot}
        treatment_pairs = {
            (row.ego_character, row.opp_character) for row in rows["fixed120"] if row.boot_index == boot
        }
        if control_pairs != treatment_pairs or len(control_pairs) != 1:
            raise ValueError(f"matchup differs at boot {boot}")
        opponent_counts[next(iter(control_pairs))[1]] += 1
    if len(opponent_counts) != 24 or set(opponent_counts.values()) != {4}:
        raise ValueError("opponent schedule does not cover 24 characters four times")

    arrays = {arm: np.array([boots[arm][index] for index in range(96)]) for arm in ("p90", "fixed120")}
    control = arrays["p90"]
    treatment = arrays["fixed120"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    index = rng.integers(0, 96, size=(BOOTSTRAP_RESAMPLES, 96))

    def rate(values: np.ndarray) -> np.ndarray:
        return 3600 * (values[..., 0].sum(axis=-1) - values[..., 1].sum(axis=-1)) / values[..., 2].sum(axis=-1)

    point = float(rate(treatment) - rate(control))
    samples = rate(treatment[index]) - rate(control[index])
    low, high = np.percentile(samples, (2.5, 97.5))
    per_character = []
    character_boot_groups = []
    for character in sorted(opponent_counts):
        boots_for_character = [
            boot
            for boot in range(96)
            if any(row.boot_index == boot and row.opp_character == character for row in rows["p90"])
        ]
        indices = np.asarray(boots_for_character, dtype=np.int64)
        character_boot_groups.append(indices)
        per_character.append(
            {
                "opponent": melee.Character(character).name,
                "boots": len(boots_for_character),
                "p90_net_stock_per_min": float(rate(control[indices])),
                "fixed120_net_stock_per_min": float(rate(treatment[indices])),
                "delta_net_stock_per_min": float(rate(treatment[indices]) - rate(control[indices])),
            }
        )
    groups = np.stack(character_boot_groups)
    stratified_indices = stratified_bootstrap_indices(groups, rng, BOOTSTRAP_RESAMPLES)
    stratified_samples = rate(treatment[stratified_indices]) - rate(control[stratified_indices])
    stratified_low, stratified_high = np.percentile(stratified_samples, (2.5, 97.5))
    report = {
        "schema_version": 1,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "control_return": control_protocol["desired_return"],
        "treatment_return": treatment_protocol["desired_return"],
        "matched_boots": 96,
        "control_matches": len(rows["p90"]),
        "treatment_matches": len(rows["fixed120"]),
        "control_net_stock_per_min_from_rows": float(rate(control)),
        "treatment_net_stock_per_min_from_rows": float(rate(treatment)),
        "pooled_net_stock_delta_from_rows": point,
        "paired_bootstrap_95_ci": [float(low), float(high)],
        "paired_character_stratified_bootstrap_95_ci": [float(stratified_low), float(stratified_high)],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "protocol_differences": sorted(differing),
        "opponent_schedule_sha256": SCHEDULE_SHA256,
        "per_character": per_character,
        "limits": "Matchups are paired, but stages and game randomness are not shared across runs.",
    }
    (ROOT / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
