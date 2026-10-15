"""Run upstream TensorFlow inference in its own Python environment."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import multiprocessing.connection
import sys
from pathlib import Path
from typing import Any

import numpy as np


def checkpoint_name(digest: str) -> str:
    names = {
        "4cb632296a5387f4a1a05f0a18cba5528ad50f3e8a343b7ed3d7f192b0ece4b7": "all_d21_imitation_v3",
        "194d95f774c8e902fcb7ba4249d1a54207d884cdee0b1a27e25a1fd803120870": "gm-v1",
    }
    if digest not in names:
        raise ValueError("expected a verified all_d21_imitation_v3 or gm-v1 artifact")
    return names[digest]


def main() -> None:
    # Optional inference dependencies belong to the separate upstream environment.
    import tensorflow as tf  # ty: ignore[unresolved-import]
    from slippi_ai import eval_lib  # ty: ignore[unresolved-import]
    from slippi_ai import saving  # ty: ignore[unresolved-import]
    from slippi_ai import utils  # ty: ignore[unresolved-import]
    from slippi_db import parse_libmelee  # ty: ignore[unresolved-import]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", default="Master Player")
    parser.add_argument("--fd", type=int)
    args = parser.parse_args()
    with args.checkpoint.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    model_name = checkpoint_name(digest)
    if importlib.metadata.version("melee") != "0.47.3":
        raise ValueError("the observation bridge is tested with upstream melee 0.47.3")
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    state = saving.load_state_from_disk(str(args.checkpoint))
    config = saving.upgrade_config(state["config"])
    if config["policy"]["delay"] != 21:
        raise ValueError("expected the released 21-frame-delay policy")
    metadata = {
        "schema_version": 1,
        "checkpoint_name": model_name,
        "checkpoint_sha256": digest,
        "config": config,
        "name": args.name,
        "name_code": eval_lib.get_name_code(state, args.name),
        "allowed_characters": [character.name for character in eval_lib.allowed_characters(config)],
        "python": sys.version,
        "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
        "tensorflow_device": "cpu",
    }

    def build(seed: int) -> Any:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        agent = eval_lib.build_delayed_agent(
            state,
            console_delay=0,
            name=args.name,
            batch_size=1,
            compile=True,
            run_on_cpu=True,
            sample_temperature=1.0,
        )
        agent.warmup()
        if agent.delay != config["policy"]["delay"]:
            raise ValueError("loaded agent delay differs from its checkpoint configuration")
        return agent

    if args.fd is None:
        with contextlib.redirect_stdout(sys.stderr):
            agent = build(0)
        metadata["loaded_delay"] = agent.delay
        print(json.dumps(metadata, indent=2, default=str))
        return
    with multiprocessing.connection.Connection(args.fd) as connection:
        connection.send(json.loads(json.dumps(metadata, default=str)))
        while True:
            request = connection.recv()
            if request is None:
                return
            port, seed = request
            agent = build(seed)
            game_parser = parse_libmelee.Parser(ports=(port, 3 - port))
            connection.send("ready")
            first = True
            last_frame = None
            while True:
                gamestate = connection.recv()
                if gamestate is None:
                    break
                if last_frame is not None and gamestate.frame != last_frame + 1:
                    raise ValueError("slippi-ai observations must be consecutive")
                last_frame = gamestate.frame
                game = game_parser.get_game(gamestate)
                game = utils.map_single_structure(lambda value: np.expand_dims(value, 0), game)
                outputs = agent.step(game, np.array([first]))
                first = False
                controller = utils.map_single_structure(lambda value: value[0], outputs.controller_state)
                action = agent.decode_controller(controller)
                connection.send(
                    (
                        float(action.main_stick.x),
                        float(action.main_stick.y),
                        float(action.c_stick.x),
                        float(action.c_stick.y),
                        float(action.shoulder),
                        tuple(name for name, pressed in action.buttons._asdict().items() if pressed),
                    )
                )


if __name__ == "__main__":
    main()
