"""The current evaluator must implement delay independently of chunk stride."""

import time
from collections.abc import Sequence

import numpy as np
import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.controller import action_vec_to_controller
from hal.controller import controller_to_action_vec
from hal.eval.policy import PolicyBatchAdapter
from hal.eval.scheduling import FrameTiming
from hal.inference.api import ActionPlan
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.api import action_plan
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import Slot


def tagged_action(source: int, target: int) -> ControllerAction:
    return ControllerAction(0.3 + (source + 1) / 256, 0.3 + (target - source) / 64, 0, 0, 0, 0, 0)


class TaggedPolicy:
    spec = PolicySpec("timing-test", "test", (), (0,))
    context_frames = 64

    def __init__(self, delay: int, pause: float) -> None:
        self.delay = delay
        self.pause = pause
        self.requests: list[PredictionRequest] = []

    def reset_prediction(self) -> None:
        self.requests.clear()

    def predict(self, requests: Sequence[PredictionRequest]) -> tuple[ActionPlan, ...]:
        time.sleep(self.pause)
        self.requests.extend(requests)
        return tuple(
            action_plan(
                request,
                tuple(
                    tagged_action(request.source_frame, request.source_frame + offset)
                    for offset in range(self.delay + 1, 13)
                ),
                state_value=0.0,
            )
            for request in requests
        )


def reference(target: int, delay: int, stride: int) -> ControllerAction:
    # Only plans whose release has passed can supply this frame.
    if target <= delay:
        return NEUTRAL_CONTROLLER_ACTION
    source = ((target - delay - 1) // stride) * stride
    return tagged_action(source, target)


@pytest.mark.parametrize("delay", (0, 1, 2, 3, 4, 6))
@pytest.mark.parametrize("stride", (1, 2, 3, 4, 6))
def test_grid_release_forcing_and_reset(delay: int, stride: int) -> None:
    policy = TaggedPolicy(delay, 0)
    adapter = PolicyBatchAdapter(policy, RuntimeConfig(1, (0,), stride), FrameTiming(0, 0, delay, stride, 12))
    slot = Slot(0, 1)
    for episode in range(2):
        actions = {0: NEUTRAL_CONTROLLER_ACTION}
        for source in range(0, 60, stride):
            frames = (0,) if source == 0 else range(source - stride + 1, source + 1)
            rows = [
                ObservationRow(frame, {}, controller_to_action_vec(actions[frame]), reset=frame == 0)
                for frame in frames
            ]
            chunk = adapter.plan_rows({slot: rows})[slot]
            request = policy.requests[-1]
            assert request.source_frame == source
            assert request.generation == episode + 1
            assert request.deadline_monotonic is None
            expected_prefix = tuple(reference(frame, delay, stride) for frame in range(source + 1, source + delay + 1))
            assert request.fixed_actions == expected_prefix
            for offset in range(stride):
                target = source + offset + 1
                actual = action_vec_to_controller(chunk[offset])
                assert actual == reference(target, delay, stride)
                actions[target] = actual


def test_inference_wall_time_cannot_change_frame_schedule() -> None:
    outputs = []
    for pause in (0.0, 0.02):
        policy = TaggedPolicy(3, pause)
        adapter = PolicyBatchAdapter(policy, RuntimeConfig(1, (0,), 2), FrameTiming(0, 0, 3, 2, 12))
        slot = Slot(0, 1)
        applied = {0: controller_to_action_vec(NEUTRAL_CONTROLLER_ACTION)}
        for source in range(0, 12, 2):
            frames = (0,) if source == 0 else (source - 1, source)
            chunk = adapter.plan_rows({slot: [ObservationRow(frame, {}, applied[frame]) for frame in frames]})[slot]
            applied.update({source + offset + 1: chunk[offset] for offset in range(2)})
        outputs.append(np.stack(list(applied.values())))
    np.testing.assert_array_equal(*outputs)
