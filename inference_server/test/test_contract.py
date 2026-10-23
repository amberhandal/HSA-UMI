"""validate_observation: every per-step channel must carry n_obs_steps rows, optional ones included."""

import numpy as np
import pytest

from polyumi_inference import Observation
from polyumi_inference.contract import validate_observation
from polyumi_inference.errors import WireFormatError


def _obs(**extra):
    channels = {'camera0_rgb': np.zeros((2, 224, 224, 3), np.uint8), 'agent_pos': np.zeros((2, 8))}
    channels.update(extra)
    return Observation(channels=channels, n_obs_steps=2, n_action_steps=16)


def test_optional_channels_are_checked_per_step():
    validate_observation(_obs(robot0_wrist_iq=np.zeros((2, 6), np.float32), camera1_rgb=np.zeros((2, 224, 224, 3), np.uint8)))
    with pytest.raises(WireFormatError, match='robot0_wrist_iq leading dim'):
        validate_observation(_obs(robot0_wrist_iq=np.zeros((3, 6), np.float32)))
