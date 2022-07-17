from typing import Union

import numpy as np
import torch as th
from gym import spaces

from buffers.reject_us_replay_buffer import _RejectUniformStateReplayBuffer
from buffers.utils import RandomProjectionEncoder


class _RejectUniformStateActionReplayBuffer(_RejectUniformStateReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        node_encoder_cls=RandomProjectionEncoder,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
            node_encoder_cls,
        )

    def _encode_obs_action(self, state: np.ndarray, action: np.ndarray):
        encoded_state = self.node_encoder(state)
        encoded_state_action = (encoded_state, action)
        return encoded_state_action


class RejectUniformStateActionReplayBuffer(_RejectUniformStateActionReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        node_encoder_cls=RandomProjectionEncoder,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
            node_encoder_cls,
        )
