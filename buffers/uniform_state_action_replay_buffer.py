from typing import Optional, Union

import numpy as np
import torch as th
from gym import spaces

from buffers.uniform_state_replay_buffer import _UniformStateReplayBuffer
from buffers.utils import RandomProjectionEncoder
from buffers.utils.encoder import obs_action_encoder


class _UniformStateActionReplayBuffer(_UniformStateReplayBuffer):
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

    def _encode_obs_action(self, obs: np.ndarray, action: np.ndarray) -> tuple:
        return obs_action_encoder(self.node_encoder, obs, action)


class UniformStateActionReplayBuffer(_UniformStateActionReplayBuffer):
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
