from collections import defaultdict
from random import sample
from typing import Any, Dict, List, Optional, Union
from unittest.mock import NonCallableMagicMock

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from buffers.utils import RandomProjectionEncoder
from buffers.utils.encoder import obs_encoder


class _UniformStateReplayBuffer(ReplayBuffer):
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
        )
        self.state_buffer_positions = defaultdict(list)
        self.add_count = 0
        self.node_encoder = node_encoder_cls(self.obs_shape)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos)

        encoded_state = self._encode_obs_action(obs, action)
        self.state_buffer_positions[encoded_state].append(self.add_count)
        self.add_count += 1

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        sampled_inds = np.array(self._sample_from_state_buffer(batch_size))

        return self._get_samples(sampled_inds, env=env)

    def _sample_from_state_buffer(self, batch_size) -> List[int]:
        keys = list(self.state_buffer_positions.keys())
        sampled_inds = []
        while len(sampled_inds) < batch_size:
            minisample_inds = np.random.randint(len(keys), size=batch_size - len(sampled_inds))
            trans_inds = map(
                lambda ind: self._sample_from_state_list(self.state_buffer_positions[keys[ind]]), minisample_inds
            )
            # Store transitions id that have been found
            for i, ind in enumerate(trans_inds):
                if ind is None:
                    # If state list is empty, delete state
                    del self.state_buffer_positions[[minisample_inds[i]]]
                else:
                    sampled_inds.append(ind)
        return sampled_inds

    def _sample_from_state_list(self, state_list) -> int:
        while len(state_list) > 0:
            ind = np.random.randint(len(state_list))
            if self._is_trans_ind_valid(state_list[ind]):
                return ind
            # Here old transitions are lazily deleted from the USR data structure
            del state_list[ind]
        # State list is empty
        return None

    def _is_trans_ind_valid(self, ind) -> bool:
        # Oldest transition in buffer has index: n added trans - buffer size
        oldest_trans = self.add_count - self.buffer_size
        return ind >= oldest_trans

    def _encode_obs_action(self, obs: np.ndarray, action: Optional[np.ndarray] = None) -> tuple:
        return obs_encoder(self.node_encoder, obs)


class UniformStateReplayBuffer(_UniformStateReplayBuffer):
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
