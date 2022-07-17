from collections import defaultdict
from random import sample
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from buffers.utils import RandomProjectionEncoder


class _RejectUniformStateReplayBuffer(ReplayBuffer):
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

        self.node_encoder = node_encoder_cls(self.observation_space.shape)
        self.state_counter = defaultdict(int)  # Counts how many transitions contain a given state

        # To maintain a record of what's min_s n(s),
        # where n(s) is the number of transitions that have state s
        self.min_count = 1
        self.count_min_count = [
            1,
            0,
        ]  # Count the number of appearences of min count and min_count + 1 in state counter

    def _encode_obs_action(self, obs: np.ndarray, action: Optional[np.ndarray] = None) -> np.ndarray:
        return self.node_encoder(obs)

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

        enc_state = self._encode_obs_action(obs, action)
        self.state_counter[enc_state] += 1

    def get_state_count(self, obs: np.ndarray, action: np.ndarray) -> int:
        enc_state = self._encode_obs_action(obs, action)
        return self.state_counter[enc_state]

    def _accept_transition(self, obs: th.Tensor, action: th.Tensor) -> bool:
        n_s = self.get_state_count(obs, action)
        u = np.random.uniform()
        return u < (self.min_count / n_s)

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        sampled_obs = []
        sampled_acts = []
        sampled_next = []
        sampled_dones = []
        sampled_rews = []
        while len(sampled_obs) < batch_size:
            n_trans = batch_size - len(sampled_obs)
            obses, acts, next_obses, dones, rews = super().sample(batch_size=n_trans)
            for i in range(n_trans):
                if self._accept_transition(obses[i], acts[i]):
                    sampled_obs.append(obses[i])
                    sampled_acts.append(acts[i])
                    sampled_next.append(next_obses[i])
                    sampled_dones.append(dones[i])
                    sampled_rews.append(rews[i])

        data = (
            th.stack(sampled_obs, dim=0),
            th.stack(sampled_acts, dim=0),
            th.stack(sampled_next, dim=0),
            th.stack(sampled_dones, dim=0),
            th.stack(sampled_rews, dim=0),
        )
        self._update_rejection_coeff()
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class RejectUniformStateReplayBuffer(_RejectUniformStateReplayBuffer):
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
