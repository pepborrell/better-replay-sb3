from collections import defaultdict
from random import sample
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from buffers.utils import RandomProjectionEncoder
from buffers.utils.encoder import obs_encoder
from buffers.utils.sample_reindexer import Reindexer, ReplayBufferTransitions, join_transitions


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

    def _encode_obs_action(self, obs: np.ndarray, action: Optional[np.ndarray] = None) -> tuple:
        return obs_encoder(self.node_encoder, obs)

    def _update_min_deleted(self, observation, action) -> None:
        # Remove the transition from the state counter
        enc_rem_t = self._encode_obs_action(observation, action)
        self.state_counter[enc_rem_t] -= 1
        if self.state_counter[enc_rem_t] == 0:
            del self.state_counter[enc_rem_t]
        # Removing a transition from a state with min_count transitions
        if self.state_counter[enc_rem_t] == self.min_count:
            self.count_min_count[0] -= 1
            # If the minimum is 1, the state was deleted and we aren't interested on it.
            # Otherwise, decrease minimum count
            if self.min_count > 1:
                self.min_count -= 1
                self.count_min_count[1] = self.count_min_count[0]
                self.count_min_count[0] = 1
        # Removing transition from state with min_count + 1 transitions
        elif self.state_counter[enc_rem_t] == self.min_count + 1:
            self.count_min_count[0] += 1
            self.count_min_count[1] -= 1

    def _update_min_added(self, encoded_state):
        # Adding a new transition to one of the states that has exactly min_count transitions
        if self.state_counter[encoded_state] == self.min_count:
            self.count_min_count[0] -= 1
            self.count_min_count[1] += 1
            if self.count_min_count[0] == 0:
                self.min_count += 1
                self.count_min_count[0] = self.count_min_count[1]
                self.count_min_count[1] = len(
                    [val for val in self.state_counter.values() if val == self.min_count + 1]
                )
        # Adding a new transition to a state that has min_count + 1 transitions
        elif self.state_counter[encoded_state] == self.min_count + 1:
            self.count_min_count[1] -= 1

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if self.full:
            removed_obs = self.observations[self.pos]
            removed_act = self.actions[self.pos]
            # Update min counts with deleted transition
            self._update_min_deleted(removed_obs, removed_act)

        super().add(obs, next_obs, action, reward, done, infos)

        enc_state = self._encode_obs_action(obs, action)
        self.state_counter[enc_state] += 1
        # Update min count with added transition
        self._update_min_added(enc_state)

    def get_state_count(self, transition: ReplayBufferTransitions) -> int:
        obs, act = transition.observations, transition.actions
        enc_state = self._encode_obs_action(obs, act)
        return self.state_counter[enc_state]

    def _accept_transition(self, n_s: int) -> bool:
        u: float = np.random.uniform()
        return u < (self.min_count / n_s)

    def _update_rejection_coeff(self) -> None:
        pass

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferTransitions:
        """Copied from stable_baselines3.common.buffers.ReplayBuffer.get_samples
        Copied because we want the output to be np arrays, not torch.
        We convert to torch later.
        """

        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferTransitions(*tuple(data))

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        sampled = []
        while len(sampled) < batch_size:
            uer_sample = Reindexer(super().sample(batch_size=batch_size, env=env))
            is_accepted = np.array(
                list(map(lambda transition: self._accept_transition(self.get_state_count(transition)), uer_sample))
            )
            if sampled:
                sampled = Reindexer(join_transitions((sampled[:], uer_sample[is_accepted])))
            else:
                sampled = Reindexer(join_transitions((uer_sample[is_accepted],)))

        self._update_rejection_coeff()
        data = sampled[:batch_size]
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
