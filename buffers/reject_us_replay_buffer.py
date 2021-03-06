from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from buffers.utils import RandomProjectionEncoder
from buffers.utils.encoder import obs_encoder


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
        rejection_coeff_change: float = 1e-6,
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
        self.rejection_exp = 1.0
        self.rejection_coeff_change = rejection_coeff_change

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
        """What happens when a transition is deleted from the buffer?
        We need to update the min_count and count_min_count variables, that offer a way to
        keep track of min_s n(s), where n(s) is the number of transitions that have state s,
        without the need to scan the entire n(s) vector.

        Two cases:
        1. The transition that was deleted has n(s) == min_count
            We need to decrement count_min_count[0] and shift the counters, because min_count is now min_count - 1
            If min_count was 1, the whole state was deleted and we don't need to do anything
        2. The transition that was deleted has n(s) == min_count + 1
            We need to decrement count_min_count[1] and increment count_min_count[0]
        """
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
        """Similar to what has been described above, but for the case of adding a transition
        that has n(s) == min_count or min_count + 1.

        Two cases:
        1. The transition that was added has n(s) == min_count
            We need to decrement count_min_count[0] and increment count_min_count[1], because the transition added one to the state counter
            If count_min_count[0] is now 0, there's no states with min_count, so we shift count_min_count and scan the entire state counter to find count_min_count[1]
        2. The transition that was added has n(s) == min_count + 1
            We need to decrement count_min_count[1]
        """
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

    def get_state_count(self, trans_ind: int) -> int:
        obs, act = self.observations[trans_ind], self.actions[trans_ind]
        enc_state = self._encode_obs_action(obs, act)
        return self.state_counter[enc_state]

    def _accept_transition(self, n_s: int) -> bool:
        """This function returns a boolean saying if the transition should be accepted or not.
        The decision is done using rejection sampling.
        In the USR case, the rejection ratio is inversely proportional to the state frequency of the transition in the buffer.
        The min count constant is added to increase sampling efficiency, and the exponent enables to shift USR to UER.
        """
        u: float = np.random.uniform()
        return u < (self.min_count / n_s) ** self.rejection_exp

    def _update_rejection_coeff(self) -> None:
        self.rejection_exp = np.min(0, self.rejection_exp - self.rejection_coeff_change)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        sampled_inds = []
        self._update_rejection_coeff()
        while len(sampled_inds) < batch_size:
            # Keep trying indices until batch_size samples have been accepted
            # Random number generation may have issues if optimize memory usage is true (check ReplayBuffer.sample)
            ind = np.random.randint(0, self.size())
            if self._accept_transition(self.get_state_count(ind)):
                sampled_inds.append(ind)
        return self._get_samples(sampled_inds, env=env)


class RejectUniformStateReplayBuffer(_RejectUniformStateReplayBuffer):
    """Uniform State Replay.
    This replay buffer performs uniform state replay on discrete state spaces.
    The implementation is done using rejection sampling, so that the buffer can be generalized to
    continuous state spaces, and the sampling coefficient can be shifted to obtain uniform sampling.

    Args:
        buffer_size: int, maximum size of the buffer
        observation_space: gym.Space, observation space of the environment
        action_space: gym.Space, action space of the environment
        device: torch.device, device on which the buffer will be stored
        n_envs: int, number of environments to sample from
        optimize_memory_usage: bool, whether to use an array less to store the next observations
        handle_timeout_termination: bool, whether to handle timeout termination or not
        rejection_coeff_change: float, change in rejection coefficient exponent. Updates are done e = e - rejection_coeff_change, it happens every time a batch is sampled.
    """

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
        rejection_coeff_change: float = 1e-6,
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
            rejection_coeff_change,
        )
