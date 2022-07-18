from re import S
from typing import NamedTuple, Sequence, Union

import numpy as np
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class ReplayBufferTransitions(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray


def index_rbuf_samples(samples: ReplayBufferSamples, ind: Union[int, np.ndarray]) -> ReplayBufferTransitions:
    return ReplayBufferTransitions(
        observations=samples.observations[ind],
        actions=samples.actions[ind],
        next_observations=samples.next_observations[ind],
        dones=samples.dones[ind],
        rewards=samples.rewards[ind],
    )


def join_transitions(transitions: Sequence[ReplayBufferTransitions]) -> ReplayBufferTransitions:
    return ReplayBufferTransitions(
        observations=np.concatenate([transitions[i].observations.copy() for i in range(len(transitions))], axis=0),
        actions=np.concatenate([transitions[i].actions.copy() for i in range(len(transitions))], axis=0),
        next_observations=np.concatenate(
            [transitions[i].next_observations.copy() for i in range(len(transitions))], axis=0
        ),
        dones=np.concatenate([transitions[i].dones.copy() for i in range(len(transitions))], axis=0),
        rewards=np.concatenate([transitions[i].rewards.copy() for i in range(len(transitions))], axis=0),
    )


class Reindexer:
    def __init__(self, samples: ReplayBufferSamples) -> None:
        self.samples = samples
        self.n = len(samples.rewards)

    def __getitem__(self, ind: Union[int, np.ndarray]) -> ReplayBufferTransitions:
        return index_rbuf_samples(self.samples, ind)

    def __len__(self) -> int:
        return self.n
