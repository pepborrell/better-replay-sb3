from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List

import numpy as np
import numpy.typing as npt


class SmallUniformReplayBuf(ABC):
    def __init__(self, trajs):
        self.repbuf = defaultdict(list)
        self.keys = []
        for transn in trajs:
            self.append(transn)

    @abstractmethod
    def _encode_state_action(self, state, action=None):
        pass

    def append(self, t):
        s = self._encode_state_action(t["state"], t["action"])
        if s not in self.repbuf:
            self.keys.append(s)
        self.repbuf[s].append(t)

    def sample(self, n=1):
        sampled = []
        for _ in range(n):
            s_id = np.random.randint(len(self.keys))
            s = self.keys[s_id]
            trans_id = np.random.randint(len(self.repbuf[s]))
            sampled.append(self.repbuf[s][trans_id])
        return sampled


class RejectionReplayBuffer(ABC):
    def __init__(self, trajs) -> None:
        self.repbuf = trajs

    def append(self, t: Dict):
        self.repbuf.append(t)

    @abstractmethod
    def accept(self, t: Dict):
        pass

    def sample(self, n: int = 1):
        sampled = []
        while len(sampled) < n:
            s_id = np.random.randint(len(self.repbuf))
            if self.accept(self.repbuf[s_id]):
                sampled.append(self.repbuf[s_id])
        return sampled


class OptimalStrategyReplay(RejectionReplayBuffer):
    def __init__(self, trajs: List[Dict], dist: npt.NDArray[np.float_]):
        super().__init__(trajs)
        self.dist = dist
        self.max_dist = np.max(dist)

    def accept(self, t):
        pos = t["state"]
        act = t["action"]
        reject_ratio = self.dist[pos[0], pos[1], act] / self.max_dist
        assert reject_ratio <= 1.0, "Rejection ratio should be <= 1"
        return np.random.random() < reject_ratio or np.random.random() < 1e-1


class SmallUSR(SmallUniformReplayBuf):
    def __init__(self, trajs):
        super().__init__(trajs)

    def _encode_state_action(self, state, action=None):
        return tuple(state)


class SmallUSAR(SmallUniformReplayBuf):
    def __init__(self, trajs):
        super().__init__(trajs)

    def _encode_state_action(self, state, action):
        return tuple(state) + (action,)


class SmallUER:
    def __init__(self, trajs):
        self.repbuf = trajs

    def append(self, t):
        self.repbuf.append(t)

    def sample(self, n=1):
        sampled = []
        ids = np.random.choice(len(self.repbuf), n, replace=True)
        return [self.repbuf[i] for i in ids]


def get_samples(trajs, n_samples=100, repbuf_cls=SmallUER):
    rep = repbuf_cls(trajs)
    all_samples = rep.sample(n_samples)
    all_samples = [t["state"] for t in all_samples]
    return all_samples


def get_UER_samples(all_trajs, n_samples=100):
    return get_samples(all_trajs, n_samples, SmallUER)


def get_USAR_samples(all_trajs, n_samples=100):
    return get_samples(all_trajs, n_samples, SmallUSAR)


def get_USR_samples(all_trajs, n_samples=100):
    return get_samples(all_trajs, n_samples, SmallUSR)
