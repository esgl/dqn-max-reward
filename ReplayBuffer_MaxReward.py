from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import random

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree

class ReplayBuffer_MaxReward(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._return = []
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def print_self(self):
        for i in range(self.__len__()):
            print(self._storage[i]," ", self._return[i])

    def add(self, obs_t, action, reward, obs_tp1, done, _return):
        data = (obs_t, action, reward, obs_tp1, done)
        if len(self._storage) < self._maxsize:
            self._storage.append(data)
            self._return.append(_return)
            self._next_idx = len(self._storage)
        else:
            if _return > np.min(self._return):
                self._next_idx = np.argmin(self._return)
                self._storage[self._next_idx] = data
                self._return[self._next_idx] = _return
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_t), np.array(dones)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class PrioritizedReplayBuffer_MaxReward(ReplayBuffer_MaxReward):
    def __init__(self, size, alpha):
        super(PrioritizedReplayBuffer_MaxReward, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0
        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):

        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
