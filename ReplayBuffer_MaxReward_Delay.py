from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import random

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
class ReplayBuffer_MaxReward_Delay(ReplayBuffer):
    def __init__(self, size, delay):
        super(ReplayBuffer_MaxReward_Delay, self).__init__(size)
        self._return = []
        self._update_step = 0
        self._delay = delay

    def add(self, obs_t, action, reward, obs_tp1, done, _return):
        self._update_step += 1
        data = (obs_t, action, reward, obs_tp1, done)
        if self.__len__() < self._maxsize:
            self._storage.append(data)
            self._return.append(_return)
            self._next_idx = (self._next_idx + 1) % self._maxsize
        else:
            if self._update_step < self._delay:
                self._storage[self._next_idx] = data
                self._return[self._next_idx] = _return
                self._next_idx = (self._next_idx + 1) % self._maxsize
            else:
                min_return = np.min(self._return)
                if _return > min_return:
                    self._next_idx = np.argmin(self._return)
                    self._storage[self._next_idx] = data
                    self._return[self._next_idx] = _return
        # self._next_idx = (self._next_idx + 1) % self._maxsize

class PrioritizedReplayBuffer_MaxReplay_Delay(ReplayBuffer_MaxReward_Delay):
    def __init__(self, size, delay, alpha):
        super(PrioritizedReplayBuffer_MaxReplay_Delay, self).__init__(size, delay)
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
        super(PrioritizedReplayBuffer_MaxReplay_Delay, self).add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, self.__len__() - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0
        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.__len__()) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.__len__()) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.__len__()
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


