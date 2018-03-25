import numpy as np

from ReplayBuffer_MaxReward import ReplayBuffer_MaxReward


replay_buffer = ReplayBuffer_MaxReward(10)
data = np.random.randn(10, 5)
return_ = np.random.randint(1, 10, 10)
print(data[:10])

for i in range(10):
    replay_buffer.add(data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i, 4], return_[i])
print("size of replay_buffer:", replay_buffer.__len__())

new_size = 5
new_data = np.random.randn(new_size, 5)
new_return_ = np.random.randint(1, 10, new_size)
print(new_return_)
for i in range(new_size):
    print("new_return:", new_return_[i])
    replay_buffer.add(data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i, 4], new_return_[i])
print("size of replay_buffer:", replay_buffer.__len__())

