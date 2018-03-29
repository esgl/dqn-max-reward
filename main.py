from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.common.atari_wrappers import make_atari
from baselines import deepq
from dqn_learning import dqn_learning
from dqn_learning_max_reward import learning_max_reward
from dqn_learning_max_reward_delay import learning_max_reward_delay
from baselines.deepq.simple import learn
from dqn_learning_max_action_reward import learning_max_action_reward
from dqn_reward_shaping import learning_reward_shaping
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--env", help="environment ID", default="BreakoutNoFrameskip-v4")
    parser.add_argument("--env", help="environment ID", default="PongNoFrameskip-v0")
    parser.add_argument("--seed", help="RNG seed", type=int, default=0)
    parser.add_argument("--prioritized", type=int, default=1)
    parser.add_argument("--dueling", type=int, default=1)
    parser.add_argument("--num_timesteps", type=int, default=int(1*10e6))
    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    option = 6
    if option == 1:
        act = learn(
            env,
            q_func=model,
            lr=1e-4,
            max_timesteps=args.num_timesteps,
            buffer_size=500000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=10000,
            target_network_update_freq=1000,
            gamma=0.99,
            prioritized_replay=bool(args.prioritized)
        )
    elif option == 2:
        act = dqn_learning(
            env,
            q_func=model,
            lr=1e-4,
            max_timesteps=args.num_timesteps,
            buffer_size=10000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=10000,
            target_network_update_freq=1000,
            gamma=0.99,
            prioritized_replay=bool(args.prioritized)
        )
    elif option == 3:
        act = learning_max_reward(
            env,
            q_func=model,
            lr=1e-4,
            max_timesteps=args.num_timesteps,
            buffer_size=10000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=10000,
            target_network_update_freq=1000,
            gamma=0.99,
            prioritized_replay=bool(args.prioritized)
        )
    elif option == 4:
        act = learning_max_reward_delay(
            env,
            q_func=model,
            lr=1e-4,
            max_timesteps=args.num_timesteps,
            buffer_size=10000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=10000,
            target_network_update_freq=1000,
            gamma=0.99,
            prioritized_replay=bool(args.prioritized)
        )
    elif option == 5:
        act = learning_max_action_reward(
            env,
            q_func=model,
            lr=1e-4,
            max_timesteps=args.num_timesteps,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=10000,
            target_network_update_freq=1000,
            gamma=0.99,
            prioritized_replay=bool(args.prioritized)
        )
    elif option == 6:
        act = learning_reward_shaping(
            env,
            q_func=model,
            lr=1e-4,
            max_timesteps=args.num_timesteps,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=10000,
            target_network_update_freq=1000,
            gamma=0.99,
            prioritized_replay=bool(args.prioritized)
        )
    env.close()

if __name__ == '__main__':
    main()
