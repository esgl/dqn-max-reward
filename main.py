from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.common.atari_wrappers import make_atari
from baselines import deepq
from dqn_learning import learning
from dqn_learning_max_reward import learning_max_reward
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--env", help="environment ID", default="BreakoutNoFrameskip-v4")
    parser.add_argument("--env", help="environment ID", default="PongNoFrameskip-v0")
    parser.add_argument("--seed", help="RNG seed", type=int, default=0)
    parser.add_argument("--prioritized", type=int, default=1)
    parser.add_argument("--dueling", type=int, default=1)
    parser.add_argument("--num_timesteps", type=int, default=int(10e6))
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
    # act = learning(
    #     env,
    #     q_func=model,
    #     lr=1e-4,
    #     max_timesteps=args.num_timesteps,
    #     buffer_size=10000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.01,
    #     train_freq=4,
    #     learning_starts=10000,
    #     target_network_update_freq=1000,
    #     gamma=0.99,
    #     prioritized_replay=bool(args.prioritized)
    # )

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
    env.close()

if __name__ == '__main__':
    main()