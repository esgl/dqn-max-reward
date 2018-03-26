from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import tempfile
import os
import sys

from baselines import logger
import baselines.common.tf_util as U
from baselines.deepq.simple import ActWrapper
from baselines.common.schedules import LinearSchedule
from ReplayBuffer_MaxReward import ReplayBuffer_MaxReward, PrioritizedReplayBuffer_MaxReward
from build_graph import build_train

def learning_max_action_reward(
        env,
        q_func,
        lr=5e-4,
        max_timesteps=10000000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=1,
        batch_size=32,
        print_freq=100,
        checkpoint_freq=10000,
        learning_starts=1000,
        gamma=1.0,
        target_network_update_freq=500,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        param_noise=False,
        callback=None
):
    logger.log(sys._getframe().f_code.co_name)
    sess = tf.Session()
    sess.__enter__()
    observation_space_shape = env.observation_space.shape
    def make_obs_ph(name):
        return U.BatchInput(observation_space_shape, name=name)

    act, train, update_target, debug, q_val_f = build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    act_params = {
        "make_obs_ph": make_obs_ph,
        "q_func": q_func,
        "num_actions": env.action_space.n
    }

    act = ActWrapper(act, act_params)

    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer_MaxReward(
            buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer_MaxReward(buffer_size)
        beta_schedule = None

    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    U.initialize()
    update_target()

    episode_reward = [0.0]
    episode = []
    saved_mean_reward = None
    obs = env.reset()
    reset = True

    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join(td, "model")
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break

            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshlod = 0.
            else:
                update_eps = 0.
                update_param_noise_threshlod = -np.log(1.0 - exploration.value(t) +
                                                       exploration.value(t) / float(env.action_space.n))
                kwargs["reset"] = reset
                kwargs["update_param_noise_threshlod"] = update_param_noise_threshlod
                kwargs["update_param_noise_scale"] = True

            action = act(np.array(obs)[None], update_eps, **kwargs)[0]

            q_vals = q_val_f(np.array(obs)[None])[0]
            q_action_val = q_vals[action]
            env_action = action
            reset = False
            new_obs, rew, done, _ = env.step(env_action)
            replay_buffer.add(obs, action, rew, new_obs, float(done), q_action_val)

            obs = new_obs

            episode_reward[-1] += rew
            if done:
                obs = env.reset()
                reset = True
                episode_reward.append(0.0)

            if t > learning_starts and t % train_freq == 0:
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weigths, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_prioritized = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_prioritized)

            if t > learning_starts and t % target_network_update_freq == 0:
                update_target()

            mean_100ep_reward = round(np.mean(episode_reward[-101: -1]), 1)
            num_episodes = len(episode_reward)
            if done and print_freq is not None and len(episode_reward) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                            saved_mean_reward, mean_100ep_reward
                        ))
                    U.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward

        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            U.load_state(model_file)

    return act


































