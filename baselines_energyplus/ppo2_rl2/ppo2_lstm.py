# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

import time
from typing import Tuple

import numpy as np
import tensorflow as tf
from baselines import logger
from baselines.common import explained_variance, set_global_seeds
from baselines.common.distributions import make_pdtype

from baselines_energyplus.ppo2_rl2.runner_lstm import RecurrentRunner


class LSTMModel(tf.Module):
    def __init__(
        self,
        ac_space,
        obs_dim: int,
        hidden_size: int = 128,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        learning_rate: float = 3e-4,
        cliprange: float = 0.2,
    ):
        super().__init__(name="PPO2LSTMModel")
        self.hidden_size = int(hidden_size)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
        self.cliprange = float(cliprange)

        self.cell = tf.keras.layers.LSTMCell(self.hidden_size, name="lstm_cell")
        self.policy_fc = tf.keras.layers.Dense(self.hidden_size, activation=tf.tanh, name="pi_fc")
        self.value_fc = tf.keras.layers.Dense(1, activation=None, name="vf")

        # Build pdtype based on latent shape
        self._latent_shape = (self.hidden_size,)
        self.pdtype = make_pdtype(self._latent_shape, ac_space)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def initial_state(self, nenv: int) -> Tuple[np.ndarray, np.ndarray]:
        h = np.zeros((nenv, self.hidden_size), dtype=np.float32)
        c = np.zeros((nenv, self.hidden_size), dtype=np.float32)
        return h, c

    def step(self, obs, states, masks):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        h, c = states
        masks = tf.convert_to_tensor(masks, dtype=tf.float32)
        h = tf.convert_to_tensor(h, dtype=tf.float32) * masks[:, None]
        c = tf.convert_to_tensor(c, dtype=tf.float32) * masks[:, None]

        output, [h2, c2] = self.cell(obs, [h, c])
        latent = self.policy_fc(output)
        pd, _ = self.pdtype.pdfromlatent(latent)
        action = pd.sample()
        neglogp = pd.neglogp(action)
        value = tf.squeeze(self.value_fc(output), axis=1)

        return (
            action.numpy(),
            value.numpy(),
            (h2.numpy(), c2.numpy()),
            neglogp.numpy(),
        )

    def value(self, obs, states, masks):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        h, c = states
        masks = tf.convert_to_tensor(masks, dtype=tf.float32)
        h = tf.convert_to_tensor(h, dtype=tf.float32) * masks[:, None]
        c = tf.convert_to_tensor(c, dtype=tf.float32) * masks[:, None]
        output, _ = self.cell(obs, [h, c])
        value = tf.squeeze(self.value_fc(output), axis=1)
        return value.numpy()

    def train(self, obs, returns, masks, actions, values, neglogp_old, states):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        masks = tf.convert_to_tensor(masks, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        values = tf.convert_to_tensor(values, dtype=tf.float32)
        neglogp_old = tf.convert_to_tensor(neglogp_old, dtype=tf.float32)

        h0, c0 = states
        h = tf.convert_to_tensor(h0, dtype=tf.float32)
        c = tf.convert_to_tensor(c0, dtype=tf.float32)

        with tf.GradientTape() as tape:
            neglogp_list = []
            entropy_list = []
            vpred_list = []

            nsteps = obs.shape[0]
            for t in range(nsteps):
                mask_t = masks[t]
                h = h * mask_t[:, None]
                c = c * mask_t[:, None]
                out, [h, c] = self.cell(obs[t], [h, c])
                latent = self.policy_fc(out)
                pd, _ = self.pdtype.pdfromlatent(latent)
                neglogp_list.append(pd.neglogp(actions[t]))
                entropy_list.append(pd.entropy())
                vpred_list.append(tf.squeeze(self.value_fc(out), axis=1))

            neglogp = tf.stack(neglogp_list, axis=0)
            entropy = tf.stack(entropy_list, axis=0)
            vpred = tf.stack(vpred_list, axis=0)

            advs = returns - values
            advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)

            ratio = tf.exp(neglogp_old - neglogp)
            pg_losses1 = -advs * ratio
            pg_losses2 = -advs * tf.clip_by_value(
                ratio, 1.0 - self.cliprange, 1.0 + self.cliprange
            )
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))

            vpredclipped = values + tf.clip_by_value(
                vpred - values, -self.cliprange, self.cliprange
            )
            vf_losses1 = tf.square(vpred - returns)
            vf_losses2 = tf.square(vpredclipped - returns)
            vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

            entropy_mean = tf.reduce_mean(entropy)
            loss = pg_loss - entropy_mean * self.ent_coef + vf_loss * self.vf_coef

        grads = tape.gradient(loss, self.trainable_variables)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        approxkl = 0.5 * tf.reduce_mean(tf.square(neglogp - neglogp_old))
        clipfrac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(ratio - 1.0), self.cliprange), tf.float32)
        )
        return (
            pg_loss.numpy(),
            vf_loss.numpy(),
            entropy_mean.numpy(),
            approxkl.numpy(),
            clipfrac.numpy(),
        )


def learn(
    *,
    env,
    total_timesteps,
    seed,
    nsteps=1024,
    ent_coef=0.0,
    lr=3e-4,
    vf_coef=0.5,
    max_grad_norm=0.5,
    gamma=0.99,
    lam=0.95,
    log_interval=10,
    nminibatches=1,
    noptepochs=4,
    hidden_size=128,
    cliprange=0.2,
):
    set_global_seeds(seed)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    if nminibatches > nenvs:
        raise ValueError("nminibatches must be <= number of environments for recurrent PPO2")

    obs_dim = int(env.observation_space.shape[0])
    ac_space = env.action_space

    model = LSTMModel(
        ac_space=ac_space,
        obs_dim=obs_dim,
        hidden_size=hidden_size,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        learning_rate=lr,
        cliprange=cliprange,
    )

    runner = RecurrentRunner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = []
    tfirststart = time.perf_counter()
    nbatch = nenvs * nsteps
    nupdates = total_timesteps // nbatch

    for update in range(1, nupdates + 1):
        tstart = time.perf_counter()
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
        epinfobuf.extend(epinfos)

        # Recurrent PPO2: minibatch over envs
        inds = np.arange(nenvs)
        for _ in range(noptepochs):
            np.random.shuffle(inds)
            for start in range(0, nenvs, nminibatches):
                end = start + nminibatches
                env_ids = inds[start:end]
                mb_obs = obs[:, env_ids]
                mb_returns = returns[:, env_ids]
                mb_masks = masks[:, env_ids]
                mb_actions = actions[:, env_ids]
                mb_values = values[:, env_ids]
                mb_neglogp = neglogpacs[:, env_ids]
                mb_states = (states[0][env_ids], states[1][env_ids])

                model.train(
                    mb_obs,
                    mb_returns,
                    mb_masks,
                    mb_actions,
                    mb_values,
                    mb_neglogp,
                    mb_states,
                )

        fps = int(nbatch / (time.perf_counter() - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values.reshape(-1), returns.reshape(-1))
            logger.logkv("misc/serial_timesteps", update * nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update * nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv("eprewmean", _safemean([ep["r"] for ep in epinfobuf]))
            logger.logkv("eplenmean", _safemean([ep["l"] for ep in epinfobuf]))
            logger.logkv("misc/time_elapsed", time.perf_counter() - tfirststart)
            logger.dumpkvs()

    return model


def _safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
