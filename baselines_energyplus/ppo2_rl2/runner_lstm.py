# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

import numpy as np


class RecurrentRunner:
    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        self.nsteps = int(nsteps)
        self.gamma = float(gamma)
        self.lam = float(lam)

        self.obs = env.reset()
        self.states = model.initial_state(env.num_envs)
        self.dones = np.zeros(env.num_envs, dtype=np.bool_)

    def run(self):
        mb_obs = []
        mb_rewards = []
        mb_actions = []
        mb_values = []
        mb_neglogpacs = []
        mb_dones = []
        epinfos = []

        states0 = (self.states[0].copy(), self.states[1].copy())

        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(
                self.obs, self.states, 1.0 - self.dones.astype(np.float32)
            )
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones.copy())

            self.obs, rewards, self.dones, infos = self.env.step(actions)
            mb_rewards.append(rewards)
            for info in infos:
                epinfo = info.get("episode")
                if epinfo:
                    epinfos.append(epinfo)

        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool_)

        last_values = self.model.value(
            self.obs, self.states, 1.0 - self.dones.astype(np.float32)
        )

        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = np.zeros(self.env.num_envs, dtype=np.float32)
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones.astype(np.float32)
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1].astype(np.float32)
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            mb_advs[t] = lastgaelam
        mb_returns = mb_advs + mb_values

        masks = 1.0 - mb_dones.astype(np.float32)
        return (
            mb_obs,
            mb_returns,
            masks,
            mb_actions,
            mb_values,
            mb_neglogpacs,
            states0,
            epinfos,
        )
