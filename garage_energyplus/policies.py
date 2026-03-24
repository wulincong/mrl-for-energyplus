import akro
import numpy as np
import torch

from garage.torch import as_torch
from garage.torch.policies.stochastic_policy import StochasticPolicy

from garage_energyplus.models import GaussianLSTMModel


class GaussianLSTMPolicy(StochasticPolicy):
    """Gaussian LSTM policy for continuous control."""

    def __init__(self,
                 env_spec,
                 hidden_dim=32,
                 output_nonlinearity=None,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 std_share_network=False,
                 hidden_state_init_trainable=False,
                 cell_state_init_trainable=False,
                 layer_normalization=False,
                 state_include_action=True,
                 name='GaussianLSTMPolicy'):
        if not isinstance(env_spec.action_space, akro.Box):
            raise ValueError('GaussianLSTMPolicy only works with '
                             'akro.Box action space, but not {}'.format(
                                 env_spec.action_space))
        super().__init__(env_spec, name)

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._state_include_action = state_include_action
        if state_include_action:
            self._input_dim = self._obs_dim + self._action_dim
        else:
            self._input_dim = self._obs_dim

        self._module = GaussianLSTMModel(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_dim=hidden_dim,
            output_nonlinearity=output_nonlinearity,
            learn_std=learn_std,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_parameterization=std_parameterization,
            std_share_network=std_share_network,
            hidden_state_init_trainable=hidden_state_init_trainable,
            cell_state_init_trainable=cell_state_init_trainable,
            layer_normalization=layer_normalization)

        self._prev_actions = None
        self._prev_hiddens = None
        self._prev_cells = None

    @property
    def input_dim(self):
        """int: Dimension of the policy input."""
        return self._input_dim

    @property
    def env_spec(self):
        """Policy environment specification."""
        return self._env_spec

    @property
    def state_info_specs(self):
        """State info specification."""
        if self._state_include_action:
            return [('prev_action', (self._action_dim, ))]
        return []

    def reset(self, do_resets=None):
        """Reset the policy state."""
        if do_resets is None:
            do_resets = np.array([True])
        if self._prev_actions is None or len(do_resets) != len(
                self._prev_actions):
            self._prev_actions = np.zeros(
                (len(do_resets), self._action_dim))
            device = next(self._module.parameters()).device
            init_hidden, init_cell = self._module.get_init_state(
                len(do_resets), device)
            self._prev_hiddens = init_hidden[0].detach()
            self._prev_cells = init_cell[0].detach()

        self._prev_actions[do_resets] = 0.
        device = next(self._module.parameters()).device
        init_hidden, init_cell = self._module.get_init_state(
            len(do_resets), device)
        reset_mask = torch.as_tensor(do_resets,
                                     dtype=torch.bool,
                                     device=device)
        self._prev_hiddens[reset_mask] = init_hidden[0][reset_mask]
        self._prev_cells[reset_mask] = init_cell[0][reset_mask]

    # pylint: disable=arguments-differ
    def forward(self, observations, hidden_state=None, cell_state=None):
        """Compute the action distributions from the observations."""
        if observations.dim() == 3:
            full_input = observations
            step_input = observations[:, -1, :]
        else:
            full_input = observations.unsqueeze(1)
            step_input = observations

        (dist, step_mean, step_log_std, _, _, _, _) = self._module(
            full_input, step_input, hidden_state, cell_state)
        return (dist, dict(mean=step_mean, log_std=step_log_std))

    def get_actions(self, observations):
        """Get multiple actions from this policy for the input observations."""
        if not isinstance(observations[0], np.ndarray) and not isinstance(
                observations[0], torch.Tensor):
            observations = self._env_spec.observation_space.flatten_n(
                observations)
        if isinstance(observations, list):
            if isinstance(observations[0], np.ndarray):
                observations = np.stack(observations)
            elif isinstance(observations[0], torch.Tensor):
                observations = torch.stack(observations)

        if isinstance(observations[0],
                      np.ndarray) and len(observations[0].shape) > 1:
            observations = self._env_spec.observation_space.flatten_n(
                observations)
        elif isinstance(observations[0],
                        torch.Tensor) and len(observations[0].shape) > 1:
            observations = torch.flatten(observations, start_dim=1)

        if not isinstance(observations, torch.Tensor):
            observations = as_torch(observations)

        if self._state_include_action:
            assert self._prev_actions is not None
            prev_actions = as_torch(self._prev_actions).to(observations.device)
            all_input = torch.cat([observations, prev_actions], dim=-1)
        else:
            all_input = observations

        if self._prev_hiddens is None or self._prev_cells is None:
            init_hidden, init_cell = self._module.get_init_state(
                all_input.shape[0], all_input.device)
            self._prev_hiddens = init_hidden[0]
            self._prev_cells = init_cell[0]

        with torch.no_grad():
            (dist, step_mean, step_log_std, next_hidden, next_cell, _, _) = (
                self._module(
                    all_input.unsqueeze(1),
                    all_input,
                    self._prev_hiddens,
                    self._prev_cells))

            step_std = self._module._std_from_log_std(step_log_std)
            actions = torch.normal(step_mean, step_std)
        actions_np = actions.detach().cpu().numpy()

        prev_actions = self._prev_actions
        self._prev_actions = actions_np
        self._prev_hiddens = next_hidden.detach()
        self._prev_cells = next_cell.detach()

        agent_infos = dict(mean=step_mean.detach().cpu().numpy(),
                           log_std=step_log_std.detach().cpu().numpy())
        if self._state_include_action:
            agent_infos['prev_action'] = np.copy(prev_actions)
        return actions_np, agent_infos

    def clone(self, name):
        """Return a clone of the policy."""
        new_policy = self.__class__(
            name=name,
            env_spec=self._env_spec,
            hidden_dim=self._module._hidden_dim,
            output_nonlinearity=self._module._output_nonlinearity,
            learn_std=self._module._learn_std,
            init_std=self._module._init_std.exp().item(),
            min_std=self._module._min_std,
            max_std=self._module._max_std,
            std_parameterization=self._module._std_parameterization,
            std_share_network=self._module._std_share_network,
            hidden_state_init_trainable=isinstance(self._module._init_hidden,
                                                   torch.nn.Parameter),
            cell_state_init_trainable=isinstance(self._module._init_cell,
                                                 torch.nn.Parameter),
            layer_normalization=self._module._layer_normalization,
            state_include_action=self._state_include_action)
        new_policy.parameters = self.parameters
        return new_policy
