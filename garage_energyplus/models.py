import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.independent import Independent


class GaussianLSTMModel(nn.Module):
    """Gaussian LSTM model for continuous control."""

    def __init__(self,
                 input_dim,
                 output_dim,
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
                 layer_normalization=False):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._output_nonlinearity = output_nonlinearity
        self._learn_std = learn_std
        self._min_std = min_std
        self._max_std = max_std
        self._std_parameterization = std_parameterization
        self._std_share_network = std_share_network
        self._layer_normalization = layer_normalization

        if self._std_parameterization not in ('exp', 'softplus'):
            raise NotImplementedError

        if self._std_share_network:
            self._lstm = nn.LSTM(input_size=self._input_dim,
                                 hidden_size=self._hidden_dim,
                                 batch_first=True)
            self._output_layer = nn.Linear(self._hidden_dim,
                                           self._output_dim * 2)
        else:
            self._lstm = nn.LSTM(input_size=self._input_dim,
                                 hidden_size=self._hidden_dim,
                                 batch_first=True)
            self._output_layer = nn.Linear(self._hidden_dim,
                                           self._output_dim)

        if self._layer_normalization:
            self._layer_norm = nn.LayerNorm(self._hidden_dim)
        else:
            self._layer_norm = None

        init_std_param = torch.tensor([init_std]).log()
        if self._learn_std:
            self._init_std = torch.nn.Parameter(init_std_param)
        else:
            self._init_std = init_std_param
            self.register_buffer('init_std', self._init_std)

        self._min_std_param = self._max_std_param = None
        if self._min_std is not None:
            self._min_std_param = torch.tensor([self._min_std]).log()
            self.register_buffer('min_std_param', self._min_std_param)
        if self._max_std is not None:
            self._max_std_param = torch.tensor([self._max_std]).log()
            self.register_buffer('max_std_param', self._max_std_param)

        hidden_init = torch.zeros(self._hidden_dim)
        if hidden_state_init_trainable:
            self._init_hidden = nn.Parameter(hidden_init)
        else:
            self._init_hidden = hidden_init
            self.register_buffer('init_hidden', self._init_hidden)

        cell_init = torch.zeros(self._hidden_dim)
        if cell_state_init_trainable:
            self._init_cell = nn.Parameter(cell_init)
        else:
            self._init_cell = cell_init
            self.register_buffer('init_cell', self._init_cell)

    def to(self, *args, **kwargs):
        """Move the module to the specified device."""
        super().to(*args, **kwargs)
        buffers = dict(self.named_buffers())
        if not isinstance(self._init_std, torch.nn.Parameter):
            self._init_std = buffers['init_std']
        if not isinstance(self._init_hidden, torch.nn.Parameter):
            self._init_hidden = buffers['init_hidden']
        if not isinstance(self._init_cell, torch.nn.Parameter):
            self._init_cell = buffers['init_cell']
        self._min_std_param = buffers.get('min_std_param')
        self._max_std_param = buffers.get('max_std_param')

    def _clamp_log_std(self, log_std):
        if self._min_std_param is None and self._max_std_param is None:
            return log_std
        return log_std.clamp(
            min=(None if self._min_std_param is None else
                 self._min_std_param.item()),
            max=(None if self._max_std_param is None else
                 self._max_std_param.item()))

    def _std_from_log_std(self, log_std):
        if self._std_parameterization == 'exp':
            return log_std.exp()
        return log_std.exp().exp().add(1.).log()

    def _format_state(self, state, batch_size, device):
        if state is None:
            return None
        if state.dim() == 2:
            return state.unsqueeze(0).to(device)
        if state.dim() == 3:
            return state.to(device)
        raise ValueError('state must have shape (B, H) or (1, B, H)')

    def get_init_state(self, batch_size, device):
        hidden = self._init_hidden.to(device).expand(batch_size, -1)
        cell = self._init_cell.to(device).expand(batch_size, -1)
        return hidden.unsqueeze(0), cell.unsqueeze(0)

    def forward(self, full_input, step_input, step_hidden, step_cell):
        """Compute distribution and updated states.

        Args:
            full_input (torch.Tensor): (B, T, input_dim)
            step_input (torch.Tensor): (B, input_dim)
            step_hidden (torch.Tensor): (B, hidden_dim) or (1, B, hidden_dim)
            step_cell (torch.Tensor): (B, hidden_dim) or (1, B, hidden_dim)

        Returns:
            torch.distributions.Distribution: Gaussian distribution.
            torch.Tensor: step_mean (B, output_dim).
            torch.Tensor: step_log_std (B, output_dim).
            torch.Tensor: step_hidden (B, hidden_dim).
            torch.Tensor: step_cell (B, hidden_dim).
            torch.Tensor: init_hidden (hidden_dim,).
            torch.Tensor: init_cell (hidden_dim,).
        """
        batch_size = full_input.shape[0]
        device = full_input.device

        init_hidden, init_cell = self.get_init_state(batch_size, device)

        full_output, _ = self._lstm(full_input, (init_hidden, init_cell))
        if self._layer_norm is not None:
            full_output = self._layer_norm(full_output)

        step_hidden = self._format_state(step_hidden, batch_size, device)
        step_cell = self._format_state(step_cell, batch_size, device)
        if step_hidden is None or step_cell is None:
            step_hidden, step_cell = init_hidden, init_cell
        step_output, (next_hidden, next_cell) = self._lstm(
            step_input.unsqueeze(1), (step_hidden, step_cell))
        step_output = step_output[:, 0, :]
        if self._layer_norm is not None:
            step_output = self._layer_norm(step_output)

        if self._std_share_network:
            full_params = self._output_layer(full_output)
            step_params = self._output_layer(step_output)
            mean = full_params[..., :self._output_dim]
            log_std = full_params[..., self._output_dim:]
            step_mean = step_params[..., :self._output_dim]
            step_log_std = step_params[..., self._output_dim:]
        else:
            mean = self._output_layer(full_output)
            step_mean = self._output_layer(step_output)
            log_std = self._init_std.expand(
                full_output.shape[0], full_output.shape[1], self._output_dim)
            step_log_std = self._init_std.expand(
                step_output.shape[0], self._output_dim)

        if self._output_nonlinearity is not None:
            mean = self._output_nonlinearity(mean)
            step_mean = self._output_nonlinearity(step_mean)

        log_std = self._clamp_log_std(log_std)
        step_log_std = self._clamp_log_std(step_log_std)

        dist = Independent(Normal(mean, self._std_from_log_std(log_std)), 1)
        return (dist, step_mean, step_log_std, next_hidden[0], next_cell[0],
                self._init_hidden, self._init_cell)
    
