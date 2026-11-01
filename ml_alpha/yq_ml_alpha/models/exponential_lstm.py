"""Shared stabilized exponential-gate LSTM cells and recurrent layers."""
from __future__ import annotations


class eLSTMCell:
    def __new__(cls, nn, input_size: int, hidden_size: int, eps: float = 1e-8):
        class Cell(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hidden_size = hidden_size
                self.eps = eps
                self.x_i = nn.Linear(input_size, hidden_size)
                self.h_i = nn.Linear(hidden_size, hidden_size, bias=False)
                self.x_f = nn.Linear(input_size, hidden_size)
                self.h_f = nn.Linear(hidden_size, hidden_size, bias=False)
                self.x_o = nn.Linear(input_size, hidden_size)
                self.h_o = nn.Linear(hidden_size, hidden_size, bias=False)
                self.x_g = nn.Linear(input_size, hidden_size)
                self.h_g = nn.Linear(hidden_size, hidden_size, bias=False)

            def forward(self, x_t, state):
                h_prev, c_prev, n_prev, m_prev = state
                i_pre = self.x_i(x_t) + self.h_i(h_prev)
                f_pre = self.x_f(x_t) + self.h_f(h_prev)
                o_pre = self.x_o(x_t) + self.h_o(h_prev)
                g_pre = self.x_g(x_t) + self.h_g(h_prev)

                m_t = torch.maximum(f_pre + m_prev, i_pre).detach()
                i_t = torch.exp(i_pre - m_t)
                f_t = torch.exp(f_pre + m_prev - m_t)
                o_t = torch.sigmoid(o_pre)
                g_t = torch.tanh(g_pre)

                c_t = f_t * c_prev + i_t * g_t
                n_t = f_t * n_prev + i_t
                h_t = o_t * (c_t / (n_t + self.eps))
                return h_t, (h_t, c_t, n_t, m_t)

        import torch

        return Cell()


class eLSTMLayer:
    def __new__(cls, nn, input_size: int, hidden_size: int, batch_first: bool = True, eps: float = 1e-8):
        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if not batch_first:
                    raise ValueError("eLSTMLayer only supports batch_first=True")
                self.hidden_size = hidden_size
                self.cell = eLSTMCell(nn, input_size, hidden_size, eps)

            def forward(self, x):
                batch_size, seq_len, _ = x.shape
                h = x.new_zeros(batch_size, self.hidden_size)
                c = x.new_zeros(batch_size, self.hidden_size)
                n = x.new_zeros(batch_size, self.hidden_size)
                m = x.new_zeros(batch_size, self.hidden_size)
                outputs = []
                state = (h, c, n, m)
                for step in range(seq_len):
                    h, state = self.cell(x[:, step, :], state)
                    outputs.append(h)
                return torch.stack(outputs, dim=1), h

        import torch

        return Layer()


class eLSTM:
    def __new__(
        cls,
        nn,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        eps: float = 1e-8,
    ):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if not batch_first:
                    raise ValueError("eLSTM only supports batch_first=True")
                self.layers = nn.ModuleList(
                    [
                        eLSTMLayer(
                            nn,
                            input_size if idx == 0 else hidden_size,
                            hidden_size,
                            batch_first=True,
                            eps=eps,
                        )
                        for idx in range(num_layers)
                    ]
                )
                self.dropout = nn.Dropout(dropout) if dropout > 0.0 and num_layers > 1 else None

            def forward(self, x):
                output = x
                h_last = None
                for idx, layer in enumerate(self.layers):
                    output, h_last = layer(output)
                    if self.dropout is not None and idx + 1 < len(self.layers):
                        output = self.dropout(output)
                return output, h_last

        return Model()
