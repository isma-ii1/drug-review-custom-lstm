"""Appendix equations implemented with nn.Linear, never nn.LSTM."""
from typing import Tuple

import torch
from torch import Tensor, nn


class LSTMCell(nn.Module):
    """Vanilla, CIFG, or WMC cell. WMC output gate reads UPDATED c."""

    def __init__(self, input_size: int, hidden_size: int, variant: str = "vanilla"):
        super().__init__()
        if variant not in {"vanilla", "cifg", "wmc"}:
            raise ValueError(f"Unknown cell: {variant}")
        self.hidden_size = hidden_size
        self.kind = {"vanilla": 0, "cifg": 1, "wmc": 2}[variant]
        self.gates = 3 if self.kind == 1 else 4
        # Gate order: (f, o, g) for CIFG; (i, f, o, g) otherwise.
        self.input_linear = nn.Linear(input_size, self.gates * hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, self.gates * hidden_size, bias=False)
        # Empty identity modules keep the module tree scriptable for every variant.
        self.memory_i = nn.Linear(hidden_size, hidden_size, bias=False) if self.kind == 2 else nn.Identity()
        self.memory_f = nn.Linear(hidden_size, hidden_size, bias=False) if self.kind == 2 else nn.Identity()
        self.memory_o = nn.Linear(hidden_size, hidden_size, bias=False) if self.kind == 2 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for block in self.input_linear.weight.chunk(self.gates, dim=0):
            nn.init.xavier_uniform_(block)
        for block in self.hidden_linear.weight.chunk(self.gates, dim=0):
            nn.init.orthogonal_(block)
        nn.init.zeros_(self.input_linear.bias)
        start = 0 if self.kind == 1 else self.hidden_size
        with torch.no_grad():
            self.input_linear.bias[start:start + self.hidden_size].fill_(1.0)
        if self.kind == 2:
            for module in (self.memory_i, self.memory_f, self.memory_o):
                nn.init.xavier_uniform_(module.weight)

    @torch.jit.export
    def step_projected(self, projected: Tensor, h: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        z = projected + self.hidden_linear(h)
        if self.kind == 1:
            fz, oz, gz = z.chunk(3, dim=-1)
            f = torch.sigmoid(fz)
            c_new = f * c + (1.0 - f) * torch.tanh(gz)
            return torch.sigmoid(oz) * torch.tanh(c_new), c_new
        iz, fz, oz, gz = z.chunk(4, dim=-1)
        if self.kind == 2:
            iz = iz + torch.tanh(self.memory_i(c))
            fz = fz + torch.tanh(self.memory_f(c))
        c_new = torch.sigmoid(fz) * c + torch.sigmoid(iz) * torch.tanh(gz)
        if self.kind == 2:
            oz = oz + torch.tanh(self.memory_o(c_new))
        return torch.sigmoid(oz) * torch.tanh(c_new), c_new

    def forward(self, x: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return self.step_projected(self.input_linear(x), state[0], state[1])


class ReviewClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_size: int = 64,
                 variant: str = "vanilla", dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.1)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        self.dropout = nn.Dropout(float(dropout))
        self.cell = LSTMCell(embedding_dim, hidden_size, variant)
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, tokens: Tensor, lengths: Tensor) -> Tensor:
        x = self.dropout(self.embedding(tokens))
        # Compute input projections for the complete batch once, then recur.
        projected = self.cell.input_linear(x)
        h = x.new_zeros((x.size(0), self.hidden_size))
        c = x.new_zeros((x.size(0), self.hidden_size))
        # Unbind once: separate per-step slice operations would materialize a
        # complete B x T x gates gradient buffer at every recurrent step.
        for t, projected_t in enumerate(projected.unbind(1)):
            h_next, c_next = self.cell.step_projected(projected_t, h, c)
            valid = (lengths > t).unsqueeze(1)
            h = torch.where(valid, h_next, h)
            c = torch.where(valid, c_next, c)
        return self.classifier(self.dropout(h))


def parameter_counts(model):
    return {
        "total": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "embedding": sum(p.numel() for p in model.embedding.parameters()),
        "recurrent": sum(p.numel() for p in model.cell.parameters()),
        "head": sum(p.numel() for p in model.classifier.parameters()),
    }
