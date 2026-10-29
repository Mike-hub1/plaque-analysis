"""Feature-tokenized Transformer regression, implemented from the stated methods."""
import math
import torch
from torch import nn


class NumericalTokenizer(nn.Module):
    def __init__(self, features, dimension):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(features, dimension))
        self.bias = nn.Parameter(torch.empty(features, dimension))
        self.cls = nn.Parameter(torch.empty(1, 1, dimension))
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -1 / math.sqrt(dimension), 1 / math.sqrt(dimension))

    def forward(self, values):
        tokens = values.unsqueeze(-1) * self.weight + self.bias
        return torch.cat([self.cls.expand(len(values), -1, -1), tokens], dim=1)


class ReGLU(nn.Module):
    def forward(self, values):
        linear, gate = values.chunk(2, dim=-1)
        return linear * torch.relu(gate)


class FeatureBlock(nn.Module):
    def __init__(self, config, *, first=False):
        super().__init__()
        d, width = config.dimension, int(config.dimension * config.feedforward_ratio)
        self.attention_norm = nn.Identity() if first else nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, config.heads, dropout=config.attention_dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(d)
        self.feedforward = nn.Sequential(nn.Linear(d, 2 * width), ReGLU(),
                                         nn.Dropout(config.feedforward_dropout), nn.Linear(width, d))
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, tokens):
        normalized = self.attention_norm(tokens)
        update, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.residual_dropout(update)
        return tokens + self.residual_dropout(self.feedforward(self.ff_norm(tokens)))


class FTRegressor(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        self.tokenizer = NumericalTokenizer(features, config.dimension)
        self.blocks = nn.ModuleList([FeatureBlock(config, first=i == 0) for i in range(config.layers)])
        self.head = nn.Sequential(nn.LayerNorm(config.dimension), nn.ReLU(), nn.Linear(config.dimension, 1))

    def forward(self, values):
        tokens = self.tokenizer(values)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, 0]).squeeze(-1)


def optimizer_groups(model, weight_decay):
    no_decay = {id(p) for module in model.modules() if isinstance(module, (nn.LayerNorm, NumericalTokenizer))
                for p in module.parameters()}
    decay, exempt = [], []
    for name, parameter in model.named_parameters():
        (exempt if id(parameter) in no_decay or "bias" in name else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": exempt, "weight_decay": 0.0}]
