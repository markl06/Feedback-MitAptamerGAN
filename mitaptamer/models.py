import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim, self.length = config.latent_dim, config.max_length
        layers, size = [], self.latent_dim
        for width in config.generator_widths:
            layers.extend([nn.Linear(size, width), nn.BatchNorm1d(width), nn.ReLU()])
            size = width
        layers.append(nn.Linear(size, self.length * 5))
        self.network = nn.Sequential(*layers)

    def forward(self, noise):
        return self.network(noise).reshape(-1, self.length, 5).softmax(dim=-1)


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers, size = [nn.Flatten()], config.max_length * 5
        for width in config.critic_widths:
            layers.extend([nn.Linear(size, width), nn.LeakyReLU(config.leaky_relu_slope)])
            size = width
        layers.append(nn.Linear(size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, encoded):
        # WGAN critic has no sigmoid and no batch normalization.
        return self.network(encoded).squeeze(-1)


class Evaluator(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_scale", torch.ones(input_dim))
        layers, size = [], input_dim
        for width in config.evaluator_widths:
            layers.extend([nn.Linear(size, width), nn.ReLU(), nn.Dropout(config.dropout)])
            size = width
        layers.append(nn.Linear(size, 1))
        self.network = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_normalization(self, training_features):
        # Fit exclusively on training data. Freeze for validation/test/generation.
        self.feature_mean.copy_(training_features.mean(0))
        self.feature_scale.copy_(training_features.std(0, unbiased=False).clamp_min(1e-6))

    def forward(self, features):
        normalized = (features - self.feature_mean) / self.feature_scale
        return self.network(normalized).squeeze(-1)

    def score(self, features):
        return self(features).sigmoid()


def gradient_penalty(critic, real, fake):
    alpha = torch.rand(real.shape[0], 1, 1, device=real.device)
    interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    scores = critic(interpolated)
    gradient = torch.autograd.grad(scores, interpolated, grad_outputs=torch.ones_like(scores),
                                   create_graph=True, only_inputs=True)[0]
    return ((gradient.flatten(1).norm(2, dim=1) - 1) ** 2).mean()
