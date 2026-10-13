import torch
from torch import nn


class PersonalizedBatchNorm1d(nn.Module):
    """Normalize inputs, map through the normal CDF, then scale each feature."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.num_features:
            raise ValueError(
                f"Expected input with shape (batch, {self.num_features}), "
                f"got {tuple(inputs.shape)}"
            )

        use_batch_stats = self.training or not self.track_running_stats
        if use_batch_stats:
            if inputs.shape[0] <= 1:
                raise ValueError(
                    "Expected more than 1 value per feature while training"
                )
            mean = inputs.mean(dim=0)
            variance = inputs.var(dim=0, unbiased=False)

            if self.training and self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    if self.momentum is None:
                        factor = 1.0 / self.num_batches_tracked.item()
                    else:
                        factor = self.momentum

                    unbiased_variance = variance * (
                        inputs.shape[0] / (inputs.shape[0] - 1)
                    )
                    self.running_mean.lerp_(mean.detach(), factor)
                    self.running_var.lerp_(unbiased_variance.detach(), factor)
        else:
            mean = self.running_mean
            variance = self.running_var

        normalized = (inputs - mean) / torch.rsqrt(variance + self.eps)
        mask = 1 - torch.special.ndtr(normalized)
        mask = mask > 0.1

        if self.affine:
            normalized = (normalized * self.weight + self.bias)*mask

        
        return normalized
