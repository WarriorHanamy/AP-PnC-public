"""Single-sample ReLU MLP with an explicit backward pass."""

from __future__ import annotations

import numpy as np


class ReluMlp:
    """Fully connected network. The last layer is linear."""

    def __init__(self, sizes: tuple[int, ...], rng: np.random.Generator):
        if len(sizes) < 2:
            raise ValueError("need at least one layer")
        self.sizes = sizes
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes, sizes[1:], strict=False):
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(rng.normal(0.0, scale, size=(fan_out, fan_in)))
            self.biases.append(np.zeros(fan_out))

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, tuple]:
        activation = np.asarray(features, dtype=float)
        if activation.shape != (self.sizes[0],):
            raise ValueError(f"expected features {(self.sizes[0],)}, got {activation.shape}")
        pre_activations: list[np.ndarray] = []
        activations = [activation]
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            linear = weight @ activation + bias
            pre_activations.append(linear)
            activation = linear if index == len(self.weights) - 1 else np.maximum(linear, 0.0)
            activations.append(activation)
        return activation, (activations, pre_activations)

    def backward(
        self, output_gradient: np.ndarray, cache: tuple
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        activations, pre_activations = cache
        gradient = np.asarray(output_gradient, dtype=float)
        if gradient.shape != (self.sizes[-1],):
            raise ValueError("output gradient has the wrong shape")
        grads: list[tuple[np.ndarray, np.ndarray]] = []
        for index in reversed(range(len(self.weights))):
            if index != len(self.weights) - 1:
                gradient = gradient * (pre_activations[index] > 0.0)
            incoming = activations[index]
            grads.append((gradient[:, None] * incoming[None, :], gradient.copy()))
            gradient = self.weights[index].T @ gradient
        grads.reverse()
        return grads

    def sgd(self, grads: list[tuple[np.ndarray, np.ndarray]], learning_rate: float) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning rate must be positive and finite")
        for weight, bias, (weight_grad, bias_grad) in zip(
            self.weights, self.biases, grads, strict=True
        ):
            weight -= learning_rate * weight_grad
            bias -= learning_rate * bias_grad
