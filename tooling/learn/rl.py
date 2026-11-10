"""On-policy actor-critic for the reduced quadrotor.

The observation is a SUPER CIRI corridor: padded h-polytope planes in the body
frame, plus the goal direction. The action is collective specific thrust and
body rates.
"""

from __future__ import annotations

import numpy as np

from tooling.learn.ciri import MAX_PLANES
from tooling.learn.net import ReluMlp
from tooling.learn.quadrotor import GRAVITY, QuadrotorState

RL_OBSERVATION_DIM = MAX_PLANES * 4 + 3
RL_SIZES = (RL_OBSERVATION_DIM, 32, 5)
ACTION_SCALE = np.array([4.0, 1.5, 1.5, 1.0])
ACTION_STD = 0.4


def initial_state(start: np.ndarray, goal: np.ndarray) -> QuadrotorState:
    delta = goal - start
    yaw = float(np.arctan2(delta[1], delta[0]))
    return QuadrotorState(start.astype(float).copy(), np.zeros(3), 0.0, 0.0, yaw)


def action_to_command(raw_action: np.ndarray) -> tuple[float, np.ndarray]:
    scaled = np.tanh(raw_action) * ACTION_SCALE
    thrust = float(GRAVITY + scaled[0])
    return thrust, scaled[1:]


def policy_update(
    net: ReluMlp,
    observations: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    learning_rate: float = 0.02,
    discount: float = 0.95,
) -> float:
    """REINFORCE with a learned value baseline. Returns the episode return."""
    returns = np.zeros(len(rewards))
    discounted = 0.0
    for index in reversed(range(len(rewards))):
        discounted = rewards[index] + discount * discounted
        returns[index] = discounted
    caches = []
    means = []
    for observation in observations:
        output, cache = net.forward(observation)
        means.append(output[:4])
        caches.append((output, cache))
    values = np.array([item[0][4] for item in caches])
    advantage = returns - values
    accumulated = [
        (np.zeros_like(weight), np.zeros_like(bias))
        for weight, bias in zip(net.weights, net.biases, strict=True)
    ]
    for index, (action, mean, (output, cache)) in enumerate(
        zip(actions, means, caches, strict=True)
    ):
        gradient = np.zeros_like(output)
        gradient[:4] = -advantage[index] * (action - mean) / ACTION_STD**2
        gradient[4] = 2.0 * (output[4] - returns[index])
        for slot, (weight_grad, bias_grad) in enumerate(net.backward(gradient, cache)):
            accumulated[slot] = (
                accumulated[slot][0] + weight_grad,
                accumulated[slot][1] + bias_grad,
            )
    scale = learning_rate / len(observations)
    net.sgd(accumulated, scale)
    return float(returns[0])


def sample_action(
    net: ReluMlp, observation: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    mean = net.forward(observation)[0][:4]
    action = mean + ACTION_STD * rng.normal(size=4)
    return action, mean
