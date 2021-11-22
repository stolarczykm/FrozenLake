import numpy as np
from gym.core import Env
from gym.spaces import Discrete

from solution.agent import Agent


class QLearningAgent(Agent):
    """
    Agent for enrionments with discrete action and observation spaces.
    Uses Q-learning with epsilon greedy exploration (optional) and step size
    decay.
    """

    def __init__(
        self,
        env: Env,
        discount: float = 0.9,
        initial_step_size: float = 0.5,
        min_step_size: float = 0.1,
        step_size_decay: float = 0.99,
        epsilon: float = 0.01,
        initial_q_value: float = 1.0,
        seed: int = 0,
    ):
        if not isinstance(env.action_space, Discrete):
            raise ValueError("env.action_space should be Discrete")

        if not isinstance(env.observation_space, Discrete):
            raise ValueError("env.observation_space should be Discrete")

        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount should be between 0 and 1 (inclusive)")

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon should be between 0 and 1 (inclusive)")

        if not 0.0 <= step_size_decay <= 1.0:
            raise ValueError("step_size_decay should be between 0 and 1 (inclusive)")

        if initial_step_size <= 0:
            raise ValueError("initial_step_size should be positive")

        if min_step_size <= 0:
            raise ValueError("min_step_size should be positive")

        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n

        self.discount = discount
        self.step_size = initial_step_size
        self.min_step_size = min_step_size
        self.step_size_decay = step_size_decay
        self.epsilon = epsilon

        self.q_values = np.full(
            (self.n_actions, self.n_states), initial_q_value, dtype="float"
        )

        self.rng = np.random.default_rng(seed)
        self.last_action = None
        self.last_state = None

    def _argmax(self, values: np.ndarray) -> int:
        values = np.asarray(values)
        max_ = np.max(values)
        max_ind = np.where(values == max_)[0]
        return self.rng.choice(max_ind)

    def select_action(self, state: int) -> tuple[int, float]:
        action_values = self.q_values[:, state]

        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.n_actions)
        else:
            action = self._argmax(action_values)

        return action, action_values[action]

    def start(self, state: int) -> int:
        action, _ = self.select_action(state)
        self.last_action = action
        self.last_state = state
        return action

    def step(self, reward: float, state: int) -> int:
        action, action_value = self.select_action(state)

        self.q_values[self.last_action, self.last_state] += self.step_size * (
            reward
            + self.discount * action_value
            - np.max(self.q_values[:, self.last_state])
        )
        self.last_action = action
        self.last_state = state

        return action

    def end(self, reward: float) -> None:
        self.q_values[self.last_action, self.last_state] += self.step_size * (
            reward - np.max(self.q_values[:, self.last_state])
        )
        self.step_size = np.maximum(
            self.min_step_size, self.step_size * self.step_size_decay
        )
