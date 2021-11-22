import pytest
from tempfile import TemporaryDirectory

import numpy as np
from gym.core import Env
from gym.spaces import Discrete, Box

from solution.qlearning import QLearningAgent


class DummyEnv(Env):
    def __init__(self, action_space=Discrete(3), observation_space=Discrete(3)) -> None:
        self.action_space = action_space
        self.observation_space = observation_space


@pytest.fixture()
def env():
    return DummyEnv()


def test_save_load():
    env = DummyEnv()
    agent = QLearningAgent(env, initial_q_value=-1.0)

    with TemporaryDirectory() as temp_dir:
        path = f"{temp_dir}/agent.pkl"
        agent.save(path)
        restored_agent = agent.load(path)

    assert (agent.q_values == restored_agent.q_values).all()


def test_constructor_value_error():
    env = DummyEnv(Box(low=0, high=1, shape=(1,)), Box(low=0, high=1, shape=(1,)))
    with pytest.raises(ValueError):
        QLearningAgent(env)


def test_greedy_action_selection(env):
    agent = QLearningAgent(env, epsilon=0.0)  # No exploration
    agent.step_size = 0.0  # No learning
    agent.q_values = np.eye(3, dtype="float")

    for i in range(3):
        assert agent.start(i) == i
        assert agent.step(0.0, i) == i


def test_update(env):
    agent = QLearningAgent(env, epsilon=0.0, initial_step_size=1.0, initial_q_value=0.0)

    first_state = 0
    first_action = agent.start(0)
    agent.step(1.0, 1)

    # Step size == 1.0 and initial_q_value == 0.0:
    assert np.isclose(agent.q_values[first_action, first_state], 1.0)


def test_step_decay(env):
    agent = QLearningAgent(
        env, initial_step_size=1.0, min_step_size=0.1, step_size_decay=0.5
    )

    agent.end(1.0)
    assert np.isclose(agent.step_size, 1.0 * agent.step_size_decay)
