import gym
from solution.agent import RandomAgent
from solution.train_agent import run


def test_run():
    env = gym.make("FrozenLake8x8-v1")
    agent = RandomAgent(env.action_space)
    rewards = run(env, agent, episodes=3)
    assert len(rewards) == 3
