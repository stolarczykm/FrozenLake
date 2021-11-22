import os

import gym
from tqdm import tqdm
from gym.core import Env

from solution.agent import Agent
from solution.qlearning import QLearningAgent
from solution.vizualize import vizualize_agent, vizualize_rewards


def run(
    env: Env, agent: Agent, episodes: int = 1000, render: bool = False
) -> list[float]:
    obs = env.reset()
    action = agent.start(obs)
    rewards = []
    episode_counter = 0
    episode_reward = 0.0
    steps = 0
    progress_bar = tqdm(total=episodes)
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(action)
        steps += 1
        episode_reward += reward
        if not done:
            action = agent.step(reward, obs)
            continue

        agent.end(reward)
        rewards.append(episode_reward)
        steps = 0
        episode_reward = 0.0
        episode_counter += 1
        progress_bar.update()
        if episode_counter >= episodes:
            break
        obs = env.reset()
        action = agent.start(obs)

    progress_bar.close()

    return rewards


def main() -> None:
    env = gym.make("FrozenLake8x8-v1")
    agent = QLearningAgent(
        env,
        epsilon=0.0,
        initial_step_size=0.5,
        discount=0.98,
        initial_q_value=1.0,
        step_size_decay=0.999,
        min_step_size=0.1
    )
    rewards = run(env, agent, episodes=10000)
    os.makedirs("output", exist_ok=True)
    vizualize_rewards("output/rewards.png", rewards)
    vizualize_agent("output/agent.png", agent)
    agent.save("output/agent.pkl")


if __name__ == "__main__":
    main()
