import matplotlib.pyplot as plt
import numpy as np

from solution.qlearning import QLearningAgent


def _draw_arrow(i: int, j: int, best_action: int) -> None:
    dx, dy = [
        (-0.2, 0.0),
        (0.0, 0.2),
        (0.2, 0.0),
        (0.0, -0.2),
    ][best_action]
    plt.arrow(
        i + dx,
        j + dy,
        dx,
        dy,
        head_width=0.05,
        head_length=0.08,
        fc="k",
        ec="k",
    )


def vizualize_agent(output_file: str, agent: QLearningAgent) -> None:
    max_q_values = agent.q_values.reshape(4, 8, 8).max(axis=0)
    best_actions = agent.q_values.reshape(4, 8, 8).argmax(axis=0)

    map = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]
    plt.figure(figsize=(10, 10))
    plt.imshow(max_q_values)
    plt.colorbar()
    plt.title("Maximal q-values and best action for states")
    for i in range(8):
        for j in range(8):
            best_action = best_actions[j, i]
            letter = map[j][i]
            plt.text(i, j, letter, ha="center", va="center")
            if letter not in "HG":
                _draw_arrow(i, j, best_action)
    plt.savefig(output_file)


def vizualize_rewards(
    output_file: str, rewards: list[float], window_size: int = 100
) -> None:
    if window_size > len(rewards):
        raise ValueError("window_size should be less than the size of rewards")
    rewards_cumsum = np.cumsum(rewards)
    plt.plot(
        np.arange(window_size, len(rewards)),
        (rewards_cumsum[window_size:] - rewards_cumsum[:-window_size]) / window_size,
    )
    plt.savefig(output_file)
