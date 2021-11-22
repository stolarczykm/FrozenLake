import abc
import pickle

from gym.spaces.space import Space


class Agent(abc.ABC):
    """
    Base class of agents for enrionments with discrete action and observation
    spaces
    """

    @abc.abstractmethod
    def start(self, obs: int) -> int:
        """Method called at the start of the episode. Returns action"""
        pass

    @abc.abstractmethod
    def step(self, reward: float, obs: int) -> int:
        """Method called after environment step. Updates agent and returns action"""
        pass

    @abc.abstractmethod
    def end(self, reward: float) -> None:
        """Method called after episode end. Updates agent."""
        pass

    def save(self, path: str) -> None:
        """Saves agent to file."""
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str) -> "Agent":
        with open(path, "rb") as file:
            obj = pickle.load(file)
            if not isinstance(obj, Agent):
                raise TypeError(f"{file} is not a pickled Agent")
            return obj


class RandomAgent(Agent):
    def __init__(self, action_space: Space) -> None:
        self.action_space: Space = action_space

    def start(self, obs: int) -> int:
        return self.action_space.sample()

    def step(self, reward: float, obs: int) -> int:
        return self.action_space.sample()

    def end(self, reward: float) -> None:
        pass
