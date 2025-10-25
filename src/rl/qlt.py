import numpy as np
import pickle
import logging
from timeit import default_timer as timer
from rl.environment import Environment

logger = logging.getLogger(__name__)

class QLearningAgentTabular:
    """
    Q-Learning agent for discrete environments.
    """

    def __init__(
        self, 
        env: Environment, 
        decay_rate: float, 
        learning_rate: float, 
        gamma: float,
        min_epsilon: float = 0.01,
        max_epsilon: float = 1.0,
        verbose: bool = True
    ):
        self.env = env
        self.q_table = np.zeros((env.get_num_states(), env.get_num_actions()))
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilons_ = []
        self.verbose = verbose
        self.history = {
            "rewards": [],
            "penalties": [],
            "steps": [],
            "epsilons": self.epsilons_
        }

        if self.verbose:
            print(f"Q-table initialized with shape {self.q_table.shape}")

    def choose_action(self, state: int, is_in_exploration_mode: bool = True) -> int:
        """ε-greedy action selection."""
        if is_in_exploration_mode and np.random.rand() < self.epsilon:
            return np.random.randint(self.env.get_num_actions())
        return np.argmax(self.q_table[state, :])

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]"""
        best_next_q = np.max(self.q_table[next_state, :])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def _epsilon_decay(self, episode: int) -> None:
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
                       np.exp(-self.decay_rate * episode)
        self.epsilons_.append(self.epsilon)

    def _run_episode(self, episode: int):
        state, _ = self.env.reset()
        state = self.env.get_state_id(state)
        total_reward, penalties, steps = 0.0, 0, 0

        while True:
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = self.env.get_state_id(next_state)
            self.update(state, action, reward, next_state)

            total_reward += reward
            if reward < 0:
                penalties += 1
            steps += 1
            state = next_state

            if terminated or truncated:
                self._epsilon_decay(episode)
                break

        return total_reward, penalties, steps

    def train(self, num_episodes: int):
        start_time = timer()
        if self.verbose:
            print("\n===========================================")
            print("Q-table before training:")
            print(self.q_table)

        for episode in range(num_episodes):
            total_reward, penalties, steps = self._run_episode(episode)
            self.history["rewards"].append(total_reward)
            self.history["penalties"].append(penalties)
            self.history["steps"].append(steps)

            if self.verbose and episode % 100 == 0:
                elapsed = timer() - start_time
                print(f"Episode {episode}/{num_episodes}")
                print(f"\tSteps: {steps}")
                print(f"\tReward: {total_reward:.2f}")
                print(f"\tPenalties: {penalties}")
                print(f"\tEpsilon: {self.epsilon:.4f}")
                print(f"\tElapsed: {elapsed:.2f}s")
                start_time = timer()

        if self.verbose:
            print("\n===========================================")
            print("Q-table after training:")
            print(self.q_table)

        return self.history

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_agent(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)
