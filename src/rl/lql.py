from timeit import default_timer as timer
import pickle
import numpy as np
from environment import Environment

from taxi_feature_extractor import TaxiFeatureExtractor
from blackjack_feature_extractor import BlackjackFeatureExtractor

feature_extractors_dict = {
    "Blackjack-v1": BlackjackFeatureExtractor,
    "Taxi-v3": TaxiFeatureExtractor
}


class QLearningAgentLinear:
    """
    Q-Learning com aproximação linear:
        Q(s,a) = w · φ(s,a)
    """

    def __init__(self,
                 gym_env: Environment,
                 epsilon_decay_rate: float,
                 learning_rate: float,
                 gamma: float):
        self.env = gym_env
        env_name = self.env.get_id()

        self.fex = feature_extractors_dict[env_name](gym_env.env)

        # Inicialização dos pesos centrada em zero (com bias explícito)
        self.w = np.random.uniform(-0.01, 0.01, size=self.fex.get_num_features() + 1)

        self.steps = 0
        self.epsilon = 0.5
        self.max_epsilon = 0.5
        self.min_epsilon = 0.05
        self.epsilon_decay_rate = epsilon_decay_rate
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_history = []

    # =========================================================
    # Seleção de ações
    # =========================================================
    def choose_action(self, state, is_in_exploration_mode=True):
        if is_in_exploration_mode and np.random.rand() < self.epsilon:
            return self.env.get_random_action()
        return self.policy(state)

    def policy(self, state):
        return self.__get_action_and_value(state)[0]

    def get_value(self, state):
        return self.__get_action_and_value(state)[1]

    def __get_action_and_value(self, state):
        q_values = [self.get_qvalue(state, a) for a in range(self.env.get_num_actions())]
        best_action = int(np.argmax(q_values))
        best_value = float(np.max(q_values))
        return best_action, best_value

    # =========================================================
    # Q-value e features
    # =========================================================
    def get_features(self, state, action):
        f = self.fex.get_features(state, action).astype(float)
        if np.max(np.abs(f)) > 0:
            f = f / np.max(np.abs(f))  # normaliza em [-1,1]
        return np.concatenate(([1.0], f))  # adiciona bias

    def get_qvalue(self, state, action):
        features = self.get_features(state, action)
        return float(np.dot(self.w, features))

    # =========================================================
    # Atualização dos pesos
    # =========================================================
    def update(self, state, action, reward, next_state, terminated):
        next_value = 0 if terminated else self.get_value(next_state)
        td_error = reward + self.gamma * next_value - self.get_qvalue(state, action)
        td_error = np.clip(td_error, -10, 10)
        features = self.get_features(state, action)
        self.w += self.learning_rate * td_error * features
        self.w = np.clip(self.w, -1e3, 1e3)

    # =========================================================
    # Treinamento
    # =========================================================
    def train(self, num_episodes: int):
        rewards_per_episode = []
        penalties_per_episode = []
        cumulative_success = []

        successful_episodes = 0
        start_time = timer()

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            terminated = truncated = False
            total_reward = 0
            total_penalties = 0
            self.steps = 0

            while not (terminated or truncated):
                self.steps += 1
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                if reward == -10:
                    total_penalties += 1

                self.update(state, action, reward, next_state, terminated)
                total_reward += reward
                state = next_state

            # Decaimento linear de epsilon
            frac = episode / max(1, num_episodes - 1)
            self.epsilon = max(self.min_epsilon, self.max_epsilon * (1 - frac))
            self.epsilon_history.append(self.epsilon)

            if terminated:
                successful_episodes += 1

            rewards_per_episode.append(total_reward)
            penalties_per_episode.append(total_penalties)
            cumulative_success.append(successful_episodes)

            if episode % 50 == 0:
                elapsed = timer() - start_time
                print(f"Episode {episode}/{num_episodes} ({successful_episodes} successful)")
                print(f"\tTotal reward: {total_reward}")
                print(f"\tTotal steps: {self.steps}")
                print(f"\tEpsilon: {self.epsilon:.4f}")
                print(f"\tWeight norm: {np.linalg.norm(self.w):.4f}")
                print(f"\tElapsed: {elapsed:.2f}s\n")
                start_time = timer()

        return penalties_per_episode, rewards_per_episode, cumulative_success

    # =========================================================
    # Utilitários
    # =========================================================
    def get_weights(self):
        return self.w.copy()

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_agent(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
