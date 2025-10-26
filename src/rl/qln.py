import random
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from timeit import default_timer as timer
import pickle
from rl.environment import Environment

from rl.qll_taxi_feature_extractor import TaxiFeatureExtractor
from rl.qll_blackjack_feature_extractor import BlackjackFeatureExtractor

feature_extractors_dict = {
    "Blackjack-v1": BlackjackFeatureExtractor,
    "Taxi-v3": TaxiFeatureExtractor
}


# ============================================================
#  Rede neural base
# ============================================================
class QNetwork(nn.Module):
    """Rede Q(s,a) com duas camadas ocultas (ReLU)."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
#  Agente DQN simplificado com Replay e vetorização
# ============================================================
class QLearningAgentNeural:
    """
    Q-Learning com aproximação neural, experience replay
    e vetorização no cálculo de Q(s,a).
    """
    def __init__(self,
                 gym_env: Environment,
                 learning_rate: float,
                 gamma: float,
                 hidden_dim: int = 64,
                 replay_size: int = 50000,
                 batch_size: int = 64,
                 train_every: int = 4,
                 device: str = None,
                 epsilon_decay_rate: float = 0.003,
                 min_epsilon: float = 0.05,
                 max_epsilon: float = 1.0,
                 checkpoint_dir: Optional[str] = None,
                 checkpoint_interval: Optional[int] = None,
                 checkpoint_prefix: Optional[str] = None):

        self.env = gym_env
        env_name = getattr(self.env, "get_id", lambda: None)()

        if env_name not in feature_extractors_dict:
            raise ValueError(f"Unsupported environment: {env_name}")

        self.fex = feature_extractors_dict[env_name](gym_env.env)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = self.fex.get_num_features()
        self.model = QNetwork(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_history = []
        self.steps = 0

        # Replay buffer
        self.replay_buffer = deque(maxlen=replay_size)
        self.batch_size = batch_size
        self.train_every = train_every
        self.checkpoint_interval = checkpoint_interval if checkpoint_interval and checkpoint_interval > 0 else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        default_prefix = env_name.lower() if isinstance(env_name, str) else "checkpoint"
        self.checkpoint_prefix = checkpoint_prefix or default_prefix
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Seleção de ações
    # =========================================================
    def choose_action(self, state, is_in_exploration_mode=True):
        if is_in_exploration_mode and np.random.rand() < self.epsilon:
            return self.env.get_random_action()
        return self.policy(state)

    def policy(self, state):
        q_values = self.get_qvalues_vectorized(state)
        return int(np.argmax(q_values))

    def get_value(self, state):
        q_values = self.get_qvalues_vectorized(state)
        return float(np.max(q_values))

    # =========================================================
    # Versão vetorizada do cálculo Q(s,a)
    # =========================================================
    def get_qvalues_vectorized(self, state):
        actions = range(self.env.get_num_actions())
        features = np.array([self.fex.get_features(state, a) for a in actions], dtype=np.float32)
        x = torch.as_tensor(features, device=self.device)
        with torch.no_grad():
            q_values = self.model(x).squeeze().cpu().numpy()
        return q_values

    # =========================================================
    # Replay buffer e atualização
    # =========================================================
    def store_transition(self, s, a, r, s2, done):
        self.replay_buffer.append((s, a, r, s2, done))

    def update_from_replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        features = np.array([self.fex.get_features(s, a) for s, a in zip(states, actions)], dtype=np.float32)
        x = torch.as_tensor(features, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device)

        # Alvos TD
        with torch.no_grad():
            next_values = torch.as_tensor(
                [self.get_value(s2) for s2 in next_states],
                dtype=torch.float32, device=self.device
            )
            targets = rewards + self.gamma * next_values * (~dones)

        self.optimizer.zero_grad()
        q_preds = self.model(x).squeeze()
        loss = self.loss_fn(q_preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

    # =========================================================
    # Treinamento
    # =========================================================
    def train(self, num_episodes: int, max_steps_per_episode: int = 500):
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

            while not (terminated or truncated) and self.steps < max_steps_per_episode:
                self.steps += 1
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if reward == -10:
                    total_penalties += 1

                self.store_transition(state, action, reward, next_state, done)

                # Atualiza modelo a cada N passos
                if self.steps % self.train_every == 0:
                    self.update_from_replay()

                total_reward += reward
                state = next_state

            # Decaimento exponencial de epsilon
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
                           np.exp(-self.epsilon_decay_rate * episode)
            self.epsilon_history.append(self.epsilon)

            if terminated:
                successful_episodes += 1

            rewards_per_episode.append(total_reward)
            penalties_per_episode.append(total_penalties)
            cumulative_success.append(successful_episodes)
            self._maybe_checkpoint(episode)

            if episode % 50 == 0:
                elapsed = timer() - start_time
                print(f"Episode {episode}/{num_episodes} ({successful_episodes} successful)")
                print(f"\tTotal reward: {total_reward}")
                print(f"\tEpsilon: {self.epsilon:.4f}")
                print(f"\tReplay size: {len(self.replay_buffer)}")
                print(f"\tElapsed: {elapsed:.2f}s\n")
                start_time = timer()

        return {
            "penalties": penalties_per_episode,
            "rewards": rewards_per_episode,
            "successes": cumulative_success,
            "epsilons": list(self.epsilon_history),
        }

    # =========================================================
    # Utilitários
    # =========================================================
    def _should_checkpoint(self, episode: int) -> bool:
        return (
            self.checkpoint_dir is not None
            and self.checkpoint_interval is not None
            and (episode + 1) % self.checkpoint_interval == 0
        )

    def _checkpoint_path(self, episode: int) -> Path:
        filename = f"{self.checkpoint_prefix}-ep{episode + 1:05d}.pkl"
        if self.checkpoint_dir is None:
            raise RuntimeError("Checkpoint directory not configured")
        return self.checkpoint_dir / filename

    def _maybe_checkpoint(self, episode: int) -> None:
        if not self._should_checkpoint(episode):
            return
        path = self._checkpoint_path(episode)
        self.save(path)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump({
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "params": {
                    "gamma": self.gamma,
                    "learning_rate": self.learning_rate,
                    "epsilon_decay_rate": self.epsilon_decay_rate
                }
            }, f)

    @staticmethod
    def load_agent(filename, gym_env):
        checkpoint = pickle.load(open(filename, "rb"))
        agent = QLearningAgentNeural(
            gym_env=gym_env,
            epsilon_decay_rate=checkpoint["params"]["epsilon_decay_rate"],
            learning_rate=checkpoint["params"]["learning_rate"],
            gamma=checkpoint["params"]["gamma"]
        )
        agent.model.load_state_dict(checkpoint["model_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent.epsilon = checkpoint["epsilon"]
        return agent
