import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from timeit import default_timer as timer

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from rl.environment_blackjack import BlackjackEnvironment
from rl.environment_taxi import TaxiEnvironment
from rl.qln import QLearningAgentNeural as QLearningAgentNeural
from rl.qll import QLearningAgentLinear
from rl.qlt import QLearningAgentTabular


EnvironmentFactory = Callable[[gym.Env], object]
AgentBuilder = Callable[[object, argparse.Namespace], object]
TrainFn = Callable[[object, argparse.Namespace], Dict[str, Iterable[float]]]


environment_dict: Dict[str, EnvironmentFactory] = {
    "Blackjack-v1": BlackjackEnvironment,
    "Taxi-v3": TaxiEnvironment,
}


def _safe_savgol(values: np.ndarray) -> np.ndarray:
    if values.size <= 10:
        return values
    max_window = 501 if values.size > 600 else 101
    window = min(values.size, max_window)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    polyorder = min(3, window - 1)
    return savgol_filter(values, window_length=window, polyorder=polyorder)


def _train_tabular(agent: QLearningAgentTabular, args: argparse.Namespace) -> Dict[str, Iterable[float]]:
    history = agent.train(args.num_episodes)
    epsilons = history.get("epsilons", getattr(agent, "epsilons_", []))
    return {
        "rewards": history.get("rewards", []),
        "penalties": history.get("penalties", []),
        "epsilons": epsilons,
        "steps": history.get("steps", []),
    }


def _train_linear(agent: QLearningAgentLinear, args: argparse.Namespace) -> Dict[str, Iterable[float]]:
    result = agent.train(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
    )
    if isinstance(result, dict):
        return result
    # Backwards compatibility: older agents returned tuples
    penalties, rewards, successes = result
    return {
        "rewards": rewards,
        "penalties": penalties,
        "successes": successes,
        "epsilons": list(getattr(agent, "epsilon_history", [])),
    }


def _train_neural(agent: QLearningAgentNeural, args: argparse.Namespace) -> Dict[str, Iterable[float]]:
    result = agent.train(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
    )
    if isinstance(result, dict):
        return result
    penalties, rewards, successes = result
    return {
        "rewards": rewards,
        "penalties": penalties,
        "successes": successes,
        "epsilons": list(getattr(agent, "epsilon_history", [])),
    }


@dataclass
class AgentSpec:
    build_agent: AgentBuilder
    train_agent: TrainFn
    basename_fn: Callable[[str], str]
    filename_fn: Callable[[str], str]
    label: str
    default_min_epsilon: float
    default_max_epsilon: float


def _build_tabular(env, args: argparse.Namespace) -> QLearningAgentTabular:
    return QLearningAgentTabular(
        env=env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_decay_rate=args.epsilon_decay_rate,
        min_epsilon=args.min_epsilon,
        max_epsilon=args.max_epsilon,
        verbose=not args.quiet,
    )


def _build_linear(env, args: argparse.Namespace) -> QLearningAgentLinear:
    return QLearningAgentLinear(
        gym_env=env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_decay_rate=args.epsilon_decay_rate,
        min_epsilon=args.min_epsilon,
        max_epsilon=args.max_epsilon,
    )


def _build_neural(env, args: argparse.Namespace) -> QLearningAgentNeural:
    return QLearningAgentNeural(
        gym_env=env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epsilon_decay_rate=args.epsilon_decay_rate,
        min_epsilon=args.min_epsilon,
        max_epsilon=args.max_epsilon,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_every if args.checkpoint_every else None,
        checkpoint_prefix=args.checkpoint_prefix or getattr(args, "model_base_name", None),
    )


AGENT_REGISTRY: MutableMapping[str, AgentSpec] = {
    "tabular": AgentSpec(
        build_agent=_build_tabular,
        train_agent=_train_tabular,
        basename_fn=lambda env_name: f"{env_name.lower()}-tabular-agent",
        filename_fn=lambda base: f"{base}.pkl",
        label="Tabular",
        default_min_epsilon=0.01,
        default_max_epsilon=1.0,
    ),
    "linear": AgentSpec(
        build_agent=_build_linear,
        train_agent=_train_linear,
        basename_fn=lambda env_name: f"{env_name.lower()}-linear-agent",
        filename_fn=lambda base: f"{base}.pkl",
        label="Linear",
        default_min_epsilon=0.05,
        default_max_epsilon=1.0,
    ),
    "neural": AgentSpec(
        build_agent=_build_neural,
        train_agent=_train_neural,
        basename_fn=lambda env_name: f"{env_name.lower()}-neural-agent",
        filename_fn=lambda base: f"{base}.pkl",
        label="Neural",
        default_min_epsilon=0.05,
        default_max_epsilon=1.0,
    ),
}

def _prepare_parser() -> argparse.ArgumentParser:
    agent_choices = sorted(set(AGENT_REGISTRY.keys()))
    parser = argparse.ArgumentParser(description="Train Q-Learning agents (tabular, linear, neural)")
    parser.add_argument("--agent", choices=agent_choices, default="tabular",
                        help="Agent variant to train")
    parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=6000, help="Number of training episodes")
    parser.add_argument("--epsilon_decay_rate", type=float, default=0.0001, help="Epsilon decay rate")
    parser.add_argument("--learning_rate", type=float, default=0.7, help="Learning rate (alpha)")
    parser.add_argument("--gamma", type=float, default=0.618, help="Discount factor (gamma)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--plot", action="store_true", help="Show plots interactively after training")
    parser.add_argument("--quiet", action="store_true", help="Run without verbose agent logging (tabular only)")

    # Agent-specific knobs (optional for tabular)
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode (most relevant for linear/neural)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch size for neural agents")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden layer size for approximate agents")
    parser.add_argument("--min_epsilon", type=float, default=None,
                        help="Minimum epsilon during training (default depends on agent)")
    parser.add_argument("--max_epsilon", type=float, default=None,
                        help="Maximum epsilon during training (default depends on agent)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to store intermediate checkpoints (neural agent only)")
    parser.add_argument("--checkpoint_every", type=int, default=0,
                        help="Number of episodes between neural checkpoints")
    parser.add_argument("--checkpoint_prefix", type=str, default=None,
                        help="Filename prefix for neural checkpoints (defaults to model base name)")
    return parser


def _to_numpy(array_like: Iterable[float]) -> np.ndarray:
    return np.asarray(list(array_like), dtype=np.float32)


def _plot_learning_curves(base_name: str,
                          env_name: str,
                          agent_label: str,
                          rewards: np.ndarray,
                          epsilons: np.ndarray,
                          show: bool) -> None:
    smooth_rewards = _safe_savgol(rewards)

    plt.figure(figsize=(10, 4))
    plt.plot(smooth_rewards, label="Smoothed reward")
    plt.title(f"Learning Curve ({env_name}, {agent_label})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_name}-learning_curve.png")
    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epsilons, color="orange")
    plt.title(f"Epsilon Decay ({env_name}, {agent_label})")
    plt.xlabel("Episode")
    plt.ylabel("ε")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{base_name}-epsilons.png")
    if show:
        plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(smooth_rewards)
    ax[0].set_title("Learning Curve")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Reward")
    ax[0].grid(True)

    ax[1].plot(epsilons, color="orange")
    ax[1].set_title("Epsilon Decay")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("ε")
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{base_name}-summary.png")
    if show:
        plt.show()
    plt.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = _prepare_parser()
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.env_name not in environment_dict:
        raise ValueError(f"Unsupported environment: {args.env_name}. "
                         f"Choose from {list(environment_dict.keys())}")

    env = gym.make(args.env_name)
    if hasattr(env, "env"):
        env = env.env
    env.reset(seed=args.seed)
    env = environment_dict[args.env_name](env)

    agent_spec = AGENT_REGISTRY[args.agent]
    base_name = agent_spec.basename_fn(args.env_name)
    args.model_base_name = base_name

    min_epsilon = agent_spec.default_min_epsilon if args.min_epsilon is None else args.min_epsilon
    max_epsilon = agent_spec.default_max_epsilon if args.max_epsilon is None else args.max_epsilon
    if max_epsilon < min_epsilon:
        raise ValueError(f"max_epsilon ({max_epsilon}) cannot be smaller than min_epsilon ({min_epsilon})")
    args.min_epsilon = min_epsilon
    args.max_epsilon = max_epsilon

    agent = agent_spec.build_agent(env, args)

    print(f"\nTraining {agent_spec.label} Q-Learning agent on {args.env_name}...\n")

    start = timer()
    metrics = agent_spec.train_agent(agent, args)
    elapsed = timer() - start
    print(f"\nTraining finished in {elapsed:.2f} seconds.\n")

    model_path = agent_spec.filename_fn(base_name)
    agent.save(model_path)
    print(f"Saved agent to {model_path}")

    rewards = _to_numpy(metrics.get("rewards", []))
    epsilons = _to_numpy(metrics.get("epsilons", []))
    _plot_learning_curves(base_name, args.env_name, agent_spec.label, rewards, epsilons, args.plot)

    if not args.plot:
        plt.close("all")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
