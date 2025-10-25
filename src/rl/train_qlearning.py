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
from rl.qln import QLearningAgentReplay as QLearningAgentNeural
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
    penalties, rewards, successes = agent.train(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
    )
    return {
        "rewards": rewards,
        "penalties": penalties,
        "epsilons": getattr(agent, "epsilon_history", []),
        "successes": successes,
    }


def _train_neural(agent: QLearningAgentNeural, args: argparse.Namespace) -> Dict[str, Iterable[float]]:
    penalties, rewards, successes = agent.train(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
    )
    return {
        "rewards": rewards,
        "penalties": penalties,
        "epsilons": getattr(agent, "epsilon_history", []),
        "successes": successes,
    }


@dataclass
class AgentSpec:
    build_agent: AgentBuilder
    train_agent: TrainFn
    basename_fn: Callable[[str], str]
    filename_fn: Callable[[str], str]
    label: str


def _build_tabular(env, args: argparse.Namespace) -> QLearningAgentTabular:
    return QLearningAgentTabular(
        env=env,
        decay_rate=args.decay_rate,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        min_epsilon=args.min_epsilon,
        max_epsilon=args.max_epsilon,
        verbose=not args.quiet,
    )


def _build_linear(env, args: argparse.Namespace) -> QLearningAgentLinear:
    return QLearningAgentLinear(
        gym_env=env,
        epsilon_decay_rate=args.decay_rate,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
    )


def _build_neural(env, args: argparse.Namespace) -> QLearningAgentNeural:
    return QLearningAgentNeural(
        gym_env=env,
        epsilon_decay_rate=args.decay_rate,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
    )


AGENT_REGISTRY: MutableMapping[str, AgentSpec] = {
    "tabular": AgentSpec(
        build_agent=_build_tabular,
        train_agent=_train_tabular,
        basename_fn=lambda env_name: f"{env_name.lower()}-tql",
        filename_fn=lambda base: f"{base}-agent.pkl",
        label="Tabular",
    ),
    "linear": AgentSpec(
        build_agent=_build_linear,
        train_agent=_train_linear,
        basename_fn=lambda env_name: f"{env_name.lower()}-linear-agent",
        filename_fn=lambda base: f"{base}.pkl",
        label="Linear",
    ),
    "neural": AgentSpec(
        build_agent=_build_neural,
        train_agent=_train_neural,
        basename_fn=lambda env_name: f"{env_name.lower()}-neural-agent",
        filename_fn=lambda base: f"{base}.pkl",
        label="Neural",
    ),
}

AGENT_ALIASES: Dict[str, str] = {
    "replay": "neural",
}


def _prepare_parser() -> argparse.ArgumentParser:
    agent_choices = sorted(set(AGENT_REGISTRY.keys()) | set(AGENT_ALIASES.keys()))
    parser = argparse.ArgumentParser(description="Train Q-Learning agents (tabular, linear, neural)")
    parser.add_argument("--agent", choices=agent_choices, default="tabular",
                        help="Agent variant to train (alias: replay -> neural)")
    parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=6000, help="Number of training episodes")
    parser.add_argument("--decay_rate", type=float, default=0.0001, help="Epsilon decay rate")
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
    parser.add_argument("--min_epsilon", type=float, default=0.01,
                        help="Minimum epsilon during training (tabular agent)")
    parser.add_argument("--max_epsilon", type=float, default=1.0,
                        help="Maximum epsilon during training (tabular agent)")
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

    agent_key = AGENT_ALIASES.get(args.agent, args.agent)
    agent_spec = AGENT_REGISTRY[agent_key]
    agent = agent_spec.build_agent(env, args)

    print(f"\nTraining {agent_spec.label} Q-Learning agent on {args.env_name}...\n")

    start = timer()
    metrics = agent_spec.train_agent(agent, args)
    elapsed = timer() - start
    print(f"\nTraining finished in {elapsed:.2f} seconds.\n")

    base_name = agent_spec.basename_fn(args.env_name)
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
