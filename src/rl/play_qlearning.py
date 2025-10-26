import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import gymnasium as gym

from rl.environment_blackjack import BlackjackEnvironment
from rl.environment_taxi import TaxiEnvironment
from rl.qlt import QLearningAgentTabular
from rl.qll import QLearningAgentLinear
from rl.qln import QLearningAgentNeural


EnvironmentWrapper = Callable[[gym.Env], object]
ActionSelector = Callable[[object, object, object], int]


ENVIRONMENT_WRAPPERS: Dict[str, EnvironmentWrapper] = {
    "Taxi-v3": TaxiEnvironment,
    "Blackjack-v1": BlackjackEnvironment,
}


@dataclass
class AgentPlaySpec:
    label: str
    default_model_path: Callable[[str], str]
    load_agent: Callable[..., object]
    requires_env_for_load: bool
    set_env_after_load: bool
    select_action: ActionSelector


def _tabular_default_model(env_name: str) -> str:
    return f"{env_name.lower()}-tql-agent.pkl"


def _linear_default_model(env_name: str) -> str:
    return f"{env_name.lower()}-linear-agent.pkl"


def _neural_default_model(env_name: str) -> str:
    return f"{env_name.lower()}-neural-agent.pkl"


def _select_action_tabular(agent: QLearningAgentTabular, env, state) -> int:
    state_id = env.get_state_id(state)
    return agent.choose_action(state_id, is_in_exploration_mode=False)


def _select_action_policy(agent, _env, state) -> int:
    return int(agent.policy(state))


AGENT_REGISTRY: Dict[str, AgentPlaySpec] = {
    "tabular": AgentPlaySpec(
        label="Tabular",
        default_model_path=_tabular_default_model,
        load_agent=lambda path, **_: QLearningAgentTabular.load_agent(path),
        requires_env_for_load=False,
        set_env_after_load=True,
        select_action=_select_action_tabular,
    ),
    "linear": AgentPlaySpec(
        label="Linear",
        default_model_path=_linear_default_model,
        load_agent=lambda path, **_: QLearningAgentLinear.load_agent(path),
        requires_env_for_load=False,
        set_env_after_load=True,
        select_action=_select_action_policy,
    ),
    "neural": AgentPlaySpec(
        label="Neural",
        default_model_path=_neural_default_model,
        load_agent=lambda path, env, **_: QLearningAgentNeural.load_agent(path, env),
        requires_env_for_load=True,
        set_env_after_load=False,
        select_action=_select_action_policy,
    ),
}


def _resolve_render_mode(request_human: bool, env_name: str) -> Tuple[str, gym.Env]:
    desired = "human" if request_human else "ansi"
    fallbacks = ["ansi", "human"] if desired == "human" else ["human", "ansi"]

    for mode in [desired] + [m for m in fallbacks if m != desired]:
        try:
            env = gym.make(env_name, render_mode=mode)
            if hasattr(env, "env"):
                env = env.env
            return mode, env
        except Exception:
            continue
    raise RuntimeError(f"Unable to create environment {env_name} with render modes human/ansi")


def _load_agent(spec: AgentPlaySpec, model_path: Path, wrapped_env) -> object:
    if spec.requires_env_for_load:
        agent = spec.load_agent(str(model_path), env=wrapped_env)
    else:
        agent = spec.load_agent(str(model_path))
    if spec.set_env_after_load:
        agent.env = wrapped_env
    return agent


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a trained Q-Learning agent")
    parser.add_argument("--agent", choices=AGENT_REGISTRY.keys(), default="tabular",
                        help="Agent variant to load (tabular, linear, neural)")
    parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=5, help="Episodes to play")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment (if supported)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Custom path to the trained agent (.pkl)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    if args.num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")

    if args.env_name not in ENVIRONMENT_WRAPPERS:
        raise ValueError(f"Unsupported environment: {args.env_name}. "
                         f"Choose from {list(ENVIRONMENT_WRAPPERS.keys())}")

    spec = AGENT_REGISTRY[args.agent]

    model_path = Path(args.model_path) if args.model_path else Path(spec.default_model_path(args.env_name))
    if not model_path.exists():
        raise FileNotFoundError(f"Trained agent not found at {model_path}")

    render_mode, base_env = _resolve_render_mode(args.render, args.env_name)
    base_env.reset(seed=args.seed)
    wrapper_cls = ENVIRONMENT_WRAPPERS[args.env_name]
    wrapped_env = wrapper_cls(base_env)

    agent = _load_agent(spec, model_path, wrapped_env)

    print(f"\nRunning {spec.label} agent on {args.env_name} ({render_mode} mode)...\n")

    total_rewards = 0.0
    total_steps = 0

    for episode in range(args.num_episodes):
        state, _ = wrapped_env.reset()
        terminated = truncated = False
        episode_reward = 0.0
        steps = 0

        while not (terminated or truncated) and steps < args.max_steps:
            if args.render and render_mode == "human":
                base_env.render()
            elif args.render and render_mode == "ansi":
                print(base_env.render())

            action = spec.select_action(agent, wrapped_env, state)
            next_state, reward, terminated, truncated, _ = wrapped_env.step(action)

            episode_reward += reward
            steps += 1
            state = next_state

        total_rewards += episode_reward
        total_steps += steps

        print(f"Episode {episode + 1}/{args.num_episodes} — reward: {episode_reward:.2f}, steps: {steps}")

    avg_reward = total_rewards / args.num_episodes
    avg_length = total_steps / args.num_episodes if args.num_episodes else 0

    print("\n******** Summary ********")
    print(f"Average episode length: {avg_length:.1f}")
    print(f"Average total reward: {avg_reward:.2f}")
    print("*************************\n")

    base_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
