import argparse
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import gymnasium as gym
from rl.qll import QLearningAgentLinear
from rl.environment_taxi import TaxiEnvironment
from rl.environment_blackjack import BlackjackEnvironment


def main():
    parser = argparse.ArgumentParser(description="Run a trained Linear Q-Learning agent")
    parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true", help="Show environment visually (if supported)")
    args = parser.parse_args()
    assert args.num_episodes > 0

    model_path = f"{args.env_name}-lql-agent.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained agent not found: {model_path}")

    agent = QLearningAgentLinear.load_agent(model_path)

    # --- Escolhe render mode ---
    render_mode = "human" if args.render else "ansi"

    # --- Cria o ambiente Gymnasium ---
    try:
        gym_env = gym.make(args.env_name, render_mode=render_mode).env
    except Exception:
        gym_env = gym.make(args.env_name, render_mode="ansi").env
        render_mode = "ansi"

    # --- Aplica o wrapper apropriado ---
    env_wrappers = {
        "Taxi-v3": TaxiEnvironment,
        "Blackjack-v1": BlackjackEnvironment
    }

    if args.env_name not in env_wrappers:
        raise ValueError(f"Unsupported environment: {args.env_name}")

    agent.env = env_wrappers[args.env_name](gym_env)

    # --- Execução dos episódios ---
    total_rewards = 0
    total_steps = 0

    print(f"\nRunning agent on {args.env_name} ({render_mode} mode)...\n")

    for episode in range(args.num_episodes):
        state, _ = agent.env.reset()
        terminated = truncated = False
        episode_reward = 0
        steps = 0

        while not (terminated or truncated) and steps < args.max_steps:
            if render_mode == "human" and args.render:
                agent.env.env.render()
            elif render_mode == "ansi" and args.render:
                print(agent.env.env.render())

            action = agent.policy(state)
            next_state, reward, terminated, truncated, _ = agent.env.step(action)
            episode_reward += reward
            steps += 1
            state = next_state

        total_rewards += episode_reward
        total_steps += steps

        print(f"Episode {episode+1}/{args.num_episodes} — reward: {episode_reward:.1f}, steps: {steps}")

    print("\n******** Summary ********")
    print(f"Average episode length: {total_steps / args.num_episodes:.1f}")
    print(f"Average total reward: {total_rewards / args.num_episodes:.2f}")
    print("*************************\n")

    agent.env.env.close()


if __name__ == "__main__":
    main()
