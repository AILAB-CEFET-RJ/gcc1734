import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import gymnasium as gym
from rl.qlt import QLearningAgentTabular


def main():
    parser = argparse.ArgumentParser(description="Run a trained Tabular Q-Learning agent")
    parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    args = parser.parse_args()
    assert args.num_episodes > 0

    # --- Load trained agent ---
    model_path = args.env_name.lower() + "-tql-agent.pkl"
    agent = QLearningAgentTabular.load_agent(model_path)

    # --- Try to create a visual environment ---
    try:
        env = gym.make(args.env_name, render_mode="human").env
        render_mode = "human"
    except Exception:
        env = gym.make(args.env_name, render_mode="ansi").env
        render_mode = "ansi"

    agent.env = env

    total_rewards = 0
    total_actions = 0

    print(f"\nRunning agent on {args.env_name} ({render_mode} mode)...\n")

    for episode in range(args.num_episodes):
        state, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0
        steps = 0

        while not (terminated or truncated) and steps < args.max_steps:
            if render_mode == "human":
                env.render()
            elif render_mode == "ansi":
                print(env.render())

            action = agent.choose_action(state, is_in_exploration_mode=False)
            state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            steps += 1

        total_rewards += episode_reward
        total_actions += steps

        print(f"Episode {episode+1}/{args.num_episodes} finished — reward: {episode_reward}, steps: {steps}")

    print("\n******** Summary ********")
    print(f"Average episode length: {total_actions / args.num_episodes:.1f}")
    print(f"Average rewards: {total_rewards / args.num_episodes:.2f}")
    print("*************************\n")

    env.close()


if __name__ == "__main__":
    main()
