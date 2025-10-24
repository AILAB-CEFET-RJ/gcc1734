import argparse
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from timeit import default_timer as timer
from scipy.signal import savgol_filter

from tql import QLearningAgentTabular
from taxi_environment import TaxiEnvironment
from blackjack_environment import BlackjackEnvironment


environment_dict = {
    "Blackjack-v1": BlackjackEnvironment,
    "Taxi-v3": TaxiEnvironment
}


def main():
    parser = argparse.ArgumentParser(description="Train a Tabular Q-Learning Agent")
    parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=6000, help="Number of episodes")
    parser.add_argument("--decay_rate", type=float, default=0.0001, help="Decay rate")
    parser.add_argument("--learning_rate", type=float, default=0.7, help="Learning rate (alpha)")
    parser.add_argument("--gamma", type=float, default=0.618, help="Discount factor (gamma)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--plot", action="store_true", help="Show plots interactively after training")
    args = parser.parse_args()

    # --- Reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- Environment setup ---
    if args.env_name not in environment_dict:
        raise ValueError(f"Unsupported environment: {args.env_name}. "
                         f"Choose from {list(environment_dict.keys())}")

    env = gym.make(args.env_name)
    if hasattr(env, "env"):  # unwrap if needed
        env = env.env

    env.reset(seed=args.seed)
    env = environment_dict[args.env_name](env)

    # --- Agent setup ---
    agent = QLearningAgentTabular(
        env=env,
        decay_rate=args.decay_rate,
        learning_rate=args.learning_rate,
        gamma=args.gamma
    )

    # --- Training ---
    start = timer()
    history = agent.train(args.num_episodes)
    elapsed = timer() - start
    print(f"\nTraining finished in {elapsed:.2f} seconds.")

    # --- Save model ---
    base_name = f"{args.env_name.lower()}-tql"
    agent.save(f"{base_name}-agent.pkl")

    # --- Rewards ---
    rewards = history["rewards"] if isinstance(history, dict) else history
    if len(rewards) > 10:
        smooth_rewards = savgol_filter(rewards, 501 if len(rewards) > 600 else 101, 3)
    else:
        smooth_rewards = rewards

    # --- Plot learning curve ---
    plt.figure(figsize=(10, 4))
    plt.plot(smooth_rewards, label="Smoothed reward")
    plt.title(f"Learning Curve ({args.env_name})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_name}-learning_curve.png")

    # --- Plot epsilon decay ---
    plt.figure(figsize=(10, 4))
    plt.plot(agent.epsilons_, color="orange")
    plt.title(f"Epsilon Decay ({args.env_name})")
    plt.xlabel("Episode")
    plt.ylabel("ε")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{base_name}-epsilons.png")

    # --- Combined summary plot ---
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(smooth_rewards)
    ax[0].set_title("Learning Curve")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Reward")
    ax[0].grid(True)

    ax[1].plot(agent.epsilons_, color="orange")
    ax[1].set_title("Epsilon Decay")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("ε")
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{base_name}-summary.png")

    # --- Optional interactive display ---
    if args.plot:
        plt.show()
    else:
        plt.close('all')


if __name__ == "__main__":
    main()
