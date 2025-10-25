from timeit import default_timer as timer
import numpy as np
import pickle
from rl.environment import Environment

class QLearningAgentTabular:


  def train(self, num_episodes: int):
    rewards_per_episode = []

    start_time = timer()  # Record the start time

    terminal_states = set()

    terminal_states_by_hand = set()

    for episode in range(num_episodes):
  
      terminated = False
      truncated = False

      state, _ = self.env.reset()
      state_id = self.env.get_state_id(state)
      state = state_id

      rewards_in_episode = []
      
      total_penalties = 0

      while not (terminated or truncated):
          
        # print(f"state: {state}")
        action = self.choose_action(state)

        # transição
        new_state, reward, terminated, truncated, info = self.env.step(action)
        new_state_id = self.env.get_state_id(new_state)
        new_state = new_state_id
        assert (not truncated)

        if reward == -10:
            total_penalties += 1

        self.update(state, action, reward, new_state)

        temp = self.env.id_to_state_dict[new_state]
        if(temp[2] == True):
          terminal_states_by_hand.add(temp)
        else:
          if temp[0] > 21:
            terminal_states_by_hand.add(temp)

        if (terminated or truncated):
          # Reduce epsilon to decrease the exploration over time
          self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
            np.exp(-self.decay_rate * episode)
          self.epsilons_.append(self.epsilon)

          if terminated:
            print(f"action: {action}, terminal state: {self.env.id_to_state_dict[new_state]}")
            terminal_states.add(self.env.id_to_state_dict[new_state])
        
        state = new_state
            
        rewards_in_episode.append(reward)

      mean_reward = np.mean(rewards_in_episode)
      rewards_per_episode.append(mean_reward)

      if episode % 1000 == 0:
        end_time = timer()  # Record the end time
        execution_time = end_time - start_time
        n_actions = len(rewards_in_episode)
        print(f"Stats for episode {episode}/{num_episodes}:") 
        print(f"\tNumber of actions: {n_actions}")
        print(f"\tMean reward: {mean_reward:#.2f}")
        print(f"\tExecution time: {execution_time:.2f}s")
        print(f"\tTotal penalties: {total_penalties}")
        start_time = end_time

    print("terminal_states:")
    print(terminal_states)
    for s in terminal_states:
      print(s)

    print("Quantities:")
    print(len(terminal_states_by_hand))
    print(len(terminal_states))

    print("Intersection:")
    print(terminal_states.intersection(terminal_states_by_hand))

    print("Diff:")
    print(terminal_states.difference(terminal_states_by_hand))

    return rewards_per_episode

  def save(self, filename):
    # open a file, where you ant to store the data
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump(self, file)

    # close the file
    file.close()

  @staticmethod
  def load_agent(filename):
    # open a file, where you stored the pickled data
    file = open(filename, 'rb')

    # dump information to that file
    agent = pickle.load(file)

    return agent
