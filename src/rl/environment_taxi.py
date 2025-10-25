from rl.environment import Environment

class TaxiEnvironment(Environment):
    def __init__(self, env):
        super().__init__(env)

    def get_num_states(self):
        return self.env.observation_space.n

    def get_num_actions(self):
        return self.env.action_space.n

    def get_state_id(self, state):
        return state

    def get_random_action(self):
        return self.env.action_space.sample()
