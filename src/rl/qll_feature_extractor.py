from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def get_num_features(self):
        pass

    @abstractmethod
    def get_action_one_hot_encoded(self):
        pass

    # @abstractmethod
    # def get_terminal_states():
    #     pass

    @abstractmethod
    def is_terminal_state():
        pass

    @staticmethod
    def __manhattanDistance(xy1, xy2):
        '''
        Computes the Manhattan distance between two points.
        '''
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
