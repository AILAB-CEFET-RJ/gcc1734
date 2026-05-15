import numpy as np
from rl.qll_feature_extractor import FeatureExtractor

class Actions:
  STICK = 0
  HIT = 1

class BlackjackFeatureExtractor(FeatureExtractor):
  __actions_one_hot_encoding = {
    Actions.STICK:   np.array([1,0]), 
    Actions.HIT:     np.array([0,1]) 
  }

  def __init__(self, env):
    '''
    Initializes the TaxiFeatureExtractor object. 
    It adds feature extraction methods to the features_list attribute.
    '''
    self.env = env
    self.features_list = []
    self.features_list.append(self.f0)
    self.features_list.append(self.f_player_sum_norm)
    self.features_list.append(self.f_dealer_card_norm)
    self.features_list.append(self.f_usable_ace)
    self.features_list.append(self.f_player_high_total)
    self.features_list.append(self.f_dealer_strong_card)
    self.features_list.append(self.f_safe_hit_margin)

  def get_num_features(self):
    '''
    Returns the number of features extracted by the feature extractor.
    '''
    return len(self.features_list) + self.get_num_actions()

  def get_num_actions(self):
    '''
    Returns the number of actions available in the environment.
    '''
    return len(self.get_actions())

  def get_action_one_hot_encoded(self, action):
    '''
    Returns the one-hot encoded representation of an action.
    '''
    return self.__actions_one_hot_encoding[action]

  def is_terminal_state(self, state):
    if state[2] == True:
      return True
    elif state[0] > 21:
      return True
    return False

  def get_actions(self):
    '''
    Returns a list of available actions in the environment.
    '''
    return [Actions.STICK, Actions.HIT]
  
  def get_features(self, state, action):
    '''
    Takes a state and an action as input and returns the feature vector for that state-action pair. 
    It calls the feature extraction methods and constructs the feature vector.
    '''
    feature_vector = np.zeros(len(self.features_list))
    for index, feature in enumerate(self.features_list):
      feature_vector[index] = feature(state, action)

    action_vector = self.get_action_one_hot_encoded(action)
    feature_vector = np.concatenate([feature_vector, action_vector])

    return feature_vector

  def f0(self, state, action):
    '''
    This is just the bias term.
    '''
    return 1.0

  def f_player_sum_norm(self, state, action):
    player_sum, _, _ = state
    return player_sum / 21.0

  def f_dealer_card_norm(self, state, action):
    _, dealer_card, _ = state
    return dealer_card / 10.0

  def f_usable_ace(self, state, action):
    _, _, usable_ace = state
    return 1.0 if usable_ace else 0.0

  def f_player_high_total(self, state, action):
    player_sum, _, _ = state
    return 1.0 if player_sum >= 17 else 0.0

  def f_dealer_strong_card(self, state, action):
    _, dealer_card, _ = state
    return 1.0 if dealer_card >= 7 else 0.0

  def f_safe_hit_margin(self, state, action):
    player_sum, _, usable_ace = state
    threshold = 21 if usable_ace else 17
    margin = max(0, threshold - player_sum)
    return margin / 10.0

