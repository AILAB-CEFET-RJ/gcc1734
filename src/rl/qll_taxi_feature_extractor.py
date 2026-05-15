import numpy as np
from rl.qll_feature_extractor import FeatureExtractor

special_locations_dict = {0: (0, 0), 1: (0, 4), 2: (4, 0), 3: (4, 3)}

class Actions:
    DOWN = 0
    UP = 1
    RIGHT = 2
    LEFT = 3
    PICK = 4
    DROP = 5


class TaxiFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for Taxi-v3 designed for didactic linear approximation.

    The features focus on the main structure of the task:
    - where the taxi is;
    - whether the passenger is already onboard;
    - how far the taxi is from the current objective;
    - whether pickup/drop actions are correct or illegal;
    - whether a movement attempts to cross a wall.
    """

    __actions_one_hot_encoding = {
        Actions.DOWN:  np.array([1, 0, 0, 0, 0, 0]),
        Actions.UP:    np.array([0, 1, 0, 0, 0, 0]),
        Actions.RIGHT: np.array([0, 0, 1, 0, 0, 0]),
        Actions.LEFT:  np.array([0, 0, 0, 1, 0, 0]),
        Actions.PICK:  np.array([0, 0, 0, 0, 1, 0]),
        Actions.DROP:  np.array([0, 0, 0, 0, 0, 1])
    }

    def __init__(self, env, debug=False):
        self.env = env
        self.debug = debug

        # Base features independent of action
        self.features_list = [
            self.f_bias,
            self.f_pos_row,
            self.f_pos_col,
            self.f_passenger_onboard,
            self.f_target_is_passenger,
            self.f_target_is_destination,
            self.f_dx_passenger,
            self.f_dy_passenger,
            self.f_dx_destination,
            self.f_dy_destination,
            self.f_dist_to_current_target,
            self.f_pick_correct,
            self.f_drop_correct,
            self.f_illegal_pick,
            self.f_illegal_drop,
            self.f_wall_bump
        ]
        for location_id in range(4):
            self.features_list.append(self._make_passenger_location_feature(location_id))
        for location_id in range(4):
            self.features_list.append(self._make_destination_location_feature(location_id))

    # ============================================================
    # Dimensionalidade
    # ============================================================
    def get_num_actions(self):
        return len(self.get_actions())

    def get_num_features(self):
        return len(self.features_list) * self.get_num_actions()

    def get_actions(self):
        return [Actions.DOWN, Actions.UP, Actions.RIGHT, Actions.LEFT, Actions.PICK, Actions.DROP]

    def get_action_one_hot_encoded(self, action):
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        return state in [0, 85, 410, 475]

    # ============================================================
    # Extração principal
    # ============================================================
    def get_features(self, state, action):
        """
        Extracts feature vector φ(s,a) combining base features and
        one-hot action encoding (via Kronecker product).
        """
        base_f = np.array([f(state, action) for f in self.features_list], dtype=float)

        one_hot = self.get_action_one_hot_encoded(action)
        full_vector = np.kron(one_hot, base_f)

        if self.debug and np.random.rand() < 0.001:  # 0.1% chance
            print(f"[DEBUG] state={state}, action={action}, features[:8]={full_vector[:8]}")

        return full_vector

    # ============================================================
    # Features base
    # ============================================================
    def f_bias(self, state, action):
        return 1.0

    def f_pos_row(self, state, action):
        l, _, _, _ = self.env.unwrapped.decode(state)
        return (l - 2) / 2.0  # centralizado em 0

    def f_pos_col(self, state, action):
        _, c, _, _ = self.env.unwrapped.decode(state)
        return (c - 2) / 2.0  # centralizado em 0

    def f_passenger_onboard(self, state, action):
        _, _, p, _ = self.env.unwrapped.decode(state)
        return 1.0 if p == 4 else 0.0

    def f_target_is_passenger(self, state, action):
        _, _, p, _ = self.env.unwrapped.decode(state)
        return 1.0 if p < 4 else 0.0

    def f_target_is_destination(self, state, action):
        _, _, p, _ = self.env.unwrapped.decode(state)
        return 1.0 if p == 4 else 0.0

    def f_dx_passenger(self, state, action):
        l, c, p, _ = self.env.unwrapped.decode(state)
        taxi = (l, c)
        passenger = (l, c) if p == 4 else special_locations_dict[p]
        dx = passenger[1] - taxi[1]
        return dx / 4.0

    def f_dy_passenger(self, state, action):
        l, c, p, _ = self.env.unwrapped.decode(state)
        taxi = (l, c)
        passenger = (l, c) if p == 4 else special_locations_dict[p]
        dy = passenger[0] - taxi[0]
        return dy / 4.0

    def f_dx_destination(self, state, action):
        l, c, _, d = self.env.unwrapped.decode(state)
        taxi = (l, c)
        dest = self.env.unwrapped.locs[d]
        dx = dest[1] - taxi[1]
        return dx / 4.0

    def f_dy_destination(self, state, action):
        l, c, _, d = self.env.unwrapped.decode(state)
        taxi = (l, c)
        dest = self.env.unwrapped.locs[d]
        dy = dest[0] - taxi[0]
        return dy / 4.0

    def f_dist_to_current_target(self, state, action):
        l, c, p, _ = self.env.unwrapped.decode(state)
        taxi = (l, c)
        _, _, _, d = self.env.unwrapped.decode(state)
        target = self.env.unwrapped.locs[d] if p == 4 else special_locations_dict[p]
        dist = self.__manhattanDistance(taxi, target)
        return dist / 8.0

    def f_pick_correct(self, state, action):
        l, c, p, _ = self.env.unwrapped.decode(state)
        if p < 4 and action == Actions.PICK:
            return 1.0 if (l, c) == self.env.unwrapped.locs[p] else 0.0
        return 0.0

    def f_drop_correct(self, state, action):
        l, c, p, d = self.env.unwrapped.decode(state)
        if p == 4 and action == Actions.DROP:
            return 1.0 if (l, c) == self.env.unwrapped.locs[d] else 0.0
        return 0.0

    def f_illegal_pick(self, state, action):
        l, c, p, _ = self.env.unwrapped.decode(state)
        if action != Actions.PICK:
            return 0.0
        if p == 4:
            return 1.0
        return 0.0 if (l, c) == self.env.unwrapped.locs[p] else 1.0

    def f_illegal_drop(self, state, action):
        l, c, p, d = self.env.unwrapped.decode(state)
        if action != Actions.DROP:
            return 0.0
        if p != 4:
            return 1.0
        if (l, c) != self.env.unwrapped.locs[d]:
            return 1.0
        return 0.0

    def f_wall_bump(self, state, action):
        l, c, _, _ = self.env.unwrapped.decode(state)
        border_bump = ((c == 0) and (action == Actions.LEFT)) or \
                      ((c == 4) and (action == Actions.RIGHT)) or \
                      ((l == 0) and (action == Actions.UP)) or \
                      ((l == 4) and (action == Actions.DOWN))
        internal_bump = ((l == 0) and (c == 1) and (action == Actions.RIGHT)) or \
                        ((l == 0) and (c == 2) and (action == Actions.LEFT)) or \
                        ((l == 3) and (c == 0) and (action == Actions.RIGHT)) or \
                        ((l == 3) and (c == 1) and (action == Actions.LEFT)) or \
                        ((l == 3) and (c == 2) and (action == Actions.RIGHT)) or \
                        ((l == 3) and (c == 3) and (action == Actions.LEFT))
        return 1.0 if (border_bump or internal_bump) else 0.0

    def _make_passenger_location_feature(self, location_id):
        def feature(state, action):
            _, _, p, _ = self.env.unwrapped.decode(state)
            return 1.0 if p == location_id else 0.0
        return feature

    def _make_destination_location_feature(self, location_id):
        def feature(state, action):
            _, _, _, d = self.env.unwrapped.decode(state)
            return 1.0 if d == location_id else 0.0
        return feature

    # ============================================================
    # Auxiliar
    # ============================================================
    @staticmethod
    def __manhattanDistance(xy1, xy2):
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
