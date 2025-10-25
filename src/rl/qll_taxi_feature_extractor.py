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
    Enhanced feature extractor for the Taxi-v3 environment.

    Key additions:
    - Directional features (Δx, Δy) from taxi to passenger.
    - Exponential epsilon decay compatibility.
    - Scaled features with stronger contrast (tanh-based).
    - Bias-centered spatial coordinates for smoother gradients.
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
            self.f_dx_passenger,
            self.f_dy_passenger,
            self.f_dist_taxi_to_passenger,
            self.f_dist_taxi_to_destiny,
            self.f_pick_correct,
            self.f_drop_correct,
            self.f_wrong_boarding,
            self.f_wall_bump
        ]

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

        # Amplia contraste e mantém estabilidade
        base_f = np.tanh(base_f * 5.0)

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

    # --- Novas features direcionais ---
    def f_dx_passenger(self, state, action):
        """Diferença horizontal (direção) entre táxi e passageiro."""
        l, c, p, _ = self.env.unwrapped.decode(state)
        taxi = (l, c)
        passenger = (l, c) if p == 4 else special_locations_dict[p]
        dx = passenger[1] - taxi[1]
        return np.tanh(dx / 2.0)

    def f_dy_passenger(self, state, action):
        """Diferença vertical (direção) entre táxi e passageiro."""
        l, c, p, _ = self.env.unwrapped.decode(state)
        taxi = (l, c)
        passenger = (l, c) if p == 4 else special_locations_dict[p]
        dy = passenger[0] - taxi[0]
        return np.tanh(dy / 2.0)

    def f_dist_taxi_to_passenger(self, state, action):
        """Distância de Manhattan inversa entre táxi e passageiro."""
        l, c, p, _ = self.env.unwrapped.decode(state)
        taxi = (l, c)
        passenger = (l, c) if p == 4 else special_locations_dict[p]
        dist = self.__manhattanDistance(taxi, passenger)
        return 1.0 / (dist + 0.5)

    def f_dist_taxi_to_destiny(self, state, action):
        """Distância de Manhattan inversa entre táxi e destino."""
        l, c, p, d = self.env.unwrapped.decode(state)
        taxi = (l, c)
        dest = self.env.unwrapped.locs[d]
        dist = self.__manhattanDistance(taxi, dest)
        return 1.0 / (dist + 0.5)

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

    def f_wrong_boarding(self, state, action):
        l, c, p, d = self.env.unwrapped.decode(state)
        if p == 4 and action == Actions.PICK:
            return 1.0
        if p < 4 and action == Actions.DROP:
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

    # ============================================================
    # Auxiliar
    # ============================================================
    @staticmethod
    def __manhattanDistance(xy1, xy2):
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
