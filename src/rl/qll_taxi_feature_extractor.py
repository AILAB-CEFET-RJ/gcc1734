import numpy as np
from rl.qll_feature_extractor import FeatureExtractor


class Actions:
    DOWN = 0
    UP = 1
    RIGHT = 2
    LEFT = 3
    PICK = 4
    DROP = 5


class TaxiFeatureExtractor(FeatureExtractor):
    """
    Sparse feature extractor for Taxi-v3.

    The representation keeps the model linear, but adds enough structure
    to distinguish:
    - the taxi exact cell;
    - passenger status/location;
    - destination;
    - the current target depot;
    - whether pickup/drop actions are correct or illegal;
    - whether a movement hits a wall.
    """

    GRID_ROWS = 5
    GRID_COLS = 5
    NUM_TAXI_CELLS = GRID_ROWS * GRID_COLS
    NUM_PASSENGER_STATES = 5  # 4 depots + onboard
    NUM_DEPOTS = 4

    __actions_one_hot_encoding = {
        Actions.DOWN: np.array([1, 0, 0, 0, 0, 0]),
        Actions.UP: np.array([0, 1, 0, 0, 0, 0]),
        Actions.RIGHT: np.array([0, 0, 1, 0, 0, 0]),
        Actions.LEFT: np.array([0, 0, 0, 1, 0, 0]),
        Actions.PICK: np.array([0, 0, 0, 0, 1, 0]),
        Actions.DROP: np.array([0, 0, 0, 0, 0, 1]),
    }

    def __init__(self, env, debug=False):
        self.env = env
        self.debug = debug

        # Base feature layout:
        #  0: bias
        #  1..25: taxi cell one-hot
        # 26..30: passenger location/onboard one-hot
        # 31..34: destination one-hot
        # 35..36: phase one-hot (target passenger / target destination)
        # 37..40: current target depot one-hot
        # 41..140: taxi cell x current target depot interaction
        # 141..145: action-context features
        self.base_dim = 146

    def get_num_actions(self):
        return len(self.get_actions())

    def get_num_features(self):
        return self.base_dim * self.get_num_actions()

    def get_actions(self):
        return [Actions.DOWN, Actions.UP, Actions.RIGHT, Actions.LEFT, Actions.PICK, Actions.DROP]

    def get_action_one_hot_encoded(self, action):
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        return state in [0, 85, 410, 475]

    def get_features(self, state, action):
        row, col, passenger, destination = self.env.unwrapped.decode(state)
        taxi_cell = self._taxi_cell_index(row, col)
        current_target = destination if passenger == 4 else passenger

        base_f = np.zeros(self.base_dim, dtype=float)
        base_f[0] = 1.0

        # Exact taxi position.
        base_f[1 + taxi_cell] = 1.0

        # Passenger state and destination.
        base_f[26 + passenger] = 1.0
        base_f[31 + destination] = 1.0

        # Phase of the task and target depot.
        if passenger == 4:
            base_f[36] = 1.0
        else:
            base_f[35] = 1.0
        base_f[37 + current_target] = 1.0

        # Interaction between taxi cell and the currently relevant target.
        base_f[41 + current_target * self.NUM_TAXI_CELLS + taxi_cell] = 1.0

        # Action-context features.
        base_f[141] = self._pick_correct(row, col, passenger, action)
        base_f[142] = self._drop_correct(row, col, passenger, destination, action)
        base_f[143] = self._illegal_pick(row, col, passenger, action)
        base_f[144] = self._illegal_drop(row, col, passenger, destination, action)
        base_f[145] = self._wall_bump(row, col, action)

        one_hot = self.get_action_one_hot_encoded(action)
        full_vector = np.kron(one_hot, base_f)

        if self.debug and np.random.rand() < 0.001:
            print(f"[DEBUG] state={state}, action={action}, nnz={np.count_nonzero(full_vector)}")

        return full_vector

    def _pick_correct(self, row, col, passenger, action):
        if action != Actions.PICK or passenger >= self.NUM_DEPOTS:
            return 0.0
        return 1.0 if (row, col) == self.env.unwrapped.locs[passenger] else 0.0

    def _drop_correct(self, row, col, passenger, destination, action):
        if action != Actions.DROP or passenger != 4:
            return 0.0
        return 1.0 if (row, col) == self.env.unwrapped.locs[destination] else 0.0

    def _illegal_pick(self, row, col, passenger, action):
        if action != Actions.PICK:
            return 0.0
        if passenger == 4:
            return 1.0
        return 0.0 if (row, col) == self.env.unwrapped.locs[passenger] else 1.0

    def _illegal_drop(self, row, col, passenger, destination, action):
        if action != Actions.DROP:
            return 0.0
        if passenger != 4:
            return 1.0
        return 0.0 if (row, col) == self.env.unwrapped.locs[destination] else 1.0

    def _wall_bump(self, row, col, action):
        border_bump = (
            ((col == 0) and (action == Actions.LEFT)) or
            ((col == 4) and (action == Actions.RIGHT)) or
            ((row == 0) and (action == Actions.UP)) or
            ((row == 4) and (action == Actions.DOWN))
        )
        internal_bump = (
            ((row == 0) and (col == 1) and (action == Actions.RIGHT)) or
            ((row == 0) and (col == 2) and (action == Actions.LEFT)) or
            ((row == 3) and (col == 0) and (action == Actions.RIGHT)) or
            ((row == 3) and (col == 1) and (action == Actions.LEFT)) or
            ((row == 3) and (col == 2) and (action == Actions.RIGHT)) or
            ((row == 3) and (col == 3) and (action == Actions.LEFT))
        )
        return 1.0 if (border_bump or internal_bump) else 0.0

    @staticmethod
    def _taxi_cell_index(row, col):
        return row * TaxiFeatureExtractor.GRID_COLS + col
