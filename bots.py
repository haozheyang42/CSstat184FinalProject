import random
import numpy as np
from itertools import product
import copy
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque


class RandomBot():
    def __init__(self):
        pass
    
    def take_step(self, observation):
        if isinstance(observation, dict) and "action_mask" in observation:
            mask = observation["action_mask"]
            indices = [i for i, value in enumerate(mask) if value == 1]
            return random.choice(indices)
        else:
            raise TypeError
        
    def learn(self, observation, reward):
        pass


class BaseBot():
    def __init__(self, board_size):
        self.board_size = board_size
        pass

    def _encode_action(self, move_idx, remove_pos):
        board_area = self.board_size[0] * self.board_size[1]
        longer_side = max(self.board_size[0], self.board_size[1])
        return move_idx * board_area \
                    + remove_pos[0] * longer_side \
                    + remove_pos[1]
    
    def _decode_action(self, action):
        board_area = self.board_size[0] * self.board_size[1]
        longer_side = max(self.board_size[0], self.board_size[1])
        move_idx = action // board_area
        remainder = action % board_area
        remove_pos_0 = remainder // longer_side
        remove_pos_1 = remainder % longer_side
        return move_idx, (remove_pos_0, remove_pos_1)
    
    def _pos_in_board(self, pos):
        assert(len(pos) == 2)
        return all(0 <= i < j for i, j in zip(pos, self.board_size))
    
    def _add_move(self, pos, move):
        return (pos[0] + move[0], pos[1] + move[1])


class HeuristicBot(BaseBot):
    """
    Always moves itself to maximize number of free tiles adjacent to self
    And removes tile to minimize number of free tiles adjacent to opponent
    """
    def __init__(self, board_size):
        super().__init__(board_size=board_size)
        pass
    
    def take_step(self, observation):
        if isinstance(observation, dict) and "action_mask" in observation:
            obs = observation["observation"]
            mask = observation["action_mask"]

        else:
            raise TypeError
        
        cur_pos = tuple(np.argwhere(obs[:, :, 0] == 1)[0]) 
        opp_pos = tuple(np.argwhere(obs[:, :, 1] == 1)[0]) 

        best_action = None
        best_score = 0
        
        for j in range(len(mask)):
            if mask[j]:
                this_score = 0

                move_idx, remove_pos = self._decode_action(j)

                MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                new_pos = self._add_move(cur_pos, MOVES[move_idx])

                pos1 = (new_pos[0] - 1, new_pos[1])
                pos2 = (new_pos[0] + 1, new_pos[1])
                pos3 = (new_pos[0], new_pos[1] - 1)
                pos4 = (new_pos[0], new_pos[1] + 1)
                
                for p in (pos1, pos2, pos3, pos4):
                    # In board, present, not occupied by opponent
                    if self._pos_in_board(p) and obs[:, :, 2][p] == 0 and obs[:, :, 1][p] == 0:
                        this_score += 1

                pos1 = (opp_pos[0] - 1, opp_pos[1])
                pos2 = (opp_pos[0] + 1, opp_pos[1])
                pos3 = (opp_pos[0], opp_pos[1] - 1)
                pos4 = (opp_pos[0], opp_pos[1] + 1)
                
                if remove_pos in (pos1, pos2, pos3, pos4):
                    this_score += 1
                
                if this_score > best_score:
                    best_action = j
                    best_score = this_score

                elif this_score == best_score:
                    best_action = random.choice([best_action, j])
        
        return best_action

    def learn(self, observation, reward):
        pass
        

class MCTSBot(BaseBot):
    """
    Every state is a 12-square "star" around the current position
        0
    1  2  3
    4  5  X  6  7
    8  9  10
        11

    There are 2**12 possible states for the current position, and
    2**12 for the opponent position

    Further reduce number of states to 1044 leveraging rotations

    An index is 1 iff it is in board, not removed, and not occupied by a player
    """
    def __init__(self, board_size, exploration_weight=1.0):
        super().__init__(board_size=board_size)
        self.exploration_weight = exploration_weight

        self.STATE_LEN = 12

        self.ROTATIONS = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 0° (no rotation)
            [7, 3, 6, 10, 0, 2, 9, 11, 1, 5, 8, 4],  # 90° CW
            [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # 180° CW
            [4, 8, 5, 1, 11, 9, 2, 0, 10, 6, 3, 7]   # 270° CW
        ])

        # binary_combinations = list(product([0, 1], repeat=self.STATE_LEN))

        # self.unique_patterns = set()
        # for comb in binary_combinations:
        #     array = np.array(comb)
        #     canonical = self._canonical_rotation(array)
        #     self.unique_patterns.add(tuple(canonical))

        # self.STAR_STATES = [np.array(p) for p in self.unique_patterns]
        self.STAR_MOVES = np.array([(-2, 0), \
                          (-1, -1), (-1, 0), (-1, 1), \
                  (0, -2), (0, -1),          (0, 1), (0, 2), \
                           (1, -1),  (1, 0), (1, 1), \
                                     (2, 0)])

        self.STATS_MOVE_VISITED = dict()
        self.STATS_MOVE_WON = dict()
        self.STATS_REMOVE_VISITED = dict()
        self.STATS_REMOVE_WON = dict()

        self.THIS_MOVE_VISITED = set()
        self.THIS_REMOVE_VISITED = set()

    def _canonical_rotation(self, array):
        """Find the lexicographically smallest rotation of a given array"""
        assert(len(array) == self.STATE_LEN)
        # assert(all(array[i] in (0,1) for i in range(self.STATE_LEN)))
        rotated = [tuple(array[rotation]) for rotation in self.ROTATIONS]
        return np.array(min(rotated))

    def _ucb_score_move(self, rotated_state):
        if all(rotated_state == -1):
            return -1

        w, v = self.STATS_MOVE_WON.get(tuple(rotated_state), 1), self.STATS_MOVE_VISITED.get(tuple(rotated_state),1)
        assert(w <= v)
        return w / v + self.exploration_weight * np.sqrt(np.log(v) / v)
    
    def _ucb_score_remove(self, rotated_state):
        w, v = self.STATS_REMOVE_WON.get(tuple(rotated_state), 1), self.STATS_REMOVE_VISITED.get(tuple(rotated_state), 1)
        assert(w <= v)
        return w / v + self.exploration_weight * np.sqrt(np.log(v) / v)
    
    def take_step(self, observation):
        if isinstance(observation, dict) and "action_mask" in observation:
            obs = observation["observation"]
            mask = observation["action_mask"]

        else:
            raise TypeError
        
        cur_pos = tuple(np.argwhere(obs[:, :, 0] == 1)[0]) 
        opp_pos = tuple(np.argwhere(obs[:, :, 1] == 1)[0]) 

        # Add visited states to THIS
        cur_star_state = np.ones(self.STATE_LEN)
        opp_star_state = np.ones(self.STATE_LEN)

        for i in range(self.STATE_LEN):
            move = self.STAR_MOVES[i]
            cur_consider = self._add_move(cur_pos, move)
            opp_consider = self._add_move(opp_pos, move)

            # If not in board, occupied by opponent, removed
            if not self._pos_in_board(cur_consider) \
            or obs[:, :, 1][cur_consider] != 0 \
            or obs[:, :, 2][cur_consider] != 0:
                cur_star_state[i] = 0 

            # If not in board, occupied by self, removed
            if not self._pos_in_board(opp_consider) \
            or obs[:, :, 0][opp_consider] != 0 \
            or obs[:, :, 2][opp_consider] != 0:
                opp_star_state[i] = 0

        rot_move = tuple(self._canonical_rotation(cur_star_state))
        rot_remove = tuple(self._canonical_rotation(opp_star_state))

        self.THIS_MOVE_VISITED.add(rot_move)
        self.THIS_REMOVE_VISITED.add(rot_remove)


        # Move: compute new stars for possible moves
        cur_next_states = []
        MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for move_dir in MOVES:
            new_pos = self._add_move(cur_pos, move_dir)

            if not self._pos_in_board(new_pos) \
                or obs[:, :, 1][new_pos] != 0 \
                or obs[:, :, 2][new_pos] != 0: 
                cur_next_states.append(np.ones(self.STATE_LEN) * -1)
                continue

            s = np.ones(self.STATE_LEN)

            for i in range(self.STATE_LEN):
                move = self.STAR_MOVES[i]
                consider = self._add_move(cur_pos, move)

                # If not in board, occupied by opponent, removed
                if not self._pos_in_board(consider) \
                or obs[:, :, 1][consider] != 0 \
                or obs[:, :, 2][consider] != 0:

                    s[i] = 0

            s_rot = self._canonical_rotation(s)
            cur_next_states.append(s_rot)

        # Move: pick action by UCB
        ucb_scores = [self._ucb_score_move(s_rot) for s_rot in cur_next_states]
        indices = [i for i, v in enumerate(ucb_scores) if v == max(ucb_scores)]
        move_idx = random.choice(indices)
        new_pos = self._add_move(cur_pos, MOVES[move_idx])

        # Remove: construct star state
        # opp_star_state = np.ones(self.STATE_LEN)

        # for i in range(self.STATE_LEN):
        #     move = self.STAR_MOVES[i]
        #     opp_consider = self._add_move(opp_pos, move)

        #     # If not in board, occupied by self, removed
        #     if not self._pos_in_board(opp_consider) \
        #         or obs[:, :, 0][opp_consider] != 0 \
        #         or obs[:, :, 2][opp_consider] != 0:
                
        #         opp_star_state[i] = 0

        # Remove: construct next star states and pick best remove
        best_remove_i = []
        best_remove_score = -np.inf

        for i in range(self.STATE_LEN):
            if opp_star_state[i] == 0:
                continue
            
            remove_pos = self._add_move(opp_pos, self.STAR_MOVES[i])
            if remove_pos == new_pos:
                continue

            next_opp_star_state = copy.deepcopy(opp_star_state)
            next_opp_star_state[i] = 0
            next_opp_star_state_rot = self._canonical_rotation(next_opp_star_state)
            ucb_score = self._ucb_score_remove(next_opp_star_state_rot)
            if ucb_score > best_remove_score:
                best_remove_i = [i]
                best_remove_score = ucb_score
            elif ucb_score == best_remove_score:
                best_remove_i.append(i)


        if best_remove_i == []:
            return np.where(mask == 1)[0][0]
        
        else:
            i = random.choice(best_remove_i)
            remove_pos = self._add_move(opp_pos, self.STAR_MOVES[i])
            return self._encode_action(move_idx, remove_pos)
        
    def learn(self, observation, reward):
        # Update statistics
        for k in self.THIS_MOVE_VISITED:
            self.STATS_MOVE_VISITED[k] = self.STATS_MOVE_VISITED.get(k, 0) + 1
            if reward == 1:
                self.STATS_MOVE_WON[k] = self.STATS_MOVE_WON.get(k, 0) + 1
        
        for k in self.THIS_REMOVE_VISITED:
            self.STATS_REMOVE_VISITED[k] = self.STATS_REMOVE_VISITED.get(k, 0) + 1
            if reward == 1:
                self.STATS_REMOVE_WON[k] = self.STATS_REMOVE_WON.get(k, 0) + 1

        # Reset trackers
        self.THIS_MOVE_VISITED = set()
        self.THIS_REMOVE_VISITED = set()

class MCTSRicherBot(MCTSBot):
    """
    -2: Not in Board
    -1: Occupied by Other Player
    0: Removed
    1: Vacant
    """
    def __init__(self, board_size, exploration_weight=1.0):
        super().__init__(board_size=board_size)
        self.exploration_weight = exploration_weight

        self.STATE_LEN = 12

        self.ROTATIONS = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 0° (no rotation)
            [7, 3, 6, 10, 0, 2, 9, 11, 1, 5, 8, 4],  # 90° CW
            [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # 180° CW
            [4, 8, 5, 1, 11, 9, 2, 0, 10, 6, 3, 7]   # 270° CW
        ])

        # permutations = list(product([-2, 0, 1], repeat=12))

        # permutations_with_one_3 = []
        # for i in range(12):
        #     for perm in product([-2, 0, 1], repeat=11):
        #         # Insert '3' at position i
        #         new_perm = list(perm)
        #         new_perm.insert(i, -1)
        #         permutations_with_one_3.append(tuple(new_perm))

        # # Combine both lists
        # binary_combinations = permutations + permutations_with_one_3
        # print(len(binary_combinations))

        # self.unique_patterns = set()
        # for comb in binary_combinations:
        #     array = np.array(comb)
        #     canonical = self._canonical_rotation(array)
        #     self.unique_patterns.add(tuple(canonical))
        # print(len(self.unique_patterns))
        # self.STAR_STATES = [np.array(p) for p in self.unique_patterns]
        self.STAR_MOVES = np.array([(-2, 0), \
                          (-1, -1), (-1, 0), (-1, 1), \
                  (0, -2), (0, -1),          (0, 1), (0, 2), \
                           (1, -1),  (1, 0), (1, 1), \
                                     (2, 0)])

        self.STATS_MOVE_VISITED = dict()
        self.STATS_MOVE_WON = dict()
        self.STATS_REMOVE_VISITED = dict()
        self.STATS_REMOVE_WON = dict()

        self.THIS_MOVE_VISITED = set()
        self.THIS_REMOVE_VISITED = set()

    def take_step(self, observation):
        if isinstance(observation, dict) and "action_mask" in observation:
            obs = observation["observation"]
            mask = observation["action_mask"]

        else:
            raise TypeError
        
        cur_pos = tuple(np.argwhere(obs[:, :, 0] == 1)[0]) 
        opp_pos = tuple(np.argwhere(obs[:, :, 1] == 1)[0]) 

        # Add visited states to THIS
        cur_star_state = np.ones(self.STATE_LEN)
        opp_star_state = np.ones(self.STATE_LEN)

        for i in range(self.STATE_LEN):
            move = self.STAR_MOVES[i]
            cur_consider = self._add_move(cur_pos, move)
            opp_consider = self._add_move(opp_pos, move)

            # If not in board, occupied by opponent, removed
            if not self._pos_in_board(cur_consider):
                cur_star_state[i] = -2
            elif obs[:, :, 1][cur_consider] != 0:
                cur_star_state[i] = -1
            elif obs[:, :, 2][cur_consider] != 0:
                cur_star_state[i] = 0 

            # If not in board, occupied by self, removed
            if not self._pos_in_board(opp_consider):
                opp_star_state[i] = -2
            elif obs[:, :, 0][opp_consider] != 0:
                opp_star_state[i] = -1
            elif obs[:, :, 2][opp_consider] != 0:
                opp_star_state[i] = 0

        rot_move = tuple(self._canonical_rotation(cur_star_state))
        rot_remove = tuple(self._canonical_rotation(opp_star_state))

        self.THIS_MOVE_VISITED.add(rot_move)
        self.THIS_REMOVE_VISITED.add(rot_remove)


        # Move: compute new stars for possible moves
        cur_next_states = []
        MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for move_dir in MOVES:
            new_pos = self._add_move(cur_pos, move_dir)

            if not self._pos_in_board(new_pos) \
                or obs[:, :, 1][new_pos] != 0 \
                or obs[:, :, 2][new_pos] != 0: 
                cur_next_states.append(np.ones(self.STATE_LEN) * -1)
                continue

            s = np.ones(self.STATE_LEN)

            for i in range(self.STATE_LEN):
                move = self.STAR_MOVES[i]
                consider = self._add_move(cur_pos, move)

                # If not in board, occupied by opponent, removed
                if not self._pos_in_board(consider):
                    s[i] = -2
                elif obs[:, :, 1][consider] != 0:
                    s[i] = -1
                elif obs[:, :, 2][consider] != 0:
                    s[i] = 0

            s_rot = self._canonical_rotation(s)
            cur_next_states.append(s_rot)

        # Move: pick action by UCB
        ucb_scores = [self._ucb_score_move(s_rot) for s_rot in cur_next_states]
        indices = [i for i, v in enumerate(ucb_scores) if v == max(ucb_scores)]
        move_idx = random.choice(indices)
        new_pos = self._add_move(cur_pos, MOVES[move_idx])

        # Remove: construct star state
        # opp_star_state = np.ones(self.STATE_LEN)

        # for i in range(self.STATE_LEN):
        #     move = self.STAR_MOVES[i]
        #     opp_consider = self._add_move(opp_pos, move)

        #     # If not in board, occupied by self, removed
        #     if not self._pos_in_board(opp_consider):

        #         or obs[:, :, 0][opp_consider] != 0 \
        #         or obs[:, :, 2][opp_consider] != 0:
                
        #         opp_star_state[i] = 0

        # Remove: construct next star states and pick best remove
        best_remove_i = []
        best_remove_score = -np.inf

        for i in range(self.STATE_LEN):
            if opp_star_state[i] != 1: # must be removable
                continue
            
            remove_pos = self._add_move(opp_pos, self.STAR_MOVES[i])
            if remove_pos == new_pos:
                continue

            next_opp_star_state = copy.deepcopy(opp_star_state)
            next_opp_star_state[i] = 0
            next_opp_star_state_rot = self._canonical_rotation(next_opp_star_state)
            ucb_score = self._ucb_score_remove(next_opp_star_state_rot)
            if ucb_score > best_remove_score:
                best_remove_i = [i]
                best_remove_score = ucb_score
            elif ucb_score == best_remove_score:
                best_remove_i.append(i)


        if best_remove_i == []:
            return np.where(mask == 1)[0][0]
        
        else:
            i = random.choice(best_remove_i)
            remove_pos = self._add_move(opp_pos, self.STAR_MOVES[i])
            return self._encode_action(move_idx, remove_pos)


class MCTSBiggerBot(MCTSBot):
    """
    Every state is a 24-square "star" around the current position
           0
        1  2  3
     4  5  6  7  8
 9  10  11 X  12 13 14
    15  16 17 18 19
        20 21 22
           23
    """
    def __init__(self, board_size, exploration_weight=1.0):
        self.board_size = board_size
        self.exploration_weight = exploration_weight

        self.STATE_LEN = 24

        self.ROTATIONS = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # 0° (no rotation)
            [14, 8, 13, 19, 3, 7, 12, 18, 22, 0, 2, 6, 17, 21, 23, 1, 5, 11, 16, 20, 4, 10, 15, 9],  # 90° CW
            [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # 180° CW
            [9, 15, 10, 4, 20, 16, 11, 5, 1, 23, 21, 17, 6, 2, 0, 22, 18, 12, 7, 3, 19, 13, 8, 14]   # 270° CW
        ])

        # self.unique_patterns = set()

        # # Generate binary combinations on the fly
        # for i in range(2**self.STATE_LEN):  # Iterates over all possible binary numbers of length STATE_LEN
        #     # Convert the number to binary representation with leading zeros
        #     binary = [int(x) for x in f"{i:0{self.STATE_LEN}b}"]
        #     array = np.array(binary)
            
        #     # Get the canonical rotation
        #     canonical = self._canonical_rotation(array)
            
        #     # Add the canonical pattern to the set
        #     self.unique_patterns.add(tuple(canonical))

        # self.STAR_STATES = [np.array(p) for p in self.unique_patterns]
        self.STAR_MOVES = np.array([(-3, 0),
                          (-2, -1), (-2, 0), (-2, 1), \
                (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),\
        (0, -3), (0, -2), (0, -1),           (0, 1), (0, 2), (0, 3),\
                (1, -2),  (1, -1),  (1, 0), (1, 1),   (1, 2),\
                           (2, -1), (2, 0), (2, 1), \
                                    (3, 0)])

        self.STATS_MOVE_VISITED = dict()
        self.STATS_MOVE_WON = dict()
        self.STATS_REMOVE_VISITED = dict()
        self.STATS_REMOVE_WON = dict()

        self.THIS_MOVE_VISITED = set()
        self.THIS_REMOVE_VISITED = set()


class DQNBot(BaseBot):
    def __init__(self, board_size, batch_size=32, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        super().__init__(board_size=board_size)
        self.action_size = 4 * (board_size[0] * board_size[1])
        self.memory = deque(maxlen=20000)
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_update_frequency = 500
        self.steps = 0
        self._update_target_model()

        self.prev_obs = None
        self.prev_action = None

    def _build_model(self):
        model = models.Sequential([
            layers.Input(shape=(*self.board_size, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='relu')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss='mse')
        return model
    
    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def take_step(self, observation):
        if isinstance(observation, dict) and "action_mask" in observation:
            obs = observation["observation"]
            mask = observation["action_mask"]
        else:
            raise TypeError("Invalid observation format")

        if np.random.rand() <= self.epsilon:
            action = random.choice([i for i, v in enumerate(mask) if v == 1])
            
        else:
            q_values = self.model.predict(np.expand_dims(obs, axis=0))[0]
            valid_q_values = (q_values + 0.0001) * mask # perturb to avoid illegal action
            action = np.argmax(valid_q_values)
        
        if self.prev_obs is not None and self.prev_action is not None:
            self._remember(self.prev_obs, self.prev_action, 0, obs, False)
        
        self.prev_obs = obs
        self.prev_action = action
        
        return action

    def _remember(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0))[0]
            if done:
                target[action] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target[action] = reward + self.gamma * np.amax(t)
            self.model.fit(np.expand_dims(state, axis=0), np.expand_dims(target, axis=0), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, observation, reward):
        if isinstance(observation, dict) and "action_mask" in observation:
            obs = observation["observation"]
            mask = observation["action_mask"]
        else:
            raise TypeError("Invalid observation format")
        
        self._remember(self.prev_obs, self.prev_action, reward, obs, True)
        self.prev_obs, self.prev_action = None, None
        self._replay()

        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self._update_target_model()