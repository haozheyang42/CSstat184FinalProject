import random
import numpy as np
from itertools import product
import copy

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
        
    def learn(self, reward):
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

    def learn(self, reward):
        pass
        

class MCTSBot(BaseBot):
    def __init__(self, board_size, exploration_weight=1.0):
        super().__init__(board_size=board_size)
        self.exploration_weight = exploration_weight

        self.ROTATIONS = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 0° (no rotation)
            [7, 3, 6, 10, 0, 2, 9, 11, 1, 5, 8, 4],  # 90° CW
            [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # 180° CW
            [4, 8, 5, 1, 11, 9, 2, 0, 10, 6, 3, 7]   # 270° CW
        ])

        binary_combinations = list(product([0, 1], repeat=12))

        self.unique_patterns = set()
        for comb in binary_combinations:
            array = np.array(comb)
            canonical = self._canonical_rotation(array)
            self.unique_patterns.add(tuple(canonical))

        self.STAR_STATES = [np.array(p) for p in self.unique_patterns]
        self.STAR_MOVES = np.array([(-2, 0), \
                          (-1, -1), (-1, 0), (-1, 1), \
                  (0, -2), (0, -1),          (0, 1), (0, 2), \
                           (1, -1),  (1, 0), (1, 1), \
                                     (2, 0)])

        self.STATS_MOVE_VISITED = {
            p: 1
            for p in self.unique_patterns
        }
        self.STATS_MOVE_WON = copy.deepcopy(self.STATS_MOVE_VISITED)
        self.STATS_REMOVE_VISITED = copy.deepcopy(self.STATS_MOVE_VISITED)
        self.STATS_REMOVE_WON = copy.deepcopy(self.STATS_MOVE_VISITED)

        self.THIS_MOVE_VISITED = {
            p: 0
            for p in self.unique_patterns
        }
        self.THIS_REMOVE_VISITED = copy.deepcopy(self.THIS_MOVE_VISITED)

    def _canonical_rotation(self, array):
        """Find the lexicographically smallest rotation of a given array"""
        assert(len(array) == 12)
        assert(all(array[i] in (0,1) for i in range(12)))
        rotated = [tuple(array[rotation]) for rotation in self.ROTATIONS]
        return np.array(min(rotated))

    def _ucb_score_move(self, rotated_state):
        if any(rotated_state == -1):
            return -1

        w, v = self.STATS_MOVE_WON[tuple(rotated_state)], self.STATS_MOVE_VISITED[tuple(rotated_state)]
        assert(w <= v)
        return w / v + self.exploration_weight * np.sqrt(np.log(v) / v)
    
    def _ucb_score_remove(self, rotated_state):
        w, v = self.STATS_REMOVE_WON[tuple(rotated_state)], self.STATS_REMOVE_VISITED[tuple(rotated_state)]
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
        cur_star_state = np.ones(12)
        opp_star_state = np.ones(12)

        for i in range(12):
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

        self.THIS_MOVE_VISITED[rot_move] += 1
        self.THIS_REMOVE_VISITED[rot_remove] += 1


        # Move: compute new stars for possible moves
        cur_next_states = []
        MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for new_pos in [self._add_move(cur_pos, move) for move in MOVES]:
            if not self._pos_in_board(new_pos) \
                or obs[:, :, 1][new_pos] != 0 \
                or obs[:, :, 2][new_pos] != 0: 
                cur_next_states.append(np.ones(12) * -1)
                continue

            s = np.ones(12)

            for i in range(12):
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
        move_idx = ucb_scores.index(max(ucb_scores))
        new_pos = self._add_move(cur_pos, MOVES[move_idx])
        # self.visited_move_states.add(cur_next_states[move_idx])

        # Remove: construct star state
        opp_star_state = np.ones(12)

        for i in range(12):
            move = self.STAR_MOVES[i]
            opp_consider = self._add_move(opp_pos, move)

            # If not in board, occupied by self, removed
            if not self._pos_in_board(opp_consider) \
                or obs[:, :, 0][opp_consider] != 0 \
                or obs[:, :, 2][opp_consider] != 0:
                
                opp_star_state[i] = 0

        # Remove: construct next star states and pick best remove
        best_remove_i = None
        best_remove_score = -np.inf

        for i in range(12):
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
                best_remove_i = i
                best_remove_score = ucb_score
        # if random.random() < 0.01:
        #     print(best_remove_score)

        if best_remove_i is None:
            return np.where(mask == 1)[0][0]
        
        else:
            remove_pos = self._add_move(opp_pos, self.STAR_MOVES[best_remove_i])
            return self._encode_action(move_idx, remove_pos)
        
    def learn(self, reward):
        # Update statistics
        for k, v in self.THIS_MOVE_VISITED.items():
            if v > 0:
                self.STATS_MOVE_VISITED[k] += 1
                if reward == 1:
                    self.STATS_MOVE_WON[k] += 1
        
        for k, v in self.THIS_REMOVE_VISITED.items():
            if v > 0:
                self.STATS_REMOVE_VISITED[k] += 1
                if reward == 1:
                    self.STATS_REMOVE_WON[k] += 1

        # Reset trackers
        for k in self.THIS_MOVE_VISITED:
            self.THIS_MOVE_VISITED[k] = 0
        
        for k in self.THIS_REMOVE_VISITED:
            self.THIS_REMOVE_VISITED[k] = 0
    