import sys
import math
import random
import numpy as np
import pygame
from typing import Tuple, List, Optional
import functools

from gymnasium.spaces import Dict, Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

# Constants for the game
BOARD_SIZE = (6, 8)
TILE_SIZE = 100
WINDOW_SIZE = (BOARD_SIZE[1] * TILE_SIZE, BOARD_SIZE[0] * TILE_SIZE)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PLAYER_COLORS = [(0, 128, 255), (255, 0, 128)]
PLAYER_STARTING_POS = [(2, 1), (3, 6)]
BLOCK_COLOR = (128, 128, 128)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTIONS_MOVE = {
    pygame.K_UP: (-1, 0),
    pygame.K_DOWN: (1, 0),
    pygame.K_LEFT: (0, -1),
    pygame.K_RIGHT: (0, 1),
}
PLAYERVSBOT = True

def env(render_mode=None, board_size=BOARD_SIZE, init_pos=PLAYER_STARTING_POS):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    e = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        e = wrappers.CaptureStdoutWrapper(e)
    e = wrappers.AssertOutOfBoundsWrapper(e)
    e = wrappers.OrderEnforcingWrapper(e)
    return e

class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "isolation_v0",
        "render_fps": 30,
    }

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("screen", None)
        state.pop("clock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()
        self.screen = None

    def __init__(self, render_mode=None, board_size=BOARD_SIZE, init_pos=PLAYER_STARTING_POS):
        super().__init__()
        self.board_size = board_size
        self.init_pos = init_pos
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = dict(zip(self.possible_agents, range(len(self.possible_agents))))

        self._action_spaces = {
            agent: Discrete(4 * self.board_size[0] * self.board_size[1])
            for agent in self.possible_agents
        }

        self._observation_spaces = {
            agent: Dict({
                "observation": Box(low=0, high=1, shape=(*self.board_size, 3), dtype=np.int8),
                "action_mask": Box(low=0, high=1, shape=(4*self.board_size[0]*self.board_size[1],), dtype=np.int8)
            }) for agent in self.possible_agents
        }

        self.screen = None
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.board = np.zeros(self.board_size, dtype=np.int8)
        self.board[self.init_pos[0]] = 1
        self.board[self.init_pos[1]] = 2

        self.agents = self.possible_agents[:]
        self.rewards = {i:0 for i in self.agents}
        self._cumulative_rewards = {name:0 for name in self.agents}
        self.terminations = {i:False for i in self.agents}
        self.truncations = {i:False for i in self.agents}
        self.infos = {i:{} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observe(self, agent):
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2
        board_cur = np.equal(self.board, cur_player + 1)
        board_opp = np.equal(self.board, opp_player + 1)
        board_removed = np.equal(self.board, -1)
        observation = np.stack([board_cur, board_opp, board_removed], axis=2).astype(np.int8)
        action_mask = np.zeros(4 * self.board_size[0] * self.board_size[1], "int8")
        legal_moves = self._legal_moves(agent) if agent == self.agent_selection else []
        for i in legal_moves:
            action_mask[i] = 1
        return {"observation": observation, "action_mask": action_mask}

    def _pos_in_board(self, pos: Tuple[int, int]) -> bool:
        return all(0 <= i < j for i, j in zip(pos, self.board_size))

    def _encode_action(self, move_idx: int, remove_pos: Tuple[int, int]) -> int:
        board_area = self.board_size[0] * self.board_size[1]
        longer_side = max(self.board_size)
        return move_idx * board_area + remove_pos[0] * longer_side + remove_pos[1]

    def _decode_action(self, action: int) -> Tuple[int, Tuple[int, int]]:
        board_area = self.board_size[0] * self.board_size[1]
        longer_side = max(self.board_size)
        move_idx = action // board_area
        remainder = action % board_area
        remove_r = remainder // longer_side
        remove_c = remainder % longer_side
        return move_idx, (remove_r, remove_c)

    def _legal_moves(self, agent: str) -> List[int]:
        cur_player = self.possible_agents.index(agent)
        positions = np.argwhere(self.board == cur_player + 1)
        if len(positions) == 0:
            return []
        cur_pos = tuple(positions[0])
        legal = []
        for move_idx, move_dir in enumerate(MOVES):
            new_pos = (cur_pos[0] + move_dir[0], cur_pos[1] + move_dir[1])
            if self._pos_in_board(new_pos) and self.board[new_pos] == 0:
                tmp_board = self.board.copy()
                tmp_board[cur_pos] = 0
                tmp_board[new_pos] = cur_player + 1
                removes = np.argwhere(tmp_board == 0)
                for rem in removes:
                    action = self._encode_action(move_idx, tuple(rem))
                    legal.append(action)
        return legal

    def step(self, action: int):
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        move_idx, remove_pos = self._decode_action(action)
        move_dir = MOVES[move_idx]
        agent = self.agent_selection
        cur_player = self.possible_agents.index(agent)
        positions = np.argwhere(self.board == cur_player + 1)
        if len(positions) == 0:
            # Player has no position, cannot move
            self._terminate_game()
            return
        cur_pos = tuple(positions[0])
        new_pos = (cur_pos[0] + move_dir[0], cur_pos[1] + move_dir[1])
        assert self._pos_in_board(new_pos) and self.board[new_pos] == 0, "Invalid move direction or target position."
        self.board[cur_pos] = 0
        self.board[new_pos] = cur_player + 1
        assert self._pos_in_board(remove_pos) and self.board[remove_pos] == 0, "Invalid remove position."
        self.board[remove_pos] = -1
        next_agent = self._agent_selector.next()
        if not self._legal_moves(next_agent):
            self.rewards[agent] += 1
            self.rewards[self.possible_agents[(cur_player + 1) % 2]] -= 1
            self._terminate_game()
        self.agent_selection = next_agent
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def _terminate_game(self):
        self.terminations = {agent: True for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

    def render(self):
        screen_width = self.board_size[1] * TILE_SIZE
        screen_height = self.board_size[0] * TILE_SIZE
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("Isolation Game")
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))
        if self.screen:
            self.screen.fill(WHITE)
            for r in range(self.board_size[0]):
                for c in range(self.board_size[1]):
                    rect = pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                    val = self.board[r, c]
                    if val == -1:
                        pygame.draw.rect(self.screen, BLOCK_COLOR, rect)
                    elif val == 1:
                        pygame.draw.rect(self.screen, PLAYER_COLORS[0], rect)
                    elif val == 2:
                        pygame.draw.rect(self.screen, PLAYER_COLORS[1], rect)
                    else:
                        pygame.draw.rect(self.screen, WHITE, rect)
                    pygame.draw.rect(self.screen, BLACK, rect, 2)
            if self.render_mode == "human":
                pygame.event.pump()
                pygame.display.update()
                self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

### MCTS Implementation with Heuristic-Based Pruning ###

class MCTSNode:
    def __init__(self, state: Tuple[np.ndarray, int], parent: Optional['MCTSNode'] = None, action: Optional[int] = None):
        self.state = state  # (board, current_player)
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: Optional[List[int]] = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def fully_expanded(self) -> bool:
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def uct(self, c=1.41) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

class MCTSAgent:
    def __init__(self, board_size=BOARD_SIZE, simulations=100, prune_top_n=8):
        self.board_size = board_size
        self.simulations = simulations
        self.prune_top_n = prune_top_n

    def pos_in_board(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.board_size[0] and 0 <= pos[1] < self.board_size[1]

    def encode_action(self, move_idx: int, remove_pos: Tuple[int, int]) -> int:
        board_area = self.board_size[0] * self.board_size[1]
        longer_side = max(self.board_size)
        return move_idx * board_area + remove_pos[0] * longer_side + remove_pos[1]

    def decode_action(self, action: int) -> Tuple[int, Tuple[int, int]]:
        board_area = self.board_size[0] * self.board_size[1]
        longer_side = max(self.board_size)
        move_idx = action // board_area
        remainder = action % board_area
        r = remainder // longer_side
        c = remainder % longer_side
        return move_idx, (r, c)

    def get_legal_actions(self, board: np.ndarray, current_player: int) -> List[int]:
        positions = np.argwhere(board == current_player + 1)
        if len(positions) == 0:
            return []
        cur_pos = tuple(positions[0])
        legal = []
        for move_idx, move_dir in enumerate(MOVES):
            new_pos = (cur_pos[0] + move_dir[0], cur_pos[1] + move_dir[1])
            if self.pos_in_board(new_pos) and board[new_pos] == 0:
                tmp_board = board.copy()
                tmp_board[cur_pos] = 0
                tmp_board[new_pos] = current_player + 1
                removes = np.argwhere(tmp_board == 0)
                for rem in removes:
                    action = self.encode_action(move_idx, tuple(rem))
                    legal.append(action)
        return legal

    def apply_action(self, board: np.ndarray, current_player: int, action: int) -> Tuple[np.ndarray, int]:
        move_idx, rem = self.decode_action(action)
        positions = np.argwhere(board == current_player + 1)
        if len(positions) == 0:
            # Player has no position, terminal state
            return board.copy(), current_player
        cur_pos = tuple(positions[0])
        move_dir = MOVES[move_idx]
        new_pos = (cur_pos[0] + move_dir[0], cur_pos[1] + move_dir[1])
        new_board = board.copy()
        new_board[cur_pos] = 0
        new_board[new_pos] = current_player + 1
        new_board[rem] = -1
        next_player = (current_player + 1) % 2
        return new_board, next_player

    def terminal_state(self, board: np.ndarray, current_player: int) -> bool:
        acts = self.get_legal_actions(board, current_player)
        return len(acts) == 0

    def simulate(self, state: Tuple[np.ndarray, int]) -> int:
        board, cur_p = state
        while True:
            acts = self.get_legal_actions(board, cur_p)
            if len(acts) == 0:
                # cur_p loses
                return 1 if cur_p == 1 else 0
            a = random.choice(acts)
            board, cur_p = self.apply_action(board, cur_p, a)

    def backpropagate(self, node: MCTSNode, result: int):
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

    def best_child(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=lambda c: c.visits)

    def select_node(self, node: MCTSNode) -> MCTSNode:
        while node.fully_expanded() and not node.is_leaf():
            node = max(node.children, key=lambda c: c.uct())
        return node

    def score_action(self, board: np.ndarray, current_player: int, action: int) -> float:
        # Heuristic: Difference between opponent's proximity and player's proximity to eliminated tiles
        new_board, new_p = self.apply_action(board, current_player, action)
        
        # Find player's new position
        player_pos = np.argwhere(new_board == current_player + 1)
        if len(player_pos) == 0:
            # Terminal state where player has no position
            return -math.inf  # Worst possible score
        player_pos = tuple(player_pos[0])
        
        # Find opponent's position
        opponent = (current_player + 1) % 2
        opp_pos = np.argwhere(new_board == opponent + 1)
        if len(opp_pos) == 0:
            # Opponent has no position, best possible score
            return math.inf  # Best possible score
        opp_pos = tuple(opp_pos[0])
        
        # Find all eliminated tiles
        eliminated_tiles = np.argwhere(new_board == -1)
        if len(eliminated_tiles) == 0:
            # No tiles eliminated, neutral score
            return 0.0
        
        # Calculate sum of inverses of distances from player's position to eliminated tiles
        sum_inv_dist_player = 0.0
        for tile in eliminated_tiles:
            tile_pos = tuple(tile)
            distance = abs(tile_pos[0] - player_pos[0]) + abs(tile_pos[1] - player_pos[1]) #math.sqrt((tile_pos[0] - player_pos[0])**2 + (tile_pos[1] - player_pos[1])**2)
            sum_inv_dist_player += 1.0 / (distance + 1)  # +1 to avoid division by zero
        
        # Calculate sum of inverses of distances from opponent's position to eliminated tiles
        sum_inv_dist_opponent = 0.0
        for tile in eliminated_tiles:
            tile_pos = tuple(tile)
            distance = abs(tile_pos[0] - opp_pos[0]) + abs(tile_pos[1] - opp_pos[1]) #math.sqrt((tile_pos[0] - opp_pos[0])**2 + (tile_pos[1] - opp_pos[1])**2)
            sum_inv_dist_opponent += 1.0 / (distance + 1)  # +1 to avoid division by zero
        
        # Heuristic: Opponent's proximity minus Player's proximity
        heuristic = sum_inv_dist_opponent - sum_inv_dist_player
        return heuristic

    def expand(self, node: MCTSNode):
        if node.untried_actions is None:
            board, cur_p = node.state
            legal = self.get_legal_actions(board, cur_p)
            if len(legal) > self.prune_top_n:
                scored_moves = []
                for a in legal:
                    score = self.score_action(board, cur_p, a)
                    scored_moves.append((score, a))
                # Sort moves by score descending (higher is better)
                scored_moves.sort(key=lambda x: x[0], reverse=True)
                # Keep only top prune_top_n moves
                legal = [m for (s, m) in scored_moves[:self.prune_top_n]]
            node.untried_actions = list(legal)

        if len(node.untried_actions) == 0:
            return node
        action = node.untried_actions.pop()
        board, cur_p = node.state
        new_board, new_p = self.apply_action(board, cur_p, action)
        child = MCTSNode((new_board, new_p), parent=node, action=action)
        node.children.append(child)
        return child

    def run(self, board: np.ndarray, current_player: int) -> int:
        root = MCTSNode((board.copy(), current_player))
        for _ in range(self.simulations):
            leaf = self.select_node(root)
            if not self.terminal_state(leaf.state[0], leaf.state[1]):
                leaf = self.expand(leaf)
            result = self.simulate(leaf.state)
            self.backpropagate(leaf, result)
        best = self.best_child(root)
        return best.action

    def choose_action(self, board: np.ndarray, current_player: int) -> int:
        return self.run(board, current_player)

### Integration with the Game ###

class IsolationGame:
    def __init__(self, board_size=BOARD_SIZE, player_starting_pos=PLAYER_STARTING_POS, player0_type="human", player1_type="mcts"):
        self.board_size = board_size
        self.env = raw_env(render_mode="human")  # Ensure rendering is enabled
        self.env.reset()

        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Isolation Game")
        self.clock = pygame.time.Clock()

        self.player_types = [player0_type, player1_type]
        self.mcts_agent = None
        if "mcts" in self.player_types:
            self.mcts_agent = MCTSAgent(board_size=self.board_size, simulations=2000, prune_top_n=6) #control the parameters as you wish

        self.chosen_direction = None

        self.sync_from_env()

    def sync_from_env(self):
        self.board = self.env.board.copy()
        pos1 = np.argwhere(self.board == 1)
        pos2 = np.argwhere(self.board == 2)
        pos1 = tuple(pos1[0]) if len(pos1) > 0 else None
        pos2 = tuple(pos2[0]) if len(pos2) > 0 else None
        self.player_positions = [pos1, pos2]
        agent_idx = self.env.possible_agents.index(self.env.agent_selection)
        self.current_player = agent_idx

    def draw_board(self):
        self.screen.fill(WHITE)
        for row in range(self.board_size[0]):
            for col in range(self.board_size[1]):
                rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                val = self.board[row, col]
                if val == -1:
                    pygame.draw.rect(self.screen, BLOCK_COLOR, rect)
                elif val == 1:
                    pygame.draw.rect(self.screen, PLAYER_COLORS[0], rect)
                elif val == 2:
                    pygame.draw.rect(self.screen, PLAYER_COLORS[1], rect)
                else:
                    pygame.draw.rect(self.screen, WHITE, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 2)
        pygame.display.flip()

    def human_move_action(self, move_dir: Tuple[int, int], remove_pos: Tuple[int, int]) -> int:
        move_idx = MOVES.index(move_dir)
        return self.env._encode_action(move_idx, remove_pos)

    def run(self):
        running = True
        while running:
            self.clock.tick(30)
            if any(self.env.terminations.values()) or any(self.env.truncations.values()):
                # Game ended
                if self.env.rewards["player_0"] > 0:
                    print("Player 1 wins! (You if you chose to play)")
                elif self.env.rewards["player_0"] < 0:
                    print("Player 2 (Bot) wins!")
                else:
                    print("Draw!")
                running = False
                break

            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                    break

            cur_type = self.player_types[self.current_player]

            if cur_type == "human":
                # Human chooses direction first, then tile
                if self.chosen_direction is None:
                    # Choose direction
                    for event in events:
                        if event.type == pygame.KEYDOWN and event.key in ACTIONS_MOVE:
                            self.chosen_direction = ACTIONS_MOVE[event.key]
                else:
                    # Direction chosen, choose tile
                    for event in events:
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            x, y = event.pos
                            col = x // TILE_SIZE
                            row = y // TILE_SIZE
                            if self.board[row, col] == 0:
                                action = self.human_move_action(self.chosen_direction, (row, col))
                                obs = self.env.observe(self.env.agent_selection)
                                if obs["action_mask"][action] == 1:
                                    self.env.step(action)
                                    self.sync_from_env()
                                    self.chosen_direction = None
                                else:
                                    print("Illegal tile chosen, try again.")
            elif cur_type == "mcts":
                # MCTS turn
                action = self.mcts_agent.choose_action(self.board, self.current_player)
                self.env.step(action)
                self.sync_from_env()
            else:
                # Random or other player types can be implemented here
                obs = self.env.observe(self.env.agent_selection)
                actions = np.where(obs["action_mask"] == 1)[0]
                if len(actions) > 0:
                    act = random.choice(actions)
                    self.env.step(act)
                    self.sync_from_env()

            self.draw_board()

        pygame.quit()
        sys.exit()

### Main Execution ###

if __name__ == "__main__":
    while True:
        user_input = input("Is this player vs. bot or bot vs. bot? Enter P or B\n")
        if user_input in ["P", "B"]:
            print("Valid input received. Proceeding...")
            if user_input == "P":
                PLAYERVSBOT = True
            if user_input == "B":
                PLAYERVSBOT = False
            break
        else:
            print("Invalid input. Please try again.")

    if PLAYERVSBOT:
        player0_type = "human"
        player1_type = "mcts"
    else:
        player0_type = "mcts"
        player1_type = "mcts"

    game = IsolationGame(player0_type=player0_type, player1_type=player1_type)
    game.run()
