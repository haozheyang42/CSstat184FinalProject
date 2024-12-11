import copy
from typing import Optional
import numpy as np
import functools
import random
import math

import pygame
from gymnasium.spaces import Dict, Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

# Constants for the game
BOARD_SIZE = (6, 8)
TILE_SIZE = 100
PLAYER_STARTING_POS = [(2, 1), (3, 6)]
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Constants for colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PLAYER_COLORS = [(0, 128, 255), (255, 0, 128)]
BLOCK_COLOR = (128, 128, 128)

class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "isolation_v0",
        "render_fps": 1,
    }

    def __getstate__(self):
        state = self.__dict__.copy()
        # Exclude non-pickleable pygame objects
        state.pop("screen", None)
        state.pop("clock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()
        self.screen = None  # Reset the screen to None

    def __init__(self, render_mode=None, board_size=BOARD_SIZE, init_pos=PLAYER_STARTING_POS):
        super().__init__()
        self.board_size = board_size
        self.init_pos = init_pos
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = dict(zip(self.possible_agents, range(len(self.possible_agents))))

        # Action space: move direction, tile to remove
        self._action_spaces = {
            agent: Discrete(4 * self.board_size[0] * self.board_size[1])
            for agent in self.possible_agents
        }

        # Observation space
        self._observation_spaces = {
            agent: Dict(
                {
                    "observation": Box(
                        low=0, high=1, shape=(*self.board_size, 3), dtype=np.int8
                    ),
                    "action_mask": Box(
                        low=0, 
                        high=1, 
                        shape=(4 * self.board_size[0] * self.board_size[1],), 
                        dtype=np.int8
                    ),
                }
            )
            for agent in self.possible_agents
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
        if seed is not None:
            self.seed(seed)

        self.board = np.zeros(self.board_size, dtype=np.int8)
        self.board[self.init_pos[0]] = 1
        self.board[self.init_pos[1]] = 2
        self.timestamp = None

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        if self.render_mode == "human":
            self.render()
        return self.observe(self.agent_selection)

    def observe(self, agent):
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        board_cur = np.equal(self.board, cur_player + 1)
        board_opp = np.equal(self.board, opp_player + 1)
        board_removed = np.equal(self.board, -1)

        observation = np.stack([board_cur, board_opp, board_removed], axis=2).astype(np.int8)
        legal_moves = self._legal_moves(agent) if agent == self.agent_selection else []

        action_mask = np.zeros(4 * self.board_size[0] * self.board_size[1], "int8")
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def _pos_in_board(self, pos):
        return all(0 <= i < j for i, j in zip(pos, self.board_size))

    def _encode_action(self, move_idx, remove_pos):
        board_area = self.board_size[0] * self.board_size[1]
        longer_side = max(self.board_size[0], self.board_size[1])
        return move_idx * board_area + remove_pos[0] * longer_side + remove_pos[1]
        
    def _decode_action(self, action):
        board_area = self.board_size[0] * self.board_size[1]
        longer_side = max(self.board_size[0], self.board_size[1])
        move_idx = action // board_area
        remainder = action % board_area
        remove_pos_0 = remainder // longer_side
        remove_pos_1 = remainder % longer_side
        return move_idx, (remove_pos_0, remove_pos_1)

    def _legal_moves(self, agent):
        cur_player = self.possible_agents.index(agent)
        positions = np.argwhere(self.board == cur_player + 1)
        if len(positions) == 0:
            return []
        cur_pos = tuple(positions[0])
        legal_moves = []

        for move_idx, move_dir in enumerate(MOVES):
            new_pos = (cur_pos[0] + move_dir[0], cur_pos[1] + move_dir[1])
            if self._pos_in_board(new_pos) and self.board[new_pos] == 0:
                # After moving, we can remove any currently free tile except the new_pos
                for r in range(self.board_size[0]):
                    for c in range(self.board_size[1]):
                        if self.board[r, c] == 0 and (r,c) != new_pos:
                            action = self._encode_action(move_idx, (r, c))
                            legal_moves.append(action)
        return legal_moves  

    def step(self, action):
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        
        move_idx, remove_pos = self._decode_action(action) 
        move_dir = MOVES[move_idx]
        
        agent = self.agent_selection
        cur_player = self.possible_agents.index(agent)
        positions = np.argwhere(self.board == cur_player + 1)
        if len(positions) == 0:
            self._was_dead_step(action)
            return
        cur_pos = tuple(positions[0])
        new_pos = (cur_pos[0] + move_dir[0], cur_pos[1] + move_dir[1])
        assert self._pos_in_board(new_pos) and self.board[new_pos] == 0

        self.board[cur_pos] = 0
        self.board[new_pos] = cur_player + 1

        # Remove tile
        assert self._pos_in_board(remove_pos) and self.board[remove_pos] == 0
        self.board[remove_pos] = -1

        # Check next agent
        next_agent = self._agent_selector.next()

        if not self._legal_moves(next_agent):
            self.rewards[self.agent_selection] += 1
            self.rewards[next_agent] -= 1
            self.terminations = {i: True for i in self.agents}

        self.agent_selection = next_agent
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def render(self):
        screen_width = self.board_size[1] * TILE_SIZE
        screen_height = self.board_size[0] * TILE_SIZE

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("Isolation Game")
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((screen_width, screen_height))

        if self.screen:  # Only proceed if the screen is successfully initialized
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

            if self.render_mode == "human":
                pygame.event.pump()
                pygame.display.update()
                self.clock.tick(self.metadata["render_fps"])

            if self.render_mode == "rgb_array":
                observation = np.array(pygame.surfarray.pixels3d(self.screen))
                return np.transpose(observation, axes=(1, 0, 2))
            else:
                return None

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

###################################
# MCTS Implementation
###################################

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state     # (board, current_agent)
        self.parent = parent
        self.action = action   # The action that led to this node
        self.children = []
        self.visits = 0
        self.value = 0.0  # sum of rewards

    def is_leaf(self):
        return len(self.children) == 0

    def uct(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)


class MCTSAgent:
    def __init__(self, env, simulations=1000000):
        # env is a raw_env environment
        self.simulations = simulations
        self.base_env = env

    def clone_env_state(self, env):
        # Use the raw_env directly (env is raw_env)
        raw_env = env
        new_env = raw_env.__class__(
            board_size=raw_env.board_size,
            init_pos=raw_env.init_pos,
            render_mode=raw_env.render_mode
        )
        # Manually copy state
        new_env.board = raw_env.board.copy()
        new_env.agents = raw_env.agents[:]
        new_env.rewards = raw_env.rewards.copy()
        new_env._cumulative_rewards = raw_env._cumulative_rewards.copy()
        new_env.terminations = raw_env.terminations.copy()
        new_env.truncations = raw_env.truncations.copy()
        new_env.infos = copy.deepcopy(raw_env.infos)

        # Rebuild _agent_selector and agent_selection
        new_env._agent_selector = agent_selector(new_env.agents)
        selected_agent = new_env._agent_selector.reset()
        # advance until we match raw_env.agent_selection
        while selected_agent != raw_env.agent_selection:
            selected_agent = new_env._agent_selector.next()
        new_env.agent_selection = raw_env.agent_selection

        return new_env

    def simulate(self, node):
        # Random rollout until terminal
        env_copy = self.clone_env_state(self.base_env)
        # Apply actions from root to this node
        path = []
        cur = node
        while cur.parent is not None:
            path.append(cur.action)
            cur = cur.parent
        path.reverse()
        for a in path:
            env_copy.step(a)
        
        # rollout
        while not any(env_copy.terminations.values()) and not any(env_copy.truncations.values()):
            # random action
            agent = env_copy.agent_selection
            obs = env_copy.observe(agent)
            action_mask = obs["action_mask"]
            legal_actions = np.where(action_mask == 1)[0]
            if len(legal_actions) == 0:
                break
            a = random.choice(legal_actions)
            env_copy.step(a)

        return env_copy.rewards

    def expand(self, node):
        # expand node by adding children for each legal action
        env_copy = self.clone_env_state(self.base_env)
        # Replay path
        actions_to_node = []
        cur = node
        while cur.parent is not None:
            actions_to_node.append(cur.action)
            cur = cur.parent
        actions_to_node.reverse()
        for a in actions_to_node:
            env_copy.step(a)

        agent = env_copy.agent_selection
        obs = env_copy.observe(agent)
        action_mask = obs["action_mask"]
        legal_actions = np.where(action_mask == 1)[0]

        for a in legal_actions:
            child = MCTSNode((env_copy.board.copy(), agent), parent=node, action=a)
            node.children.append(child)

    def backpropagate(self, node, rewards):
        # Assume zero-sum perspective from player_0 for simplicity
        perspective_agent = "player_0"
        value = rewards[perspective_agent]

        current = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent

    def best_child(self, node):
        # Choose the child with highest average value
        best_score = -float('inf')
        best_c = None
        for c in node.children:
            if c.visits > 0:
                score = c.value / c.visits
            else:
                score = -float('inf')
            if score > best_score:
                best_score = score
                best_c = c
        return best_c

    def select_node(self, node):
        # selection: descend tree by UCT
        current = node
        while not current.is_leaf() and current.children:
            current = max(current.children, key=lambda c: c.uct())
        return current

    def run(self):
        # Build root node from current environment state
        root_env = self.clone_env_state(self.base_env)
        root_node = MCTSNode((root_env.board.copy(), root_env.agent_selection))

        # expand root
        self.expand(root_node)

        for _ in range(self.simulations):
            leaf = self.select_node(root_node)
            if leaf.is_leaf():
                # expand
                self.expand(leaf)
                if leaf.is_leaf():
                    # no actions (terminal)
                    rewards = self.simulate(leaf)
                    self.backpropagate(leaf, rewards)
                    continue
            
            # simulate from a child
            child = random.choice(leaf.children)
            rewards = self.simulate(child)
            self.backpropagate(child, rewards)

        # Choose best child of root after simulations
        best = self.best_child(root_node)
        return best.action

    def choose_action(self):
        return self.run()

###################################
# Main Execution (Example)
###################################

def play_game(render_mode="human"):
    env_inst = raw_env(render_mode=render_mode)  # Using raw_env directly
    env_inst.reset()

    # player_0 uses MCTS
    mcts_agent = MCTSAgent(env=env_inst, simulations=1000000000)
    # player_1 random

    while not any(env_inst.terminations.values()) and not any(env_inst.truncations.values()):
        agent = env_inst.agent_selection
        obs = env_inst.observe(agent)
        action_mask = obs["action_mask"]
        legal_actions = np.where(action_mask == 1)[0]

        if agent == "player_0":
            # MCTS
            mcts_agent.base_env = env_inst
            action = mcts_agent.choose_action()
        else:
            # random
            if len(legal_actions) == 0:
                break
            action = random.choice(legal_actions)

        env_inst.step(action)
        if render_mode == "human":
            env_inst.render()

    print("Game finished!")
    print("Rewards:", env_inst.rewards)

    env_inst.close()


if __name__ == "__main__":
    play_game(render_mode="human")