import copy
from typing import Optional
import numpy as np
import functools

import pygame
from gymnasium.spaces import Dict, Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

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

def env(render_mode=None, board_size=BOARD_SIZE, init_pos=PLAYER_STARTING_POS):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":       # Print to terminal
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

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
        # Reinitialize pygame objects if needed
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

        # Observation space: board x 3 channels
        # Channel 1: self occupy?
        # Channel 2: opponent occupy?
        # Channel 3: present or removed?
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
                    # Discrete(4 * self.board_size[0] * self.board_size[1])
                }
            )
            for agent in self.possible_agents
        }
        
        # Render
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
        
        # Board init
        # Removed = -1
        # Unoccupied = 0
        # Player 0 = 1
        # Player 1 = 2
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


    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """

        # Observation space: board x 3 channels
        # Channel 1: self occupy?
        # Channel 2: opponent occupy?
        # Channel 3: present or removed?
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
        assert(len(pos) == 2)
        return all(0 <= i < j for i, j in zip(pos, self.board_size))


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
        

    def _legal_moves(self, agent):
        """
        Returns [0/1/2/3 x row x col]
        for every valid L/R/U/D x row x col of tile to be removed
        """
        cur_player = self.possible_agents.index(agent)
        cur_pos = tuple(np.argwhere(self.board == cur_player + 1)[0])
        legal_moves = []

        for move_idx, move_dir in enumerate(MOVES):
            new_pos = (cur_pos[0] + move_dir[0], cur_pos[1] + move_dir[1])
            if self._pos_in_board(new_pos) and self.board[new_pos] == 0:
                # Add the valid move direction and resulting free positions
                # re-create the board to find the free positions
                tmp_board = copy.deepcopy(self.board)
                tmp_board[cur_pos] = 0
                tmp_board[new_pos] = cur_player + 1

                for remove_pos in np.argwhere(tmp_board == 0):
                    action = self._encode_action(move_idx, remove_pos)
                    legal_moves.append(action)

        return legal_moves  


    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        
        # Disaggregate action
        move_idx, remove_pos = self._decode_action(action) 
        move_dir = MOVES[move_idx]
        
        # Move
        agent = self.agent_selection
        cur_player = self.possible_agents.index(agent)
        cur_pos = tuple(np.argwhere(self.board == cur_player + 1)[0])
        new_pos = (cur_pos[0] + move_dir[0], cur_pos[1] + move_dir[1])
        assert(self._pos_in_board(new_pos) and self.board[new_pos] == 0)

        self.board[cur_pos] = 0
        self.board[new_pos] = cur_player + 1

        # Remove
        assert(self._pos_in_board(remove_pos) and self.board[remove_pos] == 0)
        self.board[remove_pos] = -1

        # If next agent has no legal moves, then current agent wins
        next_agent = self._agent_selector.next()

        if not self._legal_moves(next_agent):
            self.rewards[self.agent_selection] += 1
            self.rewards[next_agent] -= 1
            self.terminations = {i: True for i in self.agents}

        # Next step
        self.agent_selection = next_agent
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
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
                    if self.board[row, col] == -1:
                        pygame.draw.rect(self.screen, BLOCK_COLOR, rect)
                    elif self.board[row, col] == 1:
                        pygame.draw.rect(self.screen, PLAYER_COLORS[0], rect)
                    elif self.board[row, col] == 2:
                        pygame.draw.rect(self.screen, PLAYER_COLORS[1], rect)
                    pygame.draw.rect(self.screen, BLACK, rect, 2)

            if self.render_mode == "human":
                pygame.event.pump()
                pygame.display.update()
                self.clock.tick(self.metadata["render_fps"])

            observation = np.array(pygame.surfarray.pixels3d(self.screen))

            # Debugging output
            if self.render_mode == "rgb_array":
                print(f"Rendered frame shape: {observation.shape}, dtype: {observation.dtype}")

            return (
                np.transpose(observation, axes=(1, 0, 2))
                if self.render_mode == "rgb_array"
                else None
            )

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None
    