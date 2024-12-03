from copy import copy
from typing import Optional
import numpy as np
import functools
import pygame
from gymnasium.spaces import Dict, Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

# Constants for the game
BOARD_SIZE = (6, 8)
PLAYER_STARTING_POS = [(2, 1), (3, 6)]

class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "isolation_v0"
    }

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
                    "action_mask": Discrete(4 * self.board_size[0] * self.board_size[1])
                }
            )
            for agent in self.possible_agents
        }
        
        # Render
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
        self.current_positions = self.init_pos
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

    @staticmethod
    def _legal_moves(self, agent):
        """
        Returns [0/1/2/3 * row * col]
        for every valid L/R/U/D * row * col of tile to be removed
        """
        cur_player = self.possible_agents.index(agent)
    
        # TODO
        pass


    

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

        # TODO
        pass
    
        

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """

        # TODO
        pass

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        # TODO
        pass



    