from isolation_env import env
import numpy as np
import copy
import random
import math
import os
from collections import defaultdict
from pettingzoo.test import api_test
import imageio

# Initialize a NumPy array
NodeValues = np.zeros(2**12)
NodeVisits = np.zeros(2**12)

# Function to map a 12-tuple to an integer index
def tuple_to_index(t):
    return int(''.join('1' if x == 1 else '0' for x in t), 2)

class MCTSNode:
    def __init__(self, boardstate, starstate, parent=None):
        self.boardstate = boardstate
        self.starstate = starstate
        self.parent = parent
        self.children = []

    def is_fully_expanded(self):
        # Compare number of children to number of legal actions
        return len(self.children) == len(self.boardstate["legal_actions"])

    def best_child(self, exploration_weight=1.0):
        # Uses UCB formula: Q(c) + U(c) = (value/visits) + exploration_weight * sqrt(2*ln(N)/n)
        parent_visits = NodeVisits[tuple_to_index(self.starstate)] + 1e-9
        best_score = -float('inf')
        best = None
        for c in self.children:
            c_idx = tuple_to_index(c.starstate)
            c_visits = NodeVisits[c_idx] + 1e-9
            c_value = NodeValues[c_idx]
            ucb = (c_value / c_visits) + exploration_weight * math.sqrt((2 * math.log(parent_visits)) / c_visits)
            if ucb > best_score:
                best_score = ucb
                best = c
        return best

    def probabilistic_exploring(self):
        # Select a child based on softmax distribution over average values
        values = []
        for c in self.children:
            c_idx = tuple_to_index(c.starstate)
            c_visits = NodeVisits[c_idx] + 1e-9
            c_value = NodeValues[c_idx]
            avg_value = c_value / c_visits
            values.append(avg_value)
        exp_values = np.exp(values - np.max(values))
        probs = exp_values / np.sum(exp_values)
        chosen_child = np.random.choice(self.children, p=probs)
        return chosen_child


def select_node(node):
    # Selection step: move down the tree using best_child until we reach a node that is not fully expanded
    while node.is_fully_expanded() and node.children:
        node = node.best_child(exploration_weight=1.0)
    return node

def expand_node(node, env):
    """This function will fully expand the children of a particular node"""
    legal_actions = node.boardstate["legal_actions"]
    existing_actions = [c.boardstate["action"] for c in node.children if "action" in c.boardstate]

    for action in legal_actions:
        # Only add a new child if it's not already there
        if action not in existing_actions:
            env_copy = copy.deepcopy(env)
            env_copy.step(action)

            child_state = {
                "observation": env_copy.observe(env_copy.agent_selection),
                "action": action,
                "legal_actions": env_copy.observe(env_copy.agent_selection)["action_mask"].nonzero()[0],
            }
            sstate = get_star_shape(env_copy, env_copy.agent_selection)
            child_node = MCTSNode(boardstate=child_state, starstate=sstate, parent=node)
            node.children.append(child_node)

    # Return a child selected probabilistically if any children were added
    if len(node.children) > 0: 
        return node.probabilistic_exploring()
    return None

def get_star_shape(env, agent):
    """
    Extract the 12-tile star shape surrounding the given agent's position.
    Empty cells = +1, blocked/out-of-bounds/occupied = -1
    """
    player = 1 if agent == "player_0" else 2
    cur_pos = tuple(np.argwhere(env.board == player)[0])
    deltas = [
        (-2, 0), (-1, -1), (-1, 0), (-1, 1),
        (0, -2), (0, -1), (0, 1), (0, 2),
        (1, -1), (1, 0), (1, 1), (2, 0),
    ]
    star_tiles = []
    for dx, dy in deltas:
        new_pos = (cur_pos[0] + dx, cur_pos[1] + dy)
        if 0 <= new_pos[0] < env.board.shape[0] and 0 <= new_pos[1] < env.board.shape[1]:
            if env.board[new_pos] == 0:
                star_tiles.append(1)
            else:
                star_tiles.append(-1)
        else:
            star_tiles.append(-1)  # Treat out-of-bounds as blocked
    return tuple(star_tiles)

def simulate(env, max_depth=10):
    # Simulation: random rollout until max_depth or terminal
    depth = 0
    cumulative_reward = 0
    env_copy = copy.deepcopy(env)
    while depth < max_depth and not any(env_copy.terminations.values()):
        legal_actions = env_copy.observe(env_copy.agent_selection)["action_mask"].nonzero()[0]
        if legal_actions.size == 0:  # no moves
            break

        # Probabilistic choice (here we just choose uniformly at random)
        action = random.choice(legal_actions)

        env_copy.step(action)
        cumulative_reward += sum(env_copy.rewards.values())
        depth += 1
    return cumulative_reward

def backpropagate(node, reward):
    # Backpropagation: update NodeValues and NodeVisits along the path back to the root
    idx = tuple_to_index(node.starstate)
    NodeValues[idx] += reward
    NodeVisits[idx] += 1
    if node.parent is not None:
        backpropagate(node.parent, reward)

def mcts(env, num_simulations=1000, exploration_weight=1.0):
    # Initialize root node
    print("MCTS fn called")
    bstate = {
        "observation": env.observe(env.agent_selection),
        "legal_actions": env.observe(env.agent_selection)["action_mask"].nonzero()[0]
    }
    sstate = get_star_shape(env, env.agent_selection)
    root = MCTSNode(boardstate=bstate, starstate=sstate)

    for _ in range(num_simulations):
        # Make a copy of the environment to move it down to the selected node
        env_copy = copy.deepcopy(env)

        # Selection: find the node to expand/simulate
        node = select_node(root)

        # Rebuild the environment state up to the selected node
        # We'll do this by tracing back to the root and replaying actions
        actions_path = []
        current = node
        while current.parent is not None:
            actions_path.append(current.boardstate["action"])
            current = current.parent
        actions_path.reverse()

        # Step through the actions_path on env_copy
        for a in actions_path:
            env_copy.step(a)

        # Expansion
        simulation_node = expand_node(node, env_copy)
        if simulation_node is None:
            simulation_node = node  # If no expansion took place

        # After expansion, we must simulate from simulation_node
        # Rebuild env state at simulation_node:
        sim_env = copy.deepcopy(env)
        # Replay actions from root to simulation_node
        sim_actions_path = []
        cur_sim = simulation_node
        while cur_sim.parent is not None:
            sim_actions_path.append(cur_sim.boardstate["action"])
            cur_sim = cur_sim.parent
        sim_actions_path.reverse()

        for a in sim_actions_path:
            sim_env.step(a)

        # Simulation: random rollout
        reward = simulate(sim_env, max_depth=10)

        # Backpropagation
        backpropagate(simulation_node, reward)

    # After all simulations, pick the best child without exploration for the final action
    best_final_child = root.best_child(exploration_weight=0.0)
    return best_final_child.boardstate["action"]

def train_and_save_videos(num_games=1, filename_prefix="gameplay"):
    for game_index in range(num_games):
        isolation_env = env(render_mode="rgb_array")
        isolation_env.reset()
        frames = []
        print(f"Starting Game {game_index + 1}...")

        while isolation_env.agents:
            agent = isolation_env.agent_selection
            observation = isolation_env.observe(agent)

            if not isolation_env.terminations[agent] and not isolation_env.truncations[agent]:
                # Use MCTS to select the action
                action = mcts(isolation_env, num_simulations=1000)
                isolation_env.step(action)
            else:
                # If the agent is terminated or truncated, just proceed
                isolation_env.step(None)

            frame = isolation_env.render()
            if frame is not None:
                frames.append(frame)

        isolation_env.close()

        # Save the frames as a video
        print(f"Game {game_index + 1} finished!")
        os.makedirs("./videos", exist_ok=True)
        video_filename = f"./videos/{filename_prefix}_game{game_index + 1}.mp4"
        imageio.mimwrite(video_filename, frames, fps=30)
        print(f"Game {game_index + 1} saved as {video_filename}")

# Run the function to train and save videos
train_and_save_videos()