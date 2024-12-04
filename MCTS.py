from isolation_env import env
import numpy as np
import copy
import random
import math
import os
from collections import defaultdict
from pettingzoo.test import api_test
import imageio

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state["legal_actions"])

    def best_child(self, exploration_weight=1.0):
        # UCB formula: Q/N + c * sqrt(log(N_parent) / N)
        return max(
            self.children,
            key=lambda child: (child.value / (child.visits + 1e-6)) +
                exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
        )

def select_node(node):
    while node.is_fully_expanded() and node.children:
        node = node.best_child()
    return node

def expand_node(node, env):
    legal_actions = env.observe(env.agent_selection)["action_mask"].nonzero()[0]
    for action in legal_actions:
        if action not in [child.state["action"] for child in node.children]:
            env_copy = copy.deepcopy(env)
            env_copy.step(action)
            child_state = {
                "observation": env_copy.observe(env_copy.agent_selection),
                "action": action,
                "legal_actions": env_copy.observe(env_copy.agent_selection)["action_mask"].nonzero()[0],
            }
            child_node = MCTSNode(state=child_state, parent=node)
            node.children.append(child_node)
            return child_node
    return None

def simulate(env, max_depth=10):
    depth = 0
    cumulative_reward = 0
    env_copy = copy.deepcopy(env)
    while depth < max_depth and not any(env_copy.terminations.values()):
        legal_actions = env_copy.observe(env_copy.agent_selection)["action_mask"].nonzero()[0]
        if legal_actions.size == 0:  # Fix: Check if legal_actions is empty
            break
        action = random.choice(legal_actions)
        env_copy.step(action)
        cumulative_reward += sum(env_copy.rewards.values())
        depth += 1
    return cumulative_reward

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.value += reward
        node = node.parent

def mcts(env, num_simulations=1000, exploration_weight=1.0):
    root = MCTSNode(state={"observation": env.observe(env.agent_selection), "legal_actions": env.observe(env.agent_selection)["action_mask"].nonzero()[0]})
    print("mcts function called")
    for i in range(num_simulations):
        node = select_node(root)
        if not node.is_fully_expanded():
            child = expand_node(node, env)
            if child:
                reward = simulate(env)
                backpropagate(child, reward)
    return root.best_child(exploration_weight).state["action"]

def train_and_save_videos(num_games=2, filename_prefix="gameplay"):
    """
    Train the agent on multiple games and save gameplay as videos.
    
    Args:
        num_games: Number of games to train the agent on.
        filename_prefix: Prefix for the video filenames.
    """
    for game_index in range(num_games):
        isolation_env = env(render_mode="rgb_array")
        isolation_env.reset()
        frames = []
        print(f"Starting Game {game_index + 1}...")
        
        # Play the game
        while isolation_env.agents:
            agent = isolation_env.agent_selection
            observation = isolation_env.observe(agent)

            if not isolation_env.terminations[agent] and not isolation_env.truncations[agent]:
                # Use MCTS to select the action
                action = mcts(isolation_env, num_simulations=1000)
                isolation_env.step(action)
            else:
                isolation_env.step(None)  # Step with None if the agent is terminated

            # Capture the frame for the video
            frame = isolation_env.render()
            if frame is not None:
                frames.append(frame)
        
        isolation_env.close()

        # Save the frames as a video
        print(f"Game {game_index + 1} finished!")
        os.makedirs("./videos", exist_ok=True)
        video_filename = f"./videos/gameplay_game{game_index + 1}.mp4"
        imageio.mimwrite(video_filename, frames, fps=30)
        print(f"Game {game_index + 1} saved as {video_filename}")

# Run the function to train and save videos
train_and_save_videos()