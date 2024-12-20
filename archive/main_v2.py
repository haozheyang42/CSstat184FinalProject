import isolation_env
from bots import RandomBot, HeuristicBot, MCTSBot, DQNBot
import matplotlib.pyplot as plt

BOARD_SIZE = (6,8)

def run_one_game(bots, env):
    for agent in env.agent_iter():
        
        id = ['player_0', 'player_1'].index(agent)

        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            bots[id].learn(observation, reward)
            action = None

            if reward == 1:
                winner = agent

        else:
            action = bots[id].take_step(observation)

        env.step(action)
    
    return winner
    

if __name__ == "__main__":
    bots = []

    p0 = input("Enter Random, Heuristic, MCTS, DQN: ")
    p1 = input("Enter Random, Heuristic, MCTS, DQN: ")

    for p in (p0, p1):
        if p == "Random":
                bots.append(RandomBot())
        elif p == "Heuristic":
                bots.append(HeuristicBot(board_size=BOARD_SIZE))
        elif p == "MCTS":
                bots.append(MCTSBot(board_size=BOARD_SIZE))
        elif p == "DQN":
                bots.append(DQNBot(board_size=BOARD_SIZE))
        else:
                raise ValueError
            
    games = int(input("Enter number of games: "))
    env = isolation_env.env(board_size=BOARD_SIZE, render_mode=None)
    env.reset()
    
    batch_win_rates = []
    batch_wins = 0
    batch_size = 100
    print("Each Batch contains {} games".format(batch_size))

    for i in range(games):
        winner = run_one_game(bots, env)
        env.reset()
        
        if winner == "player_0":
            batch_wins += 1

        if i % batch_size == batch_size - 1:
            batch = i // batch_size
            batch_win_rate = batch_wins / batch_size
            print("Batch {} Win Percentage: {:.0%}".format(batch, batch_win_rate))
            batch_win_rates.append(batch_win_rate)
            batch_wins = 0

    env.close()
    plt.plot(batch_win_rates)