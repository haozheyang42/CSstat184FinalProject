import isolation_env
from bots import RandomBot, HeuristicBot, MCTSBot

BOARD_SIZE = (6,8)

def run_one_game(bots):

    for agent in env.agent_iter():
        id = env.agents.index(agent)

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            bots[id].learn(reward) # TODO
            action = None

            if reward == 1:
                winner = agent

        else:
            action = bots[id].take_step(observation)

        env.step(action)
    
    return winner
    

if __name__ == "__main__":
    bots = []

    p0 = input("Enter Random, Heuristic, or MCTS: ")
    p1 = input("Enter Random, Heuristic, or MCTS: ")

    for p in (p0, p1):
        match p:
            case "Random":
                bots.append(RandomBot())
            case "Heuristic":
                bots.append(HeuristicBot(board_size=BOARD_SIZE))
            case "MCTS":
                bots.append(MCTSBot(board_size=BOARD_SIZE))
            case _:
                raise ValueError
            
    games = int(input("Enter number of games: "))
    env = isolation_env.env(board_size=BOARD_SIZE, render_mode=None)
    env.reset()
    
    batch_wins = 0
    batch_size = 100
    print("Each Batch contains {} games".format(batch_size))

    for i in range(games):
        winner = run_one_game(bots)
        env.reset()
        
        if winner == "player_0":
            batch_wins += 1

        if i % batch_size == batch_size - 1:
            batch = i // batch_size
            print("Batch {} Win Percentage: {:.0%}".format(batch, batch_wins / batch_size))
            batch_wins = 0

    env.close()