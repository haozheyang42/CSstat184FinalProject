import pygame
import numpy as np
import sys
import random
import copy

# Constants for the game
BOARD_SIZE = (6, 8)
TILE_SIZE = 100
WINDOW_SIZE = (BOARD_SIZE[1] * TILE_SIZE, BOARD_SIZE[0] * TILE_SIZE)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PLAYER_COLORS = [(0, 128, 255), (255, 0, 128)]
PLAYER_STARTING_POS = [(2, 1), (3, 6)]
BLOCK_COLOR = (128, 128, 128)

# Action mappings for arrow keys
ACTIONS_MOVE = {
    pygame.K_UP: (-1, 0),
    pygame.K_DOWN: (1, 0),
    pygame.K_LEFT: (0, -1),
    pygame.K_RIGHT: (0, 1),
}

# Board values mapping
VALUES = ["Present, Unoccupied", "Present, Occupied by player 1", "Present, Occupied by player 2", "Removed"]
PLAYERVSBOT = True

class IsolationGame:
    def __init__(self, board_size=BOARD_SIZE, player_starting_pos=PLAYER_STARTING_POS):
        self.board_size = board_size
        self.board = np.zeros(board_size, dtype=int)
        self.player_positions = player_starting_pos
        self.board[player_starting_pos[0]] = 1
        self.board[player_starting_pos[1]] = 2
        self.current_player = 0  # Player 1 starts

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Isolation Game")
        self.clock = pygame.time.Clock()

    def draw_board(self):
        self.screen.fill(WHITE)
        for row in range(self.board_size[0]):
            for col in range(self.board_size[1]):
                rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if self.board[row, col] == 3:
                    pygame.draw.rect(self.screen, BLOCK_COLOR, rect)
                elif self.board[row, col] == 1:
                    pygame.draw.rect(self.screen, PLAYER_COLORS[0], rect)
                elif self.board[row, col] == 2:
                    pygame.draw.rect(self.screen, PLAYER_COLORS[1], rect)
                pygame.draw.rect(self.screen, BLACK, rect, 2)

    def _pos_in_board(self, pos):
        assert(len(pos) == 2)
        return all(0 <= i < j for i, j in zip(pos, self.board_size))
    
    def player_move(self, cur_pos, move_dir):
        new_pos = (cur_pos[0] + move_dir[0], cur_pos[1] + move_dir[1])

        # Check if index is in board and move is valid
        if not (self._pos_in_board(new_pos) and self.board[new_pos] == 0):
            return False
                
        # Move
        self.board[cur_pos] = 0
        self.board[new_pos] = self.current_player + 1
        self.player_positions[self.current_player] = new_pos
        return True

    def player_remove(self, remove_pos):
        # Check if index is in board and move is valid
        if not (self._pos_in_board(remove_pos) and self.board[remove_pos] == 0):
            return False
                
        # Remove
        self.board[remove_pos] = 3
        return True

    # # TO FIX (maybe): actions_move always valid. Actions_remove depends on choice of actions_move
    # def get_valid_moves(self, position):
    #     actions_move = []
    #     for move_dir in ACTIONS_MOVE.values():
    #         new_pos = (position[0] + move_dir[0], position[1] + move_dir[1])
    #         if self._pos_in_board(new_pos) and self.board[new_pos] == 0:
    #             actions_move.append(move_dir)
        
    #     actions_remove = np.where((self.board == 0) | (self.board == self.current_player + 1))
    #     actions_remove = list(zip(actions_remove[0], actions_remove[1]))
    #     return [actions_move, actions_remove]

    def get_valid_moves(self, cur_player):
        position = self.player_positions[cur_player]
        # print("cur_player", cur_player, "position", position) #debug line
        
        valid_moves = []
        for move_dir in ACTIONS_MOVE.values():
            new_pos = (position[0] + move_dir[0], position[1] + move_dir[1])
            if self._pos_in_board(new_pos) and self.board[new_pos] == 0:
                # Add the valid move direction and resulting free positions
                # re-create the board to find the free positions
                tmp_board = copy.deepcopy(self.board)
                tmp_board[position] = 0
                tmp_board[new_pos] = cur_player+1

                actions_remove = np.where((tmp_board == 0))
                actions_remove = list(zip(actions_remove[0], actions_remove[1]))
                valid_moves.append((move_dir, actions_remove))
        
        return valid_moves

    def bot_move(self):
        cur_pos = self.player_positions[self.current_player]
        valid_moves = self.get_valid_moves(self.current_player)
        if len(valid_moves) == 0: return False

        random_move = random.choice(valid_moves)
        random_remove = random.choice(random_move[1])
        move_dir = random_move[0]

        # Move
        new_pos = (cur_pos[0] + move_dir[0], cur_pos[1] + move_dir[1])
        self.board[cur_pos] = 0
        self.board[new_pos] = self.current_player + 1
        self.player_positions[self.current_player] = new_pos

        # Remove
        self.board[random_remove] = 3
        return True

    def run(self):
        running = True
        human_move_made = False
        while running:
            self.clock.tick(30)
            if PLAYERVSBOT:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    
                    # Human player's turn (Player 1) -- move
                    if self.current_player == 0 and not human_move_made:
                        if event.type == pygame.KEYDOWN and event.key in ACTIONS_MOVE:
                            if self.player_move(self.player_positions[self.current_player], ACTIONS_MOVE[event.key]):
                                human_move_made = True
                    
                    # Human player's turn (Player 1) -- remove
                    elif self.current_player == 0 and human_move_made:
                        if len(np.where(self.board == 0)) == 0:
                            print("Bot wins!")
                            running = False
                        else:
                            if event.type == pygame.MOUSEBUTTONDOWN:
                                x, y = event.pos
                                col = x // TILE_SIZE
                                row = y // TILE_SIZE
                                if self.board[row, col] == 0:
                                    self.board[row, col] = 3
                                    human_move_made = False
                                    self.current_player = 1  # Switch to bot
                                    break

                # Bot's turn (Player 2)
                if self.current_player == 1:
                    pygame.time.delay(500)  # Add a small delay for bot move visualization
                    if not self.bot_move():
                        print("Human wins!")
                        running = False
                        break
                    
                    self.current_player = 0
                    player_validmoved = self.get_valid_moves(0)
                    if len(player_validmoved) == 0:
                        print("Bot wins!")
                        running = False
                        break
            else:
                pygame.time.delay(500)
                if not self.bot_move():
                    print("Bot", 2-self.current_player, "wins!")
                    running = False
                    break
                self.current_player = 1-self.current_player

            self.draw_board()
            pygame.display.flip()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    while True:
        user_input = input("Is this player vs. bot or bot vs. bot? Enter P or B\n")
        if user_input in ["P", "B"]:
            print("Valid input received. Proceeding...")
            if user_input == "P": PLAYERVSBOT = True
            if user_input == "B": PLAYERVSBOT = False
            break
        else:
            print("Invalid input. Please try again.")
    
    game = IsolationGame()
    game.run()