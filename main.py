import pygame
import numpy as np
import sys
import random

# Constants for the game
BOARD_SIZE = (7, 7)
TILE_SIZE = 100
WINDOW_SIZE = (BOARD_SIZE[1] * TILE_SIZE, BOARD_SIZE[0] * TILE_SIZE)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PLAYER_COLORS = [(0, 128, 255), (255, 0, 128)]
BLOCK_COLOR = (128, 128, 128)

# Action mappings for arrow keys
ACTIONS = {
    pygame.K_UP: (-1, 0),
    pygame.K_DOWN: (1, 0),
    pygame.K_LEFT: (0, -1),
    pygame.K_RIGHT: (0, 1),
}

class IsolationGame:
    def __init__(self):
        self.board = np.zeros(BOARD_SIZE, dtype=int)
        self.player_positions = [(0, 0), (BOARD_SIZE[0] - 1, BOARD_SIZE[1] - 1)]
        self.board[0, 0] = 1
        self.board[-1, -1] = 2
        self.current_player = 0  # Player 1 starts

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Isolation Game")
        self.clock = pygame.time.Clock()

    def draw_board(self):
        self.screen.fill(WHITE)
        for row in range(BOARD_SIZE[0]):
            for col in range(BOARD_SIZE[1]):
                rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if self.board[row, col] == -1:
                    pygame.draw.rect(self.screen, BLOCK_COLOR, rect)
                elif self.board[row, col] == 1:
                    pygame.draw.rect(self.screen, PLAYER_COLORS[0], rect)
                elif self.board[row, col] == 2:
                    pygame.draw.rect(self.screen, PLAYER_COLORS[1], rect)
                pygame.draw.rect(self.screen, BLACK, rect, 2)

    def move_player(self, position, direction):
        dx, dy = direction
        new_pos = (position[0] + dx, position[1] + dy)

        # Check if the move is valid
        if 0 <= new_pos[0] < BOARD_SIZE[0] and 0 <= new_pos[1] < BOARD_SIZE[1] and self.board[new_pos] == 0:
            self.board[position] = -1  # Mark current position as blocked
            self.board[new_pos] = self.current_player + 1  # Update new position
            self.player_positions[self.current_player] = new_pos
            return True
        return False

    def get_valid_moves(self, position):
        moves = []
        for direction in ACTIONS.values():
            new_pos = (position[0] + direction[0], position[1] + direction[1])
            if 0 <= new_pos[0] < BOARD_SIZE[0] and 0 <= new_pos[1] < BOARD_SIZE[1] and self.board[new_pos] == 0:
                moves.append(new_pos)
        return moves

    def bot_move(self):
        current_pos = self.player_positions[self.current_player]
        valid_moves = self.get_valid_moves(current_pos)
        if valid_moves:
            new_pos = random.choice(valid_moves)  # Pick a random valid move
            self.board[current_pos] = -1
            self.board[new_pos] = self.current_player + 1
            self.player_positions[self.current_player] = new_pos

    def check_game_over(self):
        current_pos = self.player_positions[self.current_player]
        return len(self.get_valid_moves(current_pos)) == 0

    def run(self):
        running = True
        while running:
            self.clock.tick(30)

            # Human player's turn (Player 1)
            if self.current_player == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key in ACTIONS:
                        if self.move_player(self.player_positions[self.current_player], ACTIONS[event.key]):
                            if self.check_game_over():
                                print("Bot wins!")
                                running = False
                            else:
                                self.current_player = 1  # Switch to bot

            # Bot's turn (Player 2)
            elif self.current_player == 1:
                pygame.time.delay(500)  # Add a small delay for bot move visualization
                self.bot_move()
                if self.check_game_over():
                    print("Human wins!")
                    running = False
                else:
                    self.current_player = 0  # Switch to human

            self.draw_board()
            pygame.display.flip()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = IsolationGame()
    game.run()