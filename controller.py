import os
import sys
import time
import pygame
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from snake_pygame.user_interface import UserInterface, GameOverException

sys.path.append(os.path.join(os.path.dirname(__file__), 'snake_pygame'))

def main():
    GAME_TICK = 3600
    WINDOW_HEIGHT = 800
    WINDOW_WIDTH = 800
    GRID_SIZE = (8, 8)
    RUN_HEADLESS = True
    NUM_TRIALS = 100
    
    game = GameController(GRID_SIZE, WINDOW_HEIGHT, WINDOW_WIDTH, GAME_TICK, RUN_HEADLESS)
    run = game.play_game()
    print(run)
    pygame.quit()

class SnakeNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SnakeNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    
    def set_weights(self, weights):
        fc2_weights, fc3_weights = weights
        self.fc2.weight.data = fc2_weights
        self.fc3.weight.data = fc3_weights

    def extract_weights(self):
        fc2_weights = self.fc2.weight.data
        fc3_weights = self.fc3.weight.data

        return pd.DataFrame(fc2_weights), pd.DataFrame(fc3_weights)

class GameController:
    def __init__(self, grid_size, window_height, window_width,game_tick, run_headless):
        # Instantiate the game's user interface
        self.game = UserInterface(grid_size=grid_size, window_height=window_height, window_width=window_width, game_tick=game_tick, run_headless=run_headless)
        input_size = grid_size[0] * grid_size[1] + 1
        output_size = 4  # For 4 directions
        hidden_size = 64  # tune this
        self.model = SnakeNN(input_size, hidden_size, output_size)

    def start_game(self):
        # Start the game loop
        self.game.run()


    def play_game(self):
        game_start = time.time_ns()  # Start timing
        try:
            while self.game.running:
                # Choose a new head direction at each tick
                new_direction = self.choose_direction()
                self.game.set_snake_direction(new_direction)

                # Update the game state and render
                self.game.processInput()
                self.game.game_grid.update_snake(self.game.snake)
                self.game.snake.move(self.game.game_grid)
                if self.game.snake_in_wall_or_body():
                    raise GameOverException
                if not self.game.headless:
                    self.game.render()

                self.game.clock.tick(self.game.game_tick)

        except GameOverException:
            game_end = time.time_ns()  # End timing
            runtime = game_end - game_start
            runtime_ms = round(runtime / 1e6,1)  # Convert nanoseconds to ms

            snek = self.game.snake.snake_body()
            snake_length = len(self.game.snake.snake_body())

            game_record = {
                'snake_length': snake_length,
                'grid_size': (self.game.game_grid.x_dim, self.game.game_grid.y_dim),
                'runtime_milliseconds': runtime_ms,
                'num_steps': self.game.snake.step_count,
                'fc2_weights' : self.model.fc2.weight.data,
                'fc3_weights' : self.model.fc3.weight.data
            }

            return game_record


    def choose_direction(self):
        # game state inputs
        last_head_direction = self.game.snake.head_direction
        snake_body = self.game.snake.snake_body()
        food_locations = self.game.game_grid.food
        grid_size = self.game.grid_size
        game_state = torch.zeros(grid_size)

        # Mark food locations in the game state
        for food in food_locations:
            game_state[food[0], food[1]] = 1

        # Mark snake body in the game state
        for body_part in snake_body:
            game_state[body_part[0], body_part[1]] = -1

        game_state_flat = game_state.flatten()
        direction_encoding = torch.tensor([last_head_direction], dtype=torch.float32)
        neural_net_input = torch.cat((direction_encoding, game_state_flat), 0)
        neural_net_input = neural_net_input.unsqueeze(0)  # Add a batch dimension
        head_direction = torch.argmax(self.model(neural_net_input)).item()
        head_direction = head_direction * 90
        return head_direction

if __name__ == '__main__':
    main()
