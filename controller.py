import os
import sys
import random
import time
import pygame
import pandas as pd

# Adjust the path if snake_pygame is in a different directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'snake_pygame'))

from user_interface import UserInterface, GameOverException  # Import the UserInterface class

def main():
    GAME_TICK = 3600
    WINDOW_HEIGHT = 800
    WINDOW_WIDTH = 800
    GRID_SIZE = (8, 8)
    RUN_HEADLESS = True
    NUM_TRIALS = 100
    
    tester = TrialRunner(100, GRID_SIZE, WINDOW_HEIGHT, WINDOW_WIDTH, GAME_TICK, RUN_HEADLESS)
    tester.run_trial()
    tester.print_table()

    pygame.quit()

class GameController:
    def __init__(self, grid_size, window_height, window_width,game_tick, run_headless):
        # Instantiate the game's user interface
        self.game = UserInterface(grid_size=grid_size, window_height=window_height, window_width=window_width, game_tick=game_tick, run_headless=run_headless)

    def start_game(self):
        # Start the game loop
        self.game.run()


    def play_game(self):
        try:
            game_start = time.time_ns()  # Start timing
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
                'num_steps': self.game.snake.step_count
            }

            return game_record

    def choose_direction(self):
        snake_body = self.game.snake.snake_body()
        first_seg = snake_body[0]
        second_seg = snake_body[1]
        x_dim, y_dim = self.game.game_grid.x_dim, self.game.game_grid.y_dim
        head_x = first_seg[0]
        head_y = first_seg[1]
        delta_x = head_x - second_seg[0]
        delta_y = head_y - second_seg[1]

        # Determine the invalid direction based on current movement
        body_directions = []
        if delta_x == 0:  # Vertical movement
            if delta_y > 0:  # Moving down
                body_directions.append('UP')
            else:  # Moving up
                body_directions.append('DOWN')
        else:  # Horizontal movement
            if delta_x > 0:  # Moving right
                body_directions.append('LEFT')
            else:  # Moving left
                body_directions.append('RIGHT')

        # Determine invalid directions based on other body segs
        if (head_x + 1, head_y) in snake_body:
            body_directions.append('RIGHT')
        if (head_x - 1, head_y) in snake_body:
            body_directions.append('LEFT')
        if (head_x, head_y + 1) in snake_body:
            body_directions.append('DOWN')
        if (head_x, head_y - 1) in snake_body:
            body_directions.append('UP')

        # Remove directions that would hit a wall
        wall_directions = []
        if first_seg[0] == 0:  # Leftmost column
            wall_directions.append('LEFT')
        if first_seg[0] == x_dim - 1:  # Rightmost column
            wall_directions.append('RIGHT')
        if first_seg[1] == 0:  # Top row
            wall_directions.append('UP')
        if first_seg[1] == y_dim - 1:  # Bottom row
            wall_directions.append('DOWN')

        valid_head_directions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        invalid_dirs = set(wall_directions + body_directions)
        valid_head_directions = [dir for dir in valid_head_directions if dir not in invalid_dirs]

        return random.choice(valid_head_directions) if valid_head_directions else 'DOWN'
    
import pandas as pd

class TrialRunner:
    def __init__(self, num_trials, grid_size, window_height, window_width, game_tick, run_headless):
        self.num_trials = num_trials
        self.grid_size = grid_size
        self.window_height = window_height
        self.window_width = window_width
        self.game_tick = game_tick
        self.run_headless = run_headless
        self.runs = {}

    def run_trial(self):
        for run_number in range(self.num_trials):
            game_controller = GameController(self.grid_size, self.window_height, self.window_width, self.game_tick, self.run_headless)
            self.runs[run_number] = game_controller.play_game()

    def output_table(self):
        return pd.DataFrame.from_dict(self.runs, orient='index')

    def print_table(self):
        print(self.output_table())


if __name__ == '__main__':
    main()
