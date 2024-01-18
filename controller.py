import os
import sys
import random

# Adjust the path if snake_pygame is in a different directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'snake_pygame'))

from user_interface import UserInterface  # Import the UserInterface class



def main():
    GAME_TICK = 60
    WINDOW_HEIGHT = 800
    WINDOW_WIDTH = 800
    GRID_SIZE = (16, 16)

    controller = GameController(GRID_SIZE, WINDOW_HEIGHT, WINDOW_WIDTH, GAME_TICK)
    controller.play_game()  # This now controls the game loop

class GameController:
    def __init__(self, grid_size, window_height, window_width,game_tick):
        # Instantiate the game's user interface
        self.user_interface = UserInterface(grid_size=grid_size, window_height=window_height, window_width=window_width, game_tick=game_tick)


    def start_game(self):
        # Start the game loop
        self.user_interface.run()


    def play_game(self):
        while self.user_interface.running:
            # Choose a new head direction at each tick
            new_direction = self.choose_direction()

            self.user_interface.set_snake_direction(new_direction)

            # Update the game state and render
            self.user_interface.processInput()
            self.user_interface.game_grid.update_snake(self.user_interface.snake)
            self.user_interface.snake.move(self.user_interface.game_grid)
            self.user_interface.render()
            self.user_interface.clock.tick(self.user_interface.game_tick)

    def choose_direction(self):
        valid_head_directions = {'RIGHT', 'DOWN', 'LEFT', 'UP'}  # Use a set for easier removal
        snake_body = self.user_interface.snake.snake_body()
        first_seg, second_seg = snake_body[0], snake_body[1]

        delta_x = second_seg[0] - first_seg[0]
        delta_y = second_seg[1] - first_seg[1]

        # Determine the invalid direction based on current movement
        if delta_x == 0:  # Vertical movement
            if delta_y > 0:  # Moving down
                invalid_direction = 'DOWN'
            else:  # Moving up
                invalid_direction = 'UP'
        else:  # Horizontal movement
            if delta_x > 0:  # Moving right
                invalid_direction = 'RIGHT'
            else:  # Moving left
                invalid_direction = 'LEFT'

        # Remove the invalid direction
        valid_head_directions.remove(invalid_direction)

        # Wall Logic
        # Remove directions that would hit a wall
        x_dim = self.user_interface.game_grid.x_dim
        y_dim = self.user_interface.game_grid.y_dim
        if first_seg[0] == 0:  # Leftmost column
            valid_head_directions.discard('LEFT')
        if first_seg[0] == x_dim - 1:  # Rightmost column
            valid_head_directions.discard('RIGHT')
        if first_seg[1] == 0:  # Top row
            valid_head_directions.discard('UP')
        if first_seg[1] == y_dim - 1:  # Bottom row
            valid_head_directions.discard('DOWN')
        # Return a random valid direction
        return random.choice(list(valid_head_directions))


if __name__ == '__main__':
    main()
    