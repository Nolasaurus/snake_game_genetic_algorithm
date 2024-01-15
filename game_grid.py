import random
from snake import Snake
def main():
    grid_size = (9, 9)
    game_grid = GameGrid(grid_size)
    snake = Snake(grid_size)
    game_grid.update_snake(snake)
    game_grid.spawn_food()
    game_grid.print_grid()

class GameGrid:
    '''
    Creates dict of game grid as coordinate pairs, (e.g., for a 9x9 grid: x: 0-8, y: 0-8)

    '''
    def __init__(self, grid_size=(9,9)):
        self.x_dim = grid_size[0]
        self.y_dim = grid_size[1]
        self.food = None
        self.snake = None
        grid_list = []

        for x_num in range(self.x_dim):
            for y_num in range(self.y_dim):
                grid_list.append((x_num, y_num))

        self.grid = grid_list

    def update_snake(self, snake_object):
        self.snake = snake_object
    
    def spawn_food(self):
        list_of_snake_cell_locations = self.snake.snake_body()
        food_locations = random.choice([xy_coord for xy_coord in self.grid if xy_coord not in list_of_snake_cell_locations])
        self.food = food_locations

        return food_locations
    

    def print_grid(self):
        # ANSI escape codes for colors
        BLACK = '\033[30m'
        RED = '\033[31m'
        BRIGHT_YELLOW = '\033[93m'
        RESET = '\033[0m'  # Resets the color to default

        snake_body = self.snake.snake_body()
        for y_num in range(self.y_dim):
            for x_num in range(self.x_dim):
                cell = (x_num, y_num)
                if cell in snake_body:
                    print(f'{BRIGHT_YELLOW}X{RESET}', end=' ')
                elif cell == self.food:
                    print(f'{RED}F{RESET}', end=' ')
                else:
                    print(f'{BLACK}O{RESET}', end=' ')
            print()  # New line after each row

if __name__ == "__main__":
    main()