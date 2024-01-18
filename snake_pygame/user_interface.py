import pygame

from snake import Snake
from game_grid import GameGrid

class UserInterface():
    def __init__(self, **kwargs):
        pygame.init()

        # Extract arguments from kwargs with default values
        window_width = kwargs.get('window_width', 640)  # Default width
        window_height = kwargs.get('window_height', 640)  # Default height
        grid_size = kwargs.get('grid_size', (9, 9))  # Default grid size
        self.game_tick = kwargs.get('game_tick', 6) # Default game speed

        self.window = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()
        self.game_grid = GameGrid(grid_size)
        self.snake = Snake(grid_size)
        self.game_grid.update_snake(self.snake)
        self.game_grid.spawn_food()
        self.running = True

        grid_dict = {}
        # name the grid cells and store their respective  (0,0)
        self.cell_size_x = window_height // grid_size[0]
        self.cell_size_y = window_width // grid_size[1]

        for x_num in range(grid_size[0]):
            for y_num in range(grid_size[1]):
                grid_dict[(x_num, y_num)] = (x_num * self.cell_size_x, y_num * self.cell_size_y)

        self.grid_to_pixel_dict = grid_dict.copy()

    def set_snake_direction(self, direction):
        '''
        Allow setting snake direction from outside class
        '''
        self.snake.head_direction = direction

    def processInput(self):
        '''
        handle keypress events
        QUIT or change head_direction depending on keypress
        WASD and Arrow keys
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    break
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    self.snake.head_direction = 'RIGHT'
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    self.snake.head_direction = 'LEFT'
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    self.snake.head_direction = 'DOWN'
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    self.snake.head_direction = 'UP'

    def render(self):
        '''
        Calculates each gridsquare's (0,0) top-left corner and stores in dict. 
        Key: nickname (x,y), e.g. (3,2) for the 3rd col, 2nd row.
        Value: "(0, 0)" for gridsquare
        '''        
        self.window.fill((0,0,0))
        unitsTexture = pygame.image.load("snake_pygame/units.png")
        
        # Draw snake
        snake = self.snake.snake_body()

        # Draw body
        for body_piece in snake:
            body_x, body_y = self.grid_to_pixel_dict[body_piece]
            location = pygame.Vector2(body_x, body_y) # where in the scene
            rectangle = pygame.Rect(0, 64 , 64, 64) # where in the units.png
            self.window.blit(unitsTexture,location,rectangle)

        for food_piece in self.game_grid.food:
            food_x, food_y = self.grid_to_pixel_dict[food_piece]
            location = pygame.Vector2(food_x, food_y) # where in the scene
            rectangle = pygame.Rect(320, 64, 64, 64) # where in the units.png
            self.window.blit(unitsTexture,location,rectangle)

        pygame.display.update()


    def run(self):
        while self.running:
            self.processInput()
            self.game_grid.update_snake(self.snake)
            self.snake.move(self.game_grid)
            self.render()
            self.clock.tick(self.game_tick)