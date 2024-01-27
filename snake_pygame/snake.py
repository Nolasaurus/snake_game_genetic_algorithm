import pygame
from linkedlist import LinkedList

class Snake:
    def __init__(self, grid_size):
        self.step_count = 0
        self.grid_size = grid_size
        x_mid = grid_size[0] // 2
        y_mid = grid_size[1] // 2

        self.head_direction = 'DOWN'
        self.snake = LinkedList()

        # Add initial snake segments; top-left of screen is x=0, y=0
        self.snake.insert_at_beginning((x_mid, y_mid))               # Middle segment
        self.snake.insert_at_beginning((x_mid, y_mid + 1))           # Next Middle segment
        self.snake.insert_at_beginning((x_mid, y_mid + 2))           # Head

    def snake_body(self):
        return self.snake.to_list()
    
    def snake_length(self):
        return len(self.snake_body())

    def move(self, game_grid_object):
        head_x, head_y = self.snake_body()[0]

        # Movement logic
        if self.head_direction == 'UP':
            head_y -= 1
        elif self.head_direction == 'DOWN':
            head_y += 1
        elif self.head_direction == 'LEFT':
            head_x -= 1
        elif self.head_direction == 'RIGHT':
            head_x += 1

        next_head = (head_x, head_y)
        self.snake.insert_at_beginning(next_head)
        self.step_count += 1

        # Check if the snake ate food
        snake_ate_food = next_head in game_grid_object.food
        if snake_ate_food:
            game_grid_object.spawn_food()
        else:
            self.snake.remove_last_node()
