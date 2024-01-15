import sys
from linkedlist import LinkedList

class Snake:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        x_mid = grid_size[0] // 2
        y_mid = grid_size[1] // 2

        self.head_direction = 'UP'
        self.snake = LinkedList()

        # Add initial snake segments; top-left of screen is x=0, y=0
        self.snake.insert_at_beginning((x_mid, y_mid))               # Middle segment
        self.snake.insert_at_beginning((x_mid, y_mid + 1))           # Next Middle segment
        self.snake.insert_at_beginning((x_mid, y_mid + 2))           # Head

    def snake_body(self):
        return self.snake.to_list()

    def move(self, game_grid_object):
        '''
        runs once per gametick
        '''
        head_x, head_y = self.snake_body()[0]
    
        if self.head_direction == 'UP':
            head_y -= 1
        elif self.head_direction == 'DOWN':
            head_y += 1
        elif self.head_direction == 'LEFT':
            head_x -= 1
        elif self.head_direction == 'RIGHT':
            head_x += 1

        next_head = (head_x, head_y)

        if head_x < 0 or head_y < 0:
            sys.exit('Snake hit a wall')
        if head_x >= self.grid_size[0] or head_y >= self.grid_size[1]:
            sys.exit('Snake hit a wall')

        self.snake.insert_at_beginning(next_head)

        # for collision
        #if food_collision:
        #    pass
        # elif wall_collision:
        #   pygame.endgame
        # else:
        #       remove last node
        if game_grid_object.food is None or next_head not in game_grid_object.food:
            self.snake.remove_last_node()

