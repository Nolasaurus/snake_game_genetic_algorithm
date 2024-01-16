import os

import pygame

from user_interface import UserInterface

os.environ['SDL_VIDEO_CENTERED'] = '1'

WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
GRID_SIZE = (16, 16)

def main():
    user_interface = UserInterface(grid_size = GRID_SIZE, window_height = WINDOW_HEIGHT, window_width = WINDOW_WIDTH)
    user_interface.run()
    pygame.quit()

if __name__ == '__main__':
    main()