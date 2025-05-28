# game/config.py
from os.path import abspath, dirname, join

# We need to go one level up to get the project root.
PROJECT_ROOT = abspath(join(dirname(__file__), '..')) # Go up one level

FONT_PATH = join(PROJECT_ROOT, 'fonts/')
IMAGE_PATH = join(PROJECT_ROOT, 'images/')
SOUND_PATH = join(PROJECT_ROOT, 'sounds/')

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors (R, G, B)
WHITE = (255, 255, 255)
GREEN = (78, 255, 87)
YELLOW = (241, 255, 0)
BLUE = (80, 255, 239)
PURPLE = (203, 0, 255)
RED = (237, 28, 36)

# Font
GAME_FONT_NAME = 'space_invaders.ttf'
GAME_FONT = FONT_PATH + GAME_FONT_NAME

# Images
IMG_NAMES = ['ship', 'mystery',
             'enemy1_1', 'enemy1_2',
             'enemy2_1', 'enemy2_2',
             'enemy3_1', 'enemy3_2',
             'explosionblue', 'explosiongreen', 'explosionpurple',
             'laser', 'enemylaser']

# Game settings
BLOCKERS_POSITION = 450
ENEMY_DEFAULT_POSITION = 65  # Initial y value for enemies in a new game
ENEMY_MOVE_DOWN = 35  # How much enemies move down when changing direction
FPS = 60

# Player settings
PLAYER_SPEED = 5
PLAYER_START_X = 375
PLAYER_START_Y = 540
PLAYER_LASER_SPEED = 15

# Enemy settings
ENEMY_LASER_SPEED = 5
ENEMY_SHOOT_INTERVAL = 700  # ms, original value was 700

# Mystery ship settings
MYSTERY_SHIP_START_Y = 45
MYSTERY_SHIP_MOVE_TIME = 25000  # ms, time until it appears/moves

# Score values
# Enemy rows for scoring: 0=topmost graphics (30pts), 1,2 (20pts), 3,4 (10pts)
# This matches original image naming (enemy1_x, enemy2_x, enemy3_x) vs row index.
# Original code structure for Enemy rows:
#   Row 0: type 'enemy1_x' (actually displayed as row 3/4 type)
#   Row 1, 2: type 'enemy2_x' (actually displayed as row 1/2 type)
#   Row 3, 4: type 'enemy3_x' (actually displayed as row 0 type)
# Let's use the enemy type index from Enemy.load_images for scoring:
# images = {0: ['1_2', '1_1'], -> score for self.row == 0 should be for enemy1 type (e.g. 10 pts)
#           1: ['2_2', '2_1'], -> score for self.row == 1 should be for enemy2 type (e.g. 20 pts)
#           2: ['2_2', '2_1'], -> score for self.row == 2 should be for enemy2 type (e.g. 20 pts)
#           3: ['3_1', '3_2'], -> score for self.row == 3 should be for enemy3 type (e.g. 30 pts)
#           4: ['3_1', '3_2'], -> score for self.row == 4 should be for enemy3 type (e.g. 30 pts)
# Re-mapping based on actual visual types and common Space Invaders scoring:
# Top type (smallest, often highest score or enemy3 type visually)
# Middle type (enemy2 type visually)
# Bottom type (largest, often lowest score or enemy1 type visually)
# The original code's `calculate_score` used:
# scores = {0: 30, 1: 20, 2: 20, 3: 10, 4: 10, 5: mystery}
# This implies row 0 in the enemy grid (topmost on screen) is worth 30 points.
# And row 4 (bottom-most enemy row on screen) is worth 10 points.
ENEMY_SCORES_BY_ROW_INDEX = { # Key is the 'row' index passed to Enemy constructor
    0: 30,  # Top row of enemies on screen
    1: 20,
    2: 20,
    3: 10,
    4: 10   # Bottom row of enemies on screen
}
MYSTERY_SCORES_OPTIONS = [50, 100, 150, 300]

# Action space for AI (example)
# These are just placeholders, actual action mapping depends on agent needs
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_SHOOT = 2
ACTION_NONE = 3
NUM_ACTIONS = 4 # Example: left, right, shoot, none