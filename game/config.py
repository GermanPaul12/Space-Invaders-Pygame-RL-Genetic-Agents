# game/config.py
from os.path import abspath, dirname, join

# --- Project Root and Asset Paths ---
PROJECT_ROOT = abspath(join(dirname(__file__), '..')) # Correct: game/../ -> project_root/

FONT_PATH = join(PROJECT_ROOT, 'fonts/')
IMAGE_PATH = join(PROJECT_ROOT, 'images/')
SOUND_PATH = join(PROJECT_ROOT, 'sounds/')

# --- Screen Dimensions ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# --- Colors (R, G, B) ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (78, 255, 87)
YELLOW = (241, 255, 0)
BLUE = (80, 255, 239)
PURPLE = (203, 0, 255)
RED = (237, 28, 36)

# --- Font ---
GAME_FONT_NAME = 'space_invaders.ttf' # Ensure this file is in FONT_PATH
GAME_FONT = join(FONT_PATH, GAME_FONT_NAME) if FONT_PATH else None # Fallback if no font path

# --- Images ---
# List of base image names (without .png extension)
IMG_NAMES = [
    'ship', 'mystery',
    'enemy1_1', 'enemy1_2', # Type 1 (e.g., bottom row)
    'enemy2_1', 'enemy2_2', # Type 2 (e.g., middle rows)
    'enemy3_1', 'enemy3_2', # Type 3 (e.g., top row)
    'explosionblue', 'explosiongreen', 'explosionpurple', # Generic explosion colors
    'laser', 'enemylaser',
]
# Specify which images have transparency and should use convert_alpha()
IMAGES_WITH_ALPHA = [
    'ship', 'mystery', 'enemy1_1', 'enemy1_2', 'enemy2_1', 'enemy2_2',
    'enemy3_1', 'enemy3_2', 'explosionblue', 'explosiongreen', 'explosionpurple',
    'laser', 'enemylaser'
]


# --- Game Settings ---
FPS = 120 # Frames per second for player mode
AI_TRAIN_RENDER_FPS = 480 # FPS if rendering during AI training/testing
PLAYER_LIVES = 3
BLOCKERS_POSITION = 450 # Y-coordinate for the top of the blockers
BLOCKER_PIECE_SIZE = 10 # Size of one square piece of a blocker
BLOCKER_ROWS = 4 # Number of rows of pieces in a blocker shield
BLOCKER_COLS_PER_PIECE = 3 # Number of columns for one visual "piece" of blocker shield.
                           # A full shield might be 3 such pieces wide.

ENEMY_DEFAULT_POSITION = 65  # Initial Y value for the top row of enemies
ENEMY_MOVE_DOWN_NEW_ROUND = 20 # How much enemies move down when a new round starts
PLAYER_AREA_HEIGHT = 80 # Height of the area at the bottom where player is; if enemies enter, game over.


# --- Player Settings ---
PLAYER_SPEED = 5
PLAYER_START_X = SCREEN_WIDTH // 2 - 25 # Centered, assuming player width ~50
PLAYER_START_Y = SCREEN_HEIGHT - 60
PLAYER_MIN_X = 10 # Minimum X position for player ship
PLAYER_MAX_X = SCREEN_WIDTH - 10 # Maximum X position for player ship (right edge)
PLAYER_LASER_SPEED = 10
MAX_PLAYER_BULLETS = 2 # Max number of player bullets on screen at once
PLAYER_WIDTH = 50 # Approx visual width
PLAYER_HEIGHT = 30 # Approx visual height
PLAYER_RESPAWN_DELAY_MS = 2000 # Delay in ms before player respawns
AI_PLAYER_RESPAWN_DELAY_MS = 0 # No delay for AI respawn


# --- Enemy Settings ---
ENEMY_ROWS = 5
ENEMY_COLUMNS = 10
ENEMY_WIDTH = 40
ENEMY_HEIGHT = 30
ENEMY_START_X_OFFSET = 75 # X offset for the first column of enemies
ENEMY_X_SPACING = 50    # Horizontal space between enemies
ENEMY_Y_SPACING = 45    # Vertical space between enemies

ENEMY_LASER_SPEED = 7
ENEMY_SHOOT_INTERVAL = 800  # ms, base interval for enemy shooting
ENEMY_MOVE_TIME_INITIAL = 1000 # ms, initial time per enemy move (slower)
ENEMY_MOVE_TIME_DECREMENT = 50 # ms, how much move time decreases on speedup
ENEMY_MOVE_TIME_MIN = 100      # ms, fastest enemy move time
ENEMY_SPEEDUP_THRESHOLD_KILLS = 5 # Number of enemies killed to trigger speedup
ENEMY_MOVE_DOWN_STEP = 15      # How much enemies move down when changing direction at edge
ENEMY_X_SHIFT_AMOUNT = 10      # How much enemies shift horizontally per move
ENEMY_MAX_HORIZONTAL_MOVES = 30 # Number of horizontal moves before moving down

# Defines which sprite names (from IMG_NAMES) correspond to which enemy row in formation
# Useful for Enemy class to load its correct animation frames.
ENEMY_ROW_SPRITES = {
    'row0': ['enemy3_1', 'enemy3_2'],    # Top row of formation (visually small, often high score)
    'row1_2': ['enemy2_1', 'enemy2_2'],  # Middle rows
    'row3_4': ['enemy1_1', 'enemy1_2']   # Bottom rows (visually larger, often low score)
}


# --- Mystery Ship Settings ---
MYSTERY_SHIP_START_Y = 45
MYSTERY_SPEED = 3
MYSTERY_WIDTH = 75
MYSTERY_HEIGHT = 35
MYSTERY_APPEAR_INTERVAL_MS = [15000, 20000, 25000] # Random choice from these for next appearance in ms


# --- Score Values ---
# Score for enemies based on their row index in the formation (0 = top row on screen)
ENEMY_SCORES_BY_ROW_INDEX = { 
    0: 30,  # Top row of enemies on screen (e.g., using 'enemy3_X' sprites)
    1: 20,
    2: 20,  # Middle rows (e.g., using 'enemy2_X' sprites)
    3: 10,
    4: 10   # Bottom row of enemies on screen (e.g., using 'enemy1_X' sprites)
}
MYSTERY_SCORES_OPTIONS = [50, 100, 150, 300] # Possible scores for destroying mystery ship

# --- Explosion Settings ---
EXPLOSION_ENEMY_FRAMES = ['explosionblue', 'explosiongreen'] # Example, can be more specific
EXPLOSION_SHIP_FRAMES = ['ship', 'explosionpurple'] # Player ship 'image' then an explosion
EXPLOSION_FRAME_DURATION_MS = 100 # Duration per frame of explosion animation
MYSTERY_EXPLOSION_DURATION_MS = 500 # How long score from mystery ship stays on screen

# --- Text & UI Timings ---
INTER_ROUND_DELAY_MS = 2000 # Delay for "Next Round" text
AI_INTER_ROUND_DELAY_MS = 0 # No delay for AI
GAME_OVER_TEXT_BLINK_INTERVAL_MS = 750 # Blinking interval for "Game Over"

# --- Life Sprite Info ---
LIFE_SPRITE_WIDTH = 23
LIFE_SPRITE_HEIGHT = 23

# --- Action Space for AI ---
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_SHOOT = 2
ACTION_NONE = 3 
NUM_ACTIONS = 4
# If ACTION_NONE is explicit: NUM_ACTIONS = 4

# --- Rewards for AI ---
REWARD_ENEMY_KILL = 100
REWARD_MYSTERY_KILL = 500 # WAS 50.
REWARD_LIFE_LOST = -700 # Keep penalty for dying significant.
REWARD_ROUND_CLEAR = 200 # WAS 200.
REWARD_PER_STEP_ALIVE = 0.1 # REDUCE this if it's making hiding too appealing. Or keep it small.

REWARD_UNDER_ENEMY = 10       # Small reward for being positioned under an enemy
ALIGNMENT_TOLERANCE_X = 10      # How close in X the player needs to be to an enemy column (logical units)

# --- New Reward Shaping Constants ---
PUNISHMENT_ACTION_NONE = 0        # WAS -0.05. Slightly increase penalty for doing nothing.
PUNISHMENT_SHOOT_MISS = -500      # WAS -0.1. Maybe slightly reduce miss penalty if kill reward is high.
PUNISHMENT_ENEMY_ADVANCE_BASE = -0.001 # WAS -0.001. Increase base penalty per enemy alive.
PUNISHMENT_ENEMY_PROXIMITY_SCALE = -0.005 # WAS -0.005. Increase penalty for close enemies.
PUNISHMENT_ENEMY_REACHED_PLAYER_AREA = -1000 # WAS -1000. Keep significant penalty for enemies reaching player area.