# game/sprites.py
import pygame as pg
from random import choice
from . import config # MODIFIED: Relative import
from .assets import IMAGES, DummySound # MODIFIED: Relative import
from .ui_elements import Text # MODIFIED: Relative import for MysteryExplosion

class Ship(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image = IMAGES['ship']
        self.rect = self.image.get_rect(topleft=(config.PLAYER_START_X, config.PLAYER_START_Y))
        self.speed = config.PLAYER_SPEED

    def update(self, keys, current_time, screen_surface): # keys can be None for AI
        if keys: # Only process keys if provided (for player mode)
            if keys[pg.K_LEFT] and self.rect.x > config.PLAYER_MIN_X:
                self.rect.x -= self.speed
            if keys[pg.K_RIGHT] and self.rect.x < config.PLAYER_MAX_X - self.rect.width:
                self.rect.x += self.speed
        
        if screen_surface: # Blit if a surface is provided (for rendering)
            screen_surface.blit(self.image, self.rect)

class Bullet(pg.sprite.Sprite):
    def __init__(self, xpos, ypos, direction, speed, filename, side): # side is 'center', 'left', 'right'
        pg.sprite.Sprite.__init__(self)
        self.image = IMAGES[filename]
        self.rect = self.image.get_rect(topleft=(xpos, ypos))
        self.speed = speed
        self.direction = direction # -1 for player (up), 1 for enemy (down)
        # self.side = side # Not used in current update, but kept for potential future use

    def update(self, keys, current_time, screen_surface):
        self.rect.y += self.speed * self.direction
        # Kill bullet if it goes off-screen
        if self.rect.bottom < 0 or self.rect.top > config.SCREEN_HEIGHT:
            self.kill()
        
        if screen_surface:
            screen_surface.blit(self.image, self.rect)

class Enemy(pg.sprite.Sprite):
    def __init__(self, row_in_formation, column_in_formation): # row is 0-4, col is 0-9
        pg.sprite.Sprite.__init__(self)
        self.row = row_in_formation
        self.column = column_in_formation
        self.images = [] # For animation
        self._load_images_for_row() # Load appropriate images based on row
        self.index = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        # Position is set by EnemiesGroup or Game's _make_enemies_formation

    def _load_images_for_row(self):
        # Determine image type based on row in the formation
        # config.ENEMY_SPRITE_MAP = {0: ['typeA_1', 'typeA_2'], 1: ['typeB_1', 'typeB_2'], ...}
        # Assuming config.ENEMY_SPRITE_SHEET_MAP defines which sprite names to use per row
        # Example: row 0-1 use 'enemy1_X', row 2-3 use 'enemy2_X', row 4 uses 'enemy3_X'
        # This needs to align with how points are scored and visuals are desired.
        
        # Using the provided config.ENEMY_SCORES_BY_ROW_INDEX structure as a guide:
        # Row 0 (top, 30pts) might be 'enemy3_X' visually.
        # Row 1,2 (mid, 20pts) might be 'enemy2_X' visually.
        # Row 3,4 (bottom, 10pts) might be 'enemy1_X' visually.
        # Let's map formation row index to visual type consistently.
        if self.row == 0: # Top row of formation (e.g., highest point value)
            img_names = config.ENEMY_ROW_SPRITES['row0'] # e.g., ['enemy3_1', 'enemy3_2']
        elif self.row == 1 or self.row == 2: # Middle rows
            img_names = config.ENEMY_ROW_SPRITES['row1_2'] # e.g., ['enemy2_1', 'enemy2_2']
        else: # Bottom rows (row 3, 4)
            img_names = config.ENEMY_ROW_SPRITES['row3_4'] # e.g., ['enemy1_1', 'enemy1_2']

        for name in img_names:
            self.images.append(pg.transform.scale(IMAGES[name], (config.ENEMY_WIDTH, config.ENEMY_HEIGHT)))
        
        if not self.images: # Fallback if loading fails
            fallback_surface = pg.Surface((config.ENEMY_WIDTH, config.ENEMY_HEIGHT))
            fallback_surface.fill(config.RED)
            self.images.append(fallback_surface)


    def toggle_image(self):
        self.index = (self.index + 1) % len(self.images)
        self.image = self.images[self.index]

    def update(self, keys, current_time, screen_surface): # Enemy update is mostly position (by group) and image toggle
        # Individual blitting if screen_surface is provided
        if screen_surface:
            screen_surface.blit(self.image, self.rect)


class EnemiesGroup(pg.sprite.Group):
    def __init__(self, columns, rows, initial_y_pos):
        pg.sprite.Group.__init__(self)
        # self.enemies_grid = [[None] * columns for _ in range(rows)] # If direct grid access is needed
        self.columns = columns; self.rows = rows
        
        self.moveTime = config.ENEMY_MOVE_TIME_INITIAL # ms per move
        self.direction = 1 # 1 for right, -1 for left
        self.move_counter = 0 # Number of horizontal moves made in current direction
        self.max_horizontal_moves = config.ENEMY_MAX_HORIZONTAL_MOVES
        
        self.timer = pg.time.get_ticks()
        self.bottom = initial_y_pos + (rows - 1) * config.ENEMY_Y_SPACING + config.ENEMY_HEIGHT
        
        # Speed up related
        self.num_enemies_killed_for_speedup = 0
        self.speedup_threshold = config.ENEMY_SPEEDUP_THRESHOLD_KILLS
        self.move_time_decrement = config.ENEMY_MOVE_TIME_DECREMENT
        self.min_move_time = config.ENEMY_MOVE_TIME_MIN

    def add_internal(self, sprite): # Overriding to manage grid or other properties if needed
        super().add_internal(sprite)
        # if isinstance(sprite, Enemy):
        #     self.enemies_grid[sprite.row][sprite.column] = sprite

    def remove_internal(self, sprite): # Overriding to manage speedup
        super().remove_internal(sprite)
        if isinstance(sprite, Enemy):
            # self.enemies_grid[sprite.row][sprite.column] = None
            self.num_enemies_killed_for_speedup += 1
            if self.num_enemies_killed_for_speedup >= self.speedup_threshold:
                self.moveTime = max(self.min_move_time, self.moveTime - self.move_time_decrement)
                self.num_enemies_killed_for_speedup = 0 # Reset counter for next speedup

    def update(self, current_time): # Note: no screen_surface here, group update doesn't blit
        if current_time - self.timer > self.moveTime:
            if self.move_counter >= self.max_horizontal_moves: # Time to move down
                self.direction *= -1 # Change horizontal direction
                self.move_counter = 0
                max_y_after_move = 0
                for enemy in self: # self directly iterates over sprites in the group
                    enemy.rect.y += config.ENEMY_MOVE_DOWN_STEP
                    enemy.toggle_image()
                    if enemy.rect.bottom > max_y_after_move:
                        max_y_after_move = enemy.rect.bottom
                self.bottom = max_y_after_move
            else: # Move horizontally
                horizontal_shift = config.ENEMY_X_SHIFT_AMOUNT * self.direction
                for enemy in self:
                    enemy.rect.x += horizontal_shift
                    enemy.toggle_image()
                self.move_counter += 1
            self.timer = current_time

    def random_bottom_shooter(self): # Selects a random enemy from bottom of a column to shoot
        if not self.sprites(): return None # No enemies left

        # Find all unique columns that still have enemies
        active_columns = sorted(list(set(enemy.column for enemy in self if isinstance(enemy, Enemy))))
        if not active_columns: return None
        
        chosen_column_idx = choice(active_columns)
        
        # Find the bottom-most enemy in that chosen column
        bottom_enemy_in_column = None
        max_y = -1
        for enemy in self:
            if isinstance(enemy, Enemy) and enemy.column == chosen_column_idx:
                if enemy.rect.bottom > max_y:
                    max_y = enemy.rect.bottom
                    bottom_enemy_in_column = enemy
        return bottom_enemy_in_column


class Blocker(pg.sprite.Sprite): # Represents one small piece of a blocker shield
    def __init__(self, size, color, row_in_formation, col_in_formation):
        pg.sprite.Sprite.__init__(self)
        self.image = pg.Surface((size, size))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        # self.row_in_formation = row_in_formation # For potential damage logic if needed
        # self.col_in_formation = col_in_formation

    def update(self, keys, current_time, screen_surface):
        if screen_surface:
            screen_surface.blit(self.image, self.rect)


class Mystery(pg.sprite.Sprite):
    def __init__(self, sound_manager): # sound_manager is Game's self.sounds
        pg.sprite.Sprite.__init__(self)
        self.image = IMAGES['mystery']
        self.image = pg.transform.scale(self.image, (config.MYSTERY_WIDTH, config.MYSTERY_HEIGHT))
        self.rect = self.image.get_rect(topleft=(-config.MYSTERY_WIDTH - 20, config.MYSTERY_SHIP_START_Y)) # Start off-screen
        
        self.speed = config.MYSTERY_SPEED
        self.direction = 1 # 1 for right, -1 for left (randomized on appearance)
        
        self.mysteryEnteredSound = sound_manager.get('mysteryentered', DummySound())
        # Correctly uses MYSTERY_APPEAR_INTERVAL_MS from config
        self.time_to_appear = pg.time.get_ticks() + choice(config.MYSTERY_APPEAR_INTERVAL_MS) 
        self.is_active = False # Not on screen initially

    def update(self, keys, currentTime, screen_surface):
        if not self.is_active:
            if currentTime >= self.time_to_appear:
                self.is_active = True
                self.direction = choice([-1, 1]) # Appear from left or right
                if self.direction == 1: # Moving right, start from left
                    self.rect.right = 0 
                else: # Moving left, start from right
                    self.rect.left = config.SCREEN_WIDTH
                if self.mysteryEnteredSound and hasattr(self.mysteryEnteredSound, 'play'):
                    self.mysteryEnteredSound.play()
        else: # Is active and on screen
            self.rect.x += self.speed * self.direction
            # Check if moved off-screen
            if (self.direction == 1 and self.rect.left > config.SCREEN_WIDTH) or \
               (self.direction == -1 and self.rect.right < 0):
                self.is_active = False # Reset for next appearance
                self.time_to_appear = currentTime + choice(config.MYSTERY_APPEAR_INTERVAL_MS)
                if self.mysteryEnteredSound and hasattr(self.mysteryEnteredSound, 'stop'):
                    self.mysteryEnteredSound.stop() # Stop sound if it was playing
            
            if screen_surface and self.is_active: # Blit if active and surface exists
                screen_surface.blit(self.image, self.rect)

# --- Explosion Sprites ---
class Explosion(pg.sprite.Sprite): # Base class for explosions
    def __init__(self, center_pos, animation_frames, frame_duration_ms, *groups):
        super().__init__(*groups)
        self.animation_frames = animation_frames # List of Surface objects
        self.frame_duration = frame_duration_ms
        self.current_frame_index = 0
        if not self.animation_frames: # Handle case for text-based explosions like MysteryExplosion
            self.image = pg.Surface((1,1), pg.SRCALPHA) # Minimal transparent surface if no frames
        else:
            self.image = self.animation_frames[self.current_frame_index]
        self.rect = self.image.get_rect(center=center_pos)
        self.last_frame_time = pg.time.get_ticks()
        # If animation_frames is empty (e.g. MysteryExplosion), total_duration is set by child
        self.total_duration = len(self.animation_frames) * self.frame_duration if self.animation_frames else 0
        self.start_time = pg.time.get_ticks()


    def update(self, keys, current_time, screen_surface):
        if self.total_duration > 0 and (current_time - self.start_time > self.total_duration):
            self.kill() # Animation finished
            return

        if self.animation_frames and (current_time - self.last_frame_time > self.frame_duration): # Only animate if frames exist
            self.current_frame_index = (self.current_frame_index + 1) % len(self.animation_frames)
            self.image = self.animation_frames[self.current_frame_index]
            self.last_frame_time = current_time
        
        if screen_surface:
            screen_surface.blit(self.image, self.rect)


class EnemyExplosion(Explosion):
    def __init__(self, enemy_sprite, *groups):
        frames = []
        if config.EXPLOSION_ENEMY_FRAMES and IMAGES.get(config.EXPLOSION_ENEMY_FRAMES[0]) and IMAGES.get(config.EXPLOSION_ENEMY_FRAMES[1]):
            frames = [
                pg.transform.scale(IMAGES[config.EXPLOSION_ENEMY_FRAMES[0]], (config.ENEMY_WIDTH, config.ENEMY_HEIGHT)),
                pg.transform.scale(IMAGES[config.EXPLOSION_ENEMY_FRAMES[1]], (int(config.ENEMY_WIDTH*1.2), int(config.ENEMY_HEIGHT*1.2)))
            ]
        else: # Fallback if frames not defined or images missing
            surf = pg.Surface((config.ENEMY_WIDTH, config.ENEMY_HEIGHT)); surf.fill(config.RED)
            frames.append(surf)
        super().__init__(enemy_sprite.rect.center, frames, config.EXPLOSION_FRAME_DURATION_MS, *groups)

class MysteryExplosion(Explosion): # Shows score text instead of image animation
    def __init__(self, mystery_sprite, score_value, *groups):
        # Call Explosion's init with empty frames, duration will be set for text display
        super().__init__(mystery_sprite.rect.center, [], config.MYSTERY_EXPLOSION_DURATION_MS, *groups)
        
        # This is where Text is instantiated. It MUST use center_x and center_y
        self.score_text_renderer = Text(config.GAME_FONT, 20, str(score_value), config.WHITE, 
                                   mystery_sprite.rect.centerx, # xpos for Text
                                   mystery_sprite.rect.centery, # ypos for Text
                                   center_x=True,  # <<< ENSURE THIS IS PASSED
                                   center_y=True)  # <<< ENSURE THIS IS PASSED

        self.image = self.score_text_renderer.surface # Use the text surface as the "image"
        self.rect = self.image.get_rect(center=mystery_sprite.rect.center) # Ensure final rect is centered
        self.total_duration = config.MYSTERY_EXPLOSION_DURATION_MS # Explicitly set duration for text display

    def update(self, keys, current_time, screen_surface): # Override base update
        if current_time - self.start_time > self.total_duration:
            self.kill()
            return
        if screen_surface: 
            # Blinking effect for score text
            blink_interval = self.total_duration // 4 # Example: blink twice during display
            if blink_interval == 0: blink_interval = 1 # Avoid division by zero if total_duration is too small
            
            if ((current_time - self.start_time) // blink_interval) % 2 == 0:
                 screen_surface.blit(self.image, self.rect)


class ShipExplosion(Explosion):
    def __init__(self, ship_sprite, *groups):
        frames = []
        if config.EXPLOSION_SHIP_FRAMES and IMAGES.get(config.EXPLOSION_SHIP_FRAMES[0]) and IMAGES.get(config.EXPLOSION_SHIP_FRAMES[1]):
            frames = [
                pg.transform.scale(IMAGES[config.EXPLOSION_SHIP_FRAMES[0]], (config.PLAYER_WIDTH, config.PLAYER_HEIGHT)),
                pg.transform.scale(IMAGES[config.EXPLOSION_SHIP_FRAMES[1]], (int(config.PLAYER_WIDTH*1.3), int(config.PLAYER_HEIGHT*1.3)))
            ]
        else:
            surf = pg.Surface((config.PLAYER_WIDTH, config.PLAYER_HEIGHT)); surf.fill(config.BLUE)
            frames.append(surf)
        super().__init__(ship_sprite.rect.center, frames, config.EXPLOSION_FRAME_DURATION_MS, *groups)


class Life(pg.sprite.Sprite): # Represents a life icon
    def __init__(self, xpos, ypos):
        pg.sprite.Sprite.__init__(self)
        self.image = pg.transform.scale(IMAGES['ship'], (config.LIFE_SPRITE_WIDTH, config.LIFE_SPRITE_HEIGHT))
        self.rect = self.image.get_rect(topleft=(xpos,ypos))
    
    def update(self, keys, current_time, screen_surface): # Lives icons don't do much in update
        if screen_surface:
            screen_surface.blit(self.image, self.rect)