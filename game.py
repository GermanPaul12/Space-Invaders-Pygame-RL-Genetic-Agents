import pygame as pg
import sys
from random import choice
# import time # Use pg.time instead

import config # Import all configurations

# Initialize Pygame's font and mixer modules (mixer pre_init handled in Game)
pg.font.init()

# Load Images (global for access by sprite classes)
# convert_alpha() is good for images with transparency.
IMAGES = {name: pg.image.load(config.IMAGE_PATH + '{}.png'.format(name)).convert_alpha()
          for name in config.IMG_NAMES}

class Ship(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image = IMAGES['ship']
        self.rect = self.image.get_rect(topleft=(config.PLAYER_START_X, config.PLAYER_START_Y))
        self.speed = config.PLAYER_SPEED

    def update(self, keys, current_time, screen_surface): # Added current_time, screen_surface
        if keys[pg.K_LEFT] and self.rect.x > 10:
            self.rect.x -= self.speed
        if keys[pg.K_RIGHT] and self.rect.x < config.SCREEN_WIDTH - self.rect.width - 10:
            self.rect.x += self.speed
        screen_surface.blit(self.image, self.rect)


class Bullet(pg.sprite.Sprite):
    def __init__(self, xpos, ypos, direction, speed, filename, side):
        pg.sprite.Sprite.__init__(self)
        self.image = IMAGES[filename]
        self.rect = self.image.get_rect(topleft=(xpos, ypos))
        self.speed = speed
        self.direction = direction
        self.side = side
        self.filename = filename

    def update(self, keys, current_time, screen_surface): # Added current_time, screen_surface (keys not used)
        screen_surface.blit(self.image, self.rect)
        self.rect.y += self.speed * self.direction
        if self.rect.y < 15 or self.rect.y > config.SCREEN_HEIGHT: # Use config for screen height
            self.kill()


class Enemy(pg.sprite.Sprite):
    def __init__(self, row, column):
        pg.sprite.Sprite.__init__(self)
        self.row = row
        self.column = column
        self.images = []
        self.load_images()
        self.index = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()

    def toggle_image(self):
        self.index += 1
        if self.index >= len(self.images):
            self.index = 0
        self.image = self.images[self.index]

    def update(self, keys, current_time, screen_surface): # Added current_time, screen_surface (keys not used)
        screen_surface.blit(self.image, self.rect)

    def load_images(self):
        # This mapping determines which enemy type appears in which row
        # Row 0,1 use enemy1 images; Row 2,3 use enemy2 images; Row 4,5 use enemy3 images.
        # Original code:
        # images = {0: ['1_2', '1_1'], 1: ['2_2', '2_1'], 2: ['2_2', '2_1'], 3: ['3_1', '3_2'], 4: ['3_1', '3_2']}
        # This means: grid row 0 uses 'enemy1' type, grid row 1&2 use 'enemy2', grid row 3&4 use 'enemy3'
        # This seems to map to screen rows from top to bottom.
        img_map = {
            0: ['enemy1_2', 'enemy1_1'],  # Top rows of enemies (visually type 1)
            1: ['enemy2_2', 'enemy2_1'],  # Middle rows (visually type 2)
            2: ['enemy2_2', 'enemy2_1'],
            3: ['enemy3_1', 'enemy3_2'],  # Bottom rows (visually type 3)
            4: ['enemy3_1', 'enemy3_2'],
        }
        img_indices = img_map[self.row]
        img1, img2 = (IMAGES[img_name] for img_name in img_indices)
        self.images.append(pg.transform.scale(img1, (40, 35)))
        self.images.append(pg.transform.scale(img2, (40, 35)))


class EnemiesGroup(pg.sprite.Group):
    def __init__(self, columns, rows, initial_y_position): # Pass initial_y_position
        pg.sprite.Group.__init__(self)
        self.enemies = [[None] * columns for _ in range(rows)]
        self.columns = columns
        self.rows = rows
        self.leftAddMove = 0
        self.rightAddMove = 0
        self.moveTime = 600  # Initial speed, gets faster
        self.direction = 1
        self.rightMoves = 30 # Number of steps to take right
        self.leftMoves = 30  # Number of steps to take left
        self.moveNumber = 15 # Current step count in current direction
        self.timer = pg.time.get_ticks()
        self.bottom = initial_y_position + ((rows - 1) * 45) + 35 # Calculate bottom based on start Y
        self._aliveColumns = list(range(columns))
        self._leftAliveColumn = 0
        self._rightAliveColumn = columns - 1

    def update(self, current_time): # This method is for group logic, not drawing individual enemies
        if current_time - self.timer > self.moveTime:
            if self.direction == 1:
                max_move = self.rightMoves + self.rightAddMove
            else:
                max_move = self.leftMoves + self.leftAddMove

            if self.moveNumber >= max_move:
                self.leftMoves = 30 + self.rightAddMove
                self.rightMoves = 30 + self.leftAddMove
                self.direction *= -1
                self.moveNumber = 0
                # self.bottom = 0 # Reset bottom, will be recalculated
                max_y = 0
                for enemy in self:
                    enemy.rect.y += config.ENEMY_MOVE_DOWN
                    enemy.toggle_image()
                    if max_y < enemy.rect.y + 35: # Update actual bottom
                        max_y = enemy.rect.y + 35
                self.bottom = max_y
            else:
                velocity = 10 if self.direction == 1 else -10
                for enemy in self:
                    enemy.rect.x += velocity
                    enemy.toggle_image()
                self.moveNumber += 1
            self.timer = current_time # Reset timer based on current_time

    def add_internal(self, *sprites):
        super(EnemiesGroup, self).add_internal(*sprites)
        for s in sprites:
            if isinstance(s, Enemy): # Ensure it's an Enemy sprite
                 self.enemies[s.row][s.column] = s

    def remove_internal(self, *sprites):
        super(EnemiesGroup, self).remove_internal(*sprites)
        for s in sprites:
            if isinstance(s, Enemy):
                 self.kill_enemy_sprite(s) # Use a renamed method for clarity
        self.update_speed()

    def is_column_dead(self, column):
        return not any(self.enemies[row][column] for row in range(self.rows))

    def random_bottom(self):
        if not self._aliveColumns: # No columns left to shoot from
            return None
        col = choice(self._aliveColumns)
        # Iterate from bottom row upwards in that column
        for row_idx in range(self.rows - 1, -1, -1):
            enemy = self.enemies[row_idx][col]
            if enemy is not None:
                return enemy
        return None # Should not happen if _aliveColumns is maintained correctly

    def update_speed(self):
        num_enemies = len(self)
        if num_enemies == 1:
            self.moveTime = 200
        elif num_enemies <= 10:
            self.moveTime = 400
        # else original speed (600 or adjusted by game difficulty)

    def kill_enemy_sprite(self, enemy): # Renamed from 'kill' to avoid conflict with Sprite.kill()
        self.enemies[enemy.row][enemy.column] = None
        is_col_dead = self.is_column_dead(enemy.column)
        
        if is_col_dead and enemy.column in self._aliveColumns: # Check if it was already removed
            self._aliveColumns.remove(enemy.column)

        # Adjust horizontal movement bounds if the killed enemy was on an edge column
        # This logic allows the formation to move further horizontally as columns are cleared
        if enemy.column == self._rightAliveColumn:
            while self._rightAliveColumn >= 0 and self.is_column_dead(self._rightAliveColumn):
                self._rightAliveColumn -= 1
                if self._rightAliveColumn < 0: break # All columns cleared
                self.rightAddMove += 5
        
        elif enemy.column == self._leftAliveColumn:
            while self._leftAliveColumn < self.columns and self.is_column_dead(self._leftAliveColumn):
                self._leftAliveColumn += 1
                if self._leftAliveColumn >= self.columns : break # All columns cleared
                self.leftAddMove += 5


class Blocker(pg.sprite.Sprite):
    def __init__(self, size, color, row, column):
        pg.sprite.Sprite.__init__(self)
        self.height = size
        self.width = size
        self.color = color # Not used if image is surface
        self.image = pg.Surface((self.width, self.height))
        self.image.fill(color) # Use passed color
        self.rect = self.image.get_rect()
        self.row = row
        self.column = column

    def update(self, keys, current_time, screen_surface): # Added current_time, screen_surface (keys, time not used)
        screen_surface.blit(self.image, self.rect)


class Mystery(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image = IMAGES['mystery']
        self.image = pg.transform.scale(self.image, (75, 35))
        self.rect = self.image.get_rect(topleft=(-80, config.MYSTERY_SHIP_START_Y))
        self.row = 5  # Special row index for scoring
        self.moveTime = config.MYSTERY_SHIP_MOVE_TIME
        self.direction = 1
        self.timer = pg.time.get_ticks()
        self.mysteryEnteredSound = pg.mixer.Sound(config.SOUND_PATH + 'mysteryentered.wav')
        self.mysteryEnteredSound.set_volume(0.3)
        self.playSound = True

    def update(self, keys, currentTime, screen_surface): # Added screen_surface (keys not used)
        resetTimer = False
        passed = currentTime - self.timer
        if passed > self.moveTime:
            if (self.rect.x < 0 or self.rect.x > config.SCREEN_WIDTH) and self.playSound: # Use config
                self.mysteryEnteredSound.play()
                self.playSound = False
            
            # Move logic
            if self.direction == 1:
                if self.rect.x < config.SCREEN_WIDTH + 40: # Move across screen
                    self.rect.x += 2
                else: # Off screen right
                    self.playSound = True
                    self.direction = -1
                    resetTimer = True
            else: # direction == -1
                if self.rect.x > -100: # Move back across screen
                    self.rect.x -= 2
                else: # Off screen left
                    self.playSound = True
                    self.direction = 1
                    resetTimer = True
            
            if self.rect.x > 0 and self.rect.x < config.SCREEN_WIDTH : # Only fadeout and blit if on screen
                 self.mysteryEnteredSound.fadeout(4000) # Original had this inside move conditions
                 screen_surface.blit(self.image, self.rect)

        if resetTimer: # Reset timer only when it goes off screen and direction flips
            self.timer = currentTime


class EnemyExplosion(pg.sprite.Sprite):
    def __init__(self, enemy, *groups): # Pass enemy sprite
        super(EnemyExplosion, self).__init__(*groups) # Add to specified groups
        # Determine color based on enemy's original row (type)
        img_colors = ['purple', 'blue', 'blue', 'green', 'green'] # Corresponds to Enemy.load_images map
        # enemy.row 0,1 -> type1 -> explosion green
        # enemy.row 2,3 -> type2 -> explosion blue
        # enemy.row 4,5 -> type3 -> explosion purple
        # This needs to align with how Enemy assigns types based on self.row
        # If Enemy.row 0 means top screen row (30pts type), it used enemy3_x images.
        # The original staticmethod mapping was:
        # img_colors = ['purple', 'blue', 'blue', 'green', 'green']
        # This implies row 0 (highest enemy on screen) -> purple explosion
        # And row 4 (lowest enemy on screen) -> green explosion
        # Let's use the enemy's `row` attribute which defines its type.
        # Mapping from enemy.row to explosion color, consistent with visual types if:
        # row 0,1 (type 1 visual, e.g. 'enemy1_x') -> green
        # row 2,3 (type 2 visual, e.g. 'enemy2_x') -> blue
        # row 4   (type 3 visual, e.g. 'enemy3_x') -> purple
        # Recheck Enemy.load_images:
        # self.row 0 -> enemy1_type -> green
        # self.row 1,2 -> enemy2_type -> blue
        # self.row 3,4 -> enemy3_type -> purple
        
        # Corrected mapping based on assumed visual types from Enemy.load_images structure
        explosion_color_map = {
            0: 'green',  # For enemies from self.row 0 (e.g., enemy1 type)
            1: 'blue',   # For enemies from self.row 1 (e.g., enemy2 type)
            2: 'blue',   # For enemies from self.row 2 (e.g., enemy2 type)
            3: 'purple', # For enemies from self.row 3 (e.g., enemy3 type)
            4: 'purple'  # For enemies from self.row 4 (e.g., enemy3 type)
        }
        color = explosion_color_map.get(enemy.row, 'blue') # Default to blue

        self.image_orig = IMAGES['explosion{}'.format(color)]
        self.image = pg.transform.scale(self.image_orig, (40, 35))
        self.image2 = pg.transform.scale(self.image_orig, (50, 45)) # Larger explosion frame
        self.rect = self.image.get_rect(topleft=(enemy.rect.x, enemy.rect.y))
        self.timer = pg.time.get_ticks()

    def update(self, keys, current_time, screen_surface): # Added keys, current_time, screen_surface
        passed = current_time - self.timer
        if passed <= 100:
            screen_surface.blit(self.image, self.rect)
        elif passed <= 200:
            # Adjust rect for larger image to keep it centered on explosion point
            screen_surface.blit(self.image2, (self.rect.x - 6, self.rect.y - 6))
        elif 200 < passed <= 400: # Adding a brief pause before kill
            pass # Or blit image2 again if you want longer visibility
        else: # passed > 400
            self.kill()


class MysteryExplosion(pg.sprite.Sprite):
    def __init__(self, mystery, score, *groups): # Pass mystery sprite and score gained
        super(MysteryExplosion, self).__init__(*groups)
        self.text_renderer = Text(config.GAME_FONT, 20, str(score), config.WHITE,
                                  mystery.rect.x + 20, mystery.rect.y + 6)
        self.timer = pg.time.get_ticks()

    def update(self, keys, current_time, screen_surface): # Added keys, current_time, screen_surface
        passed = current_time - self.timer
        if passed <= 200 or (400 < passed <= 600): # Blink effect
            self.text_renderer.draw(screen_surface)
        elif passed > 600:
            self.kill()


class ShipExplosion(pg.sprite.Sprite):
    def __init__(self, ship, *groups): # Pass ship sprite
        super(ShipExplosion, self).__init__(*groups)
        self.image = IMAGES['ship'] # Use player ship image for explosion start
        self.rect = self.image.get_rect(topleft=(ship.rect.x, ship.rect.y))
        self.timer = pg.time.get_ticks()

    def update(self, keys, current_time, screen_surface): # Added keys, current_time, screen_surface
        passed = current_time - self.timer
        # Simple flicker effect: show ship, then nothing, then kill
        if 300 < passed <= 600: # Show ship briefly during explosion frames
            screen_surface.blit(self.image, self.rect)
        # elif 600 < passed <= 900:
            # Could add more frames here, e.g. different explosion image
        elif passed > 900: # Duration of explosion effect
            self.kill()


class Life(pg.sprite.Sprite):
    def __init__(self, xpos, ypos):
        pg.sprite.Sprite.__init__(self)
        self.image = IMAGES['ship']
        self.image = pg.transform.scale(self.image, (23, 23))
        self.rect = self.image.get_rect(topleft=(xpos, ypos))

    def update(self, keys, current_time, screen_surface): # Added keys, current_time, screen_surface
        screen_surface.blit(self.image, self.rect)


class Text(object): # Not a sprite, a helper for rendering text
    def __init__(self, textFontPath, size, message, color, xpos, ypos):
        self.font = pg.font.Font(textFontPath, size)
        self.surface = self.font.render(message, True, color)
        self.rect = self.surface.get_rect(topleft=(xpos, ypos))

    def draw(self, surface):
        surface.blit(self.surface, self.rect)


class Game:
    def __init__(self):
        pg.mixer.pre_init(44100, -16, 1, 4096) # As in original
        pg.init() # Initialize all pygame modules
        self.clock = pg.time.Clock()
        pg.display.set_caption('Space Invaders')
        self.screen = pg.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        self.background = pg.image.load(config.IMAGE_PATH + 'background.jpg').convert()
        
        # Game state variables
        self.mainScreenActive = True # Start with main menu
        self.gameplayActive = False  # True when playing a round
        self.gameOverActive = False  # True when game over sequence is shown
        
        self.enemy_start_y = config.ENEMY_DEFAULT_POSITION
        self.score = 0
        self.lives = 3

        self._make_static_text_objects()
        self._create_audio_assets()

        # Game elements that are reset per game/round
        self.player = None
        self.playerGroup = pg.sprite.Group()
        self.enemies = None # This will be an EnemiesGroup instance
        self.bullets = pg.sprite.Group() # Player bullets
        self.enemyBullets = pg.sprite.Group()
        self.mysteryShip = None
        self.mysteryGroup = pg.sprite.Group()
        self.explosionsGroup = pg.sprite.Group()
        self.allBlockers = pg.sprite.Group()
        self.livesSpritesGroup = pg.sprite.Group()
        
        # allSprites group will hold most visible game objects for easier updating/drawing
        self.allSprites = pg.sprite.Group()

        # Timers
        self.general_timer = pg.time.get_ticks() # For enemy shooting, etc.
        self.noteTimer = pg.time.get_ticks()     # For background music beats
        self.shipRespawnTimer = pg.time.get_ticks()
        self.roundOverTimer = pg.time.get_ticks() # For "Next Round" / "Game Over" screen timing
        
        self.makeNewShipNext = False # Flag for respawning ship
        self.shipCurrentlyAlive = True

    def _make_static_text_objects(self):
        self.titleText = Text(config.GAME_FONT, 50, 'Space Invaders', config.WHITE, 164, 155)
        self.titleText2 = Text(config.GAME_FONT, 25, 'Press any key to continue', config.WHITE, 201, 225)
        self.gameOverTextDisplay = Text(config.GAME_FONT, 50, 'Game Over', config.WHITE, 250, 270)
        self.nextRoundTextDisplay = Text(config.GAME_FONT, 50, 'Next Round', config.WHITE, 240, 270)
        
        self.scoreLabelText = Text(config.GAME_FONT, 20, 'Score', config.WHITE, 5, 5)
        self.livesLabelText = Text(config.GAME_FONT, 20, 'Lives ', config.WHITE, 640, 5)
        
        # Text for main menu point values
        self.enemy1ScoreText = Text(config.GAME_FONT, 25, '   =   30 pts', config.PURPLE, 368, 270) # Top type
        self.enemy2ScoreText = Text(config.GAME_FONT, 25, '   =  20 pts', config.BLUE, 368, 320)   # Middle type
        self.enemy3ScoreText = Text(config.GAME_FONT, 25, '   =  10 pts', config.GREEN, 368, 370)  # Bottom type
        self.mysteryScoreText = Text(config.GAME_FONT, 25, '   =  ?????', config.RED, 368, 420)
        self.scoreValueText = None # Will be updated dynamically

    def _create_audio_assets(self):
        self.sounds = {}
        for sound_name in ['shoot', 'shoot2', 'invaderkilled', 'mysterykilled', 'shipexplosion']:
            self.sounds[sound_name] = pg.mixer.Sound(config.SOUND_PATH + '{}.wav'.format(sound_name))
            self.sounds[sound_name].set_volume(0.2)

        self.musicNotes = [pg.mixer.Sound(config.SOUND_PATH + '{}.wav'.format(i)) for i in range(4)]
        for sound in self.musicNotes:
            sound.set_volume(0.5)
        self.noteIndex = 0

    def _full_game_reset(self, start_score=0, start_lives=3):
        self.score = start_score
        self.lives = start_lives
        self.enemy_start_y = config.ENEMY_DEFAULT_POSITION
        
        self.allBlockers.empty() # Clear existing blockers
        for i in range(4): # Create 4 sets of blockers
            self.allBlockers.add(self._make_blocker_group(i))
        
        self._reset_round_state(start_score)


    def _reset_round_state(self, current_score):
        self.score = current_score # Carry over score for new round
        
        # Clear old sprites from groups that are repopulated
        self.playerGroup.empty()
        self.mysteryGroup.empty()
        self.bullets.empty()
        self.enemyBullets.empty()
        self.explosionsGroup.empty()
        if self.enemies: self.enemies.empty() # Clear EnemiesGroup
        self.allSprites.empty() # Clear all sprites before re-adding

        self.player = Ship()
        self.playerGroup.add(self.player)
        
        self.mysteryShip = Mystery()
        self.mysteryGroup.add(self.mysteryShip)
        
        self._make_enemies_formation() # Creates and populates self.enemies (EnemiesGroup)
        
        self._update_lives_sprites() # Create/recreate life icons based on self.lives

        # Populate allSprites group (order can matter for drawing if not using layers)
        self.allSprites.add(self.player)
        self.allSprites.add(self.enemies) # EnemiesGroup is a Sprite Group, add its members
        self.allSprites.add(self.mysteryShip)
        self.allSprites.add(self.allBlockers) # Blocker sprites
        self.allSprites.add(self.livesSpritesGroup) # Life icon sprites
        # Bullets and explosions are added to self.allSprites when created.

        self.keys = pg.key.get_pressed()
        self.general_timer = pg.time.get_ticks() # Reset general timer (e.g. for enemy shooting)
        self.noteTimer = pg.time.get_ticks()
        
        self.makeNewShipNext = False
        self.shipCurrentlyAlive = True
        if self.lives <= 0: # Should not happen if reset is for new round, but safety for full reset
            self.shipCurrentlyAlive = False


    def _make_blocker_group(self, number): # Helper for creating one set of blockers
        one_blocker_set = pg.sprite.Group()
        for row in range(4):
            for column in range(9):
                blocker = Blocker(10, config.GREEN, row, column)
                blocker.rect.x = 50 + (200 * number) + (column * blocker.width)
                blocker.rect.y = config.BLOCKERS_POSITION + (row * blocker.height)
                one_blocker_set.add(blocker)
        return one_blocker_set

    def _make_enemies_formation(self):
        self.enemies = EnemiesGroup(10, 5, self.enemy_start_y)
        for row_idx in range(5): # 5 rows of enemies
            for col_idx in range(10): # 10 columns
                enemy = Enemy(row_idx, col_idx) # Enemy now determines its type from row_idx
                enemy.rect.x = 157 + (col_idx * 50)
                enemy.rect.y = self.enemy_start_y + (row_idx * 45)
                self.enemies.add(enemy) # Add to the EnemiesGroup

    def _update_lives_sprites(self):
        self.livesSpritesGroup.empty() # Clear current life icons
        life_positions = [715, 742, 769]
        for i in range(self.lives):
            if i < len(life_positions): # Max 3 lives displayed
                life_sprite = Life(life_positions[i], 3)
                self.livesSpritesGroup.add(life_sprite)
        # Add to allSprites so they are drawn/updated
        if self.allSprites: # Ensure allSprites exists
             self.allSprites.add(self.livesSpritesGroup)


    def run_player_mode(self):
        self._full_game_reset() # Initial setup for a brand new game session

        running = True
        while running:
            currentTime = pg.time.get_ticks()
            self.keys = pg.key.get_pressed()

            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYUP and e.key == pg.K_ESCAPE):
                    running = False
                
                if self.mainScreenActive:
                    if e.type == pg.KEYUP: # Any key to start from main menu
                        self.mainScreenActive = False
                        self.gameplayActive = True
                        self.gameOverActive = False
                        self._full_game_reset(start_score=0, start_lives=3) # Start new game
                elif self.gameplayActive:
                    if e.type == pg.KEYDOWN and e.key == pg.K_SPACE:
                        self._handle_player_shooting()
                elif self.gameOverActive:
                    if e.type == pg.KEYUP: # Any key to return to main menu after game over
                        self.gameOverActive = False
                        self.mainScreenActive = True
                        # Reset is handled when starting new game from main menu
            
            # --- Game State Logic & Updates ---
            self.screen.blit(self.background, (0,0)) # Draw background first

            if self.mainScreenActive:
                self._draw_main_menu_elements()
            elif self.gameplayActive:
                self._update_gameplay_state(currentTime)
                self._draw_gameplay_elements(currentTime)
            elif self.gameOverActive:
                self._update_game_over_state(currentTime)
                self._draw_game_over_elements(currentTime)

            pg.display.update()
            self.clock.tick(config.FPS)
        
        pg.quit()

    def _handle_player_shooting(self):
        if self.shipCurrentlyAlive and len(self.bullets) == 0 : # Original single shot logic
            if self.score < 1000:
                bullet = Bullet(self.player.rect.x + 23, self.player.rect.y + 5, -1,
                                config.PLAYER_LASER_SPEED, 'laser', 'center')
                self.bullets.add(bullet)
                self.allSprites.add(bullet)
                self.sounds['shoot'].play()
            else: # Double shot for score >= 1000
                l_bullet = Bullet(self.player.rect.x + 8, self.player.rect.y + 5, -1, config.PLAYER_LASER_SPEED, 'laser', 'left')
                r_bullet = Bullet(self.player.rect.x + 38, self.player.rect.y + 5, -1, config.PLAYER_LASER_SPEED, 'laser', 'right')
                self.bullets.add(l_bullet, r_bullet)
                self.allSprites.add(l_bullet, r_bullet)
                self.sounds['shoot2'].play()

    def _draw_main_menu_elements(self):
        self.titleText.draw(self.screen)
        self.titleText2.draw(self.screen)
        self.enemy1ScoreText.draw(self.screen)
        self.enemy2ScoreText.draw(self.screen)
        self.enemy3ScoreText.draw(self.screen)
        self.mysteryScoreText.draw(self.screen)
        # Draw enemy icons for score display
        # Using enemy type from load_images: enemy1_2 (bottom), enemy2_2 (middle), enemy3_1 (top)
        e1_img = pg.transform.scale(IMAGES['enemy3_1'], (40, 40)) # Corresponds to 30pts
        e2_img = pg.transform.scale(IMAGES['enemy2_2'], (40, 40)) # Corresponds to 20pts
        e3_img = pg.transform.scale(IMAGES['enemy1_2'], (40, 40)) # Corresponds to 10pts
        mystery_img = pg.transform.scale(IMAGES['mystery'], (80, 40))
        self.screen.blit(e1_img, (318, 270)) # 30 pts
        self.screen.blit(e2_img, (318, 320)) # 20 pts
        self.screen.blit(e3_img, (318, 370)) # 10 pts
        self.screen.blit(mystery_img, (299, 420)) # Mystery

    def _update_gameplay_state(self, currentTime):
        if not self.enemies and not self.explosionsGroup: # Round cleared
            self.gameplayActive = False # Pause gameplay for "Next Round" message
            self.roundOverTimer = currentTime # Start timer for message display
            return # Skip further gameplay updates this frame

        # Update game elements
        self.enemies.update(currentTime) # EnemiesGroup movement logic
        
        # Update all sprites. Pass (keys, currentTime, screen_surface)
        # Individual sprite update methods will use what they need and draw themselves.
        self.allSprites.update(self.keys, currentTime, self.screen)
        self.bullets.update(self.keys, currentTime, self.screen)
        self.enemyBullets.update(self.keys, currentTime, self.screen)
        self.explosionsGroup.update(self.keys, currentTime, self.screen) # Explosions draw themselves

        self._check_collisions_and_deaths()
        self._respawn_player_if_needed(currentTime)
        self._trigger_enemy_shooting(currentTime)
        self._play_background_music(currentTime)

        # Check for game over by enemies reaching bottom
        if self.enemies.bottom >= config.SCREEN_HEIGHT - 80 : # Near player's starting line
            if self.shipCurrentlyAlive and pg.sprite.spritecollideany(self.player, self.enemies):
                self._player_death() # Player hit by descending enemies
            if self.enemies.bottom >= config.SCREEN_HEIGHT: # Off screen
                self.lives = 0 # Game over if enemies reach very bottom
                self._player_death(final_death=True) # Ensure game over transition

    def _draw_gameplay_elements(self, currentTime):
        # If round just ended, show "Next Round"
        if not self.gameplayActive and not self.gameOverActive and not self.mainScreenActive:
            if currentTime - self.roundOverTimer < 3000: # Display for 3 seconds
                self.nextRoundTextDisplay.draw(self.screen)
            else: # Time to start next round
                self.enemy_start_y = min(self.enemy_start_y + config.ENEMY_MOVE_DOWN, config.BLOCKERS_POSITION - 100) # Move down, but not too far
                if self.enemy_start_y >= config.BLOCKERS_POSITION - 100: # Check if enemies start too low
                    self.lives = 0
                    self._player_death(final_death=True) # Game over condition
                    return
                self._reset_round_state(self.score) # Setup next round
                self.gameplayActive = True # Resume gameplay
                return # Skip drawing regular HUD this frame
        
        # Regular HUD and sprite drawing is implicitly handled by allSprites.update if sprites draw themselves
        # If not, explicit draw calls would be needed:
        # self.allSprites.draw(self.screen)
        # self.explosionsGroup.draw(self.screen)
        # For now, assuming sprites draw in their update.

        # Draw score and lives text (dynamic parts)
        self.scoreValueText = Text(config.GAME_FONT, 20, str(self.score), config.GREEN, 85, 5)
        self.scoreLabelText.draw(self.screen)
        self.scoreValueText.draw(self.screen)
        self.livesLabelText.draw(self.screen)
        # Lives icons are in allSprites, so they are drawn via its update call.
        
    def _play_background_music(self, currentTime):
        if self.enemies and self.enemies.moveTime > 0: # Ensure enemies exist and moveTime is valid
            if currentTime - self.noteTimer > self.enemies.moveTime:
                note = self.musicNotes[self.noteIndex]
                self.noteIndex = (self.noteIndex + 1) % len(self.musicNotes)
                note.play()
                self.noteTimer = currentTime # More accurate timing

    def _trigger_enemy_shooting(self, currentTime):
        if (currentTime - self.general_timer) > config.ENEMY_SHOOT_INTERVAL and self.enemies:
            enemy_to_shoot = self.enemies.random_bottom()
            if enemy_to_shoot:
                bullet = Bullet(enemy_to_shoot.rect.x + 14, enemy_to_shoot.rect.y + 20, 1,
                                config.ENEMY_LASER_SPEED, 'enemylaser', 'center')
                self.enemyBullets.add(bullet)
                self.allSprites.add(bullet)
                self.general_timer = currentTime

    def _calculate_score_for_kill(self, killed_sprite):
        if isinstance(killed_sprite, Mystery):
            points = choice(config.MYSTERY_SCORES_OPTIONS)
        elif isinstance(killed_sprite, Enemy):
            points = config.ENEMY_SCORES_BY_ROW_INDEX.get(killed_sprite.row, 0)
        else:
            points = 0
        self.score += points
        return points

    def _check_collisions_and_deaths(self):
        pg.sprite.groupcollide(self.bullets, self.enemyBullets, True, True) # Player vs Enemy bullets

        for enemy in pg.sprite.groupcollide(self.enemies, self.bullets, True, True).keys():
            self.sounds['invaderkilled'].play()
            self._calculate_score_for_kill(enemy)
            expl = EnemyExplosion(enemy, self.explosionsGroup, self.allSprites)
            self.roundOverTimer = pg.time.get_ticks() # Used for "Next Round" screen delay

        for mystery in pg.sprite.groupcollide(self.mysteryGroup, self.bullets, True, True).keys():
            mystery.mysteryEnteredSound.stop()
            self.sounds['mysterykilled'].play()
            score_val = self._calculate_score_for_kill(mystery)
            MysteryExplosion(mystery, score_val, self.explosionsGroup, self.allSprites)
            # Respawn mystery ship
            self.mysteryShip = Mystery()
            self.mysteryGroup.add(self.mysteryShip)
            self.allSprites.add(self.mysteryShip)

        if self.shipCurrentlyAlive:
            if pg.sprite.spritecollide(self.player, self.enemyBullets, True): # True to kill bullet
                self._player_death()
        
        # Blockers vs bullets and enemies
        pg.sprite.groupcollide(self.bullets, self.allBlockers, True, True)
        pg.sprite.groupcollide(self.enemyBullets, self.allBlockers, True, True)
        if self.enemies.bottom >= config.BLOCKERS_POSITION:
            pg.sprite.groupcollide(self.enemies, self.allBlockers, False, True) # Enemies destroy blockers


    def _player_death(self, final_death=False):
        if not self.shipCurrentlyAlive and not final_death: # Already processing a death
            return

        self.sounds['shipexplosion'].play()
        if self.player: # Check if player instance exists
             ShipExplosion(self.player, self.explosionsGroup, self.allSprites)
             self.player.kill() # Remove from all groups
        
        self.shipCurrentlyAlive = False
        self.lives -= 1
        self._update_lives_sprites()

        if self.lives <= 0 or final_death:
            self.gameOverActive = True
            self.gameplayActive = False
            self.roundOverTimer = pg.time.get_ticks() # For Game Over screen timing
        else:
            self.makeNewShipNext = True
            self.shipRespawnTimer = pg.time.get_ticks()

    def _respawn_player_if_needed(self, currentTime):
        if self.makeNewShipNext and (currentTime - self.shipRespawnTimer > 900): # Respawn delay
            if self.lives > 0:
                self.player = Ship()
                self.playerGroup.add(self.player) # Should be empty before this
                self.allSprites.add(self.player)
                self.shipCurrentlyAlive = True
                self.makeNewShipNext = False
            else: # No lives left, ensure game over state
                if not self.gameOverActive:
                    self.gameOverActive = True
                    self.gameplayActive = False
                    self.roundOverTimer = pg.time.get_ticks()


    def _update_game_over_state(self, currentTime): # Manages transition after Game Over period
        if currentTime - self.roundOverTimer > 3000: # After 3s of game over screen
            self.gameOverActive = False
            self.mainScreenActive = True
            # Reset game variables for a potential new game initiated from main menu
            self.enemy_start_y = config.ENEMY_DEFAULT_POSITION
            # Full reset will happen if player starts new game from main menu

    def _draw_game_over_elements(self, currentTime):
        # Blinking "Game Over" text
        time_in_game_over = currentTime - self.roundOverTimer
        if (time_in_game_over // 750) % 2 == 0: # Blink on/off every 750ms
            self.gameOverTextDisplay.draw(self.screen)
        # Score can be displayed too
        self.scoreValueText = Text(config.GAME_FONT, 20, str(self.score), config.GREEN, 85, 5)
        self.scoreLabelText.draw(self.screen)
        self.scoreValueText.draw(self.screen)


    # --- AI Interface Methods (STUBS) ---
    def reset_for_ai(self):
        """Resets the game for an AI agent to start playing."""
        self._full_game_reset(start_score=0, start_lives=3)
        self.mainScreenActive = False
        self.gameplayActive = True
        self.gameOverActive = False
        print("Game reset for AI.")
        return self._get_observation_for_ai()

    def step_ai(self, action):
        """
        Processes one AI action and advances the game by one frame.
        Returns: observation, reward, done, info
        """
        if not self.gameplayActive: # Game might be in "Next Round" pause or "Game Over"
            if self.gameOverActive:
                return self._get_observation_for_ai(), 0, True, {'lives': self.lives, 'score': self.score}
            else: # e.g. "Next Round" screen
                 # We can either fast-forward this or return current state with no reward.
                 # For simplicity, let's assume AI step implies active gameplay.
                 # This part needs careful handling for robust AI training.
                 print("Warning: AI step called when gameplay not fully active.")
                 # Potentially advance timers to skip non-interactive parts:
                 current_time = pg.time.get_ticks()
                 if not self.enemies and not self.explosionsGroup : # Round cleared, in "Next Round" phase
                    if current_time - self.roundOverTimer >= 3000:
                        self.enemy_start_y = min(self.enemy_start_y + config.ENEMY_MOVE_DOWN, config.BLOCKERS_POSITION - 100)
                        if self.enemy_start_y >= config.BLOCKERS_POSITION - 100:
                             self.gameOverActive = True # Game over if enemies start too low.
                             return self._get_observation_for_ai(), 0, True, {'lives': self.lives, 'score': self.score}
                        self._reset_round_state(self.score)
                        self.gameplayActive = True


        prev_score = self.score
        prev_lives = self.lives

        # --- Apply AI action ---
        # Simulate key presses based on AI action
        # This is a simplified mapping. More complex actions might be needed.
        current_keys_state = pg.key.get_pressed() # Get current real keys (mostly for human override/debug)
        simulated_action_keys = {pg.K_LEFT: False, pg.K_RIGHT: False, pg.K_SPACE: False}

        if action == config.ACTION_LEFT:
            simulated_action_keys[pg.K_LEFT] = True
        elif action == config.ACTION_RIGHT:
            simulated_action_keys[pg.K_RIGHT] = True
        elif action == config.ACTION_SHOOT:
            simulated_action_keys[pg.K_SPACE] = True
        # config.ACTION_NONE results in no simulated key presses for movement/shooting

        # Update player based on simulated action (direct manipulation for AI)
        if self.shipCurrentlyAlive and self.player:
            if simulated_action_keys[pg.K_LEFT]:
                self.player.rect.x = max(10, self.player.rect.x - self.player.speed)
            if simulated_action_keys[pg.K_RIGHT]:
                self.player.rect.x = min(config.SCREEN_WIDTH - self.player.rect.width - 10, 
                                         self.player.rect.x + self.player.speed)
            if simulated_action_keys[pg.K_SPACE]:
                self._handle_player_shooting() # Uses self.score, self.shipCurrentlyAlive, self.bullets


        # --- Update game state for one frame (logic from _update_gameplay_state) ---
        currentTime = pg.time.get_ticks() # Or use a fixed dt for AI steps
        
        self.enemies.update(currentTime)
        
        # Update sprites (excluding player if already handled, or pass simulated keys)
        # For AI, drawing on screen_surface might be skipped.
        # So, sprite updates might need to function without screen_surface if not rendering.
        # For now, assume they are robust or AI implies rendering.
        # For allSprites.update, pass the AI's simulated keys, not human's.
        # Player's movement is already handled. The player.update method might conflict if called again by allSprites.
        # Better: player has move_left(), move_right(), shoot() methods for AI.
        # Temporary: For now, assume player update in allSprites is benign or player has been moved.
        
        # The player's visual update (drawing) still needs to happen.
        # Let's make `allSprites.update` pass a `keys` dict that AI controls for player part.
        # For this, Ship.update should accept this dict instead of pg.key.get_pressed().
        # This is a deeper refactor. For now, player handled, other sprites update:
        
        # Simplified sprite updates for AI step (non-player)
        # These calls will include drawing if update methods draw.
        # This should be fine even if AI doesn't "see" pixels, for game logic.
        for sprite in self.allSprites:
            if sprite != self.player: # Player was handled by AI action
                # Sprite.update needs a consistent signature: (self, keys, current_time, screen_surface)
                # Pass None for keys if not relevant for these sprites.
                sprite.update(None, currentTime, self.screen) 
        
        self.bullets.update(None, currentTime, self.screen)
        self.enemyBullets.update(None, currentTime, self.screen)
        self.explosionsGroup.update(None, currentTime, self.screen)


        self._check_collisions_and_deaths() # This updates score, lives, gameOverActive
        self._respawn_player_if_needed(currentTime)
        self._trigger_enemy_shooting(currentTime)
        # Skipping _play_background_music for AI speed

        # --- Calculate reward ---
        reward = (self.score - prev_score)  # Score change
        if self.lives < prev_lives: # Lost a life
            reward -= 50 # Example penalty, tune as needed

        # --- Check if game is done ---
        done = self.gameOverActive
        if not self.enemies and not self.explosionsGroup and not done: # Round cleared
            reward += 100 # Bonus for clearing round
            # For simple AI tasks, one round might be "done". Or continue.
            # If continuing, need to handle the "Next Round" screen logic briefly.
            # (This is partly handled at start of step_ai for now)
            pass


        # --- Get new observation and info ---
        observation = self._get_observation_for_ai()
        info = {'lives': self.lives, 'score': self.score, 'is_round_cleared': (not self.enemies and not self.explosionsGroup)}
        
        # Tick clock for AI too, to keep Pygame events processing and allow rendering if enabled
        self.clock.tick(config.FPS) # Or a different tick rate for AI headless mode

        return observation, reward, done, info

    def _get_observation_for_ai(self):
        """Returns the current game state as an observation for the AI."""
        # For pixel-based agents, render to a surface and get its array.
        # For feature-based, extract relevant game variables.
        # Placeholder: return raw pixel data of the screen.
        # Important: Ensure the screen is up-to-date before getting pixels.
        # This might mean calling draw methods if AI step doesn't draw.
        # self.screen.blit(self.background, (0,0))
        # self.allSprites.draw(self.screen) # If sprites don't draw in update
        # self.explosionsGroup.draw(self.screen)
        # ... draw HUD ...
        return pg.surfarray.array3d(pg.display.get_surface())


    def render_for_ai(self):
        """Renders the current game state. Useful if AI runs headless but occasional render is needed."""
        self.screen.blit(self.background, (0,0))
        
        # Simplest is to call the sprite update methods which also handle drawing
        # Passing None for keys as AI actions are handled in step_ai directly
        # This might be slow if AI calls render frequently.
        current_time = pg.time.get_ticks()
        self.allSprites.update(None, current_time, self.screen)
        self.bullets.update(None, current_time, self.screen)
        self.enemyBullets.update(None, current_time, self.screen)
        self.explosionsGroup.update(None, current_time, self.screen)

        # Draw HUD
        self.scoreValueText = Text(config.GAME_FONT, 20, str(self.score), config.GREEN, 85, 5)
        self.scoreLabelText.draw(self.screen)
        self.scoreValueText.draw(self.screen)
        self.livesLabelText.draw(self.screen)
        # Lives icons are in allSprites and should have been updated/drawn.

        pg.display.update()

    def get_action_size(self):
        return config.NUM_ACTIONS