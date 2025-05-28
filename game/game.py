import pygame as pg
import sys
import os
from random import choice
from . import config
import numpy as np

# Initialize Pygame's font module (mixer pre_init handled in Game)
pg.font.init()

# Global dictionary for images, to be populated after pygame display is initialized
IMAGES = {}
_images_loaded_and_converted = False # Module-level flag

# Define DummySound class at the module level for global access if needed as a fallback
class DummySound:
    def play(self, *args, **kwargs): pass
    def stop(self, *args, **kwargs): pass
    def fadeout(self, *args, **kwargs): pass
    def set_volume(self, *args, **kwargs): pass

def load_all_game_images(force_convert=False):
    global IMAGES, _images_loaded_and_converted
    if _images_loaded_and_converted and not force_convert: # Only load and convert once per process
        return

    print(f"Process {os.getpid()}: Loading images...")
    display_initialized = pg.display.get_init() and pg.display.get_surface() is not None

    for name in config.IMG_NAMES:
        try:
            image_path = config.IMAGE_PATH + '{}.png'.format(name)
            loaded_image = pg.image.load(image_path)
            if display_initialized or force_convert: # Only convert if display is set or forced
                IMAGES[name] = loaded_image.convert_alpha()
                # print(f"  Converted {name}")
            else:
                IMAGES[name] = loaded_image # Store unconverted surface
                # print(f"  Loaded (unconverted) {name}")
        except pg.error as e:
            # ... (your existing error handling for missing images) ...
            print(f"Error loading image '{image_path}': {e}")
            IMAGES[name] = pg.Surface((30,30)); IMAGES[name].fill(config.RED)

    if display_initialized or force_convert:
        _images_loaded_and_converted = True
    print(f"Process {os.getpid()}: Image loading complete. Converted: {_images_loaded_and_converted}")

class Ship(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image = IMAGES['ship']
        self.rect = self.image.get_rect(topleft=(config.PLAYER_START_X, config.PLAYER_START_Y))
        self.speed = config.PLAYER_SPEED

    def update(self, keys, current_time, screen_surface):
        if keys and keys[pg.K_LEFT] and self.rect.x > 10: # Check if keys is not None
            self.rect.x -= self.speed
        if keys and keys[pg.K_RIGHT] and self.rect.x < config.SCREEN_WIDTH - self.rect.width - 10: # Check if keys is not None
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

    def update(self, keys, current_time, screen_surface):
        screen_surface.blit(self.image, self.rect)
        self.rect.y += self.speed * self.direction
        if self.rect.y < 15 or self.rect.y > config.SCREEN_HEIGHT:
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

    def update(self, keys, current_time, screen_surface):
        screen_surface.blit(self.image, self.rect)

    def load_images(self):
        img_map = {
            0: ['enemy1_2', 'enemy1_1'],
            1: ['enemy2_2', 'enemy2_1'],
            2: ['enemy2_2', 'enemy2_1'],
            3: ['enemy3_1', 'enemy3_2'],
            4: ['enemy3_1', 'enemy3_2'],
        }
        img_indices = img_map[self.row]
        img1, img2 = (IMAGES[img_name] for img_name in img_indices)
        self.images.append(pg.transform.scale(img1, (40, 35)))
        self.images.append(pg.transform.scale(img2, (40, 35)))


class EnemiesGroup(pg.sprite.Group):
    def __init__(self, columns, rows, initial_y_position):
        pg.sprite.Group.__init__(self)
        self.enemies = [[None] * columns for _ in range(rows)]
        self.columns = columns
        self.rows = rows
        self.leftAddMove = 0
        self.rightAddMove = 0
        self.moveTime = 600
        self.direction = 1
        self.rightMoves = 30
        self.leftMoves = 30
        self.moveNumber = 15
        self.timer = pg.time.get_ticks()
        self.bottom = initial_y_position + ((rows - 1) * 45) + 35
        self._aliveColumns = list(range(columns))
        self._leftAliveColumn = 0
        self._rightAliveColumn = columns - 1

    def update(self, current_time):
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
                max_y = 0
                for enemy in self:
                    enemy.rect.y += config.ENEMY_MOVE_DOWN
                    enemy.toggle_image()
                    if enemy.rect.bottom > max_y: # Use rect.bottom for accurate position
                        max_y = enemy.rect.bottom
                self.bottom = max_y
            else:
                velocity = 10 if self.direction == 1 else -10
                for enemy in self:
                    enemy.rect.x += velocity
                    enemy.toggle_image()
                self.moveNumber += 1
            self.timer = current_time

    def add_internal(self, *sprites):
        super(EnemiesGroup, self).add_internal(*sprites)
        for s in sprites:
            if isinstance(s, Enemy):
                 self.enemies[s.row][s.column] = s

    def remove_internal(self, *sprites):
        super(EnemiesGroup, self).remove_internal(*sprites)
        for s in sprites:
            if isinstance(s, Enemy):
                 self.kill_enemy_sprite(s)
        self.update_speed()

    def is_column_dead(self, column):
        return not any(self.enemies[row][column] for row in range(self.rows))

    def random_bottom(self):
        if not self._aliveColumns:
            return None
        col = choice(self._aliveColumns)
        for row_idx in range(self.rows - 1, -1, -1):
            enemy = self.enemies[row_idx][col]
            if enemy is not None:
                return enemy
        return None

    def update_speed(self):
        num_enemies = len(self)
        if num_enemies == 1:
            self.moveTime = 200
        elif num_enemies <= 10:
            self.moveTime = 400

    def kill_enemy_sprite(self, enemy):
        self.enemies[enemy.row][enemy.column] = None
        is_col_dead = self.is_column_dead(enemy.column)
        
        if is_col_dead and enemy.column in self._aliveColumns:
            self._aliveColumns.remove(enemy.column)

        if enemy.column == self._rightAliveColumn:
            while self._rightAliveColumn >= 0 and self.is_column_dead(self._rightAliveColumn):
                self._rightAliveColumn -= 1
                if self._rightAliveColumn < 0: break
                self.rightAddMove += 5
        
        elif enemy.column == self._leftAliveColumn:
            while self._leftAliveColumn < self.columns and self.is_column_dead(self._leftAliveColumn):
                self._leftAliveColumn += 1
                if self._leftAliveColumn >= self.columns : break
                self.leftAddMove += 5


class Blocker(pg.sprite.Sprite):
    def __init__(self, size, color, row, column):
        pg.sprite.Sprite.__init__(self)
        self.height = size
        self.width = size
        self.color = color
        self.image = pg.Surface((self.width, self.height))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.row = row
        self.column = column

    def update(self, keys, current_time, screen_surface):
        screen_surface.blit(self.image, self.rect)


class Mystery(pg.sprite.Sprite):
    def __init__(self, sound_manager): # Expects the game's self.sounds dictionary
        pg.sprite.Sprite.__init__(self)
        self.image = IMAGES['mystery']
        self.image = pg.transform.scale(self.image, (75, 35))
        self.rect = self.image.get_rect(topleft=(-80, config.MYSTERY_SHIP_START_Y))
        self.row = 5 # Special row index for scoring (used in original, not in current score calc)
        self.moveTime = config.MYSTERY_SHIP_MOVE_TIME
        self.direction = 1
        self.timer = pg.time.get_ticks()
        
        self.mysteryEnteredSound = sound_manager.get('mysteryentered', DummySound())
        self.playSound = True

    def update(self, keys, currentTime, screen_surface):
        resetTimer = False
        passed = currentTime - self.timer
        if passed > self.moveTime:
            if (self.rect.x < 0 or self.rect.x > config.SCREEN_WIDTH) and self.playSound:
                self.mysteryEnteredSound.play()
                self.playSound = False
            
            if self.direction == 1:
                if self.rect.x < config.SCREEN_WIDTH + 40:
                    self.rect.x += 2
                else:
                    self.playSound = True
                    self.direction = -1
                    resetTimer = True
            else: # direction == -1
                if self.rect.x > -100:
                    self.rect.x -= 2
                else:
                    self.playSound = True
                    self.direction = 1
                    resetTimer = True
            
            if self.rect.x > 0 and self.rect.x < config.SCREEN_WIDTH :
                 self.mysteryEnteredSound.fadeout(4000)
                 screen_surface.blit(self.image, self.rect)

        if resetTimer:
            self.timer = currentTime


class EnemyExplosion(pg.sprite.Sprite):
    def __init__(self, enemy, *groups):
        super(EnemyExplosion, self).__init__(*groups)
        explosion_color_map = {
            0: 'green', 1: 'blue', 2: 'blue', 3: 'purple', 4: 'purple'
        }
        color = explosion_color_map.get(enemy.row, 'blue')
        self.image_orig = IMAGES['explosion{}'.format(color)]
        self.image = pg.transform.scale(self.image_orig, (40, 35))
        self.image2 = pg.transform.scale(self.image_orig, (50, 45))
        self.rect = self.image.get_rect(topleft=(enemy.rect.x, enemy.rect.y))
        self.timer = pg.time.get_ticks()

    def update(self, keys, current_time, screen_surface):
        passed = current_time - self.timer
        if passed <= 100:
            screen_surface.blit(self.image, self.rect)
        elif passed <= 200:
            screen_surface.blit(self.image2, (self.rect.x - 6, self.rect.y - 6))
        elif 200 < passed <= 400:
            pass
        else:
            self.kill()


class MysteryExplosion(pg.sprite.Sprite):
    def __init__(self, mystery, score, *groups):
        super(MysteryExplosion, self).__init__(*groups)
        self.text_renderer = Text(config.GAME_FONT, 20, str(score), config.WHITE,
                                  mystery.rect.x + 20, mystery.rect.y + 6)
        self.timer = pg.time.get_ticks()

    def update(self, keys, current_time, screen_surface):
        passed = current_time - self.timer
        if passed <= 200 or (400 < passed <= 600):
            self.text_renderer.draw(screen_surface)
        elif passed > 600:
            self.kill()


class ShipExplosion(pg.sprite.Sprite):
    def __init__(self, ship, *groups):
        super(ShipExplosion, self).__init__(*groups)
        self.image = IMAGES['ship']
        self.rect = self.image.get_rect(topleft=(ship.rect.x, ship.rect.y))
        self.timer = pg.time.get_ticks()

    def update(self, keys, current_time, screen_surface):
        passed = current_time - self.timer
        if 300 < passed <= 600:
            screen_surface.blit(self.image, self.rect)
        elif passed > 900:
            self.kill()


class Life(pg.sprite.Sprite):
    def __init__(self, xpos, ypos):
        pg.sprite.Sprite.__init__(self)
        self.image = IMAGES['ship']
        self.image = pg.transform.scale(self.image, (23, 23))
        self.rect = self.image.get_rect(topleft=(xpos, ypos))

    def update(self, keys, current_time, screen_surface):
        screen_surface.blit(self.image, self.rect)


class Text(object):
    def __init__(self, textFontPath, size, message, color, xpos, ypos):
        try:
            self.font = pg.font.Font(textFontPath, size)
        except pg.error: # Fallback if font not found
            print(f"Warning: Font '{textFontPath}' not found. Using default system font.")
            self.font = pg.font.Font(None, size) # Default system font
        self.surface = self.font.render(message, True, color)
        self.rect = self.surface.get_rect(topleft=(xpos, ypos))

    def draw(self, surface):
        surface.blit(self.surface, self.rect)


class Game:
    def __init__(self, silent_mode=False, ai_training_mode=False, headless_worker_mode=False): # New flag
        self.silent_mode = silent_mode
        self.ai_training_mode = ai_training_mode
        self.headless_worker_mode = headless_worker_mode

        if not self.silent_mode:
            pg.mixer.pre_init(44100, -16, 1, 4096)
        else:
            try: pg.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            except pg.error: print("Mixer init failed in silent (this is often ok).")

        pg.init()
        if not self.headless_worker_mode:
            self.screen = pg.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            pg.display.set_caption('Space Invaders')
        else:
            # For headless workers, create an in-memory surface of the same dimensions
            # This surface will be used by _get_observation_for_ai
            self.screen = pg.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            # No display caption needed for headless
        load_all_game_images()
        self.clock = pg.time.Clock()
        try:
            self.background = pg.image.load(config.IMAGE_PATH + 'background.jpg').convert()
        except pg.error as e:
            print(f"Warning: Could not load background image '{config.IMAGE_PATH + 'background.jpg'}': {e}")
            self.background = pg.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            self.background.fill((0,0,0)) # Black background

        self.mainScreenActive = True
        self.gameplayActive = False
        self.gameOverActive = False
        self.enemy_start_y = config.ENEMY_DEFAULT_POSITION
        self.score = 0
        self.lives = 3

        self._make_static_text_objects()
        self._create_audio_assets() # Call BEFORE sprites that need sounds are made

        self.player_respawn_delay = 900
        self.inter_round_delay = 3000
        if self.ai_training_mode:
            self.player_respawn_delay = 0
            self.inter_round_delay = 0

        self.player = None
        self.playerGroup = pg.sprite.Group()
        self.enemies = None
        self.bullets = pg.sprite.Group()
        self.enemyBullets = pg.sprite.Group()
        self.mysteryShip = None # Will be created in _reset_round_state
        self.mysteryGroup = pg.sprite.Group()
        self.explosionsGroup = pg.sprite.Group()
        self.allBlockers = pg.sprite.Group()
        self.livesSpritesGroup = pg.sprite.Group()
        self.allSprites = pg.sprite.Group()

        self.general_timer = pg.time.get_ticks()
        self.noteTimer = pg.time.get_ticks()
        self.shipRespawnTimer = pg.time.get_ticks()
        self.roundOverTimer = pg.time.get_ticks()
        self.makeNewShipNext = False
        self.shipCurrentlyAlive = True
        
        # A flag to indicate if render_for_ai was called in the current step_ai cycle
        # This helps decide if clock.tick needs to be conditional in step_ai
        self._is_rendering_for_ai_this_step = False


    def _make_static_text_objects(self):
        self.titleText = Text(config.GAME_FONT, 50, 'Space Invaders', config.WHITE, 164, 155)
        self.titleText2 = Text(config.GAME_FONT, 25, 'Press any key to continue', config.WHITE, 201, 225)
        self.gameOverTextDisplay = Text(config.GAME_FONT, 50, 'Game Over', config.WHITE, 250, 270)
        self.nextRoundTextDisplay = Text(config.GAME_FONT, 50, 'Next Round', config.WHITE, 240, 270)
        self.scoreLabelText = Text(config.GAME_FONT, 20, 'Score', config.WHITE, 5, 5)
        self.livesLabelText = Text(config.GAME_FONT, 20, 'Lives ', config.WHITE, 640, 5)
        self.enemy1ScoreText = Text(config.GAME_FONT, 25, '   =   30 pts', config.PURPLE, 368, 270)
        self.enemy2ScoreText = Text(config.GAME_FONT, 25, '   =  20 pts', config.BLUE, 368, 320)
        self.enemy3ScoreText = Text(config.GAME_FONT, 25, '   =  10 pts', config.GREEN, 368, 370)
        self.mysteryScoreText = Text(config.GAME_FONT, 25, '   =  ?????', config.RED, 368, 420)
        self.scoreValueText = None

    def _create_audio_assets(self):
        self.sounds = {}
        self.musicNotes = []
        self.noteIndex = 0
        sound_names_to_load = ['shoot', 'shoot2', 'invaderkilled', 'mysterykilled', 'shipexplosion', 'mysteryentered']

        if self.silent_mode:
            for sound_name in sound_names_to_load:
                self.sounds[sound_name] = DummySound()
            for i in range(4):
                self.musicNotes.append(DummySound())
            return

        for sound_name in sound_names_to_load:
            try:
                path = config.SOUND_PATH + f'{sound_name}.wav'
                self.sounds[sound_name] = pg.mixer.Sound(path)
                volume = 0.3 if sound_name == 'mysteryentered' else 0.2
                self.sounds[sound_name].set_volume(volume)
            except pg.error as e:
                print(f"Warning: Could not load sound '{path}': {e}")
                self.sounds[sound_name] = DummySound()
        try:
            self.musicNotes = [pg.mixer.Sound(config.SOUND_PATH + f'{i}.wav') for i in range(4)]
            for sound in self.musicNotes:
                sound.set_volume(0.5)
        except pg.error as e:
            print(f"Warning: Could not load music notes: {e}")
            self.musicNotes = [DummySound() for _ in range(4)]

    def _full_game_reset(self, start_score=0, start_lives=3):
        self.score = start_score
        self.lives = start_lives
        self.enemy_start_y = config.ENEMY_DEFAULT_POSITION
        self.allBlockers.empty()
        for i in range(4):
            self.allBlockers.add(self._make_blocker_group(i))
        self._reset_round_state(start_score)

    def _reset_round_state(self, current_score):
        self.score = current_score
        self.playerGroup.empty()
        self.mysteryGroup.empty()
        self.bullets.empty()
        self.enemyBullets.empty()
        self.explosionsGroup.empty()
        if self.enemies: self.enemies.empty()
        self.allSprites.empty()

        self.player = Ship()
        self.playerGroup.add(self.player)
        
        self.mysteryShip = Mystery(sound_manager=self.sounds)
        self.mysteryGroup.add(self.mysteryShip)
        
        self._make_enemies_formation()
        self._update_lives_sprites()

        self.allSprites.add(self.player, list(self.enemies), self.mysteryShip, list(self.allBlockers), list(self.livesSpritesGroup))

        self.keys = pg.key.get_pressed()
        self.general_timer = pg.time.get_ticks()
        self.noteTimer = pg.time.get_ticks()
        self.makeNewShipNext = False
        self.shipCurrentlyAlive = True
        if self.lives <= 0:
            self.shipCurrentlyAlive = False

    def _make_blocker_group(self, number):
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
        for row_idx in range(5):
            for col_idx in range(10):
                enemy = Enemy(row_idx, col_idx)
                enemy.rect.x = 157 + (col_idx * 50)
                enemy.rect.y = self.enemy_start_y + (row_idx * 45)
                self.enemies.add(enemy)

    def _update_lives_sprites(self):
        self.livesSpritesGroup.empty()
        life_positions = [715, 742, 769]
        for i in range(self.lives):
            if i < len(life_positions):
                life_sprite = Life(life_positions[i], 3)
                self.livesSpritesGroup.add(life_sprite)
        # No need to re-add to self.allSprites if _reset_round_state handles it initially
        # and this is only called during _reset_round_state.
        # If called mid-game (it's not currently), then re-adding might be needed.

    def run_player_mode(self):
        self._full_game_reset()
        running = True
        while running:
            currentTime = pg.time.get_ticks()
            self.keys = pg.key.get_pressed()
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYUP and e.key == pg.K_ESCAPE):
                    running = False
                if self.mainScreenActive:
                    if e.type == pg.KEYUP:
                        self.mainScreenActive = False
                        self.gameplayActive = True
                        self.gameOverActive = False
                        self._full_game_reset(start_score=0, start_lives=3)
                elif self.gameplayActive:
                    if e.type == pg.KEYDOWN and e.key == pg.K_SPACE:
                        self._handle_player_shooting()
                elif self.gameOverActive:
                    if e.type == pg.KEYUP:
                        self.gameOverActive = False
                        self.mainScreenActive = True
            
            self.screen.blit(self.background, (0,0))
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
        if self.shipCurrentlyAlive and len(self.bullets) == 0:
            if self.score < 1000:
                bullet = Bullet(self.player.rect.x + 23, self.player.rect.y + 5, -1, config.PLAYER_LASER_SPEED, 'laser', 'center')
                self.bullets.add(bullet)
                self.allSprites.add(bullet) # Add new bullets to allSprites
                self.sounds['shoot'].play()
            else:
                l_bullet = Bullet(self.player.rect.x + 8, self.player.rect.y + 5, -1, config.PLAYER_LASER_SPEED, 'laser', 'left')
                r_bullet = Bullet(self.player.rect.x + 38, self.player.rect.y + 5, -1, config.PLAYER_LASER_SPEED, 'laser', 'right')
                self.bullets.add(l_bullet, r_bullet)
                self.allSprites.add(l_bullet, r_bullet) # Add new bullets to allSprites
                self.sounds['shoot2'].play()

    def _draw_main_menu_elements(self):
        self.titleText.draw(self.screen)
        self.titleText2.draw(self.screen)
        self.enemy1ScoreText.draw(self.screen)
        self.enemy2ScoreText.draw(self.screen)
        self.enemy3ScoreText.draw(self.screen)
        self.mysteryScoreText.draw(self.screen)
        e1_img = pg.transform.scale(IMAGES['enemy3_1'], (40, 40))
        e2_img = pg.transform.scale(IMAGES['enemy2_2'], (40, 40))
        e3_img = pg.transform.scale(IMAGES['enemy1_2'], (40, 40))
        mystery_img = pg.transform.scale(IMAGES['mystery'], (80, 40))
        self.screen.blit(e1_img, (318, 270))
        self.screen.blit(e2_img, (318, 320))
        self.screen.blit(e3_img, (318, 370))
        self.screen.blit(mystery_img, (299, 420))

    def _update_gameplay_state(self, currentTime):
        if not self.enemies and not self.explosionsGroup:
            self.gameplayActive = False
            self.roundOverTimer = currentTime
            return

        if self.enemies: self.enemies.update(currentTime)
        
        # Pass self.keys for player Ship to react in human mode.
        # For AI mode, player movement is handled directly in step_ai before this.
        # If step_ai calls this, keys could be the AI's simulated keys or None.
        keys_for_update = self.keys if not self.ai_training_mode else None
        self.allSprites.update(keys_for_update, currentTime, self.screen)
        self.bullets.update(keys_for_update, currentTime, self.screen)
        self.enemyBullets.update(keys_for_update, currentTime, self.screen)
        self.explosionsGroup.update(keys_for_update, currentTime, self.screen)

        self._check_collisions_and_deaths()
        self._respawn_player_if_needed(currentTime)
        self._trigger_enemy_shooting(currentTime)
        if not self.ai_training_mode: # Only play background music in human mode
            self._play_background_music(currentTime)

        if self.enemies and self.enemies.bottom >= config.SCREEN_HEIGHT - 80:
            if self.shipCurrentlyAlive and self.player and pg.sprite.spritecollideany(self.player, self.enemies):
                self._player_death()
            if self.enemies.bottom >= config.SCREEN_HEIGHT:
                self._player_death(final_death=True)

    def _draw_gameplay_elements(self, currentTime):
        # "Next Round" screen logic
        if not self.gameplayActive and not self.gameOverActive and not self.mainScreenActive:
            proceed_to_next_round = False
            if self.ai_training_mode: # For AI, inter_round_delay is 0, so this should be true immediately
                if currentTime - self.roundOverTimer >= self.inter_round_delay:
                    proceed_to_next_round = True
            else: # For human player, wait for the actual delay
                if currentTime - self.roundOverTimer < self.inter_round_delay:
                    self.nextRoundTextDisplay.draw(self.screen)
                else:
                    proceed_to_next_round = True
            
            if proceed_to_next_round:
                self.enemy_start_y = min(self.enemy_start_y + config.ENEMY_MOVE_DOWN, config.BLOCKERS_POSITION - 100)
                if self.enemy_start_y >= config.BLOCKERS_POSITION - 100:
                    self._player_death(final_death=True)
                    return # Game over, no further drawing for this phase
                self._reset_round_state(self.score)
                self.gameplayActive = True
                return # Skip drawing HUD this frame as game state just reset

        # Dynamic HUD elements
        self.scoreValueText = Text(config.GAME_FONT, 20, str(self.score), config.GREEN, 85, 5)
        self.scoreLabelText.draw(self.screen)
        self.scoreValueText.draw(self.screen)
        self.livesLabelText.draw(self.screen)
        # Lives icons are drawn via self.allSprites.update

    def _play_background_music(self, currentTime):
        # This check for self.silent_mode is redundant if self.musicNotes contains DummySound objects
        # but doesn't hurt.
        if self.silent_mode: return

        if self.enemies and self.enemies.moveTime > 0 and len(self.musicNotes) > 0:
            if currentTime - self.noteTimer > self.enemies.moveTime:
                note = self.musicNotes[self.noteIndex]
                self.noteIndex = (self.noteIndex + 1) % len(self.musicNotes)
                note.play()
                self.noteTimer = currentTime

    def _trigger_enemy_shooting(self, currentTime):
        if (currentTime - self.general_timer) > config.ENEMY_SHOOT_INTERVAL and self.enemies:
            enemy_to_shoot = self.enemies.random_bottom()
            if enemy_to_shoot:
                bullet = Bullet(enemy_to_shoot.rect.x + 14, enemy_to_shoot.rect.y + 20, 1,
                                config.ENEMY_LASER_SPEED, 'enemylaser', 'center')
                self.enemyBullets.add(bullet)
                self.allSprites.add(bullet) # Add new bullets to allSprites
                self.general_timer = currentTime

    def _calculate_score_for_kill(self, killed_sprite):
        points = 0
        if isinstance(killed_sprite, Mystery):
            points = choice(config.MYSTERY_SCORES_OPTIONS)
        elif isinstance(killed_sprite, Enemy):
            points = config.ENEMY_SCORES_BY_ROW_INDEX.get(killed_sprite.row, 0)
        self.score += points
        return points

    def _check_collisions_and_deaths(self):
        pg.sprite.groupcollide(self.bullets, self.enemyBullets, True, True)

        if self.enemies: # Ensure enemies group exists
            for enemy in pg.sprite.groupcollide(self.enemies, self.bullets, True, True).keys():
                self.sounds['invaderkilled'].play()
                self._calculate_score_for_kill(enemy)
                EnemyExplosion(enemy, self.explosionsGroup, self.allSprites)

        if self.mysteryGroup: # Ensure mystery group exists
            for hit_mystery_sprite in pg.sprite.groupcollide(self.mysteryGroup, self.bullets, True, True).keys():
                if hasattr(hit_mystery_sprite, 'mysteryEnteredSound') and hit_mystery_sprite.mysteryEnteredSound:
                    hit_mystery_sprite.mysteryEnteredSound.stop()
                self.sounds['mysterykilled'].play()
                score_val = self._calculate_score_for_kill(hit_mystery_sprite)
                MysteryExplosion(hit_mystery_sprite, score_val, self.explosionsGroup, self.allSprites)
                
                new_mystery_ship = Mystery(sound_manager=self.sounds)
                self.mysteryShip = new_mystery_ship
                self.mysteryGroup.add(self.mysteryShip)
                self.allSprites.add(self.mysteryShip)

        if self.shipCurrentlyAlive and self.player:
            if pg.sprite.spritecollide(self.player, self.enemyBullets, True):
                self._player_death()
        
        pg.sprite.groupcollide(self.bullets, self.allBlockers, True, True)
        pg.sprite.groupcollide(self.enemyBullets, self.allBlockers, True, True)
        if self.enemies and self.allBlockers and self.enemies.bottom >= config.BLOCKERS_POSITION:
            pg.sprite.groupcollide(self.enemies, self.allBlockers, False, True)


    def _player_death(self, final_death=False):
        if not self.shipCurrentlyAlive and not final_death:
            return
        self.sounds['shipexplosion'].play()
        if self.player:
             ShipExplosion(self.player, self.explosionsGroup, self.allSprites)
             self.player.kill()
        
        self.shipCurrentlyAlive = False
        self.lives -= 1
        self._update_lives_sprites() # Updates icons

        # Add existing lives sprites back to allSprites if they were cleared or not added correctly
        # This ensures they are drawn even if _reset_round_state isn't called immediately.
        self.allSprites.add(list(self.livesSpritesGroup))


        if self.lives <= 0 or final_death:
            self.gameOverActive = True
            self.gameplayActive = False
            self.roundOverTimer = pg.time.get_ticks()
        else:
            self.makeNewShipNext = True
            self.shipRespawnTimer = pg.time.get_ticks()

    def _respawn_player_if_needed(self, currentTime):
        if self.makeNewShipNext and (currentTime - self.shipRespawnTimer > self.player_respawn_delay):
            if self.lives > 0:
                self.player = Ship()
                self.playerGroup.add(self.player)
                self.allSprites.add(self.player)
                self.shipCurrentlyAlive = True
                self.makeNewShipNext = False
            else:
                if not self.gameOverActive: # Should already be handled by _player_death
                    self.gameOverActive = True
                    self.gameplayActive = False
                    self.roundOverTimer = pg.time.get_ticks()

    def _update_game_over_state(self, currentTime):
        if self.ai_training_mode or (currentTime - self.roundOverTimer > self.inter_round_delay):
            self.gameOverActive = False # Transition out of game over screen
            if not self.ai_training_mode: # For human, go to main menu
                 self.mainScreenActive = True
            # For AI, reset_for_ai will handle the next state
            self.enemy_start_y = config.ENEMY_DEFAULT_POSITION # Reset for potential new game

    def _draw_game_over_elements(self, currentTime):
        # For human player, show "Game Over" message
        if not self.ai_training_mode:
            time_in_game_over = currentTime - self.roundOverTimer
            if (time_in_game_over // 750) % 2 == 0:
                self.gameOverTextDisplay.draw(self.screen)
            self.scoreValueText = Text(config.GAME_FONT, 20, str(self.score), config.GREEN, 85, 5)
            self.scoreLabelText.draw(self.screen)
            self.scoreValueText.draw(self.screen)


    def reset_for_ai(self):
        self._full_game_reset(start_score=0, start_lives=3)
        self.mainScreenActive = False # AI doesn't see main menu
        self.gameplayActive = True
        self.gameOverActive = False
        return self._get_observation_for_ai()

    def step_ai(self, action):
        # If game is paused for "Next Round" or "Game Over" (for AI, these pauses are near zero)
        current_time_step = pg.time.get_ticks()

        # Handle transitions between game states (Next Round, Game Over)
        # This logic allows the game to advance out of pause states quickly for AI.
        if not self.gameplayActive:
            if self.gameOverActive:
                # If game over, and in AI mode, the inter_round_delay (which is 0 for AI) should have passed.
                # The 'done_flag' will be True, and the training loop will reset.
                # No special action needed here beyond returning the state.
                # The 'done_flag' will be set to self.gameOverActive later.
                pass # Let the main logic and done_flag handle termination
            
            elif not self.enemies and not self.explosionsGroup: # "Next Round" phase
                # This means all enemies are cleared, and no explosions are active.
                # The roundOverTimer was set when the last enemy/explosion cleared.
                if self.ai_training_mode and (current_time_step - self.roundOverTimer >= self.inter_round_delay):
                    # AI mode: "Next Round" delay has passed (instantly if delay is 0)
                    self.enemy_start_y = min(self.enemy_start_y + config.ENEMY_MOVE_DOWN, config.BLOCKERS_POSITION - 100)
                    if self.enemy_start_y >= config.BLOCKERS_POSITION - 100:
                        self.gameOverActive = True # Mark as game over if enemies start too low
                        # game will effectively end here due to done_flag being True
                    else:
                        self._reset_round_state(self.score) # Prepare for the next round
                        self.gameplayActive = True         # Resume gameplay
                
                # If still not gameplayActive (e.g., human player waiting for visual delay, or AI mode about to transition)
                # and not game over, return current state indicating round was cleared.
                if not self.gameplayActive and not self.gameOverActive:
                     observation_val = self._get_observation_for_ai()
                     info_dict = {'lives': self.lives, 'score': self.score, 'is_round_cleared': True}
                     # For this intermediate state, reward is 0, game is not "done" (game over).
                     return observation_val, 0, False, info_dict
            # If none of the above, game might be in an unexpected paused state. 
            # For AI, we generally want to keep it moving or end the episode.
            # If it's truly stuck, done_flag will eventually be caught by max_steps.

        # --- If gameplay is active or just became active ---
        prev_score = self.score
        prev_lives = self.lives

        simulated_action_keys = {pg.K_LEFT: False, pg.K_RIGHT: False, pg.K_SPACE: False}
        if action == config.ACTION_LEFT: simulated_action_keys[pg.K_LEFT] = True
        elif action == config.ACTION_RIGHT: simulated_action_keys[pg.K_RIGHT] = True
        elif action == config.ACTION_SHOOT: simulated_action_keys[pg.K_SPACE] = True

        if self.shipCurrentlyAlive and self.player:
            if simulated_action_keys[pg.K_LEFT]:
                self.player.rect.x = max(10, self.player.rect.x - self.player.speed)
            if simulated_action_keys[pg.K_RIGHT]:
                self.player.rect.x = min(config.SCREEN_WIDTH - self.player.rect.width - 10, 
                                         self.player.rect.x + self.player.speed)
            if simulated_action_keys[pg.K_SPACE]:
                self._handle_player_shooting()
        
        current_time_update = pg.time.get_ticks() # Use a consistent time for all updates in this step

        if self.enemies: 
            self.enemies.update(current_time_update)
        
        # For AI step, pass None as keys to allSprites.update.
        # Player was already moved. Ship.update needs to be robust to keys=None.
        self.allSprites.update(None, current_time_update, self.screen)
        self.bullets.update(None, current_time_update, self.screen)
        self.enemyBullets.update(None, current_time_update, self.screen)
        self.explosionsGroup.update(None, current_time_update, self.screen)

        self._check_collisions_and_deaths()
        self._respawn_player_if_needed(current_time_update) # Uses self.player_respawn_delay
        self._trigger_enemy_shooting(current_time_update)

        # Check for enemies reaching bottom (game over condition)
        if self.enemies and self.enemies.bottom >= config.SCREEN_HEIGHT - 80:
            if self.shipCurrentlyAlive and self.player and pg.sprite.spritecollideany(self.player, self.enemies):
                self._player_death() 
            if self.enemies.bottom >= config.SCREEN_HEIGHT:
                if not self.gameOverActive: # Ensure _player_death is called only once for this condition
                    self._player_death(final_death=True)

        # --- Calculate reward and done status ---
        reward_val = (self.score - prev_score)
        if self.lives < prev_lives: 
            reward_val -= 50 # Penalty for losing a life

        done_flag = self.gameOverActive # This is the primary "game over" condition
        
        round_cleared_this_step = False
        # Check if a round was just cleared AND the game is not already over
        if not self.enemies and not self.explosionsGroup and not done_flag and self.gameplayActive:
            # self.gameplayActive check ensures we were in active play before clearing.
            # If round clear logic already set gameplayActive=False, this won't run again,
            # which is intended. The roundOverTimer would have been set.
            round_cleared_this_step = True
            reward_val += 100 # Bonus for clearing a round
            self.gameplayActive = False # Transition to "Next Round" pause state
            self.roundOverTimer = pg.time.get_ticks() # Set timer for the (potentially zero) pause

        observation_val = self._get_observation_for_ai()
        info_dict = {'lives': self.lives, 'score': self.score, 'is_round_cleared': round_cleared_this_step}
        
        # --- Clock Tick Logic ---
        if self.headless_worker_mode:
            # Headless workers in multiprocessing run as fast as possible.
            # Pygame's internal timers still advance based on pg.time.get_ticks().
            pass 
        elif self._is_rendering_for_ai_this_step:
            # If render_for_ai() was called in this conceptual frame (by train.py/test.py when args.render is True)
            # Use a potentially higher FPS for faster visual training/testing.
            self.clock.tick(getattr(config, 'AI_TRAIN_RENDER_FPS', 120)) # Use game.config
        elif not self.ai_training_mode:
            # If not in AI training mode at all (e.g., human play via run_player_mode, though this uses its own loop)
            # or if step_ai was used for some other mode that needs standard FPS.
            self.clock.tick(config.FPS) # Use game.config
        # Else (self.ai_training_mode is True AND self._is_rendering_for_ai_this_step is False):
        #   This means AI training is running headless (no render call from train.py/test.py).
        #   Do NOT call clock.tick() to run at maximum computational speed.
        #   This case is covered by the above conditions (headless_worker_mode or the other two failing).

        self._is_rendering_for_ai_this_step = False # Reset for the next call to step_ai

        return observation_val, reward_val, done_flag, info_dict

    def _get_observation_for_ai(self):
        # If truly headless (no display.set_mode), this needs to draw to a specific surface.
        # If display.set_mode was called (even with dummy driver), self.screen exists.
        # All sprite updates blit to self.screen.
        # If self.screen is a pg.Surface and not the display screen, this is fine.
        
        # If self.screen was a dummy surface, we need to manually draw everything to it here.
        # However, the current structure where step_ai calls allSprites.update(..., self.screen)
        # means self.screen (whether it's the display or an in-memory Surface) gets updated.
        if self.screen: # Check if self.screen was initialized
            return pg.surfarray.array3d(self.screen)
        else: # Should not happen if __init__ always creates a surface or display
            return np.zeros((config.SCREEN_HEIGHT, config.SCREEN_WIDTH, 3), dtype=np.uint8)

    def render_for_ai(self):
        if self.headless_worker_mode: # Workers should not render to physical display
            # They might "render" to their internal self.screen for _get_observation_for_ai
            # but no pg.display.update()
            # The blitting happens in sprite updates to self.screen (in-memory surface for worker)
            # So, this method might just be a pass for headless, or ensure self.screen is updated.
            # Let's ensure self.screen is drawn onto for observation purposes.
            self.screen.blit(self.background, (0,0))
            current_time = pg.time.get_ticks()
            self.allSprites.update(None, current_time, self.screen) # Update and blit to self.screen
            self.bullets.update(None, current_time, self.screen)
            self.enemyBullets.update(None, current_time, self.screen)
            self.explosionsGroup.update(None, current_time, self.screen)
            # Draw HUD to self.screen
            self.scoreValueText = Text(config.GAME_FONT, 20, str(self.score), config.GREEN, 85, 5)
            self.scoreLabelText.draw(self.screen)
            self.scoreValueText.draw(self.screen)
            self.livesLabelText.draw(self.screen)
            return # No pg.display.update()
        # Original rendering logic for non-headless
        self._is_rendering_for_ai_this_step = True
        self.screen.blit(self.background, (0,0))
        current_time = pg.time.get_ticks()
        self.allSprites.update(None, current_time, self.screen)
        self.bullets.update(None, current_time, self.screen)
        self.enemyBullets.update(None, current_time, self.screen)
        self.explosionsGroup.update(None, current_time, self.screen)
        self.scoreValueText = Text(config.GAME_FONT, 20, str(self.score), config.GREEN, 85, 5)
        self.scoreLabelText.draw(self.screen)
        self.scoreValueText.draw(self.screen)
        self.livesLabelText.draw(self.screen)
        pg.display.update() # Only update physical display if not headless worker

    def get_action_size(self):
        return config.NUM_ACTIONS