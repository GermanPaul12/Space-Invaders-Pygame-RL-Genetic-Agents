# game/game_manager.py
import pygame as pg
import sys
import os 
from random import choice
import numpy as np 

from . import config 
from .assets import load_all_game_images, DummySound, IMAGES 
from .ui_elements import Text 
from .sprites import ( 
    Ship, Bullet, Enemy, EnemiesGroup, Blocker, Mystery, 
    EnemyExplosion, MysteryExplosion, ShipExplosion, Life
)

class Game:
    def __init__(self, silent_mode=False, ai_training_mode=False, headless_worker_mode=False):
        self.silent_mode = silent_mode
        self.ai_training_mode = ai_training_mode
        self.headless_worker_mode = headless_worker_mode
        
        if not pg.get_init(): 
            pg.init()
        
        if not self.headless_worker_mode:
            self.screen = pg.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            pg.display.set_caption('Space Invaders')
        else: 
            try:
                self.screen = pg.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            except pg.error as e_disp:
                self.screen = pg.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        
        load_all_game_images() 

        if not self.silent_mode:
            if not pg.mixer.get_init():
                try: 
                    pg.mixer.pre_init(44100, -16, 2, 512) 
                    pg.mixer.init()
                except pg.error as e_mix: 
                    self.silent_mode = True 
        else: 
            if not pg.mixer.get_init():
                 try: pg.mixer.init(frequency=22050, size=-16, channels=2, buffer=512) 
                 except: pass 

        self.clock = pg.time.Clock()
        try: 
            bg_path = os.path.join(config.IMAGE_PATH, 'background.jpg')
            if pg.display.get_init() and pg.display.get_surface() and not self.headless_worker_mode: 
                self.background = pg.image.load(bg_path).convert()
            else: 
                self.background = pg.image.load(bg_path)
        except Exception as e_bg_load: 
            print(f"ERROR: Failed to load background image '{bg_path}': {e_bg_load}", flush=True)
            self.background = pg.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            self.background.fill((0,0,0)) 

        self.mainScreenActive = True
        self.gameplayActive = False
        self.gameOverActive = False
        self.enemy_start_y = config.ENEMY_DEFAULT_POSITION
        self.score = 0
        self.lives = config.PLAYER_LIVES 

        self._make_static_text_objects()
        self._create_audio_assets()

        self.player_respawn_delay = config.PLAYER_RESPAWN_DELAY_MS 
        self.inter_round_delay = config.INTER_ROUND_DELAY_MS
        if self.ai_training_mode:
            self.player_respawn_delay = config.AI_PLAYER_RESPAWN_DELAY_MS
            self.inter_round_delay = config.AI_INTER_ROUND_DELAY_MS

        self.player = None
        self.playerGroup = pg.sprite.GroupSingle() 
        self.enemies = None 
        self.bullets = pg.sprite.Group()
        self.enemyBullets = pg.sprite.Group()
        self.mysteryShip = None 
        self.mysteryGroup = pg.sprite.GroupSingle() 
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
        self._is_rendering_for_ai_this_step = False

    def _make_static_text_objects(self):
        # Centering Text using the new center_x, center_y flags
        self.titleText = Text(config.GAME_FONT, 50, 'Space Invaders', config.WHITE, 
                              config.SCREEN_WIDTH // 2, 155, center_x=True)
        self.titleText2 = Text(config.GAME_FONT, 25, 'Press any key to continue', config.WHITE,
                               config.SCREEN_WIDTH // 2, 225, center_x=True)
        self.gameOverTextDisplay = Text(config.GAME_FONT, 50, 'Game Over', config.WHITE,
                                        config.SCREEN_WIDTH // 2, 270, center_x=True)
        self.nextRoundTextDisplay = Text(config.GAME_FONT, 50, 'Next Round', config.WHITE,
                                         config.SCREEN_WIDTH // 2, 270, center_x=True)
        
        self.scoreLabelText = Text(config.GAME_FONT, 20, 'Score', config.WHITE, 5, 5)

        estimated_icons_width = (config.PLAYER_LIVES * (config.LIFE_SPRITE_WIDTH + 5)) - 5 
        lives_label_x = config.SCREEN_WIDTH - estimated_icons_width - Text(config.GAME_FONT, 20, 'Lives', config.WHITE, 0,0).rect.width - 10
        self.livesLabelText = Text(config.GAME_FONT, 20, 'Lives', config.WHITE, lives_label_x, 5)

        text_x_pos_anchor = config.SCREEN_WIDTH // 2 
        self.enemy1ScoreText = Text(config.GAME_FONT, 25, '   =   30 pts', config.PURPLE, 
                                    text_x_pos_anchor, 270, center_x=True)
        self.enemy2ScoreText = Text(config.GAME_FONT, 25, '   =  20 pts', config.BLUE, 
                                    text_x_pos_anchor, 320, center_x=True)
        self.enemy3ScoreText = Text(config.GAME_FONT, 25, '   =  10 pts', config.GREEN, 
                                    text_x_pos_anchor, 370, center_x=True)
        self.mysteryScoreText = Text(config.GAME_FONT, 25, '   =  ?????', config.RED, 
                                     text_x_pos_anchor, 420, center_x=True)
        self.scoreValueText = None

    def _create_audio_assets(self):
        self.sounds = {}
        self.musicNotes = []
        self.noteIndex = 0
        sound_names_to_load = ['shoot', 'shoot2', 'invaderkilled', 'mysterykilled', 'shipexplosion', 'mysteryentered']
        if self.silent_mode:
            for name in sound_names_to_load: self.sounds[name] = DummySound()
            for _ in range(4): self.musicNotes.append(DummySound())
            return
        for name in sound_names_to_load:
            try:
                path = os.path.join(config.SOUND_PATH, f'{name}.wav')
                self.sounds[name] = pg.mixer.Sound(path)
                self.sounds[name].set_volume(0.3 if name == 'mysteryentered' else 0.2)
            except pg.error: self.sounds[name] = DummySound()
        try:
            self.musicNotes = [pg.mixer.Sound(os.path.join(config.SOUND_PATH, f'{i}.wav')) for i in range(4)]
            for sound in self.musicNotes: sound.set_volume(0.5)
        except pg.error: self.musicNotes = [DummySound() for _ in range(4)]

    def _full_game_reset(self, start_score=0, start_lives=config.PLAYER_LIVES):
        self.score = start_score
        self.lives = start_lives
        self.enemy_start_y = config.ENEMY_DEFAULT_POSITION
        self.allBlockers.empty() 
        for i in range(4): 
            blocker_formation = self._make_blocker_group(i)
            self.allBlockers.add(blocker_formation) 
        self._reset_round_state(current_score=self.score) 
        self.mainScreenActive = False 
        self.gameplayActive = True 
        self.gameOverActive = False

    def _reset_round_state(self, current_score):
        self.score = current_score 
        if self.player: self.player.kill()
        self.playerGroup.empty()
        if self.mysteryShip: self.mysteryShip.kill()
        self.mysteryGroup.empty()
        self.bullets.empty(); self.enemyBullets.empty(); self.explosionsGroup.empty()
        if self.enemies: self.enemies.empty() 
        
        self.player = Ship(); self.playerGroup.add(self.player)
        self.mysteryShip = Mystery(sound_manager=self.sounds); self.mysteryGroup.add(self.mysteryShip)
        self._make_enemies_formation() 
        self._update_lives_sprites() 

        self.allSprites.empty()
        self.allSprites.add(self.player)
        if self.enemies: self.allSprites.add(self.enemies.sprites())
        self.allSprites.add(self.mysteryShip)
        self.allSprites.add(self.allBlockers.sprites()) 
        self.allSprites.add(self.livesSpritesGroup.sprites())

        self.keys = pg.key.get_pressed() 
        self.general_timer = pg.time.get_ticks(); self.noteTimer = pg.time.get_ticks()     
        self.makeNewShipNext = False 
        self.shipCurrentlyAlive = (self.lives > 0)

    def _make_blocker_group(self, number):
        blocker_formation = pg.sprite.Group()
        # Simplified blocker creation for now - makes solid rectangles
        base_x = 75 + (175 * number) # Adjusted for potentially wider blockers
        for row in range(config.BLOCKER_ROWS):
            for col in range(config.BLOCKER_COLS_PER_PIECE * 2): # Example: 2 "columns" wide
                 # Original blocker shape logic was complex. This is a placeholder.
                 # A better way involves a template for blocker shapes.
                blocker_piece = Blocker(config.BLOCKER_PIECE_SIZE, config.GREEN, row, col)
                blocker_piece.rect.x = base_x + (col * blocker_piece.rect.width)
                blocker_piece.rect.y = config.BLOCKERS_POSITION + (row * blocker_piece.rect.height)
                blocker_formation.add(blocker_piece)
        return blocker_formation

    def _make_enemies_formation(self):
        self.enemies = EnemiesGroup(config.ENEMY_COLUMNS, config.ENEMY_ROWS, self.enemy_start_y)
        for r_idx in range(config.ENEMY_ROWS):
            for c_idx in range(config.ENEMY_COLUMNS):
                enemy = Enemy(r_idx, c_idx)
                enemy.rect.x = config.ENEMY_START_X_OFFSET + (c_idx * config.ENEMY_X_SPACING)
                enemy.rect.y = self.enemy_start_y + (r_idx * config.ENEMY_Y_SPACING)
                self.enemies.add(enemy)

    def _update_lives_sprites(self):
        self.livesSpritesGroup.empty()
        start_x_for_icons = self.livesLabelText.rect.right + 10 
        for i in range(self.lives):
            x_pos = start_x_for_icons + (i * (config.LIFE_SPRITE_WIDTH + 5)) 
            life_sprite = Life(x_pos, self.livesLabelText.rect.top + (self.livesLabelText.rect.height - config.LIFE_SPRITE_HEIGHT)//2 ) 
            self.livesSpritesGroup.add(life_sprite)
        if hasattr(self, 'allSprites'):
            for s in self.allSprites:
                if isinstance(s, Life): s.kill() 
            self.allSprites.add(self.livesSpritesGroup.sprites())

    def run_player_mode(self):
        self.mainScreenActive = True; self.gameplayActive = False; self.gameOverActive = False
        running = True
        while running:
            currentTime = pg.time.get_ticks()
            self.keys = pg.key.get_pressed() 
            for event in pg.event.get():
                if event.type == pg.QUIT: running = False
                if event.type == pg.KEYUP:
                    if event.key == pg.K_ESCAPE: running = False
                    if self.mainScreenActive: self._full_game_reset() 
                    elif self.gameOverActive: self.gameOverActive = False; self.mainScreenActive = True 
                if self.gameplayActive and event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                    self._handle_player_shooting()
            
            self.screen.blit(self.background, (0,0)) 
            if self.mainScreenActive: self._draw_main_menu_elements()
            elif self.gameplayActive:
                self._update_gameplay_state(currentTime) 
                self._draw_gameplay_elements(currentTime)  
            elif self.gameOverActive: self._draw_game_over_elements(currentTime)
            elif not self.mainScreenActive and not self.gameplayActive and not self.gameOverActive:
                self._draw_gameplay_elements(currentTime) 
            
            if not self.headless_worker_mode: pg.display.update()
            self.clock.tick(config.FPS) 
        if pg.get_init(): pg.quit()

    def _handle_player_shooting(self):
        if self.shipCurrentlyAlive and self.player and len(self.bullets) < config.MAX_PLAYER_BULLETS:
            bullet_x = self.player.rect.centerx - IMAGES['laser'].get_width() // 2 
            bullet_y = self.player.rect.top
            sound_to_play = self.sounds['shoot2'] if self.score >= 1000 and 'shoot2' in self.sounds else self.sounds['shoot']
            b = Bullet(bullet_x, bullet_y, -1, config.PLAYER_LASER_SPEED, 'laser', 'center')
            self.bullets.add(b); self.allSprites.add(b); sound_to_play.play()

    def _draw_main_menu_elements(self):
        self.titleText.draw(self.screen); self.titleText2.draw(self.screen)
        img_text_pairs = [
            (IMAGES['enemy3_1'], self.enemy1ScoreText), (IMAGES['enemy2_1'], self.enemy2ScoreText),
            (IMAGES['enemy1_1'], self.enemy3ScoreText), (IMAGES['mystery'], self.mysteryScoreText)
        ]
        for img_prototype, text_obj in img_text_pairs:
            img_scaled = pg.transform.scale(img_prototype, (40,35) if text_obj != self.mysteryScoreText else (70,35))
            img_x = text_obj.rect.centerx - (img_scaled.get_width() // 2) - 60 # Image left of text center
            text_obj.draw(self.screen)
            self.screen.blit(img_scaled, (img_x, text_obj.rect.top))


    def _update_gameplay_state(self, currentTime):
        screen_for_updates = self.screen if not self.headless_worker_mode else None

        if not self.enemies and not self.explosionsGroup: 
            self.gameplayActive = False; self.roundOverTimer = currentTime; return 

        if self.player: 
            self.playerGroup.update(self.keys if not self.ai_training_mode else None, currentTime, screen_for_updates)
        if self.enemies: self.enemies.update(currentTime) 

        # Pass screen_for_updates so sprites can blit themselves if not headless
        self.bullets.update(None, currentTime, screen_for_updates)
        self.enemyBullets.update(None, currentTime, screen_for_updates)
        self.mysteryGroup.update(None, currentTime, screen_for_updates) 
        self.explosionsGroup.update(None, currentTime, screen_for_updates) 
        if screen_for_updates: # Also update blockers if there's a screen
            self.allBlockers.update(None, currentTime, screen_for_updates)


        self._check_collisions_and_deaths()
        self._respawn_player_if_needed(currentTime)
        self._trigger_enemy_shooting(currentTime)
        if not self.ai_training_mode: self._play_background_music(currentTime)

        if self.enemies and self.enemies.bottom >= config.SCREEN_HEIGHT - config.PLAYER_AREA_HEIGHT:
            if self.shipCurrentlyAlive and self.player and pg.sprite.spritecollideany(self.player, self.enemies):
                self._player_death() 
            if self.enemies.bottom >= config.SCREEN_HEIGHT: self._player_death(final_death=True) 
                
    def _draw_gameplay_elements(self, currentTime):
        # Inter-round transition
        if not self.gameplayActive and not self.gameOverActive and not self.mainScreenActive:
            if currentTime - self.roundOverTimer < self.inter_round_delay:
                if not self.headless_worker_mode: self.nextRoundTextDisplay.draw(self.screen)
                return 
            else: 
                self.enemy_start_y = min(self.enemy_start_y + config.ENEMY_MOVE_DOWN_NEW_ROUND, 
                                         config.BLOCKERS_POSITION - (config.ENEMY_ROWS * config.ENEMY_Y_SPACING) - 20)
                max_y = config.BLOCKERS_POSITION - (config.ENEMY_ROWS * config.ENEMY_Y_SPACING) - 10
                if self.enemy_start_y >= max_y: self._player_death(final_death=True); return
                self._reset_round_state(self.score); self.gameplayActive = True 

        # Active gameplay drawing (only if not headless, or if AI rendering this step)
        if self.gameplayActive and (not self.headless_worker_mode or self._is_rendering_for_ai_this_step):
            self.scoreValueText = Text(config.GAME_FONT, 20, str(self.score), config.GREEN, 
                                       self.scoreLabelText.rect.right + 5, 5) 
            self.scoreLabelText.draw(self.screen)
            self.scoreValueText.draw(self.screen)
            self.livesLabelText.draw(self.screen)
            # self.livesSpritesGroup.draw(self.screen) # Drawn by allSprites
            
            # Sprites are blitted by their own update methods if screen_surface is provided.
            # If a strict draw order is needed, or if sprites don't blit in update:
            # self.allBlockers.draw(self.screen)
            # self.enemies.draw(self.screen)
            # self.playerGroup.draw(self.screen)
            # self.bullets.draw(self.screen)
            # self.enemyBullets.draw(self.screen)
            # self.mysteryGroup.draw(self.screen)
            # self.explosionsGroup.draw(self.screen)
            # self.livesSpritesGroup.draw(self.screen)
            # OR simply:
            self.allSprites.draw(self.screen) # If allSprites contains everything in correct layers


    def _play_background_music(self, currentTime):
        if self.silent_mode or not self.musicNotes or not self.enemies: return
        if self.enemies.moveTime > 0 and (currentTime - self.noteTimer > self.enemies.moveTime):
            self.musicNotes[self.noteIndex % len(self.musicNotes)].play()
            self.noteIndex += 1; self.noteTimer = currentTime

    def _trigger_enemy_shooting(self, currentTime):
        if self.enemies and (currentTime - self.general_timer) > config.ENEMY_SHOOT_INTERVAL:
            shooter = self.enemies.random_bottom_shooter()
            if shooter:
                b = Bullet(shooter.rect.centerx - IMAGES['enemylaser'].get_width()//2, shooter.rect.bottom, 
                           1, config.ENEMY_LASER_SPEED, 'enemylaser', 'center')
                self.enemyBullets.add(b); self.allSprites.add(b)
                self.general_timer = currentTime

    def _calculate_score_for_kill(self, sprite_obj):
        points = choice(config.MYSTERY_SCORES_OPTIONS) if isinstance(sprite_obj, Mystery) else \
                 config.ENEMY_SCORES_BY_ROW_INDEX.get(sprite_obj.row, 10) if isinstance(sprite_obj, Enemy) else 0
        self.score += points; return points

    def _check_collisions_and_deaths(self):
        pg.sprite.groupcollide(self.bullets, self.enemyBullets, True, True)
        if self.enemies:
            for e in pg.sprite.groupcollide(self.enemies, self.bullets, True, True).keys():
                self.sounds['invaderkilled'].play(); self._calculate_score_for_kill(e)
                EnemyExplosion(e, self.explosionsGroup, self.allSprites)
        if self.mysteryShip and self.mysteryShip.alive() and pg.sprite.spritecollide(self.mysteryShip, self.bullets, True):
            self.mysteryShip.mysteryEnteredSound.stop(); self.sounds['mysterykilled'].play()
            score = self._calculate_score_for_kill(self.mysteryShip)
            MysteryExplosion(self.mysteryShip, score, self.explosionsGroup, self.allSprites)
            self.mysteryShip.kill()
            self.mysteryShip = Mystery(self.sounds); self.mysteryGroup.add(self.mysteryShip); self.allSprites.add(self.mysteryShip)
        if self.shipCurrentlyAlive and self.player and pg.sprite.spritecollide(self.player, self.enemyBullets, True):
            self._player_death()
        pg.sprite.groupcollide(self.bullets, self.allBlockers, True, True)
        pg.sprite.groupcollide(self.enemyBullets, self.allBlockers, True, True)
        if self.enemies and self.allBlockers and self.enemies.bottom >= config.BLOCKERS_POSITION:
            pg.sprite.groupcollide(self.enemies, self.allBlockers, False, True)

    def _player_death(self, final_death=False):
        if not self.shipCurrentlyAlive and not final_death: return
        self.sounds['shipexplosion'].play()
        if self.player: ShipExplosion(self.player, self.explosionsGroup, self.allSprites); self.player.kill(); self.player = None
        self.shipCurrentlyAlive = False; self.lives -= 1; self._update_lives_sprites()
        if self.lives <= 0 or final_death:
            self.gameOverActive = True; self.gameplayActive = False; self.roundOverTimer = pg.time.get_ticks()
        else: self.makeNewShipNext = True; self.shipRespawnTimer = pg.time.get_ticks()

    def _respawn_player_if_needed(self, currentTime):
        if self.makeNewShipNext and (currentTime - self.shipRespawnTimer > self.player_respawn_delay):
            if self.lives > 0:
                self.player = Ship(); self.playerGroup.add(self.player); self.allSprites.add(self.player)
                self.shipCurrentlyAlive, self.makeNewShipNext = True, False
            elif not self.gameOverActive:
                self.gameOverActive = True; self.gameplayActive = False; self.roundOverTimer = pg.time.get_ticks()

    def _update_game_over_state(self, currentTime):
        if self.ai_training_mode: self.gameOverActive = False
        elif (currentTime - self.roundOverTimer > self.inter_round_delay):
            self.gameOverActive = False; self.mainScreenActive = True

    def _draw_game_over_elements(self, currentTime):
        if self.headless_worker_mode and not self._is_rendering_for_ai_this_step: return
        if ((currentTime - self.roundOverTimer) // config.GAME_OVER_TEXT_BLINK_INTERVAL_MS) % 2 == 0:
            self.gameOverTextDisplay.draw(self.screen)
        self.scoreValueText = Text(config.GAME_FONT,20,str(self.score),config.GREEN, self.scoreLabelText.rect.right + 5,5)
        self.scoreLabelText.draw(self.screen); self.scoreValueText.draw(self.screen)

    def reset_for_ai(self):
        self._full_game_reset(); return self._get_observation_for_ai()
    
    def set_render_for_ai_this_step(self, should_render: bool):
        if not self.headless_worker_mode: self._is_rendering_for_ai_this_step = should_render
        else: self._is_rendering_for_ai_this_step = False 
    
    def step_ai(self, action):
        current_time_step = pg.time.get_ticks()
        # ... (screen preparation and inter-round logic as before) ...
        if not self.gameplayActive and not self.gameOverActive:
            # ... (inter-round logic for AI) ...
            if not self.gameplayActive:
                obs = self._get_observation_for_ai()
                reward = 0 
                done = self.gameOverActive
                info = {'lives': self.lives, 'score': self.score, 'is_round_cleared': not self.enemies and not self.explosionsGroup and not done}
                # ... (display update if rendering) ...
                self._is_rendering_for_ai_this_step = False 
                return obs, reward, done, info
        
        prev_score = self.score
        prev_lives = self.lives
        player_shot_this_step = False # Flag to track if player shot

        # --- Apply AI Action & Update Game Logic ---
        if self.shipCurrentlyAlive and self.player:
            if action == config.ACTION_LEFT:
                self.player.rect.x = max(config.PLAYER_MIN_X, self.player.rect.x - self.player.speed)
            elif action == config.ACTION_RIGHT:
                self.player.rect.x = min(config.PLAYER_MAX_X - self.player.rect.width, self.player.rect.x + self.player.speed)
            elif action == config.ACTION_SHOOT:
                # Check if a bullet was actually fired (e.g., not max bullets on screen)
                num_bullets_before_shot = len(self.bullets)
                self._handle_player_shooting()
                if len(self.bullets) > num_bullets_before_shot:
                    player_shot_this_step = True
            # ACTION_NONE: do nothing explicitly

        # --- Store state needed for reward calculation BEFORE game state updates ---
        # For "shoot and miss", we need to know if a bullet hit something *this step*.
        # This is tricky because collisions are checked after bullet movement.
        # A simpler proxy: if player_shot_this_step and score did not increase from enemy kill.
        
        # Store current number of enemies for "enemy advance" penalty
        num_enemies_before_update = len(self.enemies.sprites()) if self.enemies else 0


        # --- Update Sprite Logic ---
        # ... (playerGroup.update, enemies.update, bullets.update, etc. as before) ...
        screen_for_sprite_updates = None
        if self.player: self.playerGroup.update(None, current_time_step, screen_for_sprite_updates)
        if self.enemies: self.enemies.update(current_time_step) 
        self.bullets.update(None, current_time_step, screen_for_sprite_updates)
        self.enemyBullets.update(None, current_time_step, screen_for_sprite_updates)
        self.mysteryGroup.update(None, current_time_step, screen_for_sprite_updates) 
        self.explosionsGroup.update(None, current_time_step, screen_for_sprite_updates) 
        self.allBlockers.update(None, current_time_step, screen_for_sprite_updates)
        
        # --- Collision Checks and Game State Changes ---
        # We need to know if a player bullet hit an enemy *this step* to avoid penalizing "shoot and miss" for successful shots.
        score_from_kills_this_step = 0
        
        # Temporarily store bullet-enemy collisions result
        # Note: This is a simplified way. A more robust method would track individual bullets.
        # For now, we check if score increased due to enemy kill.
        
        self._check_collisions_and_deaths() # This updates self.score
        score_from_kills_this_step = self.score - prev_score # score change due to kills (and potentially mystery ship)
                                                          # Excludes penalties/bonuses we add later.

        self._respawn_player_if_needed(current_time_step)
        self._trigger_enemy_shooting(current_time_step)
        # ... (enemy boundary checks and game over conditions as before) ...
        if self.enemies and self.enemies.bottom >= config.SCREEN_HEIGHT - config.PLAYER_AREA_HEIGHT:
            if self.shipCurrentlyAlive and self.player and pg.sprite.spritecollideany(self.player, self.enemies): self._player_death()
            if self.enemies.bottom >= config.SCREEN_HEIGHT and not self.gameOverActive: self._player_death(final_death=True)
        
        # --- Calculate Reward & Done Status ---
        reward = 0.0 # Start with a base reward for the step

        # 1. Reward for score increase (from kills this step)
        reward += score_from_kills_this_step # This is already positive from enemy/mystery points

        # 2. Penalty for losing a life
        if self.lives < prev_lives:
            reward += config.REWARD_LIFE_LOST

        # 3. Bonus for clearing a round
        round_cleared_this_step = False
        if not self.enemies and not self.explosionsGroup and not done and self.gameplayActive:
            round_cleared_this_step = True
            reward += config.REWARD_ROUND_CLEAR 
            self.gameplayActive = False; self.roundOverTimer = current_time_step

        # 4. Reward for surviving this step
        if not self.gameOverActive: # Only give survival reward if not game over
            reward += config.REWARD_PER_STEP_ALIVE

        # 5. Penalty for inaction (ACTION_NONE) if enemies are present
        if action == config.ACTION_NONE and self.enemies and len(self.enemies.sprites()) > 0:
            reward += config.PUNISHMENT_ACTION_NONE
        
        # 6. Penalty for shooting and missing
        # A simple proxy: if player shot and score_from_kills_this_step is 0 (or very small, e.g. <10 if only mystery ship gives small points)
        # This is imperfect as mystery ship destruction also increases score.
        # A better way: track if any player bullet specifically hit an enemy/mystery this step.
        # For now, simplified: if shot and score increase was 0 (or less than typical enemy kill)
        if player_shot_this_step and score_from_kills_this_step <= 0: # Assuming kills give >0 score
            # Check if bullets list is empty (meaning bullet hit something, even a blocker, or went off-screen)
            # This is still tricky. A more direct way would be for Bullet to set a flag if it hit an Enemy/Mystery.
            # For simplicity, let's assume for now if shot and no kill score, it's a miss penalty.
            # This might unfairly penalize hitting a blocker.
            reward += config.PUNISHMENT_SHOOT_MISS

        # 7. Penalty for enemies advancing / being present
        if self.enemies and len(self.enemies.sprites()) > 0:
            # Basic penalty for each enemy alive
            reward += config.PUNISHMENT_ENEMY_ADVANCE_BASE * len(self.enemies.sprites())
            
            # Additional penalty based on proximity of the lowest enemy row
            # self.enemies.bottom is the y-coordinate of the bottom-most edge of any enemy.
            # Closer to player (bottom of screen) means higher self.enemies.bottom value.
            # Danger zone starts, say, at config.BLOCKERS_POSITION
            if self.enemies.bottom > config.BLOCKERS_POSITION - 50: # Example threshold
                # Proximity factor: increases as enemies get lower.
                # Normalize this: (current_bottom - start_bottom) / (screen_height - start_bottom)
                # For simplicity, scale directly with how far down they are past a certain point.
                # Let's make the penalty larger the closer enemies.bottom is to SCREEN_HEIGHT.
                # A simple way: (self.enemies.bottom / config.SCREEN_HEIGHT) is a factor from ~0.1 to 1.0
                # proximity_factor = self.enemies.bottom / config.SCREEN_HEIGHT
                # More sensitive when close:
                distance_to_bottom_screen = config.SCREEN_HEIGHT - self.enemies.bottom
                # Avoid division by zero, ensure penalty increases as distance_to_bottom_screen decreases.
                # Let's use a simpler scaling:
                # Penalty increases linearly as enemies move from halfway down to the bottom.
                max_penalty_range_y_start = config.SCREEN_HEIGHT / 2 
                if self.enemies.bottom > max_penalty_range_y_start:
                    progress_into_danger = (self.enemies.bottom - max_penalty_range_y_start) / (config.SCREEN_HEIGHT - max_penalty_range_y_start)
                    reward += config.PUNISHMENT_ENEMY_PROXIMITY_SCALE * progress_into_danger
        
        done = self.gameOverActive

        # --- Drawing, Observation, Final Display Update ---
        # ... (Drawing logic, _get_observation_for_ai, display update, _is_rendering_for_ai_this_step reset as before) ...
        if not self.headless_worker_mode and self._is_rendering_for_ai_this_step:
            self.allSprites.draw(self.screen) 
        obs = self._get_observation_for_ai()
        info = {'lives':self.lives,'score':self.score,'is_round_cleared':round_cleared_this_step} # Use updated round_cleared_this_step
        if not self.headless_worker_mode:
            if self._is_rendering_for_ai_this_step:
                self.clock.tick(config.AI_TRAIN_RENDER_FPS)
                if pg.display.get_init() and pg.display.get_surface(): pg.display.update()
        self._is_rendering_for_ai_this_step = False
        
        return obs, reward, done, info


    def _get_observation_for_ai(self):
        is_display_rendering_step = (not self.headless_worker_mode and self._is_rendering_for_ai_this_step)

        if not is_display_rendering_step: # If NOT a display rendering step, draw to memory surface
            self.screen.blit(self.background, (0,0))
            # Draw UI for the observation frame
            if self.gameplayActive or self.gameOverActive: 
                self.scoreValueText = Text(config.GAME_FONT, 20, str(self.score), config.GREEN, 
                                           self.scoreLabelText.rect.right + 5, 5)
                self.scoreLabelText.draw(self.screen); self.scoreValueText.draw(self.screen)
                self.livesLabelText.draw(self.screen) 
                # livesSpritesGroup is in allSprites, so allSprites.draw() covers it.
                self.allSprites.draw(self.screen) # Draw all game sprites to the memory surface
            elif self.mainScreenActive: 
                self._draw_main_menu_elements()
            elif not self.gameplayActive and not self.gameOverActive and not self.mainScreenActive: # Inter-round
                 current_time_step_for_obs = pg.time.get_ticks() 
                 if current_time_step_for_obs - self.roundOverTimer < self.inter_round_delay:
                      self.nextRoundTextDisplay.draw(self.screen)
        # If it *was* a display_rendering_step, self.screen (the display) is assumed to be up-to-date from step_ai.
        try:
            return pg.surfarray.array3d(self.screen) 
        except pg.error: 
            return np.zeros((config.SCREEN_HEIGHT, config.SCREEN_WIDTH, 3), dtype=np.uint8)

    def get_action_size(self): return config.NUM_ACTIONS
    def close(self): pass