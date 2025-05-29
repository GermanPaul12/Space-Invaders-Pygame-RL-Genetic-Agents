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
                # Attempt to set mode for headless to allow image conversions
                self.screen = pg.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            except pg.error: # Fallback if display mode fails (e.g., no X server)
                self.screen = pg.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        
        load_all_game_images() 

        if not self.silent_mode:
            if not pg.mixer.get_init():
                try: 
                    pg.mixer.pre_init(44100, -16, 2, 512) 
                    pg.mixer.init()
                except pg.error: 
                    self.silent_mode = True # Force silent if mixer fails
        else: 
            if not pg.mixer.get_init(): # Minimal init if explicitly silent
                 try: pg.mixer.init(frequency=22050, size=-16, channels=2, buffer=512) 
                 except: pass 

        self.clock = pg.time.Clock()
        try: 
            bg_path = os.path.join(config.IMAGE_PATH, 'background.jpg')
            # Convert only if there's a display and not headless (performance for player mode)
            if pg.display.get_init() and pg.display.get_surface() and not self.headless_worker_mode: 
                self.background = pg.image.load(bg_path).convert()
            else: # Load plain for headless or if convert fails
                self.background = pg.image.load(bg_path)
        except Exception as e_bg_load: 
            print(f"ERROR: Failed to load background image '{bg_path}': {e_bg_load}", flush=True)
            self.background = pg.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            self.background.fill(config.BLACK) # Fallback to black

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
        self.allSprites = pg.sprite.Group() # Master group for drawing

        self.general_timer = pg.time.get_ticks() 
        self.noteTimer = pg.time.get_ticks()     
        self.shipRespawnTimer = pg.time.get_ticks()
        self.roundOverTimer = pg.time.get_ticks()

        self.makeNewShipNext = False 
        self.shipCurrentlyAlive = True
        self._is_rendering_for_ai_this_step = False # Flag for AI rendering

    def _make_static_text_objects(self):
        self.titleText = Text(config.GAME_FONT, 50, 'Space Invaders', config.WHITE, 
                              config.SCREEN_WIDTH // 2, 155, center_x=True)
        self.titleText2 = Text(config.GAME_FONT, 25, 'Press any key to continue', config.WHITE,
                               config.SCREEN_WIDTH // 2, 225, center_x=True)
        self.gameOverTextDisplay = Text(config.GAME_FONT, 50, 'Game Over', config.WHITE,
                                        config.SCREEN_WIDTH // 2, 270, center_x=True)
        self.nextRoundTextDisplay = Text(config.GAME_FONT, 50, 'Next Round', config.WHITE,
                                         config.SCREEN_WIDTH // 2, 270, center_x=True)
        self.scoreLabelText = Text(config.GAME_FONT, 20, 'Score', config.WHITE, 5, 5)

        text_lives_width = Text(config.GAME_FONT, 20, 'Lives', config.WHITE, 0,0).rect.width
        estimated_icons_total_width = (config.PLAYER_LIVES * config.LIFE_SPRITE_WIDTH) + ((config.PLAYER_LIVES -1) * 5 if config.PLAYER_LIVES > 0 else 0)
        lives_label_x = config.SCREEN_WIDTH - estimated_icons_total_width - text_lives_width - 15 # Padding from right edge
        self.livesLabelText = Text(config.GAME_FONT, 20, 'Lives', config.WHITE, lives_label_x, 5)

        text_x_pos_anchor = config.SCREEN_WIDTH // 2 
        self.enemy1ScoreText = Text(config.GAME_FONT, 25, '   =   30 pts', config.PURPLE, text_x_pos_anchor, 270, center_x=True)
        self.enemy2ScoreText = Text(config.GAME_FONT, 25, '   =  20 pts', config.BLUE, text_x_pos_anchor, 320, center_x=True)
        self.enemy3ScoreText = Text(config.GAME_FONT, 25, '   =  10 pts', config.GREEN, text_x_pos_anchor, 370, center_x=True)
        self.mysteryScoreText = Text(config.GAME_FONT, 25, '   =  ?????', config.RED, text_x_pos_anchor, 420, center_x=True)
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
        self.score = start_score; self.lives = start_lives
        self.enemy_start_y = config.ENEMY_DEFAULT_POSITION
        self.allBlockers.empty() 
        for i in range(4): self.allBlockers.add(self._make_blocker_group(i))
        self._reset_round_state(self.score) 
        self.mainScreenActive = False; self.gameplayActive = True; self.gameOverActive = False

    def _reset_round_state(self, current_score):
        self.score = current_score 
        for group in [self.playerGroup, self.mysteryGroup, self.bullets, self.enemyBullets, self.explosionsGroup]: group.empty()
        if self.player: self.player.kill()
        if self.mysteryShip: self.mysteryShip.kill()
        if self.enemies: self.enemies.empty()
        
        self.player = Ship(); self.playerGroup.add(self.player)
        self.mysteryShip = Mystery(self.sounds); self.mysteryGroup.add(self.mysteryShip)
        self._make_enemies_formation() 
        self._update_lives_sprites() 

        self.allSprites.empty()
        self.allSprites.add(self.player, self.mysteryShip)
        if self.enemies: self.allSprites.add(self.enemies.sprites())
        self.allSprites.add(self.allBlockers.sprites(), self.livesSpritesGroup.sprites())

        self.keys = pg.key.get_pressed() 
        self.general_timer = pg.time.get_ticks(); self.noteTimer = pg.time.get_ticks()     
        self.makeNewShipNext = False; self.shipCurrentlyAlive = (self.lives > 0)

    def _make_blocker_group(self, number): # number is 0-3
        formation = pg.sprite.Group()
        blocker_base_x = 75 + (175 * number) 
        for r in range(config.BLOCKER_ROWS):
            for c in range(config.BLOCKER_COLS_PER_PIECE * 2): # Blocker width
                # This is a simplified solid blocker. Original game has shaped blockers.
                b = Blocker(config.BLOCKER_PIECE_SIZE, config.GREEN, r, c)
                b.rect.x = blocker_base_x + (c * b.rect.width)
                b.rect.y = config.BLOCKERS_POSITION + (r * b.rect.height)
                formation.add(b)
        return formation

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
        icons_start_x = self.livesLabelText.rect.right + 10
        for i in range(self.lives):
            x = icons_start_x + (i * (config.LIFE_SPRITE_WIDTH + 5))
            y_center_offset = (self.livesLabelText.rect.height - config.LIFE_SPRITE_HEIGHT) // 2
            life = Life(x, self.livesLabelText.rect.top + y_center_offset)
            self.livesSpritesGroup.add(life)
        
        # Ensure allSprites reflects the change if already populated
        if hasattr(self, 'allSprites') and self.allSprites:
             for s in self.allSprites:
                if isinstance(s, Life): s.kill() 
             self.allSprites.add(self.livesSpritesGroup.sprites())


    def _draw_main_game_frame(self, currentTime):
        """Helper to draw a complete active gameplay frame."""
        self.screen.blit(self.background, (0,0))
        self.scoreValueText = Text(config.GAME_FONT, 20, str(self.score), config.GREEN, 
                                   self.scoreLabelText.rect.right + 5, 5) 
        self.scoreLabelText.draw(self.screen); self.scoreValueText.draw(self.screen)
        self.livesLabelText.draw(self.screen)
        # Lives icons are in allSprites, so they are drawn by allSprites.draw()
        self.allSprites.draw(self.screen) # Draws all sprites in their current state

    def run_player_mode(self):
        self.mainScreenActive = True; self.gameplayActive = False; self.gameOverActive = False
        running = True
        while running:
            currentTime = pg.time.get_ticks()
            self.keys = pg.key.get_pressed() 
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYUP and event.key == pg.K_ESCAPE): running = False
                if event.type == pg.KEYUP:
                    if self.mainScreenActive: self._full_game_reset() 
                    elif self.gameOverActive: self.gameOverActive = False; self.mainScreenActive = True 
                if self.gameplayActive and event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                    self._handle_player_shooting()
            
            # --- Player Mode Drawing Logic ---
            if self.mainScreenActive:
                self.screen.blit(self.background, (0,0)) 
                self._draw_main_menu_elements()
            elif self.gameplayActive:
                self._update_gameplay_state_player(currentTime) # Use player-specific update
                self._draw_main_game_frame(currentTime)
            elif self.gameOverActive:
                self.screen.blit(self.background, (0,0)) 
                self._draw_game_over_elements(currentTime)
            elif not self.mainScreenActive and not self.gameplayActive and not self.gameOverActive: # Inter-round
                self.screen.blit(self.background, (0,0)) 
                if currentTime - self.roundOverTimer < self.inter_round_delay:
                    self.nextRoundTextDisplay.draw(self.screen)
                else: 
                    self.enemy_start_y = min(self.enemy_start_y + config.ENEMY_MOVE_DOWN_NEW_ROUND, 
                                             config.BLOCKERS_POSITION - (config.ENEMY_ROWS * config.ENEMY_Y_SPACING) - 20)
                    max_y = config.BLOCKERS_POSITION - (config.ENEMY_ROWS * config.ENEMY_Y_SPACING) - 10
                    if self.enemy_start_y >= max_y: self._player_death(final_death=True) 
                    else: self._reset_round_state(self.score); self.gameplayActive = True
            
            if not self.headless_worker_mode: pg.display.update()
            self.clock.tick(config.FPS) 
        if pg.get_init(): pg.quit()

    def _handle_player_shooting(self):
        if self.shipCurrentlyAlive and self.player and len(self.bullets) < config.MAX_PLAYER_BULLETS:
            bullet_x = self.player.rect.centerx - IMAGES['laser'].get_width() // 2 
            bullet_y = self.player.rect.top
            sound = self.sounds['shoot2'] if self.score >= 1000 and 'shoot2' in self.sounds else self.sounds['shoot']
            b = Bullet(bullet_x, bullet_y, -1, config.PLAYER_LASER_SPEED, 'laser', 'center')
            self.bullets.add(b); self.allSprites.add(b); sound.play()

    def _draw_main_menu_elements(self):
        self.titleText.draw(self.screen); self.titleText2.draw(self.screen)
        img_text_pairs = [
            (IMAGES['enemy3_1'], self.enemy1ScoreText), (IMAGES['enemy2_1'], self.enemy2ScoreText),
            (IMAGES['enemy1_1'], self.enemy3ScoreText), (IMAGES['mystery'], self.mysteryScoreText)
        ]
        default_size = (40,35); mystery_size = (70,35); img_offset = -75 # Image left of text center
        for img_proto, text_obj in img_text_pairs:
            size = mystery_size if text_obj == self.mysteryScoreText else default_size
            img_s = pg.transform.scale(img_proto, size)
            img_x_pos = text_obj.rect.centerx + img_offset - (img_s.get_width()//2) 
            text_obj.draw(self.screen)
            self.screen.blit(img_s, (img_x_pos, text_obj.rect.top + (text_obj.rect.height - img_s.get_height())//2))


    def _update_gameplay_state_player(self, currentTime): # Renamed for clarity
        """Handles game logic updates specifically for player mode."""
        screen_for_updates = None # Sprites update logic, don't blit

        if not self.gameplayActive: return # Should be handled by main loop

        if not self.enemies and not self.explosionsGroup: 
            self.gameplayActive = False; self.roundOverTimer = currentTime; return 

        if self.player: self.playerGroup.update(self.keys, currentTime, screen_for_updates) # Player uses self.keys
        if self.enemies: self.enemies.update(currentTime) 

        self.bullets.update(None, currentTime, screen_for_updates)
        self.enemyBullets.update(None, currentTime, screen_for_updates)
        self.mysteryGroup.update(None, currentTime, screen_for_updates) 
        self.explosionsGroup.update(None, currentTime, screen_for_updates) 
        self.allBlockers.update(None, currentTime, screen_for_updates)

        self._check_collisions_and_deaths()
        self._respawn_player_if_needed(currentTime)
        self._trigger_enemy_shooting(currentTime)
        self._play_background_music(currentTime) # Music only for player mode

        if self.enemies and self.enemies.bottom >= config.SCREEN_HEIGHT - config.PLAYER_AREA_HEIGHT:
            if self.shipCurrentlyAlive and self.player and pg.sprite.spritecollideany(self.player, self.enemies):
                self._player_death() 
            if self.enemies.bottom >= config.SCREEN_HEIGHT: self._player_death(final_death=True) 
                
    def _draw_game_over_elements(self, currentTime):
        if self.headless_worker_mode and not self._is_rendering_for_ai_this_step: return
        if ((currentTime - self.roundOverTimer) // config.GAME_OVER_TEXT_BLINK_INTERVAL_MS) % 2 == 0:
            self.gameOverTextDisplay.draw(self.screen)
        self.scoreValueText = Text(config.GAME_FONT,20,str(self.score),config.GREEN, self.scoreLabelText.rect.right + 5,5)
        self.scoreLabelText.draw(self.screen); self.scoreValueText.draw(self.screen)
        # Lives icons are part of allSprites and typically not re-drawn on game over screen unless explicitly added

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
            elif not self.gameOverActive: # Should be covered by _player_death but defensive
                self.gameOverActive = True; self.gameplayActive = False; self.roundOverTimer = pg.time.get_ticks()

    def reset_for_ai(self):
        self._full_game_reset(); return self._get_observation_for_ai()
    
    def set_render_for_ai_this_step(self, should_render: bool):
        if not self.headless_worker_mode: self._is_rendering_for_ai_this_step = should_render
        else: self._is_rendering_for_ai_this_step = False 
    
    def step_ai(self, action):
        current_time_step = pg.time.get_ticks()
        screen_for_sprite_updates = None # Sprites update logic, don't blit in their update

        # --- AI Inter-Round Logic (before primary game step logic) ---
        if not self.gameplayActive and not self.gameOverActive:
            # Initial screen clear if rendering this inter-round/game-over step
            if not self.headless_worker_mode and self._is_rendering_for_ai_this_step:
                self.screen.blit(self.background, (0,0))
                if current_time_step - self.roundOverTimer < self.inter_round_delay:
                    self.nextRoundTextDisplay.draw(self.screen)
            
            # Logic to transition to next round or game over
            if current_time_step - self.roundOverTimer >= self.inter_round_delay:
                self.enemy_start_y = min(self.enemy_start_y + config.ENEMY_MOVE_DOWN_NEW_ROUND, 
                                         config.BLOCKERS_POSITION - (config.ENEMY_ROWS*config.ENEMY_Y_SPACING)-20)
                max_y = config.BLOCKERS_POSITION - (config.ENEMY_ROWS*config.ENEMY_Y_SPACING)-10 # Max allowable start y
                if self.enemy_start_y >= max_y: 
                    self.gameOverActive = True 
                else: 
                    self._reset_round_state(self.score)
                    self.gameplayActive = True
            
            if not self.gameplayActive: # Still not active (waiting for delay or became game over)
                # For observation, ensure the state is drawn if it's a non-display-render step
                obs = self._get_observation_for_ai() 
                reward = 0.0
                done = self.gameOverActive
                info = {'lives':self.lives,'score':self.score,'is_round_cleared': not self.enemies and not self.explosionsGroup and not done}
                
                # If rendering this inter-round/game-over step, ensure display is updated
                if not self.headless_worker_mode and self._is_rendering_for_ai_this_step:
                    if self.gameOverActive: # If became game over this step, draw game over screen
                        # Screen might have been cleared above, or clear again if needed
                        self.screen.blit(self.background, (0,0)) 
                        self._draw_game_over_elements(current_time_step)
                    # If it was just "Next Round" text, it was drawn above.
                    if pg.display.get_init() and pg.display.get_surface(): pg.display.update()
                
                self._is_rendering_for_ai_this_step = False 
                return obs, reward, done, info
        
        prev_score = self.score
        prev_lives = self.lives 
        player_shot_this_step = False

        # --- Apply AI Action & Update Game Logic ---
        if self.shipCurrentlyAlive and self.player: # Use shipCurrentlyAlive
            if action == config.ACTION_LEFT: 
                self.player.rect.x = max(config.PLAYER_MIN_X, self.player.rect.x - self.player.speed)
            elif action == config.ACTION_RIGHT: 
                self.player.rect.x = min(config.PLAYER_MAX_X - self.player.rect.width, self.player.rect.x + self.player.speed)
            elif action == config.ACTION_SHOOT:
                num_bullets_before_shot = len(self.bullets)
                self._handle_player_shooting()
                if len(self.bullets) > num_bullets_before_shot:
                    player_shot_this_step = True
        
        # Update sprite logic (no drawing directly in these update calls)
        if self.player: self.playerGroup.update(None, current_time_step, screen_for_sprite_updates)
        if self.enemies: self.enemies.update(current_time_step) 
        self.bullets.update(None, current_time_step, screen_for_sprite_updates)
        self.enemyBullets.update(None, current_time_step, screen_for_sprite_updates)
        self.mysteryGroup.update(None, current_time_step, screen_for_sprite_updates) 
        self.explosionsGroup.update(None, current_time_step, screen_for_sprite_updates) 
        self.allBlockers.update(None, current_time_step, screen_for_sprite_updates)
        
        score_before_collisions = self.score 
        self._check_collisions_and_deaths() 
        score_increase_from_kills = self.score - score_before_collisions

        self._respawn_player_if_needed(current_time_step) # This method correctly uses/updates self.shipCurrentlyAlive
        self._trigger_enemy_shooting(current_time_step)
        
        if self.enemies and self.enemies.bottom >= config.SCREEN_HEIGHT - config.PLAYER_AREA_HEIGHT:
            if self.shipCurrentlyAlive and self.player and pg.sprite.spritecollideany(self.player, self.enemies): # Use shipCurrentlyAlive
                self._player_death()
            if self.enemies.bottom >= config.SCREEN_HEIGHT and not self.gameOverActive: 
                self._player_death(final_death=True)
        
        # --- Calculate Reward & Done Status ---
        reward = 0.0 
        reward += score_increase_from_kills 
        if self.lives < prev_lives: 
            reward += config.REWARD_LIFE_LOST

        round_cleared_this_step = False
        if not self.enemies and not self.explosionsGroup and not self.gameOverActive and self.gameplayActive:
            round_cleared_this_step = True
            reward += config.REWARD_ROUND_CLEAR 
            self.gameplayActive = False; self.roundOverTimer = current_time_step

        if not self.gameOverActive: 
            reward += config.REWARD_PER_STEP_ALIVE

        if action == config.ACTION_NONE and self.enemies and len(self.enemies.sprites()) > 0:
            reward += config.PUNISHMENT_ACTION_NONE
        
        if player_shot_this_step and score_increase_from_kills <= 0 : 
            reward += config.PUNISHMENT_SHOOT_MISS

        # --- REWARD FOR BEING UNDER ENEMY ---
        is_under_enemy = False
        if self.shipCurrentlyAlive and self.player and self.enemies and len(self.enemies.sprites()) > 0: # Use shipCurrentlyAlive
            player_center_x = self.player.rect.centerx
            for enemy in self.enemies.sprites(): 
                if abs(player_center_x - enemy.rect.centerx) < config.ALIGNMENT_TOLERANCE_X \
                   and self.player.rect.top > enemy.rect.bottom: 
                    is_under_enemy = True
                    break 
            if is_under_enemy:
                reward += config.REWARD_UNDER_ENEMY
        # --- END REWARD FOR BEING UNDER ENEMY ---

        if self.enemies and len(self.enemies.sprites()) > 0:
            reward += config.PUNISHMENT_ENEMY_ADVANCE_BASE * len(self.enemies.sprites())
            if hasattr(self.enemies, 'bottom'):
                max_penalty_range_y_start = config.SCREEN_HEIGHT / 2 
                if self.enemies.bottom > max_penalty_range_y_start:
                    denominator = (config.SCREEN_HEIGHT - max_penalty_range_y_start)
                    if denominator <= 0: denominator = 1 
                    progress_into_danger = (self.enemies.bottom - max_penalty_range_y_start) / denominator
                    reward += config.PUNISHMENT_ENEMY_PROXIMITY_SCALE * max(0, min(1, progress_into_danger)) 
        
        done = self.gameOverActive

        # ... (Drawing logic, _get_observation_for_ai, display update, _is_rendering_for_ai_this_step reset as before) ...
        if not self.headless_worker_mode and self._is_rendering_for_ai_this_step:
            # Ensure screen is cleared and UI drawn before sprites
            self.screen.blit(self.background, (0,0)) 
            self.scoreValueText = Text(config.GAME_FONT, 20, str(self.score), config.GREEN, 
                                       self.scoreLabelText.rect.right + 5, 5) 
            self.scoreLabelText.draw(self.screen); self.scoreValueText.draw(self.screen)
            self.livesLabelText.draw(self.screen)
            # livesSpritesGroup is part of allSprites
            self.allSprites.draw(self.screen) 

        obs = self._get_observation_for_ai() 
        info = {'lives':self.lives,'score':self.score,'is_round_cleared':round_cleared_this_step}
        
        if not self.headless_worker_mode and self._is_rendering_for_ai_this_step:
            self.clock.tick(config.AI_TRAIN_RENDER_FPS)
            if pg.display.get_init() and pg.display.get_surface(): 
                pg.display.update() 
        
        self._is_rendering_for_ai_this_step = False 
        
        return obs, reward, done, info

    def _get_observation_for_ai(self):
        # If this call is part of a display rendering step, self.screen is already up-to-date.
        # Otherwise (headless, or AI step without display render), draw the current state.
        if not (not self.headless_worker_mode and self._is_rendering_for_ai_this_step):
            self.screen.blit(self.background, (0,0))
            current_time_for_obs = pg.time.get_ticks() # For potential text drawing
            if self.gameplayActive: 
                self._draw_main_game_frame(current_time_for_obs) 
            elif self.gameOverActive:
                self._draw_game_over_elements(current_time_for_obs)
            elif self.mainScreenActive: 
                self._draw_main_menu_elements()
            elif not self.gameplayActive and not self.mainScreenActive and not self.gameOverActive: # Inter-round
                 if current_time_for_obs - self.roundOverTimer < self.inter_round_delay:
                      self.nextRoundTextDisplay.draw(self.screen)
        try:
            return pg.surfarray.array3d(self.screen) 
        except pg.error: 
            return np.zeros((config.SCREEN_HEIGHT, config.SCREEN_WIDTH, 3), dtype=np.uint8)

    def get_action_size(self): return config.NUM_ACTIONS
    def close(self): 
        if pg.get_init(): pg.quit() # Ensure Pygame quits if game instance is closed explicitly