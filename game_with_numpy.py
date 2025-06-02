import pygame as pg
import numpy as np
from random import choice
from game import config
import time

from game.game_manager import Game

class NumpyGame:
    def __init__(self, render_mode='rgb_array'):
        self.screen = np.zeros((config.SCREEN_WIDTH, config.SCREEN_HEIGHT, 3), dtype=np.uint8)
        
        self.render_mode = render_mode
        if self.render_mode == 'human':
            pg.init()
            self.window = pg.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            pg.display.set_caption("Space Invaders")
        
        self.score = 0
        self.lives = config.PLAYER_LIVES
        self.lives_current = config.PLAYER_LIVES
        self.enemies_velocity = config.ENEMY_X_SHIFT_AMOUNT
        self.enemy_direction = 1
        self.player_position = [config.PLAYER_START_X, config.PLAYER_START_Y]
        self.enemies_positions = self.initialize_enemies()
        self.bullets_positions = []
        self.enemy_bullets_positions = []
        self.blockers_positions = self.initialize_blockers()
        self.previous_enemy_count = len(self.enemies_positions)
        self.wall_touches = 0  # Anzahl der Wandberührungen
    
    def initialize_enemies(self):
        return np.array([[config.ENEMY_START_X_OFFSET + col * config.ENEMY_X_SPACING,
                          config.ENEMY_DEFAULT_POSITION + row * config.ENEMY_Y_SPACING,
                          config.ENEMY_WIDTH, config.ENEMY_HEIGHT, row]
                         for row in range(config.ENEMY_ROWS) for col in range(config.ENEMY_COLUMNS)])

    def initialize_blockers(self):
        return [[75 + (175 * i) + (c * config.BLOCKER_PIECE_SIZE),
                config.BLOCKERS_POSITION + (r * config.BLOCKER_PIECE_SIZE)]
                for i in range(4)  # 4 Blocker formations
                for r in range(config.BLOCKER_ROWS)
                for c in range(config.BLOCKER_COLS_PER_PIECE * 2)]

    def reset_game_state(self):
        self.screen.fill(0)
        self.draw_player()
        self.draw_enemies()
        self.draw_blockers()
        self.draw_bullets()
        self.render()

    def draw_player(self):
        x, y = self.player_position
        # self.screen[y:y + config.PLAYER_HEIGHT, x:x + config.PLAYER_WIDTH] = config.BLUE
        self.screen[x:x + config.PLAYER_WIDTH, y:y + config.PLAYER_HEIGHT] = config.BLUE

    def draw_enemies(self):
        for x, y, w, h, _ in self.enemies_positions:
            # self.screen[y:y + h, x:x + w] = config.RED
            self.screen[x:x + w, y:y + h] = config.RED

    def draw_blockers(self):
        for x, y in self.blockers_positions:
            # self.screen[y:y + h, x:x + w] = config.GREEN
            self.screen[x:x + config.BLOCKER_PIECE_SIZE, y:y + config.BLOCKER_PIECE_SIZE] = config.GREEN
    
    def draw_bullets(self):
        for x, y in self.bullets_positions:
            # self.screen[y:y + 5, x:x + 2] = config.YELLOW
            self.screen[x:x + 2, y:y + 5] = config.YELLOW
        
        for x, y in self.enemy_bullets_positions:
            # self.screen[y:y + 5, x:x + 2] = config.RED
            self.screen[x:x + 2, y:y + 5] = config.RED

    def move_player(self, action):
        x, y = self.player_position
        if action == config.ACTION_LEFT:
            self.player_position[0] = max(config.PLAYER_MIN_X, x - config.PLAYER_SPEED)
        elif action == config.ACTION_RIGHT:
            self.player_position[0] = min(config.PLAYER_MAX_X - config.PLAYER_WIDTH, x + config.PLAYER_SPEED)
        elif action == config.ACTION_SHOOT:
            bullet_x = x + config.PLAYER_WIDTH // 2
            bullet_y = y
            self.bullets_positions.append([bullet_x, bullet_y])
        else:
            pass  # No action

    def update_bullets(self):
        new_bullets = []
        Punishment = 0
        for x, y in self.bullets_positions:
            y -= config.PLAYER_LASER_SPEED
            if y > 0:
                new_bullets.append([x, y])
            else:
                Punishment += config.PUNISHMENT_SHOOT_MISS  # Missed shot penalty
        self.bullets_positions = new_bullets

        new_enemy_bullets = []
        for x, y in self.enemy_bullets_positions:
            y += config.ENEMY_LASER_SPEED
            new_enemy_bullets.append([x, y])
        self.enemy_bullets_positions = new_enemy_bullets
        
        return Punishment
            

    def update_enemies(self):
        # Move enemies and change direction if necessary
        chance_to_shoot = 0.01  # Random chance to shoot
        touched_edge = False
        new_enemies = []

        # Überprüfe, ob ein Gegner eine Wand berührt hat
        for x, y, w, h, row in self.enemies_positions:
            # Check for edge contact
            if x + w >= config.SCREEN_WIDTH or x - w <= 0:
                touched_edge = True

            # Normal movement
            x += self.enemies_velocity * self.enemy_direction
            
            # Schießen mit Zufallswahrscheinlichkeit
            if np.random.rand() < chance_to_shoot:
                self.enemy_bullets_positions.append([x + w // 2, y + h])
            
            new_enemies.append([x, y, w, h, row])
        
        # Wenn ein Gegner die Bildschirmkante berührt hat, ändere die Richtung und bewege eine Stufe nach unten
        if touched_edge:
            if self.wall_touches % 5 == 0:
                self.enemy_direction *= -1  # Wechsel der Horizontalrichtung
                for i in range(len(new_enemies)):
                    new_enemies[i][1] += config.ENEMY_MOVE_DOWN_STEP  # Nur eine Stufe nach unten bewegen
            self.wall_touches += 1
        self.enemies_positions = new_enemies
    
    def check_collisions(self):
        score_increase = 0
        hit = False
        
        # Check player bullets with enemies
        remaining_enemies = []
        for ex, ey, ew, eh, row in self.enemies_positions:
            remember_hit = False
            for bx, by in self.bullets_positions:
                if (ex <= bx <= ex + ew) and (ey <= by <= ey + eh):
                    score_increase += config.ENEMY_SCORES_BY_ROW_INDEX[row]
                    self.bullets_positions.remove([bx, by])
                    hit, remember_hit = True, True
                    continue
                for bx2, by2 in self.blockers_positions:
                    if (bx >= bx2 and bx <= bx2 + config.BLOCKER_PIECE_SIZE) and (by >= by2 and by <= by2 + config.BLOCKER_PIECE_SIZE):
                        if [bx2, by2] in self.blockers_positions:
                            self.blockers_positions.remove([bx2, by2])
                            continue
                        self.bullets_positions.remove([bx, by])
            if not remember_hit:
                remaining_enemies.append([ex, ey, ew, eh, row])

        # Belohnung für das Töten eines Gegners
        if score_increase > 0:
            self.score += score_increase
        self.enemies_positions = remaining_enemies

        # Check enemy bullets with player
        px, py = self.player_position
        for bx, by in self.enemy_bullets_positions:
            if (px <= bx <= px + config.PLAYER_WIDTH) and (py <= by <= py + config.PLAYER_HEIGHT):
                self.lives -= 1
                self.enemy_bullets_positions.remove([bx, by])
                continue
            # Check if enemy bullet hit blockers
            for bx2, by2 in self.blockers_positions:
                if (bx >= bx2 and bx <= bx2 + config.BLOCKER_PIECE_SIZE) and (by >= by2 and by <= by2 + config.BLOCKER_PIECE_SIZE):
                    if [bx2, by2] in self.blockers_positions:
                        self.blockers_positions.remove([bx2, by2])
                        continue
                    self.enemy_bullets_positions.remove([bx, by])

        # Check if any enemies pushed past blockers and reach the player area
        for ex, ey, ew, eh, _ in self.enemies_positions:
            if ey + eh >= config.SCREEN_HEIGHT - config.PLAYER_AREA_HEIGHT:
                self.lives = 0
                score_increase += config.PUNISHMENT_ENEMY_REACHED_PLAYER_AREA
        

        return score_increase, hit

    def calculate_reward(self, score_increase, hit, is_under_enemy):
        reward = 0

        # Belohnung für das Töten eines Gegners
        reward += score_increase

        # # Belohnung für eine Backstellung unter den Feinden
        # if is_under_enemy:
        #     reward += config.REWARD_UNDER_ENEMY

        # # Bestrafung für das Verfehlen
        # if not hit and config.ACTION_SHOOT in self.bullets_positions:
        #     reward += config.PUNISHMENT_SHOOT_MISS

        # # Belohnung für einen vollständigen Rundenzug
        # if not self.enemies_positions:
        #     reward += config.REWARD_ROUND_CLEAR

        # Bestrafung für das Verlieren eines Lebens
        if self.lives < self.lives_current:
            self.lives_current = self.lives
            reward += config.REWARD_LIFE_LOST

        # Belohnung pro Schritt am Leben
        reward += config.REWARD_PER_STEP_ALIVE

        return reward

    def is_under_enemy(self):
        player_center_x = self.player_position[0] + config.PLAYER_WIDTH // 2
        for x, y, w, h, _ in self.enemies_positions:
            if abs(player_center_x - (x + w // 2)) < config.ALIGNMENT_TOLERANCE_X:
                return True
        return False

    def get_observation(self):
        return self.screen.copy()

    def step(self, action):
        self.move_player(action)
        self.update_enemies()
        Penalty = self.update_bullets()
        score_increase, hit = self.check_collisions()
        is_under_enemy = self.is_under_enemy()
        self.reset_game_state()  # Refresh the screen with updated positions
        
        observation = self.get_observation()
        done = not self.enemies_positions or self.lives <= 0
        reward = self.calculate_reward(score_increase, hit, is_under_enemy) + Penalty
        info = {'score': self.score, 'lives': self.lives}

        self.render()
        
        return observation, reward, done, info
    
    def render(self):
        if self.render_mode == 'human':
            pg.surfarray.blit_array(self.window, self.screen)
            pg.display.flip()
    
    def close(self):
        if self.render_mode == 'human':
            pg.quit()

# =====================================================================
# =====================================================================

class GeneticAlgorithm:
    def __init__(self, num_generations, population_size, mutation_rate, crossover_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.action_space = list(range(config.NUM_ACTIONS))
        self.observation_size = config.SCREEN_WIDTH * config.SCREEN_HEIGHT * 3  # RGB observation
        self.action_size = config.NUM_ACTIONS
        self.population = self.initialize_population()
        self.fitness_cache = {}

    def initialize_population(self):
        return [np.random.rand(self.action_size, self.observation_size) for _ in range(self.population_size)]

    def evaluate(self, individual, index):
        
        env = NumpyGame()
        
        # env = Game(
        #     silent_mode=True,
        #     ai_training_mode=True,
        #     headless_worker_mode=False,
        # )
        
        total_reward = 0
        observation = env.get_observation()
        # observation = env.reset_for_ai()
        
        for _ in range(30000):  # Adjust steps to make evaluation faster
            action = np.argmax(np.dot(individual, observation.flatten()))
            observation, reward, done, info = env.step(action)
            # observation, reward, done, info = env.step_ai(action)
            # env.set_render_for_ai_this_step(True)  # No rendering during evaluation
            total_reward += reward
            if done:
                break
        print(f'Reward Genom {index}: {total_reward}')

        return total_reward

    def mutate(self, individual):
        for i in range(individual.shape[0]):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.normal(0, 1)
        return individual

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(parent1.shape[0])
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def select_parents(self, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)[::-1]
        self.population = [self.population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        return self.population, fitness_scores

    def create_next_generation(self, fitness_scores):
        population, scores = self.select_parents(fitness_scores)
        new_population = population[:int(self.population_size * 0.1)]  # Elitismus
        while len(new_population) < self.population_size:
            parent_indices = np.random.choice(int(self.population_size * 0.2), 2, replace=False)
            parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def evolve(self):
        for generation in range(self.num_generations):
            # with mp.Pool(mp.cpu_count()) as pool:
            #     fitness_scores = pool.map(self.evaluate, self.population)
            
            fitness_scores = [self.evaluate(ind, i) for i, ind in enumerate(self.population)]
            
            print(f"Generation {generation}: Best Score = {max(fitness_scores)}")
            self.create_next_generation(fitness_scores)
        
        best_individual = self.population[0]
        np.save('best_strategy.npy', best_individual)

def run_best_strategy(n_times=10, render=True):
    # Geladene beste Strategie
    best_strategy = np.load('best_strategy.npy')

    environment = Game(
        silent_mode=True,
        ai_training_mode=False,
        headless_worker_mode=False,
    )

    for _ in range(n_times):
        observation = environment.reset_for_ai()
        
        # env = NumpyGame(render_mode='human')
        # observation = env.get_observation()
        
        total_reward = 0
        done = False
        
        for _ in range(2000000):
            
            # time.sleep(1/60)
            
            for event in pg.event.get():
                if event.type == pg.QUIT: break
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: break
            
            action = np.argmax(np.dot(best_strategy, observation.flatten()))
            
            environment.set_render_for_ai_this_step(render)
            
            observation, reward, done, info = environment.step_ai(action)
            # observation, reward, done, info = env.step(action)
            
            total_reward += reward
            if done:
                break
        
        print(f"Total Reward after completion of the game: {total_reward}")
    environment.close()
    # env.close()


# Testen der Logik
if __name__ == "__main__":
    
    ga = GeneticAlgorithm(num_generations=40,
                          population_size=25, 
                          mutation_rate=0.1, 
                          crossover_rate=0.7)
    ga.evolve()
    
    run_best_strategy()
