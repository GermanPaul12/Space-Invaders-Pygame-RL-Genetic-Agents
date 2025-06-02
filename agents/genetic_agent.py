# agents/genetic_agent.py
import os
import time
import json
import random
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from copy import deepcopy

from .agent import Agent # Your base Agent class

# --- Preprocessing ---
def preprocess_observation_ga(obs, new_size=(84, 84)):
    if obs is None:
        return np.zeros((1, new_size[0], new_size[1]), dtype=np.float32)
    if isinstance(obs, tuple):
        obs = obs[0]

    img = Image.fromarray(obs)
    img = img.convert('L')
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Individual Neural Network ---
class IndividualNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(IndividualNetwork, self).__init__()
        self.channels, self.height, self.width = input_shape

        self.conv1 = nn.Conv2d(self.channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        def conv_output_size(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv_output_size(conv_output_size(self.width, 8, 4), 4, 2)
        convh = conv_output_size(conv_output_size(self.height, 8, 4), 4, 2)
        linear_input_size = convw * convh * 32

        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_weights_biases(self):
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_weights_biases(self, flat_params):
        offset = 0
        for param in self.parameters():
            param_shape = param.data.shape
            param_size = np.prod(param_shape)
            # Ensure flat_params has enough elements
            if offset + param_size > len(flat_params):
                print(f"Error: Not enough elements in flat_params to set weights for layer. Needed {param_size}, got {len(flat_params) - offset}")
                # This indicates a mismatch in network structure or saved weights
                # You might want to raise an error or handle this more gracefully
                return # Stop setting weights to prevent further errors
            param_values = flat_params[offset : offset + param_size].reshape(param_shape)
            param.data = torch.from_numpy(param_values).to(param.data.device)
            offset += param_size


class GeneticAgent(Agent):
    _default_ga_hparams = {
        "population_size": 30, # Reduced for faster example runs
        "mutation_rate": 0.15,
        "mutation_strength": 0.05,
        "crossover_rate": 0.7,
        "num_elites": 3,      # Adjusted
        "tournament_size": 5,
        "max_generations": 50, # Default if `episodes` in train is used as generations
        "max_steps_per_eval": 1500 # Default steps for evaluating one individual
    }
    DEFAULT_ENV_ID = "ALE/SpaceInvaders-v5"

    def __init__(self, env_id, hparams, mode, models_dir_for_agent, gifs_dir_for_agent):
        super().__init__(env_id if env_id else self.DEFAULT_ENV_ID,
                         hparams, mode, models_dir_for_agent, gifs_dir_for_agent)

        self.merged_hparams = self._default_ga_hparams.copy()
        self.merged_hparams.update(self.hparams)

        temp_env = gym.make(self.env_id)
        self.action_size = temp_env.action_space.n
        self.preprocessed_obs_shape = (1, 84, 84)
        temp_env.close()

        print(f"  Genetic Agent: Env '{self.env_id}', Action Size: {self.action_size}, Obs Shape: {self.preprocessed_obs_shape}")

        self.population = self._initialize_population()
        self.current_generation = 0
        self.best_fitness_overall = -float('inf')
        self.best_individual_overall_weights = None

        self.active_model = IndividualNetwork(self.preprocessed_obs_shape, self.action_size)
        if self.population: # Initialize active_model with first individual's weights
             self.active_model.set_weights_biases(deepcopy(self.population[0].get_weights_biases()))

        self.model_save_path = self.get_next_version_save_path("genetic") # Default save path


    def get_model_file_extension(self):
        return ".pth"

    def _initialize_population(self):
        population = []
        pop_size = self.merged_hparams['population_size']
        print(f"  Initializing population of {pop_size} individuals...")
        for _ in range(pop_size):
            individual = IndividualNetwork(self.preprocessed_obs_shape, self.action_size)
            population.append(individual)
        return population

    def _evaluate_individual(self, individual_network, eval_env, max_steps):
        individual_network.eval()
        obs, info = eval_env.reset()
        current_obs_p = preprocess_observation_ga(obs)
        total_reward = 0
        for _ in range(max_steps):
            with torch.no_grad():
                state_tensor = torch.from_numpy(current_obs_p).float().unsqueeze(0)
                action_logits = individual_network(state_tensor)
                action = torch.argmax(action_logits, dim=1).item()
            next_obs_raw, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            current_obs_p = preprocess_observation_ga(next_obs_raw)
            if terminated or truncated: break
        return total_reward

    def _select_parents_tournament(self, pop_with_fitness):
        parents = []
        for _ in range(2):
            tournament_contenders = random.sample(pop_with_fitness, self.merged_hparams['tournament_size'])
            winner = sorted(tournament_contenders, key=lambda x: x[1], reverse=True)[0]
            parents.append(winner[0])
        return parents[0], parents[1]

    def _crossover(self, parent1_net, parent2_net):
        child_net = IndividualNetwork(self.preprocessed_obs_shape, self.action_size)
        p1_weights = parent1_net.get_weights_biases()
        p2_weights = parent2_net.get_weights_biases()
        new_weights = (p1_weights + p2_weights) / 2.0
        child_net.set_weights_biases(new_weights)
        return child_net

    def _mutate(self, individual_net):
        weights = individual_net.get_weights_biases()
        mutated_weights = weights.copy()
        for i in range(len(weights)):
            if random.random() < self.merged_hparams['mutation_rate']:
                mutation = np.random.normal(0, self.merged_hparams['mutation_strength'])
                mutated_weights[i] += mutation
        individual_net.set_weights_biases(mutated_weights)
        return individual_net

    def train(self, episodes, max_steps_per_episode, render_mode_str,
              path_to_load_model, force_new_training_if_model_exists,
              save_interval_eps, print_interval_steps, **kwargs): # print_interval_steps is for gens here
        
        num_generations = episodes if episodes > 0 else self.merged_hparams['max_generations']
        steps_for_eval = max_steps_per_episode if max_steps_per_episode > 0 else self.merged_hparams['max_steps_per_eval']

        print(f"\n--- Genetic Algorithm Training Started ---")
        # ... (parameter printing as before)

        if path_to_load_model and os.path.exists(path_to_load_model) and not force_new_training_if_model_exists:
            print(f"  Loading GA state from: {path_to_load_model}")
            self.load(path_to_load_model)
            self.model_save_path = path_to_load_model # Continue saving to this path
        else:
            if force_new_training_if_model_exists and path_to_load_model and os.path.exists(path_to_load_model):
                print(f"  Force new training: existing model at {path_to_load_model} will be ignored for loading population.")
            elif path_to_load_model: print(f"  Load path '{path_to_load_model}' not found. Starting fresh.")
            # Determine a new save path if not loading or forcing new
            self.model_save_path = self.get_next_version_save_path("genetic")
        
        print(f"  Models will be saved to: {os.path.basename(self.model_save_path)}")

        eval_env = gym.make(self.env_id, render_mode=render_mode_str if render_mode_str=="human" else "rgb_array", full_action_space=False)
        eval_env = RecordEpisodeStatistics(eval_env)

        start_generation = self.current_generation
        for gen in range(start_generation, num_generations):
            self.current_generation = gen
            gen_time_start = time.time()
            print(f"\n--- Generation {gen + 1}/{num_generations} ---")

            population_with_fitness = []
            current_gen_fitness_scores = []
            print(f"  Evaluating {len(self.population)} individuals...")
            for i, individual_net in enumerate(self.population):
                fitness = self._evaluate_individual(individual_net, eval_env, steps_for_eval)
                population_with_fitness.append((individual_net, fitness))
                current_gen_fitness_scores.append(fitness)
            
            sorted_population_tuples = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
            
            best_fitness_this_gen = sorted_population_tuples[0][1] if sorted_population_tuples else -float('inf')
            avg_fitness_this_gen = np.mean(current_gen_fitness_scores) if current_gen_fitness_scores else -float('inf')
            
            print(f"  Generation {gen + 1} Results: Best Fitness: {best_fitness_this_gen:.2f}, Avg Fitness: {avg_fitness_this_gen:.2f}")

            if best_fitness_this_gen > self.best_fitness_overall:
                self.best_fitness_overall = best_fitness_this_gen
                # Save weights of the best individual network
                self.best_individual_overall_weights = deepcopy(sorted_population_tuples[0][0].get_weights_biases())
                print(f"  New Overall Best Fitness! {self.best_fitness_overall:.2f}")

            new_population = []
            num_elites = self.merged_hparams['num_elites']
            elites = [item[0] for item in sorted_population_tuples[:num_elites]] # Get the networks
            new_population.extend(deepcopy(elites)) # Store copies

            num_offspring_needed = self.merged_hparams['population_size'] - len(new_population)
            for _ in range(num_offspring_needed):
                parent1_net, parent2_net = self._select_parents_tournament(population_with_fitness)
                child_net = self._crossover(parent1_net, parent2_net) if random.random() < self.merged_hparams['crossover_rate'] else deepcopy(parent1_net)
                child_net = self._mutate(child_net)
                new_population.append(child_net)
            
            self.population = new_population
            gen_time_end = time.time()
            print(f"  Generation {gen+1} took {gen_time_end - gen_time_start:.2f} seconds.")

            if save_interval_eps > 0 and (gen + 1) % save_interval_eps == 0:
                print(f"  Saving model at generation {gen+1}...")
                self.save(self.model_save_path) # Overwrites the current file
        
        eval_env.close()
        print(f"\n--- Genetic Algorithm Training Finished ---")
        print(f"Overall Best Fitness Achieved: {self.best_fitness_overall:.2f}")
        self.save(self.model_save_path)


    def choose_action(self, observation_preprocessed, deterministic=True): # Deterministic is implicit for GA eval
        self.active_model.eval() # Ensure it's in eval mode
        with torch.no_grad():
            state_tensor = torch.from_numpy(observation_preprocessed).float().unsqueeze(0)
            action_logits = self.active_model(state_tensor)
            action = torch.argmax(action_logits, dim=1).item()
        return action

    def test(self, model_path_to_load, episodes, max_steps_per_episode,
             render_during_test, record_video_flag, video_fps, **kwargs):
        print(f"\n--- Genetic Algorithm Testing ---")
        # Load best model for testing
        if model_path_to_load and os.path.exists(model_path_to_load):
            print(f"  Loading GA state from: {model_path_to_load} for testing.")
            self.load(model_path_to_load) # This sets self.best_individual_overall_weights
            if self.best_individual_overall_weights is not None:
                self.active_model.set_weights_biases(self.best_individual_overall_weights)
            else:
                print("  Warning: Loaded file, but no 'best_individual_overall_weights' found. Testing with a potentially random/untrained model.")
        elif model_path_to_load:
            print(f"  Warning: Test model path '{model_path_to_load}' not found.")
            if self.best_individual_overall_weights is None and self.population: # Fallback to first in pop if no best overall and no load
                 self.active_model.set_weights_biases(self.population[0].get_weights_biases())
        else: # No model specified, use current best if available
            print("  No model path for testing. Using current best in memory if available.")
            if self.best_individual_overall_weights is not None:
                self.active_model.set_weights_biases(self.best_individual_overall_weights)
            elif self.population:
                self.active_model.set_weights_biases(self.population[0].get_weights_biases())


        test_env_render_mode = "human" if render_during_test else "rgb_array"
        test_env = gym.make(self.env_id, render_mode=test_env_render_mode, full_action_space=False)
        if record_video_flag:
            ts = time.strftime("%Y%m%d_%H%M%S")
            video_folder = os.path.join(self.gifs_dir if self.gifs_dir else "videos", f"genetic_test_{ts}")
            test_env = RecordVideo(test_env, video_folder=video_folder, name_prefix=f"ga_test",
                                   episode_trigger=lambda ep_id: True, fps=video_fps)
        test_env = RecordEpisodeStatistics(test_env)

        all_rewards = []
        for i in range(episodes):
            obs, info = test_env.reset()
            current_obs_p = preprocess_observation_ga(obs)
            episode_reward, current_steps = 0, 0
            for _ in range(max_steps_per_episode):
                action = self.choose_action(current_obs_p, deterministic=True)
                next_obs_raw, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward
                current_obs_p = preprocess_observation_ga(next_obs_raw)
                current_steps +=1
                if render_during_test and test_env_render_mode=="human" and not record_video_flag: test_env.render()
                if terminated or truncated: break
            actual_ep_reward = info.get('episode', {}).get('r', episode_reward)
            all_rewards.append(actual_ep_reward)
            print(f"  GA Test Episode {i+1}/{episodes} - Score: {actual_ep_reward:.0f}, Steps: {current_steps}")
        
        test_env.close()
        avg_r = np.mean(all_rewards) if all_rewards else 0
        std_r = np.std(all_rewards) if all_rewards else 0
        print(f"\n  GA Test Summary: Avg Score: {avg_r:.2f} +/- {std_r:.2f}")
        print(f"--- Genetic Algorithm Testing Ended ---")

    def evaluate(self, model_path_to_load, episodes, max_steps_per_episode, **kwargs):
        print(f"\n--- Genetic Algorithm Evaluation ---")
        if model_path_to_load and os.path.exists(model_path_to_load):
            self.load(model_path_to_load)
            if self.best_individual_overall_weights is not None:
                self.active_model.set_weights_biases(self.best_individual_overall_weights)
            else:
                print("  Warning: Loaded file for eval, but no 'best_individual_overall_weights'. Using first in population if exists.")
                if self.population: self.active_model.set_weights_biases(self.population[0].get_weights_biases())
        elif model_path_to_load:
             print(f"  Warning: Eval model path '{model_path_to_load}' not found. Using current best if any.")
             if self.best_individual_overall_weights is not None:
                self.active_model.set_weights_biases(self.best_individual_overall_weights)
             elif self.population: self.active_model.set_weights_biases(self.population[0].get_weights_biases())
        else: # No model path, evaluate current best
            print("  No model path for eval. Using current best in memory if available.")
            if self.best_individual_overall_weights is not None:
                self.active_model.set_weights_biases(self.best_individual_overall_weights)
            elif self.population: self.active_model.set_weights_biases(self.population[0].get_weights_biases())


        eval_env = gym.make(self.env_id, render_mode="rgb_array", full_action_space=False)
        eval_env = RecordEpisodeStatistics(eval_env)
        all_ep_rewards, all_ep_steps = [], []
        for i in range(episodes):
            obs, info = eval_env.reset()
            current_obs_p = preprocess_observation_ga(obs)
            ep_r, ep_s = 0, 0
            for _ in range(max_steps_per_episode):
                action = self.choose_action(current_obs_p, deterministic=True)
                next_obs_raw, reward, terminated, truncated, info = eval_env.step(action)
                ep_r += reward; ep_s += 1
                current_obs_p = preprocess_observation_ga(next_obs_raw)
                if terminated or truncated: break
            actual_ep_reward = info.get('episode', {}).get('r', ep_r)
            actual_ep_steps = info.get('episode', {}).get('l', ep_s)
            all_ep_rewards.append(actual_ep_reward)
            all_ep_steps.append(actual_ep_steps)
            if (i + 1) % max(1, episodes // 5) == 0:
                 print(f"    GA Eval Ep {i+1}/{episodes}: Score={actual_ep_reward:.0f}, Steps={actual_ep_steps}")
        eval_env.close()

        results = {}
        if all_ep_rewards:
            results = {
                "num_episodes_eval": episodes,
                "avg_score": round(np.mean(all_ep_rewards), 2),
                "std_dev_score": round(np.std(all_ep_rewards), 2),
                "min_score": round(np.min(all_ep_rewards), 2),
                "max_score": round(np.max(all_ep_rewards), 2),
                "avg_steps": round(np.mean(all_ep_steps), 1)
            }
        print(f"--- Genetic Algorithm Evaluation Ended ---")
        return results

    def save(self, path):
        if path is None:
            print("  GA Save Error: No path provided.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        population_weights = [ind.get_weights_biases() for ind in self.population]
        state = {
            'population_weights': population_weights,
            'best_individual_overall_weights': self.best_individual_overall_weights,
            'best_fitness_overall': self.best_fitness_overall,
            'current_generation': self.current_generation,
            'hparams': self.merged_hparams # Save hyperparameters used for this run
        }
        try:
            torch.save(state, path)
            print(f"  Genetic Agent state saved to {os.path.basename(path)}")
        except Exception as e:
            print(f"  Error saving Genetic Agent state to {path}: {e}")

    def load(self, path):
        if not os.path.exists(path):
            print(f"  GA Load Error: Path does not exist: {path}")
            return
        try:
            state = torch.load(path)
            loaded_hparams = state.get('hparams', self.merged_hparams) # Load hparams if saved
            
            # Re-initialize population with correct size from loaded hparams or current
            self.merged_hparams['population_size'] = loaded_hparams.get('population_size', self.merged_hparams['population_size'])
            self.population = []
            for weights_array in state['population_weights']:
                individual = IndividualNetwork(self.preprocessed_obs_shape, self.action_size)
                individual.set_weights_biases(weights_array)
                self.population.append(individual)
            
            self.best_individual_overall_weights = state.get('best_individual_overall_weights')
            self.best_fitness_overall = state.get('best_fitness_overall', -float('inf'))
            self.current_generation = state.get('current_generation', 0)
            
            # If best weights are loaded, set active_model
            if self.best_individual_overall_weights is not None:
                self.active_model.set_weights_biases(self.best_individual_overall_weights)
            elif self.population: # Fallback to first in loaded population
                self.active_model.set_weights_biases(self.population[0].get_weights_biases())

            print(f"  Genetic Agent state loaded from {os.path.basename(path)}. Resuming from generation {self.current_generation + 1}.")
        except Exception as e:
            print(f"  Error loading Genetic Agent state from {path}: {e}")
            # Fallback to re-initializing if load fails badly
            self.population = self._initialize_population()
            self.current_generation = 0
            self.best_fitness_overall = -float('inf')
            self.best_individual_overall_weights = None