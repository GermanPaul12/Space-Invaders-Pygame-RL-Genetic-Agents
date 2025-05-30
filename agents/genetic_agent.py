# agents/genetic_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy
from .agent import Agent
from .dqn_agent import preprocess_observation # Reuse preprocessing

class GeneticNetwork(nn.Module):
    def __init__(self, input_channels, num_actions, h=84, w=84): # h,w are target preprocessed dimensions
        super(GeneticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2) # Simpler CNN
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        
        def conv_output_size(size, kernel_size, stride, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        conv_h = conv_output_size(conv_output_size(h, 5, 2), 3, 2)
        conv_w = conv_output_size(conv_output_size(w, 5, 2), 3, 2)
        flattened_size = conv_h * conv_w * 32
        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GeneticAgent(Agent):
    def __init__(self, action_size, observation_shape, # (C,H,W) e.g. (1,84,84)
                 population_size=50, mutation_rate=0.05, mutation_strength=0.1,
                 crossover_rate=0.7, num_elites=5):
        super().__init__(action_size, observation_shape)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Genetic Agent using device: {self.device}")

        self.input_channels = observation_shape[0]
        self.processed_h = observation_shape[1]
        self.processed_w = observation_shape[2]

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_elites = num_elites

        self.population = [
            GeneticNetwork(self.input_channels, action_size, h=self.processed_h, w=self.processed_w).to(self.device)
            for _ in range(population_size)
        ]
        self.fitness_scores = np.zeros(population_size)
        self.current_individual_idx = 0
        self.best_fitness_overall = -float('inf') # Track best fitness across generations
        self.best_individual_overall_state_dict = None # Store state_dict of the best individual overall

    def get_current_individual(self):
        return self.population[self.current_individual_idx]

    def choose_action(self, raw_observation):
        individual = self.get_current_individual()
        individual.eval()

        state_p_np = preprocess_observation(raw_observation, new_size=(self.processed_h, self.processed_w))
        state_tensor = torch.from_numpy(state_p_np).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = individual(state_tensor)
        action = q_values.max(1)[1].item()
        return action

    def record_fitness(self, score):
        self.fitness_scores[self.current_individual_idx] = score
        if score > self.best_fitness_overall:
            self.best_fitness_overall = score
        self.current_individual_idx += 1

    def learn(self):
        if not np.any(self.fitness_scores) and self.current_individual_idx < self.population_size : # Check if any scores are actually set
             print(f"GA: Waiting for all individuals to be evaluated. {self.current_individual_idx}/{self.population_size} done based on internal counter.")
             # This condition might not be hit if train.py manages current_individual_idx to pop_size before calling learn
             return

        # --- UPDATE BEST OVERALL FITNESS BASED ON CURRENT GENERATION ---
        current_gen_max_fitness = np.max(self.fitness_scores)
        if current_gen_max_fitness > self.best_fitness_overall:
            self.best_fitness_overall = current_gen_max_fitness
            best_idx_this_gen = np.argmax(self.fitness_scores)
            # Store the state_dict of the best individual overall for saving
            self.best_individual_overall_state_dict = deepcopy(self.population[best_idx_this_gen].state_dict())
        # --- END UPDATE ---

        print(f"GA Generation finished. Max fitness in gen: {current_gen_max_fitness:.2f}, Avg fitness: {np.mean(self.fitness_scores):.2f}, Best overall: {self.best_fitness_overall:.2f}")
        new_population = []
        # Elitism
        elite_indices = np.argsort(self.fitness_scores)[-self.num_elites:]
        for idx in elite_indices:
            elite_model = GeneticNetwork(self.input_channels, self.action_size, h=self.processed_h, w=self.processed_w).to(self.device)
            elite_model.load_state_dict(self.population[idx].state_dict())
            new_population.append(elite_model)

        # Crossover and Mutation
        num_offspring = self.population_size - self.num_elites
        for _ in range(num_offspring // 2): # Create two children per pair of parents
            if len(new_population) >= self.population_size: break
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            child1_params, child2_params = parent1.state_dict(), parent2.state_dict() # Default to no crossover
            if random.random() < self.crossover_rate:
                child1_params, child2_params = self._crossover(parent1.state_dict(), parent2.state_dict())

            child1 = GeneticNetwork(self.input_channels, self.action_size, h=self.processed_h, w=self.processed_w).to(self.device)
            child1.load_state_dict(self._mutate(child1_params))
            new_population.append(child1)

            if len(new_population) < self.population_size:
                child2 = GeneticNetwork(self.input_channels, self.action_size, h=self.processed_h, w=self.processed_w).to(self.device)
                child2.load_state_dict(self._mutate(child2_params))
                new_population.append(child2)
        
        # Fill remaining spots if population_size is odd or offspring count wasn't exact
        while len(new_population) < self.population_size:
            parent = self._tournament_selection() # Select one parent
            child = GeneticNetwork(self.input_channels, self.action_size, h=self.processed_h, w=self.processed_w).to(self.device)
            # Mutate a clone of the parent's parameters
            child.load_state_dict(self._mutate(deepcopy(parent.state_dict())))
            new_population.append(child)

        self.population = new_population
        self.fitness_scores = np.zeros(self.population_size) # Reset for next generation
        self.current_individual_idx = 0 # Reset for next generation's evaluations
        print("GA: New generation created.")


    def _tournament_selection(self, tournament_size=5):
        selected_indices = random.sample(range(self.population_size), tournament_size)
        best_idx_in_tournament = selected_indices[0] # Initialize with first
        best_fitness = -float('inf')
        for idx in selected_indices:
            if self.fitness_scores[idx] > best_fitness:
                best_fitness = self.fitness_scores[idx]
                best_idx_in_tournament = idx
        return self.population[best_idx_in_tournament]

    def _crossover(self, params1, params2):
        child1_params = {}
        child2_params = {}
        for name in params1.keys():
            p1 = params1[name]
            p2 = params2[name]
            if p1.ndim > 0 and p1.numel() > 1: # Crossover for non-scalar tensors with more than 1 element
                split_point = random.randint(1, p1.numel() - 1) # Ensure split_point is not 0 or numel
                
                p1_flat = p1.flatten()
                p2_flat = p2.flatten()

                c1_flat = torch.cat((p1_flat[:split_point], p2_flat[split_point:]))
                c2_flat = torch.cat((p2_flat[:split_point], p1_flat[split_point:]))
                
                child1_params[name] = c1_flat.view_as(p1)
                child2_params[name] = c2_flat.view_as(p2)
            else:
                child1_params[name] = p1.clone()
                child2_params[name] = p2.clone()
        return child1_params, child2_params

    def _mutate(self, params):
        mutated_params = {}
        for name, param_tensor in params.items():
            if param_tensor.ndim > 0:
                mutation_mask = (torch.rand_like(param_tensor) < self.mutation_rate).float().to(self.device)
                mutation = torch.randn_like(param_tensor).to(self.device) * self.mutation_strength
                mutated_params[name] = param_tensor + mutation_mask * mutation
            else:
                mutated_params[name] = param_tensor.clone()
        return mutated_params
    
    def save(self, path):
        # Save the best individual encountered so far across all generations
        if self.best_individual_overall_state_dict is not None:
            torch.save(self.best_individual_overall_state_dict, path)
            print(f"Best GA individual overall (fitness: {self.best_fitness_overall:.2f}) saved to {path}")
        elif self.population: # Fallback: if no overall best tracked, save best of current or first
            # This part may not be hit if best_individual_overall_state_dict is always maintained
            current_gen_best_idx = 0
            if np.any(self.fitness_scores): # If current gen has scores
                current_gen_best_idx = np.argmax(self.fitness_scores)
            best_individual_to_save = self.population[current_gen_best_idx]
            torch.save(best_individual_to_save.state_dict(), path)
            fitness_to_print = self.fitness_scores[current_gen_best_idx] if np.any(self.fitness_scores) else "N/A"
            print(f"GA: Saved current best/first individual (fitness: {fitness_to_print}) to {path}")
        else:
            print("GA: No individuals to save.")


    def load(self, path):
        # When loading, we are essentially starting a new evolutionary run seeded by this individual.
        # The loaded individual's fitness isn't directly transferred to best_fitness_overall yet.
        # best_fitness_overall will be updated as this new population is evaluated.
        try:
            loaded_state_dict = torch.load(path, map_location=self.device)
            self.population = []
            for _ in range(self.population_size):
                individual = GeneticNetwork(self.input_channels, self.action_size, h=self.processed_h, w=self.processed_w).to(self.device)
                individual.load_state_dict(loaded_state_dict)
                self.population.append(individual)
            
            self.current_individual_idx = 0
            self.fitness_scores = np.zeros(self.population_size)
            # Reset best_fitness_overall; it will be re-established by the new evaluations.
            self.best_fitness_overall = -float('inf') 
            self.best_individual_overall_state_dict = None # Also reset this
            print(f"GA population initialized with model from {path}. Best overall fitness will be re-evaluated.")
        except Exception as e:
            print(f"Error loading GA model from {path}: {e}. Initializing new random population.")
            # Re-initialize population if load fails
            self.population = [
                GeneticNetwork(self.input_channels, self.action_size, h=self.processed_h, w=self.processed_w).to(self.device)
                for _ in range(self.population_size)
            ]
            self.current_individual_idx = 0
            self.fitness_scores = np.zeros(self.population_size)
            self.best_fitness_overall = -float('inf')
            self.best_individual_overall_state_dict = None