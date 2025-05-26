# agents/genetic_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
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
        if self.current_individual_idx < self.population_size:
            print(f"GA: Waiting for all individuals to be evaluated. {self.current_individual_idx}/{self.population_size} done.")
            return

        print(f"GA Generation finished. Max fitness in gen: {np.max(self.fitness_scores)}, Avg fitness: {np.mean(self.fitness_scores)}, Best overall: {self.best_fitness_overall}")
        
        new_population = []
        
        elite_indices = np.argsort(self.fitness_scores)[-self.num_elites:]
        for idx in elite_indices:
            # Create a new network and load state_dict for elites to avoid issues with shared references if not careful
            elite_model = GeneticNetwork(self.input_channels, self.action_size, h=self.processed_h, w=self.processed_w).to(self.device)
            elite_model.load_state_dict(self.population[idx].state_dict())
            new_population.append(elite_model)


        num_offspring = self.population_size - self.num_elites
        for _ in range(num_offspring // 2):
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            if random.random() < self.crossover_rate:
                child1_params, child2_params = self._crossover(parent1.state_dict(), parent2.state_dict())
            else:
                child1_params, child2_params = parent1.state_dict(), parent2.state_dict()

            child1 = GeneticNetwork(self.input_channels, self.action_size, h=self.processed_h, w=self.processed_w).to(self.device)
            child1.load_state_dict(self._mutate(child1_params))
            new_population.append(child1)

            if len(new_population) < self.population_size:
                child2 = GeneticNetwork(self.input_channels, self.action_size, h=self.processed_h, w=self.processed_w).to(self.device)
                child2.load_state_dict(self._mutate(child2_params))
                new_population.append(child2)
        
        while len(new_population) < self.population_size: # Fill remaining spots if population_size is odd
            parent = self._tournament_selection()
            child = GeneticNetwork(self.input_channels, self.action_size, h=self.processed_h, w=self.processed_w).to(self.device)
            child.load_state_dict(self._mutate(parent.state_dict()))
            new_population.append(child)

        self.population = new_population
        self.fitness_scores = np.zeros(self.population_size)
        self.current_individual_idx = 0
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
        if np.any(self.fitness_scores) or self.current_individual_idx > 0 : # If any fitness recorded for current gen
            best_idx_current_gen = np.argmax(self.fitness_scores[:self.current_individual_idx]) if self.current_individual_idx > 0 else 0
            # Check if this best is truly the overall best if saving mid-generation for some reason
            # Typically save is called after a full generation, where np.argmax(self.fitness_scores) is fine
            best_individual = self.population[best_idx_current_gen]
            torch.save(best_individual.state_dict(), path)
            print(f"Best GA individual (fitness: {self.fitness_scores[best_idx_current_gen]:.2f}) saved to {path}")
        elif self.population: # If no scores, but population exists (e.g. before first eval), save first one
            torch.save(self.population[0].state_dict(), path)
            print(f"GA: Saved first individual (no fitness scores yet) to {path}")
        else:
            print("GA: No individuals to save.")


    def load(self, path):
        loaded_state_dict = torch.load(path, map_location=self.device)
        self.population = []
        for _ in range(self.population_size):
            individual = GeneticNetwork(self.input_channels, self.action_size, h=self.processed_h, w=self.processed_w).to(self.device)
            individual.load_state_dict(loaded_state_dict)
            self.population.append(individual)
        self.current_individual_idx = 0
        self.fitness_scores = np.zeros(self.population_size)
        self.best_fitness_overall = -float('inf') # Reset best overall as we loaded a specific model
        print(f"GA population initialized with model from {path}")