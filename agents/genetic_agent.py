# agents/genetic_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .agent import Agent
from .dqn_agent import preprocess_observation # Reuse preprocessing

# Simple Neural Network for Genetic Algorithm
class GeneticNetwork(nn.Module):
    def __init__(self, input_channels, num_actions, h=84, w=84):
        super(GeneticNetwork, self).__init__()
        # Using a simpler CNN than DQN for GA to speed up evaluation
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2)
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
        return self.fc2(x) # Output Q-values or action logits

class GeneticAgent(Agent):
    def __init__(self, action_size, observation_shape, # (C,H,W)
                 population_size=50, mutation_rate=0.05, mutation_strength=0.1,
                 crossover_rate=0.7, num_elites=5):
        super().__init__(action_size, observation_shape)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Genetic Agent using device: {self.device}")

        self.input_channels = observation_shape[0] if observation_shape else 1
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_elites = num_elites

        self.population = [GeneticNetwork(self.input_channels, action_size).to(self.device)
                           for _ in range(population_size)]
        self.fitness_scores = np.zeros(population_size)
        self.current_individual_idx = 0 # To track which individual is currently playing

    def get_current_individual(self):
        return self.population[self.current_individual_idx]

    def choose_action(self, observation): # observation is raw
        individual = self.get_current_individual()
        individual.eval() # Set to evaluation mode

        state_p = preprocess_observation(observation)
        state_tensor = torch.from_numpy(state_p).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = individual(state_tensor)
        action = q_values.max(1)[1].item() # Choose action with max Q-value
        return action

    def record_fitness(self, score):
        self.fitness_scores[self.current_individual_idx] = score
        self.current_individual_idx += 1

    def learn(self): # This will be called after evaluating the whole population
        if self.current_individual_idx < self.population_size:
            # Not all individuals evaluated yet for this generation
            print(f"Waiting for all individuals to be evaluated. {self.current_individual_idx}/{self.population_size} done.")
            return # Return or raise error, depends on how main loop calls it

        print(f"Generation finished. Max fitness: {np.max(self.fitness_scores)}, Avg fitness: {np.mean(self.fitness_scores)}")
        
        new_population = []
        
        # Elitism: carry over the best individuals
        elite_indices = np.argsort(self.fitness_scores)[-self.num_elites:]
        for idx in elite_indices:
            new_population.append(self.population[idx]) # Keep existing elite models

        # Selection, Crossover, Mutation
        # Using tournament selection for simplicity
        num_offspring = self.population_size - self.num_elites
        for _ in range(num_offspring // 2): # Create two offspring per crossover
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            if random.random() < self.crossover_rate:
                child1_params, child2_params = self._crossover(parent1.state_dict(), parent2.state_dict())
            else:
                child1_params, child2_params = parent1.state_dict(), parent2.state_dict() # No crossover

            child1 = GeneticNetwork(self.input_channels, self.action_size).to(self.device)
            child1.load_state_dict(self._mutate(child1_params))
            new_population.append(child1)

            if len(new_population) < self.population_size:
                child2 = GeneticNetwork(self.input_channels, self.action_size).to(self.device)
                child2.load_state_dict(self._mutate(child2_params))
                new_population.append(child2)
        
        # If odd number of offspring needed, add one more mutated elite or random
        while len(new_population) < self.population_size:
            parent = self._tournament_selection()
            child = GeneticNetwork(self.input_channels, self.action_size).to(self.device)
            child.load_state_dict(self._mutate(parent.state_dict()))
            new_population.append(child)


        self.population = new_population
        self.fitness_scores = np.zeros(self.population_size) # Reset fitness for new generation
        self.current_individual_idx = 0 # Reset for next generation's evaluation
        print("New generation created.")


    def _tournament_selection(self, tournament_size=5):
        selected_indices = random.sample(range(self.population_size), tournament_size)
        best_idx_in_tournament = -1
        best_fitness = -float('inf')
        for idx in selected_indices:
            if self.fitness_scores[idx] > best_fitness:
                best_fitness = self.fitness_scores[idx]
                best_idx_in_tournament = idx
        return self.population[best_idx_in_tournament]

    def _crossover(self, params1, params2):
        # Single point crossover for each parameter tensor for simplicity
        child1_params = {}
        child2_params = {}
        for name in params1.keys():
            p1 = params1[name]
            p2 = params2[name]
            if p1.ndim > 0: # Crossover for weights/biases, not scalars
                split_point = random.randint(0, p1.numel() -1) # numel is total elements
                
                p1_flat = p1.flatten()
                p2_flat = p2.flatten()

                c1_flat = torch.cat((p1_flat[:split_point], p2_flat[split_point:]))
                c2_flat = torch.cat((p2_flat[:split_point], p1_flat[split_point:]))
                
                child1_params[name] = c1_flat.view_as(p1)
                child2_params[name] = c2_flat.view_as(p2)
            else: # Scalar parameters (if any)
                child1_params[name] = p1.clone()
                child2_params[name] = p2.clone()
        return child1_params, child2_params

    def _mutate(self, params):
        mutated_params = {}
        for name, param_tensor in params.items():
            if param_tensor.ndim > 0: # Mutate weights/biases
                mutation_mask = (torch.rand_like(param_tensor) < self.mutation_rate).float()
                mutation = torch.randn_like(param_tensor) * self.mutation_strength
                mutated_params[name] = param_tensor + mutation_mask * mutation
            else: # Scalar parameters
                mutated_params[name] = param_tensor.clone()
        return mutated_params
    
    def save(self, path): # Saves the best individual of the current population
        if np.any(self.fitness_scores): # If any fitness recorded
            best_idx = np.argmax(self.fitness_scores)
            best_individual = self.population[best_idx]
            torch.save(best_individual.state_dict(), path)
            print(f"Best Genetic Algorithm individual saved to {path}")
        else:
            print("No fitness scores recorded yet, cannot save best GA individual.")


    def load(self, path): # Loads a single individual and makes it the whole population (for testing)
        loaded_state_dict = torch.load(path, map_location=self.device)
        self.population = []
        for _ in range(self.population_size):
            individual = GeneticNetwork(self.input_channels, self.action_size).to(self.device)
            individual.load_state_dict(loaded_state_dict)
            self.population.append(individual)
        self.current_individual_idx = 0
        self.fitness_scores = np.zeros(self.population_size)
        print(f"Genetic Algorithm population initialized with model from {path}")