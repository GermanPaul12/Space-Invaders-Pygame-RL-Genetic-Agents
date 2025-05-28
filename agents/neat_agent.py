# agents/neat_agent.py
import numpy as np
import random
import math
from collections import defaultdict
from copy import deepcopy
import pickle # Moved import here

from .agent import Agent
from .dqn_agent import preprocess_observation

# --- Remove module-level NEAT constants or keep them only as comments for default values ---
# Default values can now be handled in the __init__ method or in the default JSON config.

# --- Innovation Tracking (keep as is, or make it an instance member of NEATAgent) ---
# For simplicity, we'll keep _global_innovation_tracker as module level for now,
# but it's reset in NEATAgent.__init__.
class GlobalInnovation:
    def __init__(self):
        self.innovation_number = 0
        # NEAT_INPUT_SIZE_FLAT and NEAT_OUTPUT_SIZE will be passed to NEATAgent
        # and then used to initialize node_innovation_count
        self.node_innovation_count = 0 # Will be set based on input/output size
        self.node_innovations = {}
        self.connection_innovations = {}

    def reset(self, num_input_nodes, num_output_nodes):
        self.innovation_number = 0
        self.node_innovation_count = num_input_nodes + num_output_nodes
        self.node_innovations = {}
        self.connection_innovations = {}

    def get_connection_innov(self, in_node_id, out_node_id):
        key = (in_node_id, out_node_id)
        if key not in self.connection_innovations:
            self.connection_innovations[key] = self.innovation_number
            self.innovation_number += 1
        return self.connection_innovations[key]

    def get_node_innov(self, connection_gene_to_split):
        key = (connection_gene_to_split.in_node, connection_gene_to_split.out_node)
        if key not in self.node_innovations:
            self.node_innovations[key] = self.node_innovation_count
            self.node_innovation_count += 1
        return self.node_innovations[key]

_global_innovation_tracker = GlobalInnovation()


# --- Gene and Genome Definitions ---
class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled=True, innovation_tracker=None):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        tracker = innovation_tracker if innovation_tracker else _global_innovation_tracker
        self.innovation = tracker.get_connection_innov(in_node, out_node)

    def clone(self):
        # Cloning does not re-assign innovation number
        cloned_gene = ConnectionGene(self.in_node, self.out_node, self.weight, self.enabled)
        cloned_gene.innovation = self.innovation # Preserve original innovation
        return cloned_gene

class NodeGene:
    def __init__(self, id, type="hidden", activation_func_name="tanh"):
        self.id = id
        self.type = type
        self.value = 0.0
        self.inputs_received = []
        
        if activation_func_name == "tanh": self.activation_func = np.tanh
        elif activation_func_name == "sigmoid": self.activation_func = lambda x: 1 / (1 + np.exp(-x))
        elif activation_func_name == "relu": self.activation_func = lambda x: np.maximum(0,x)
        else: self.activation_func = np.tanh # Default

    def activate(self):
        if self.type == "input": pass
        else: self.value = self.activation_func(sum(self.inputs_received))
        self.inputs_received = []
        return self.value

class GenomeNEAT:
    def __init__(self, num_inputs, num_outputs, agent_ref=None): # agent_ref to access config and innovation
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.agent_ref = agent_ref # Reference to the NEATAgent instance for params
        
        self.connection_genes = {}
        self.node_genes = {}
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.species_id = None

        innov_tracker = self.agent_ref.innovation_tracker if self.agent_ref else _global_innovation_tracker

        for i in range(self.num_inputs):
            self.node_genes[i] = NodeGene(i, type="input")
        for i in range(self.num_inputs, self.num_inputs + self.num_outputs):
            self.node_genes[i] = NodeGene(i, type="output", activation_func_name=self.agent_ref.output_activation if self.agent_ref else "tanh")
        
        # Initial minimal connections (or fully connected based on a config param)
        if self.agent_ref and self.agent_ref.initial_connection_type == "full":
            for i_node_id in range(self.num_inputs):
                for o_node_id in range(self.num_inputs, self.num_inputs + self.num_outputs):
                    weight = np.random.uniform(-1, 1)
                    gene = ConnectionGene(i_node_id, o_node_id, weight, innovation_tracker=innov_tracker)
                    self.connection_genes[gene.innovation] = gene
        # else: start with no connections, mutations will add them (more typical NEAT start)


    def feed_forward(self, inputs_flat):
        # ... (Feedforward logic as before, ensure it uses self.num_inputs) ...
        if len(inputs_flat) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs_flat)}")
        for node_id, node in self.node_genes.items():
            if node.type != "input": node.value = 0.0; node.inputs_received = []
        for i in range(self.num_inputs): self.node_genes[i].value = inputs_flat[i]
        node_eval_order = sorted(self.node_genes.keys())
        for _ in range(self.agent_ref.ff_passes if self.agent_ref else 3): 
            for node_id in node_eval_order:
                node = self.node_genes[node_id]
                if node.type == "input": continue
                node.activate()
                for gene in self.connection_genes.values(): # Iterate through values
                    if gene.enabled and gene.in_node == node_id:
                        self.node_genes[gene.out_node].inputs_received.append(gene.weight * node.value)
        outputs = []
        for i in range(self.num_inputs, self.num_inputs + self.num_outputs):
            # Ensure output nodes are activated if they received inputs in the last pass
            if self.node_genes[i].inputs_received : self.node_genes[i].activate()
            outputs.append(self.node_genes[i].value)
        return outputs


    def mutate(self):
        innov_tracker = self.agent_ref.innovation_tracker if self.agent_ref else _global_innovation_tracker
        # Use self.agent_ref.mutation_rate, etc.
        for gene in self.connection_genes.values():
            if np.random.rand() < self.agent_ref.weight_mutate_rate:
                if np.random.rand() < self.agent_ref.weight_perturb_uniform_rate:
                    gene.weight += np.random.uniform(-0.1, 0.1) # Use a config for perturb strength
                else:
                    gene.weight = np.random.uniform(-2, 2) # Use config for weight range
            if gene.enabled and np.random.rand() < self.agent_ref.disable_mutate_rate: gene.enabled = False
            elif not gene.enabled and np.random.rand() < self.agent_ref.enable_mutate_rate: gene.enabled = True

        if np.random.rand() < self.agent_ref.add_connection_mutate_rate:
            self._mutate_add_connection(innov_tracker)
        if np.random.rand() < self.agent_ref.add_node_mutate_rate:
            self._mutate_add_node(innov_tracker)
            
    def _mutate_add_connection(self, innovation_tracker):
        # ... (logic as before, but use innovation_tracker passed in) ...
        # Ensure new_gene uses the passed innovation_tracker
        # gene = ConnectionGene(..., innovation_tracker=innovation_tracker)
        possible_in_nodes = [nid for nid, n in self.node_genes.items() if n.type != "output"]
        possible_out_nodes = [nid for nid, n in self.node_genes.items() if n.type != "input"]
        if not possible_in_nodes or not possible_out_nodes: return
        for _ in range(10):
            in_node_id = random.choice(possible_in_nodes)
            out_node_id = random.choice(possible_out_nodes)
            if in_node_id == out_node_id: continue
            if self.node_genes[in_node_id].type == "output" or self.node_genes[out_node_id].type == "input": continue
            connection_exists = any(
                (g.in_node == in_node_id and g.out_node == out_node_id) or \
                (g.in_node == out_node_id and g.out_node == in_node_id and self.agent_ref.allow_recurrent_connections_on_mutate) # Example for config
                for g in self.connection_genes.values()
            )
            if not connection_exists:
                weight = np.random.uniform(-1, 1) # Configurable weight range
                new_gene = ConnectionGene(in_node_id, out_node_id, weight, innovation_tracker=innovation_tracker)
                # Check if innovation already exists (shouldn't if tracker is global and consistent)
                # but if it does due to complex history, this gene might be a duplicate structurally
                if new_gene.innovation not in self.connection_genes:
                     self.connection_genes[new_gene.innovation] = new_gene
                     break

    def _mutate_add_node(self, innovation_tracker):
        # ... (logic as before, but use innovation_tracker for new node ID and new ConnectionGenes) ...
        if not self.connection_genes: return
        enabled_genes = [g for g in self.connection_genes.values() if g.enabled]
        if not enabled_genes: return
        gene_to_split = random.choice(enabled_genes)
        gene_to_split.enabled = False
        new_node_id = innovation_tracker.get_node_innov(gene_to_split)
        if new_node_id not in self.node_genes:
             self.node_genes[new_node_id] = NodeGene(new_node_id, type="hidden", activation_func_name=self.agent_ref.hidden_activation if self.agent_ref else "tanh")
        else: gene_to_split.enabled = True; return # ID collision, abort
        gene1 = ConnectionGene(gene_to_split.in_node, new_node_id, 1.0, innovation_tracker=innovation_tracker)
        gene2 = ConnectionGene(new_node_id, gene_to_split.out_node, gene_to_split.weight, innovation_tracker=innovation_tracker)
        self.connection_genes[gene1.innovation] = gene1
        self.connection_genes[gene2.innovation] = gene2

    def clone(self):
        cloned_genome = GenomeNEAT(self.num_inputs, self.num_outputs, self.agent_ref)
        cloned_genome.node_genes = {}
        for nid, node_obj in self.node_genes.items():
            cloned_genome.node_genes[nid] = NodeGene(node_obj.id, node_obj.type, 
                                                     activation_func_name=self.agent_ref.get_node_activation_name(node_obj.id) if self.agent_ref else "tanh") # Need a way to get func name

        cloned_genome.connection_genes = {}
        for innov, gene in self.connection_genes.items():
            cloned_gene = gene.clone() # Clone already preserves innovation
            cloned_genome.connection_genes[innov] = cloned_gene
            
        cloned_genome.fitness = 0.0; cloned_genome.adjusted_fitness = 0.0; cloned_genome.species_id = None
        return cloned_genome


def crossover_neat(parent1, parent2, agent_ref): # agent_ref for num_inputs/outputs
    if parent2.fitness > parent1.fitness: parent1, parent2 = parent2, parent1
    child = GenomeNEAT(parent1.num_inputs, parent1.num_outputs, agent_ref)
    child.connection_genes = {}
    for innov1, gene1 in parent1.connection_genes.items():
        gene2 = parent2.connection_genes.get(innov1)
        child_gene = (random.choice([gene1, gene2]) if gene2 else gene1).clone()
        if gene2 and (not gene1.enabled or not gene2.enabled) and np.random.rand() < 0.75: # Chance to inherit disabled
            child_gene.enabled = False
        child.connection_genes[child_gene.innovation] = child_gene
    
    child.node_genes = {} # Rebuild node genes from connections
    all_node_ids_in_connections = set()
    for gene in child.connection_genes.values():
        all_node_ids_in_connections.add(gene.in_node)
        all_node_ids_in_connections.add(gene.out_node)
    
    # Add back explicit input/output nodes if not covered by connections (should always be)
    for i in range(parent1.num_inputs): all_node_ids_in_connections.add(i)
    for i in range(parent1.num_inputs, parent1.num_inputs + parent1.num_outputs): all_node_ids_in_connections.add(i)

    for node_id in all_node_ids_in_connections:
        # Determine type and activation from parents or defaults
        p1_node = parent1.node_genes.get(node_id)
        p2_node = parent2.node_genes.get(node_id)
        
        if node_id < parent1.num_inputs: node_type = "input"
        elif node_id < parent1.num_inputs + parent1.num_outputs: node_type = "output"
        else: node_type = "hidden"
        
        # Activation: inherit from parent1 if possible, else parent2, else default
        activation_name = "tanh" # Default
        if p1_node: activation_name = agent_ref.get_node_activation_name(p1_node.id) if agent_ref else "tanh" # Requires helper
        elif p2_node: activation_name = agent_ref.get_node_activation_name(p2_node.id) if agent_ref else "tanh"
            
        child.node_genes[node_id] = NodeGene(node_id, type=node_type, activation_func_name=activation_name)
    return child


def compatibility_distance(genome1, genome2, agent_ref): # agent_ref for coefficients
    # ... (logic as before, but use agent_ref.excess_coef, etc.) ...
    innov1 = set(genome1.connection_genes.keys()); innov2 = set(genome2.connection_genes.keys())
    max_innov_g1 = max(innov1) if innov1 else -1; max_innov_g2 = max(innov2) if innov2 else -1
    disjoint, excess, matching_innovs = 0,0,[]
    all_innovs = innov1 | innov2
    for innov in sorted(list(all_innovs)): # Iterate through all unique innovations in order
        g1_has = innov in innov1
        g2_has = innov in innov2
        if g1_has and g2_has: matching_innovs.append(innov)
        elif g1_has and innov > max_innov_g2: excess +=1 # Excess in g1
        elif g2_has and innov > max_innov_g1: excess +=1 # Excess in g2
        else: disjoint +=1 # Disjoint if one has it and it's not excess for the other
    
    weight_diff_sum = sum(abs(genome1.connection_genes[i].weight - genome2.connection_genes[i].weight) for i in matching_innovs)
    avg_w_diff = (weight_diff_sum / len(matching_innovs)) if matching_innovs else 0
    N = max(len(genome1.connection_genes), len(genome2.connection_genes)); N = 1 if N == 0 else N
    return ( (agent_ref.excess_coef * excess / N) + 
             (agent_ref.disjoint_coef * disjoint / N) + 
             (agent_ref.weight_diff_coef * avg_w_diff) )


class SpeciesNEAT:
    def __init__(self, representative_genome, species_id, agent_ref):
        self.id = species_id
        self.agent_ref = agent_ref # For accessing compatibility_threshold, etc.
        self.representative = representative_genome.clone()
        self.members = [representative_genome]
        self.fitness_history = [] # Could store (generation, max_fitness)
        self.generations_since_improvement = 0
        self.max_fitness_achieved = -float('inf')
        representative_genome.species_id = self.id


    def add_member(self, genome): # ... (as before, sets genome.species_id) ...
        self.members.append(genome)
        genome.species_id = self.id

    def calculate_shared_fitness(self): # ... (as before) ...
        for member in self.members:
            member.adjusted_fitness = member.fitness / len(self.members) if len(self.members) > 0 else 0.0

    def get_average_fitness(self): # ... (as before) ...
        if not self.members: return 0.0
        return sum(m.fitness for m in self.members) / len(self.members)
    
    def get_champion(self): # ... (as before) ...
        if not self.members: return None
        return max(self.members, key=lambda m: m.fitness)

    def sort_members_by_fitness(self): self.members.sort(key=lambda m: m.fitness, reverse=True)

    def check_stagnation(self):
        # ... (logic as before, uses self.agent_ref.stagnation_threshold) ...
        current_max_fitness = self.get_champion().fitness if self.get_champion() else -float('inf')
        if current_max_fitness > self.max_fitness_achieved :
            self.max_fitness_achieved = current_max_fitness
            self.generations_since_improvement = 0
        else: self.generations_since_improvement +=1
        return self.generations_since_improvement > self.agent_ref.stagnation_threshold


class NEATAgent(Agent):
    def __init__(self, action_size, observation_shape, **kwargs): # Use kwargs for config
        super().__init__(action_size, observation_shape)
        
        # --- Set NEAT parameters from kwargs or use defaults ---
        self.population_size = kwargs.get("population_size", 50)
        self.compatibility_threshold = kwargs.get("compatibility_threshold", 3.0)
        self.excess_coef = kwargs.get("excess_coef", 1.0)
        self.disjoint_coef = kwargs.get("disjoint_coef", 1.0)
        self.weight_diff_coef = kwargs.get("weight_diff_coef", 0.4)
        self.stagnation_threshold = kwargs.get("stagnation_threshold", 15)
        self.min_species_size_for_champion = kwargs.get("min_species_size_for_champion", 5)
        self.add_connection_mutate_rate = kwargs.get("add_connection_mutate_rate", 0.1)
        self.add_node_mutate_rate = kwargs.get("add_node_mutate_rate", 0.05)
        self.weight_mutate_rate = kwargs.get("weight_mutate_rate", 0.8)
        self.weight_perturb_uniform_rate = kwargs.get("weight_perturb_uniform_rate", 0.9)
        self.enable_mutate_rate = kwargs.get("enable_mutate_rate", 0.05)
        self.disable_mutate_rate = kwargs.get("disable_mutate_rate", 0.01)
        self.crossover_rate = kwargs.get("crossover_rate", 0.75)
        self.interspecies_mate_rate = kwargs.get("interspecies_mate_rate", 0.001)
        self.elitism_species_percent = kwargs.get("elitism_species_percent", 0.1)
        self.elitism_genome_percent_in_species = kwargs.get("elitism_genome_percent_in_species", 0.1)
        self.initial_connection_type = kwargs.get("initial_connection_type", "minimal") # "full" or "minimal"
        self.output_activation = kwargs.get("output_activation", "tanh") # e.g. "tanh", "sigmoid"
        self.hidden_activation = kwargs.get("hidden_activation", "tanh")
        self.ff_passes = kwargs.get("ff_passes", 3) # Number of feedforward passes
        self.allow_recurrent_connections_on_mutate = kwargs.get("allow_recurrent_connections_on_mutate", True)


        # Input/Output sizes based on preprocessed observation and game actions
        # Assumes observation_shape is (C, H, W) and C=1 for grayscale
        self.num_inputs = observation_shape[1] * observation_shape[2] # H * W
        self.num_outputs = action_size

        # Global Innovation Tracker (reset for this agent instance)
        self.innovation_tracker = GlobalInnovation()
        self.innovation_tracker.reset(self.num_inputs, self.num_outputs)
        # Make _global_innovation_tracker an alias or remove its direct use in Genome/ConnectionGene
        # For now, GenomeNEAT and ConnectionGene will use the agent_ref.innovation_tracker

        self.population = [GenomeNEAT(self.num_inputs, self.num_outputs, agent_ref=self) for _ in range(self.population_size)]
        self.species = []
        self.current_genome_idx = 0 # Correct attribute name
        self.current_generation = 0
        self.best_fitness_overall = -float('inf')
        self.best_genome_overall = None

        self.processed_h = observation_shape[1]
        self.processed_w = observation_shape[2]
        print(f"NEAT Agent initialized. Pop: {self.population_size}, Inputs: {self.num_inputs}, Outputs: {self.num_outputs}")

    # Helper for GenomeNEAT to get activation function names
    def get_node_activation_name(self, node_id):
        if node_id < self.num_inputs: return None # Inputs don't have activation in same sense
        if node_id < self.num_inputs + self.num_outputs: return self.output_activation
        return self.hidden_activation


    def choose_action(self, raw_observation):
        # ... (logic as before, uses self.processed_h, self.processed_w) ...
        current_genome = self.population[self.current_genome_idx]
        processed_obs_np = preprocess_observation(raw_observation, new_size=(self.processed_h, self.processed_w))
        flat_input = processed_obs_np.flatten()
        network_outputs = current_genome.feed_forward(flat_input)
        action = np.argmax(network_outputs) 
        return action

    def record_fitness(self, score):
        # ... (logic as before, uses self.current_genome_idx) ...
        self.population[self.current_genome_idx].fitness = score
        if score > self.best_fitness_overall:
            self.best_fitness_overall = score
            self.best_genome_overall = self.population[self.current_genome_idx].clone()
            # print_f for train.py if needed: print_f(f"  NEAT New Best: {self.best_fitness_overall:.2f}")
        self.current_genome_idx += 1

    def learn(self): # Evolution step
        if self.current_genome_idx < self.population_size:
            # print_f(f"NEAT learn called mid-evaluation. Waiting...") # train.py will print this
            return None

        self.current_generation += 1
        # print_f for train.py: print_f(f"--- NEAT Gen {self.current_generation} Evolution ---")

        self._speciate_population()
        
        total_adj_fitness_sum_for_spawn = 0
        active_species_count = 0
        for s in self.species:
            if s.members:
                s.calculate_shared_fitness()
                species_total_adj_fitness = sum(m.adjusted_fitness for m in s.members)
                total_adj_fitness_sum_for_spawn += species_total_adj_fitness
                active_species_count +=1
        
        if active_species_count == 0: # All species died out or empty
            print(f"  All species are empty! Re-initializing population for NEAT.")
            self.population = [GenomeNEAT(self.num_inputs, self.num_outputs, agent_ref=self) for _ in range(self.population_size)]
            self.current_genome_idx = 0
            self.species = []
            self.innovation_tracker.reset(self.num_inputs, self.num_outputs) # Reset innovation
            return None


        new_population = []
        # Elitism: Carry over champions of top species
        self.species.sort(key=lambda s: s.get_average_fitness(), reverse=True) # Sort by raw average fitness
        
        num_elite_species_to_preserve_champ = max(1, int(len(self.species) * self.elitism_species_percent))
        for i in range(min(num_elite_species_to_preserve_champ, len(self.species))):
            species_to_preserve = self.species[i]
            if species_to_preserve.members and len(species_to_preserve.members) >= self.min_species_size_for_champion :
                champion = species_to_preserve.get_champion()
                if champion and len(new_population) < self.population_size:
                    new_population.append(champion.clone())
            # else species is too small or empty, its champ is not auto-preserved this way

        # Generate offspring
        spawn_amounts = []
        if total_adj_fitness_sum_for_spawn > 0:
            for s in self.species:
                if s.members:
                    species_total_adj_fitness = sum(m.adjusted_fitness for m in s.members)
                    num_to_spawn = math.floor( (species_total_adj_fitness / total_adj_fitness_sum_for_spawn) * (self.population_size - len(new_population)) )
                    spawn_amounts.append(num_to_spawn)
                else:
                    spawn_amounts.append(0)
        else: # All adjusted fitness is zero, spawn somewhat evenly from existing (if any)
            num_surviving_species = len([s for s in self.species if s.members])
            if num_surviving_species > 0:
                base_spawn = (self.population_size - len(new_population)) // num_surviving_species
                spawn_amounts = [base_spawn if s.members else 0 for s in self.species]
            else: # Should have been caught by active_species_count == 0
                spawn_amounts = [0] * len(self.species)


        # Distribute remaining spawn slots due to floor rounding
        remaining_spawn_slots = (self.population_size - len(new_population)) - sum(spawn_amounts)
        s_idx = 0
        while remaining_spawn_slots > 0 and any(s.members for s in self.species):
            if self.species[s_idx % len(self.species)].members: # Only add to non-empty species
                spawn_amounts[s_idx % len(self.species)] += 1
                remaining_spawn_slots -= 1
            s_idx += 1


        for i, s in enumerate(self.species):
            if not s.members or spawn_amounts[i] == 0: continue
            
            s.sort_members_by_fitness() # Sort for elitism within species
            num_elites_in_species = max(1, int(len(s.members) * self.elitism_genome_percent_in_species))
            
            for j in range(min(num_elites_in_species, len(s.members))): # Elites from this species
                if len(new_population) < self.population_size:
                    new_population.append(s.members[j].clone())
            
            # Crossover and mutation for remaining spawns for this species
            for _ in range(spawn_amounts[i] - min(num_elites_in_species, len(s.members))):
                if len(new_population) >= self.population_size: break
                
                parent1 = random.choice(s.members) # Or tournament selection within species
                child = None
                if np.random.rand() < self.interspecies_mate_rate and len(self.species) > 1:
                    other_species = random.choice([sp for sp_idx, sp in enumerate(self.species) if sp_idx != i and sp.members])
                    parent2 = random.choice(other_species.members) if other_species else random.choice(s.members)
                else:
                    parent2 = random.choice(s.members)

                if np.random.rand() < self.crossover_rate:
                    child = crossover_neat(parent1, parent2, agent_ref=self)
                else:
                    child = parent1.clone()
                child.mutate()
                new_population.append(child)
        
        # Fill population if still not full (e.g. all species died out early)
        while len(new_population) < self.population_size:
            # print_f("  NEAT: Filling population with new random genomes.")
            new_population.append(GenomeNEAT(self.num_inputs, self.num_outputs, agent_ref=self))


        self.population = new_population[:self.population_size] # Ensure correct size
        self.current_genome_idx = 0

        surviving_species_next_gen = []
        for s in self.species:
            if s.check_stagnation() or not s.members: # Remove species if stagnated or empty
                 # print_f for train.py: print_f(f"  Species {s.id} removed (stagnation/empty).")
                 pass
            else:
                # Update representative to a random member of the species for next gen
                # Or keep the old one if preferred. Random helps explore species space.
                s.representative = random.choice(s.members).clone()
                surviving_species_next_gen.append(s)
        self.species = surviving_species_next_gen
        
        # print_f for train.py: print_f(f"--- NEAT Gen {self.current_generation} Complete. Pop: {len(self.population)} ---")
        return None # No single loss for NEAT generation

    def _speciate_population(self):
        # ... (logic as before, but use self.compatibility_threshold and pass self to compatibility_distance)
        for s in self.species: s.members = [] # Clear members, keep representatives
        for genome in self.population:
            placed = False
            for s in self.species:
                if compatibility_distance(genome, s.representative, agent_ref=self) < self.compatibility_threshold:
                    s.add_member(genome); placed = True; break
            if not placed:
                new_s = SpeciesNEAT(genome, len(self.species), agent_ref=self)
                self.species.append(new_s)
        self.species = [s for s in self.species if s.members] # Remove empty species
        # print_f for train.py: print_f(f"  Speciation: {len(self.species)} species.")

    def save(self, path):
        if self.best_genome_overall:
            try:
                with open(path, 'wb') as f: pickle.dump(self.best_genome_overall, f)
                # print_f for train.py: print_f(f"NEAT: Best genome saved (Fit: {self.best_genome_overall.fitness:.2f})")
            except Exception as e: print(f"Error saving NEAT: {e}")
        # else: print_f for train.py: print_f("NEAT: No best to save.")

    def load(self, path):
        try:
            with open(path, 'rb') as f: loaded_genome = pickle.load(f)
            # For NEAT, loading a single genome means this genome becomes the basis for a new population
            # The loaded genome needs its agent_ref set to this NEAT agent instance
            loaded_genome.agent_ref = self
            self.population = [loaded_genome.clone() for _ in range(self.population_size)]
            self.best_genome_overall = loaded_genome.clone() # Ensure it's a clone
            self.best_fitness_overall = loaded_genome.fitness
            self.current_genome_idx = 0
            self.species = [] # Reset species for new evolution from this loaded seed
            self.current_generation = 0 # Reset generation count as we are starting fresh from this genome

            # Reset global innovation tracker based on the loaded genome's complexity
            max_conn_innov = 0; max_node_id = self.num_inputs + self.num_outputs -1
            if loaded_genome.connection_genes:
                max_conn_innov = max(loaded_genome.connection_genes.keys()) if loaded_genome.connection_genes else -1
            if loaded_genome.node_genes:
                 max_node_id = max(loaded_genome.node_genes.keys()) if loaded_genome.node_genes else (self.num_inputs + self.num_outputs -1)
            
            self.innovation_tracker.innovation_number = max_conn_innov + 1
            self.innovation_tracker.node_innovation_count = max_node_id + 1
            # Rebuild innovation history dictionaries in tracker if possible (complex)
            # For now, this simplified reset is okay for starting from a loaded genome.
            print(f"NEAT: Loaded genome from {path} (Fit: {loaded_genome.fitness:.2f}). Population seeded.")
            print(f"  Innovation reset: conn_next={self.innovation_tracker.innovation_number}, node_next={self.innovation_tracker.node_innovation_count}")

        except Exception as e:
            print(f"Error loading NEAT genome: {e}. New population.")
            self.population = [GenomeNEAT(self.num_inputs, self.num_outputs, agent_ref=self) for _ in range(self.population_size)]