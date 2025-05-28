# agents/neat_agent.py
import numpy as np
import random
import math
from collections import defaultdict
from copy import deepcopy
import pickle

from .agent import Agent
from .dqn_agent import preprocess_observation # Assuming this is picklable

# GlobalInnovation class remains the same, it's instantiated per NEATAgent now.
class GlobalInnovation:
    def __init__(self): self.innovation_number = 0; self.node_innovation_count = 0; self.node_innovations = {}; self.connection_innovations = {}
    def reset(self, num_input_nodes, num_output_nodes): self.innovation_number = 0; self.node_innovation_count = num_input_nodes + num_output_nodes; self.node_innovations = {}; self.connection_innovations = {}
    def get_connection_innov(self, in_node_id, out_node_id):
        key = (in_node_id, out_node_id)
        if key not in self.connection_innovations: self.connection_innovations[key] = self.innovation_number; self.innovation_number += 1
        return self.connection_innovations[key]
    def get_node_innov(self, connection_gene_to_split):
        key = (connection_gene_to_split.in_node, connection_gene_to_split.out_node)
        if key not in self.node_innovations: self.node_innovations[key] = self.node_innovation_count; self.node_innovation_count += 1
        return self.node_innovations[key]

class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled=True, innovation_tracker_ref=None): # Changed tracker
        self.in_node, self.out_node, self.weight, self.enabled = in_node, out_node, weight, enabled
        if innovation_tracker_ref: self.innovation = innovation_tracker_ref.get_connection_innov(in_node, out_node)
        else: raise ValueError("ConnectionGene requires an innovation_tracker_ref") # Must be provided
    def clone(self):
        cloned = ConnectionGene(self.in_node, self.out_node, self.weight, self.enabled)
        cloned.innovation = self.innovation # Crucial: preserve innovation on clone
        return cloned

class NodeGene: # (As in your last correct version, using activation_func_name)
    def __init__(self, id, type="hidden", activation_func_name="tanh"):
        self.id, self.type = id, type
        self.value, self.inputs_received = 0.0, []
        if activation_func_name == "tanh": self.activation_func = np.tanh
        elif activation_func_name == "sigmoid": self.activation_func = lambda x: 1/(1 + np.exp(-x))
        elif activation_func_name == "relu": self.activation_func = lambda x: np.maximum(0,x)
        elif activation_func_name == "linear": self.activation_func = lambda x: x
        else: self.activation_func = np.tanh 
    def activate(self):
        if self.type == "input": pass
        else: self.value = self.activation_func(sum(self.inputs_received))
        self.inputs_received = []
        return self.value

class GenomeNEAT:
    def __init__(self, num_inputs, num_outputs, agent_ref): # agent_ref is now mandatory
        self.num_inputs, self.num_outputs = num_inputs, num_outputs
        self.agent_ref = agent_ref # NEATAgent instance
        self.connection_genes, self.node_genes = {}, {}
        self.fitness, self.adjusted_fitness, self.species_id = 0.0, 0.0, None

        for i in range(self.num_inputs): self.node_genes[i] = NodeGene(i, type="input")
        for i in range(self.num_inputs, self.num_inputs + self.num_outputs):
            self.node_genes[i] = NodeGene(i, type="output", activation_func_name=self.agent_ref.output_activation)
        
        if self.agent_ref.initial_connection_type == "full":
            for i_node_id in range(self.num_inputs):
                for o_node_id in range(self.num_inputs, self.num_inputs + self.num_outputs):
                    weight = np.random.uniform(-1, 1) # Use configured range
                    gene = ConnectionGene(i_node_id, o_node_id, weight, innovation_tracker_ref=self.agent_ref.innovation_tracker)
                    self.connection_genes[gene.innovation] = gene

    def feed_forward(self, inputs_flat):
        if len(inputs_flat) != self.num_inputs: raise ValueError(f"Input size mismatch")
        for node in self.node_genes.values(): 
            if node.type != "input": node.value = 0.0; node.inputs_received = []
        for i in range(self.num_inputs): self.node_genes[i].value = inputs_flat[i]
        
        # More robust feedforward using node depths or multiple passes
        # For now, using the pass count from agent_ref
        for _ in range(self.agent_ref.ff_passes):
            for node_id in sorted(self.node_genes.keys()): # Process in ID order (basic topological for non-recurrent)
                node = self.node_genes[node_id]
                if node.type == "input": continue
                current_val_before_activate = sum(node.inputs_received) # Sum before clearing
                node.inputs_received = [] # Clear for this node for this pass
                node.value = node.activation_func(current_val_before_activate)

                # Propagate output
                for gene in self.connection_genes.values():
                    if gene.enabled and gene.in_node == node_id:
                        # Check if out_node exists; essential for dynamic topologies
                        if gene.out_node in self.node_genes:
                             self.node_genes[gene.out_node].inputs_received.append(gene.weight * node.value)
                        # else: print_f(f"Warning: Genome {id(self)} feed_forward found connection to non-existent node {gene.out_node}")


        outputs = []
        for i in range(self.num_inputs, self.num_inputs + self.num_outputs):
            # Final activation for output nodes if they received new inputs
            # This ensures they use the latest summed inputs from the final pass
            out_node = self.node_genes[i]
            if out_node.inputs_received : # Only activate if there are new inputs from last propagation
                out_node.activate() # This will sum inputs_received and clear it
            outputs.append(out_node.value)
        return outputs

    def mutate(self): # Uses self.agent_ref for rates and innovation_tracker
        # ... (Weight mutation logic, using self.agent_ref.weight_mutate_rate etc.) ...
        for gene in self.connection_genes.values():
            if np.random.rand() < self.agent_ref.weight_mutate_rate:
                gene.weight += np.random.uniform(-self.agent_ref.weight_perturb_strength, self.agent_ref.weight_perturb_strength) if np.random.rand() < self.agent_ref.weight_perturb_uniform_rate else np.random.uniform(-self.agent_ref.weight_random_range, self.agent_ref.weight_random_range)
                gene.weight = np.clip(gene.weight, -self.agent_ref.max_weight, self.agent_ref.max_weight) # Clip weights
            if gene.enabled and np.random.rand() < self.agent_ref.disable_mutate_rate: gene.enabled = False
            elif not gene.enabled and np.random.rand() < self.agent_ref.enable_mutate_rate: gene.enabled = True

        if np.random.rand() < self.agent_ref.add_connection_mutate_rate:
            self._mutate_add_connection() # Will use self.agent_ref.innovation_tracker
        if np.random.rand() < self.agent_ref.add_node_mutate_rate:
            self._mutate_add_node() # Will use self.agent_ref.innovation_tracker

    def _mutate_add_connection(self):
        # ... (Logic as before, but ConnectionGene now requires innovation_tracker_ref)
        possible_in_nodes = [nid for nid, n in self.node_genes.items() if n.type != "output"]
        possible_out_nodes = [nid for nid, n in self.node_genes.items() if n.type != "input"]
        if not possible_in_nodes or not possible_out_nodes: return
        for _ in range(10): # Attempts
            in_id = random.choice(possible_in_nodes)
            out_id = random.choice(possible_out_nodes)
            if in_id == out_id: continue # No self-loops on same node via this mutation
            # Check if this connection would create a direct cycle to an input node (not allowed)
            # or if it's from an output node (also typically not allowed for initial connections)
            if self.node_genes[out_id].type == "input": continue
            if self.node_genes[in_id].type == "output" and self.node_genes[out_id].type != "output" and not self.agent_ref.allow_recurrent_connections_on_mutate : continue


            existing = False
            for gene in self.connection_genes.values():
                if gene.in_node == in_id and gene.out_node == out_id: existing = True; break
            if existing: continue

            weight = np.random.uniform(-self.agent_ref.weight_random_range, self.agent_ref.weight_random_range)
            new_gene = ConnectionGene(in_id, out_id, weight, innovation_tracker_ref=self.agent_ref.innovation_tracker)
            self.connection_genes[new_gene.innovation] = new_gene
            break # Added connection

    def _mutate_add_node(self):
        # ... (Logic as before, ConnectionGene requires innovation_tracker_ref)
        if not self.connection_genes: return
        enabled_genes = [g for g in self.connection_genes.values() if g.enabled]
        if not enabled_genes: return
        gene_to_split = random.choice(enabled_genes)
        gene_to_split.enabled = False
        
        new_node_id = self.agent_ref.innovation_tracker.get_node_innov(gene_to_split)
        if new_node_id in self.node_genes: gene_to_split.enabled = True; return # Collision
        
        self.node_genes[new_node_id] = NodeGene(new_node_id, type="hidden", activation_func_name=self.agent_ref.hidden_activation)
        
        gene1 = ConnectionGene(gene_to_split.in_node, new_node_id, 1.0, innovation_tracker_ref=self.agent_ref.innovation_tracker)
        gene2 = ConnectionGene(new_node_id, gene_to_split.out_node, gene_to_split.weight, innovation_tracker_ref=self.agent_ref.innovation_tracker)
        self.connection_genes[gene1.innovation] = gene1
        self.connection_genes[gene2.innovation] = gene2

    def clone(self):
        # ... (Logic as before, ensure agent_ref is handled, and NodeGene takes activation_func_name)
        cloned_genome = GenomeNEAT(self.num_inputs, self.num_outputs, self.agent_ref)
        cloned_genome.node_genes = {}
        for nid, node_obj in self.node_genes.items():
            # Get original activation name string
            activation_name_str = "tanh" # default
            if node_obj.type == "output": activation_name_str = self.agent_ref.output_activation
            elif node_obj.type == "hidden": activation_name_str = self.agent_ref.hidden_activation
            cloned_genome.node_genes[nid] = NodeGene(node_obj.id, node_obj.type, activation_func_name=activation_name_str)
        cloned_genome.connection_genes = {innov: gene.clone() for innov, gene in self.connection_genes.items()}
        return cloned_genome


def crossover_neat(parent1, parent2, agent_ref_for_child): # Pass agent_ref for child
    # ... (Logic as before, GenomeNEAT child needs agent_ref)
    if parent2.fitness > parent1.fitness: parent1, parent2 = parent2, parent1
    child = GenomeNEAT(parent1.num_inputs, parent1.num_outputs, agent_ref_for_child) # Pass agent_ref
    child.connection_genes = {} # Clear any default connections
    # ... (rest of crossover for connections, ensure child_gene.innovation is preserved from cloned parent gene)
    for innov1, gene1 in parent1.connection_genes.items():
        gene2 = parent2.connection_genes.get(innov1)
        if gene2 is not None: # Matching gene
            chosen_parent_gene = random.choice([gene1, gene2])
            child_gene = chosen_parent_gene.clone()
            # Handle disabled genes
            if not gene1.enabled or not gene2.enabled:
                if np.random.rand() < 0.75: # Chance to inherit disabled status from either parent
                    child_gene.enabled = False
                # else: it remains enabled (from the clone)
        else: # Disjoint or excess gene from parent1
            child_gene = gene1.clone()
        child.connection_genes[child_gene.innovation] = child_gene # Use innovation of the gene itself

    # Rebuild node gene dictionary for child based on connections and I/O nodes
    child.node_genes = {}
    # Add input nodes
    for i in range(child.num_inputs):
        child.node_genes[i] = NodeGene(i, type="input")
    # Add output nodes
    for i in range(child.num_inputs, child.num_inputs + child.num_outputs):
        child.node_genes[i] = NodeGene(i, type="output", activation_func_name=agent_ref_for_child.output_activation)
    # Add hidden nodes implied by connections
    for gene in child.connection_genes.values():
        for node_id in [gene.in_node, gene.out_node]:
            if node_id not in child.node_genes: # If it's a hidden node not yet added
                 child.node_genes[node_id] = NodeGene(node_id, type="hidden", activation_func_name=agent_ref_for_child.hidden_activation)
    return child


def compatibility_distance(genome1, genome2, agent_ref_for_coeffs): # Pass agent_ref
    # ... (Logic as before, using agent_ref_for_coeffs.excess_coef etc.)
    innov1 = set(genome1.connection_genes.keys()); innov2 = set(genome2.connection_genes.keys())
    max_innov_g1 = max(innov1) if innov1 else -1; max_innov_g2 = max(innov2) if innov2 else -1
    disjoint, excess, matching_innovs = 0,0,[]
    all_innovs = innov1 | innov2
    for innov in sorted(list(all_innovs)):
        g1_has = innov in innov1; g2_has = innov in innov2
        if g1_has and g2_has: matching_innovs.append(innov)
        elif g1_has and innov > max_innov_g2: excess +=1
        elif g2_has and innov > max_innov_g1: excess +=1
        else: disjoint +=1
    weight_diff_sum = sum(abs(genome1.connection_genes[i].weight - genome2.connection_genes[i].weight) for i in matching_innovs)
    avg_w_diff = (weight_diff_sum / len(matching_innovs)) if matching_innovs else 0
    N = max(len(genome1.connection_genes), len(genome2.connection_genes)); N = 1 if N < 1 else N # N should be at least 1
    return ( (agent_ref_for_coeffs.excess_coef * excess / N) + 
             (agent_ref_for_coeffs.disjoint_coef * disjoint / N) + 
             (agent_ref_for_coeffs.weight_diff_coef * avg_w_diff) )

class SpeciesNEAT: # (As before, ensure it uses agent_ref for its parameters)
    def __init__(self, representative_genome, species_id, agent_ref):
        self.id, self.agent_ref = species_id, agent_ref
        self.representative = representative_genome.clone()
        self.members = [representative_genome]
        self.generations_since_improvement, self.max_fitness_achieved = 0, -float('inf')
        representative_genome.species_id = self.id # Assign species ID to the genome
    def add_member(self, genome): self.members.append(genome); genome.species_id = self.id
    def calculate_shared_fitness(self):
        for m in self.members: m.adjusted_fitness = m.fitness / len(self.members) if self.members else 0.0
    def get_average_fitness(self): return sum(m.fitness for m in self.members) / len(self.members) if self.members else 0.0
    def get_champion(self): return max(self.members, key=lambda m: m.fitness) if self.members else None
    def sort_members_by_fitness(self): self.members.sort(key=lambda m: m.fitness, reverse=True)
    def check_stagnation(self):
        champ = self.get_champion()
        current_max = champ.fitness if champ else -float('inf')
        if current_max > self.max_fitness_achieved: self.max_fitness_achieved = current_max; self.generations_since_improvement = 0
        else: self.generations_since_improvement += 1
        return self.generations_since_improvement > self.agent_ref.stagnation_threshold


class NEATAgent(Agent): # (As in your last correct version, accepting **kwargs)
    def __init__(self, action_size, observation_shape, **kwargs):
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
        self.weight_perturb_strength = kwargs.get("weight_perturb_strength", 0.1) # Added for mutate
        self.weight_random_range = kwargs.get("weight_random_range", 2.0) # Added for mutate
        self.max_weight = kwargs.get("max_weight", 5.0) # Added for mutate (weight clipping)
        self.enable_mutate_rate = kwargs.get("enable_mutate_rate", 0.05)
        self.disable_mutate_rate = kwargs.get("disable_mutate_rate", 0.01)
        self.crossover_rate = kwargs.get("crossover_rate", 0.75)
        self.interspecies_mate_rate = kwargs.get("interspecies_mate_rate", 0.001)
        self.elitism_species_percent = kwargs.get("elitism_species_percent", 0.1) # Elites from best species
        self.elitism_genome_percent_in_species = kwargs.get("elitism_genome_percent_in_species", 0.1) # Elites within a species
        self.initial_connection_type = kwargs.get("initial_connection_type", "minimal")
        self.output_activation = kwargs.get("output_activation", "tanh")
        self.hidden_activation = kwargs.get("hidden_activation", "tanh")
        self.ff_passes = kwargs.get("ff_passes", 3)
        self.allow_recurrent_connections_on_mutate = kwargs.get("allow_recurrent_connections_on_mutate", False) # Default to False for simpler initial networks

        self.num_inputs = observation_shape[1] * observation_shape[2]
        self.num_outputs = action_size
        self.innovation_tracker = GlobalInnovation() # Instance specific tracker
        self.innovation_tracker.reset(self.num_inputs, self.num_outputs)

        self.population = [GenomeNEAT(self.num_inputs, self.num_outputs, agent_ref=self) for _ in range(self.population_size)]
        self.species = []
        self.current_genome_idx = 0
        self.current_generation = 0
        self.best_fitness_overall = -float('inf')
        self.best_genome_overall = None
        self.processed_h, self.processed_w = observation_shape[1], observation_shape[2]
        print(f"NEAT Agent: Pop: {self.population_size}, Inputs: {self.num_inputs}, Outputs: {self.num_outputs}")

    def get_node_activation_name(self, node_id): # Helper for GenomeNEAT.clone
        if node_id < self.num_inputs: return "linear" # Or None, inputs don't 'activate' via func
        if node_id < self.num_inputs + self.num_outputs: return self.output_activation
        return self.hidden_activation

    def choose_action(self, raw_observation):
        current_genome = self.population[self.current_genome_idx]
        processed_obs_np = preprocess_observation(raw_observation, new_size=(self.processed_h, self.processed_w))
        flat_input = processed_obs_np.flatten()
        network_outputs = current_genome.feed_forward(flat_input)
        
        action = np.argmax(network_outputs)
        # --- ADD THIS ---
        if self.current_generation < 5 and self.current_genome_idx < 3: # Print for early phase
            print(f"  Gen {self.current_generation} Genome {self.current_genome_idx} Outputs: {network_outputs}, Action: {action}")
        # You'll need to pass step_count_in_episode or have the agent track it.
        # Or just print for every action for a few genomes:
        if self.current_genome_idx == 0 and self.current_generation % 5 == 0: # Print for genome 0 every 5 gens
            print(f"  NEAT G{self.current_generation} I0 Outputs: {network_outputs} -> Act: {action}")
        # --- END ADD ---
        return action

    def record_fitness(self, score): # Called by train.py after worker evaluates a genome
        # This method now assumes train.py will assign fitness to self.population[some_index].fitness
        # and then update current_genome_idx. This method is more for bookkeeping if called directly.
        # In the parallel setup, train.py will directly update fitnesses.
        # This method is effectively superseded by train.py's direct fitness assignment for parallel NEAT.
        # However, NEATAgent still needs to track its best_fitness_overall.
        # The train.py loop for NEAT will update self.best_fitness_overall and self.best_genome_overall.
        
        # This is called by train.py for EACH genome after its fitness is known
        # The `current_genome_idx` here refers to the conceptual index if we were iterating serially.
        # Since train.py assigns all fitnesses, then calls learn(), current_genome_idx here isn't the driver.
        # Let's assume train.py has updated the fitness for population[self.current_genome_idx]
        # current_genome_fitness = self.population[self.current_genome_idx].fitness
        # if current_genome_fitness > self.best_fitness_overall:
        #     self.best_fitness_overall = current_genome_fitness
        #     self.best_genome_overall = self.population[self.current_genome_idx].clone()
        # self.current_genome_idx += 1 # This will be reset in learn() or managed by train.py's generation loop
        pass # Fitness assignment and best tracking is now handled in train.py's NEAT loop

    def learn(self): # Evolution step - called by train.py AFTER all genomes have fitness assigned
        # No check for current_genome_idx needed here, train.py ensures all are evaluated.
        self.current_generation += 1
        self._speciate_population()
        if not self.species: # Check if all species died out
            print(f"  All species died out! Re-initializing NEAT population.")
            self.population = [GenomeNEAT(self.num_inputs, self.num_outputs, agent_ref=self) for _ in range(self.population_size)]
            self.current_genome_idx = 0
            self.species = []
            self.innovation_tracker.reset(self.num_inputs, self.num_outputs)
            return None

        # ... (The rest of your detailed speciation, shared fitness, elitism, crossover, mutation logic) ...
        # This part is complex and largely remains the same as your previous neat_agent.py's learn method.
        # Ensure it uses self.innovation_tracker via self.agent_ref in child genomes and for mutations.
        # Calculate shared fitness
        total_adjusted_fitness_sum = 0
        for s in self.species: s.calculate_shared_fitness(); total_adjusted_fitness_sum += sum(m.adjusted_fitness for m in s.members)
        if total_adjusted_fitness_sum == 0: total_adjusted_fitness_sum = 1e-6

        new_population = []
        # Elitism from best species
        self.species.sort(key=lambda s: s.get_average_fitness(), reverse=True)
        num_elite_species_champs = max(1, int(len(self.species) * self.elitism_species_percent))
        for i in range(min(num_elite_species_champs, len(self.species))):
            s = self.species[i]
            if s.members and len(s.members) >= self.min_species_size_for_champion:
                champ = s.get_champion()
                if champ and len(new_population) < self.population_size: new_population.append(champ.clone())
        
        # Offspring
        num_offspring_needed = self.population_size - len(new_population)
        spawn_counts = []
        for s in self.species:
            if s.members and total_adjusted_fitness_sum > 0: # Check total_adjusted_fitness_sum
                 species_adj_fit_sum = sum(m.adjusted_fitness for m in s.members)
                 spawns = math.floor(species_adj_fit_sum / total_adjusted_fitness_sum * num_offspring_needed)
                 spawn_counts.append(spawns)
            else: spawn_counts.append(0)
        
        # Distribute remaining due to floor
        current_total_spawns = sum(spawn_counts)
        s_idx_distribute = 0
        while current_total_spawns < num_offspring_needed and any(s.members for s in self.species):
            species_for_extra_spawn = self.species[s_idx_distribute % len(self.species)]
            if species_for_extra_spawn.members: # Only add to non-empty species
                 spawn_counts[s_idx_distribute % len(self.species)] +=1
                 current_total_spawns +=1
            s_idx_distribute +=1
            if s_idx_distribute > 2 * len(self.species) and current_total_spawns < num_offspring_needed: # Safety break
                break 


        for i, s in enumerate(self.species):
            if not s.members or spawn_counts[i] == 0: continue
            s.sort_members_by_fitness()
            num_elites_this_species = max(0,int(len(s.members) * self.elitism_genome_percent_in_species)) # Can be 0 if small species
            
            # Add elites from this species (if not already added by top species elitism, and if distinct)
            for j in range(min(num_elites_this_species, len(s.members))):
                elite_genome = s.members[j]
                # Avoid adding exact same elite if already added from top species elitism
                # This check is tricky if clones are involved. Simple check:
                already_added = any(new_pop_genome.fitness == elite_genome.fitness and 
                                    compatibility_distance(new_pop_genome, elite_genome, self) < 0.1 
                                    for new_pop_genome in new_population)
                if len(new_population) < self.population_size and not already_added:
                    new_population.append(elite_genome.clone())

            # Reproduce for remaining spawn count for this species
            num_to_reproduce = spawn_counts[i] - (len(new_population) - (self.population_size - num_offspring_needed - current_total_spawns + sum(spawn_counts[:i+1]) )) # Complex to track exact elites added
            # Simpler: num_to_reproduce = spawn_counts[i] - number_of_elites_added_from_this_species_in_this_step
            # For now, just fill up to spawn_counts[i] for this species in total (elites + offspring)
            
            current_pop_len_before_repro = len(new_population)
            for _ in range(spawn_counts[i]): # This species needs to contribute spawn_counts[i] individuals total
                if len(new_population) >= self.population_size: break
                if not s.members: break # Safety if species became empty during elite selection

                parent1 = random.choice(s.members) # Simplified selection
                child = None
                if np.random.rand() < self.interspecies_mate_rate and len(self.species) > 1:
                    other_species_list = [sp for sp_idx, sp in enumerate(self.species) if sp_idx != i and sp.members]
                    if other_species_list:
                        parent2 = random.choice(random.choice(other_species_list).members)
                    else: parent2 = random.choice(s.members) # Fallback
                else: parent2 = random.choice(s.members)

                if np.random.rand() < self.crossover_rate: child = crossover_neat(parent1, parent2, self)
                else: child = random.choice([parent1, parent2]).clone() # Or just parent1.clone()
                
                child.agent_ref = self # Critical: ensure child has ref to NEATAgent for mutation
                child.mutate()
                new_population.append(child)
        
        while len(new_population) < self.population_size:
            new_population.append(GenomeNEAT(self.num_inputs, self.num_outputs, agent_ref=self)) # Fill with new random

        self.population = new_population[:self.population_size]
        self.current_genome_idx = 0 # Reset for next generation's fitness evaluation cycle

        surviving_species = [] # Stagnation check
        for s in self.species:
            s.members = [g for g in self.population if g.species_id == s.id] # Re-assign members based on full new pop
            if not s.members: continue # Skip if no members assigned in new pop
            if s.check_stagnation():
                # print_f(f"  Species {s.id} removed (stagnation).") # train.py can print this
                pass
            else:
                if s.members : s.representative = random.choice(s.members).clone() # Update representative
                surviving_species.append(s)
        self.species = surviving_species
        return None

    def _speciate_population(self): # Uses self.compatibility_threshold, self.innovation_tracker
        for s in self.species: s.members = []
        for genome in self.population:
            placed = False
            for s in self.species:
                if compatibility_distance(genome, s.representative, self) < self.compatibility_threshold:
                    s.add_member(genome); placed = True; break
            if not placed:
                self.species.append(SpeciesNEAT(genome, len(self.species), self))
        self.species = [s for s in self.species if s.members]

    def save(self, path): # Saves best_genome_overall
        if self.best_genome_overall:
            try:
                with open(path, 'wb') as f: pickle.dump(self.best_genome_overall, f)
            except Exception as e: print(f"Error saving NEAT: {e}")

    def load(self, path): # Loads a single genome and seeds population
        # ... (logic as before, ensuring agent_ref is set and innovation_tracker is updated)
        try:
            with open(path, 'rb') as f: loaded_genome = pickle.load(f)
            loaded_genome.agent_ref = self # VERY IMPORTANT
            self.population = [loaded_genome.clone() for _ in range(self.population_size)]
            self.best_genome_overall = loaded_genome.clone()
            self.best_fitness_overall = loaded_genome.fitness
            self.current_genome_idx, self.current_generation, self.species = 0, 0, []
            max_conn_innov, max_node_id = -1, self.num_inputs + self.num_outputs -1
            if loaded_genome.connection_genes: max_conn_innov = max(loaded_genome.connection_genes.keys() or [-1])
            if loaded_genome.node_genes: max_node_id = max(loaded_genome.node_genes.keys() or [max_node_id])
            self.innovation_tracker.reset(self.num_inputs, self.num_outputs) # Reset first
            self.innovation_tracker.innovation_number = max_conn_innov + 1
            self.innovation_tracker.node_innovation_count = max_node_id + 1
            # Rebuild innovation history for loaded genome's connections/nodes
            # This is complex. For simplicity, the above reset might lead to some innovation number reuse
            # if evolution continues from a loaded genome AND new structures identical to old ones (from pre-load) are formed.
            # A truly robust resume would save/load the innovation_tracker's state too.
            print(f"NEAT: Loaded genome from {path} (Fit: {loaded_genome.fitness:.2f}). Population seeded.")
            print(f"  Innovation reset: conn_next={self.innovation_tracker.innovation_number}, node_next={self.innovation_tracker.node_innovation_count}")
        except Exception as e:
            print(f"Error loading NEAT: {e}. New random population.")
            self.population = [GenomeNEAT(self.num_inputs, self.num_outputs, agent_ref=self) for _ in range(self.population_size)]