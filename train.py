# train.py
import pygame as pg
import argparse
import sys
import os
import numpy as np
from collections import deque
import json
from copy import deepcopy 
import torch # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ADDED IMPORT

try:
    from game.game_manager import Game 
    from game import config as game_config_module 
    
    from utils.model_helpers import ( 
        get_existing_model_versions, get_next_model_save_path,
        get_latest_model_path, MODELS_DIR
    )
    from utils.cli_args import add_common_training_args, add_agent_config_path_args, ALL_AGENT_TYPES_AVAILABLE
    from utils.agent_factory import create_agent
    from agents.dqn_agent import preprocess_observation 
    from utils.serial_training_loop import run_serial_agent_training 
except ImportError as e:
    print(f"Error importing files in train.py: {e}", flush=True)
    sys.exit(1)

def print_f(*args, **kwargs): print(*args, **kwargs); sys.stdout.flush()

def load_agent_config(config_path_cli, agent_name):
    config_to_try = config_path_cli
    if not config_to_try: 
        default_config_filename = f"{agent_name}_default.json"
        config_to_try = os.path.join(os.path.dirname(__file__), "configs", default_config_filename)
    
    if config_to_try and os.path.exists(config_to_try):
        try:
            with open(config_to_try, 'r') as f:
                print_f(f"  Loading config for {agent_name} from: {os.path.basename(config_to_try)}")
                return json.load(f)
        except Exception as e: 
            print_f(f"  Warning: Error loading config '{os.path.basename(config_to_try)}': {e}.")
    # Check if CLI path was given but failed (either not found or error during loading)
    # _did_config_load_successfully is a helper to check if loading the file would succeed
    if config_path_cli and (not os.path.exists(config_path_cli) or \
                            (config_path_cli == config_to_try and not _did_config_load_successfully(config_to_try))):
         print_f(f"  Specified config '{os.path.basename(config_path_cli)}' not found or failed to load.")
    
    print_f(f"  Using agent's internal default hyperparameters for {agent_name}.")
    return {}

def _did_config_load_successfully(path): # Helper for load_agent_config
    if path and os.path.exists(path):
        try:
            with open(path, 'r') as f: json.load(f) # Try to load to check for errors
            return True
        except: return False
    return False


def evaluate_single_genome_serially(genome_to_evaluate, 
                                    agent_ref, 
                                    agent_hparams, 
                                    game_instance_eval, 
                                    max_steps_eval, 
                                    preprocessed_obs_shape_eval,
                                    is_render_run=False,
                                    agent_type_for_eval='neat',
                                    print_interval_steps_eval=0, 
                                    episode_info_str="" 
                                    ):
    observation_raw = game_instance_eval.reset_for_ai()
    current_game_score = 0.0 # This will be the fitness for NEAT/GA
    done_eval = False
    info_eval = {}
    total_accumulated_reward_this_episode = 0.0 # Use this for tracking reward for printing

    original_current_genome_idx = -1 # For NEAT
    original_ga_idx = -1 # For GA

    if agent_type_for_eval == 'neat' and hasattr(agent_ref, 'population'):
        original_current_genome_idx = agent_ref.current_genome_idx
        try:
            eval_genome_idx = agent_ref.population.index(genome_to_evaluate)
            agent_ref.current_genome_idx = eval_genome_idx
        except ValueError:
            print_f(f"  ERROR: Genome {episode_info_str} not found in NEATAgent population. Using original index {original_current_genome_idx}.")
            # If genome not found, it can't be evaluated properly by agent_ref.choose_action
            # This indicates a logic error in how genomes are passed or managed.
            # For safety, might return very low fitness or raise an error.
            # For now, we'll let it proceed but it will likely use the wrong genome via agent_ref.current_genome_idx
            pass 
    elif agent_type_for_eval == 'genetic' and hasattr(agent_ref, 'current_individual_idx') and hasattr(agent_ref, 'population'):
        original_ga_idx = agent_ref.current_individual_idx
        try:
            # The train.py loop for GA ALREADY sets agent_ref.current_individual_idx correctly.
            # This block is more of a sanity check or for if this function was called differently.
            # Here, genome_to_evaluate IS the network. agent_ref.population contains these networks.
            current_eval_idx = agent_ref.population.index(genome_to_evaluate)
            agent_ref.current_individual_idx = current_eval_idx # Ensure it's set
        except ValueError:
            print_f(f"  ERROR: GA Individual {episode_info_str} not found in population. Using original GA index {original_ga_idx}.")
            pass


    for step_count in range(max_steps_eval):
        actual_step_count = step_count + 1
        if is_render_run: 
            for event in pg.event.get():
                if event.type == pg.QUIT or \
                   (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    print_f(f"    {episode_info_str} Rendered serial evaluation interrupted by user.")
                    # Restore original indices before returning
                    if agent_type_for_eval == 'neat' and original_current_genome_idx != -1:
                        agent_ref.current_genome_idx = original_current_genome_idx
                    if agent_type_for_eval == 'genetic' and original_ga_idx != -1:
                        agent_ref.current_individual_idx = original_ga_idx
                    return -float('inf'), 0.0 # Fitness, Reward

        if observation_raw is None: 
            current_game_score = -float('inf'); total_accumulated_reward_this_episode = -float('inf'); break
        
        action = -1 
        try:
            # For both NEAT and GA, agent_ref.choose_action will now be used,
            # relying on the correct internal index (current_genome_idx for NEAT, current_individual_idx for GA)
            # being set prior to or at the start of this function call by the main training loop.
            action = agent_ref.choose_action(observation_raw)

        except Exception as e_action: 
            print_f(f"  Error during {agent_type_for_eval} {episode_info_str} action choice: {e_action}")
            current_game_score = -float('inf'); total_accumulated_reward_this_episode = -float('inf'); break
        
        if is_render_run and hasattr(game_instance_eval, 'set_render_for_ai_this_step'):
            game_instance_eval.set_render_for_ai_this_step(True)

        observation_raw, reward_step, done_eval, info_eval = game_instance_eval.step_ai(action)
        total_accumulated_reward_this_episode += reward_step # Correctly accumulate reward
        
        if print_interval_steps_eval > 0 and actual_step_count % print_interval_steps_eval == 0:
            # Get score from info if available, otherwise use the accumulating current_game_score (though for GA/NEAT fitness is end-of-episode score)
            score_for_print = info_eval.get('score', 0) if done_eval else game_instance_eval.score # Get current score from game
            print_f(f"    {episode_info_str} Step {actual_step_count}: EvalRwdSoFar={total_accumulated_reward_this_episode:.0f}, GameScoreSoFar={score_for_print}")
        
        if done_eval:
            current_game_score = float(info_eval.get('score', 0))
            break
            
    if not done_eval: # If loop finished due to max_steps
        current_game_score = float(info_eval.get('score', game_instance_eval.score)) # Get final score
        
    # Restore original current_genome_idx for NEAT agent after evaluation
    if agent_type_for_eval == 'neat' and original_current_genome_idx != -1:
        agent_ref.current_genome_idx = original_current_genome_idx
    # Restore original current_individual_idx for GA (train.py loop manages this primarily)
    if agent_type_for_eval == 'genetic' and original_ga_idx != -1:
        agent_ref.current_individual_idx = original_ga_idx
        
    return current_game_score, total_accumulated_reward_this_episode


def train_all_agents():
    parser = argparse.ArgumentParser(description="Train Space Invaders AI Agents")
    parser = add_common_training_args(parser)
    parser = add_agent_config_path_args(parser)
    args = parser.parse_args()
    
    valid_agents_to_train = [a.strip() for a in args.agents.split(',') if a.strip() in ALL_AGENT_TYPES_AVAILABLE and a != 'random']
    if not valid_agents_to_train: print_f("No valid agents. Exiting."); sys.exit(1)
    
    print_f(f"--- Starting Training Session (Saving to: {MODELS_DIR}) ---")
    print_f(f"Agents: {', '.join(valid_agents_to_train)}, Episodes/Generations: {args.episodes}")
    print_f(f"Note: NEAT and Genetic Algorithm training will run SERIALLY.")

    print_f(f"Load LATEST models: {args.load_models}")
    print_f(f"Force train (new version): {args.force_train}")
    print_f(f"Render game content: {args.render}")
    effective_silent_mode = not args.render # Default: silent if not rendering
    if args.render:
        # If rendering, be silent only if --silent_training is explicitly passed
        effective_silent_mode = args.silent_training 
        print_f(f"Sounds during rendered training: {not effective_silent_mode}")
    else:
        print_f(f"Running headless (no sound, no visuals for main training loop).")
    print_f(f"Max steps per episode/eval: {args.max_steps_per_episode}")
    print_f(f"Save interval (NNs): {args.save_interval} episodes")
    print_f(f"Print interval (steps): {args.print_interval_steps}")
    print_f(f"AI Training Mode (in Game): ON")
    print_f("---------------------------------")

    game_instance_main = Game(
        silent_mode=effective_silent_mode, # Use the calculated effective_silent_mode
        ai_training_mode=True, 
        headless_worker_mode=not args.render # headless if not rendering on display
    )
    preprocessed_obs_shape = (1, 84, 84) 
    action_size = game_instance_main.get_action_size()
    
    overall_episode_rewards_or_fitness = deque(maxlen=100) 
    stop_training_global_ref = [False] 

    for agent_name in valid_agents_to_train:
        if stop_training_global_ref[0]: break
        print_f(f"\n--- Preparing Agent: {agent_name.upper()} ---")
        
        existing_versions = get_existing_model_versions(agent_name) 
        model_to_load_path, model_to_save_path = None, None
        if existing_versions:
            print_f(f"  Found: {', '.join([os.path.basename(p) for p in existing_versions])}")
            if args.load_models:
                model_to_load_path = existing_versions[-1]; model_to_save_path = model_to_load_path
                print_f(f"  Continue. Load: {os.path.basename(model_to_load_path)}. Save to same.")
            elif not args.force_train: print_f(f"  Skip {agent_name}."); continue
            else: model_to_save_path = get_next_model_save_path(agent_name); print_f(f"  Force. New: {os.path.basename(model_to_save_path)}")
        else: model_to_save_path = get_next_model_save_path(agent_name); print_f(f"  No models. New: {os.path.basename(model_to_save_path)}")
        if not model_to_save_path and not model_to_load_path: print_f(f"  Path error for {agent_name}. Skip."); continue
            
        cli_config_path = getattr(args, f"{agent_name}_config_path", None)
        agent_hparams = load_agent_config(cli_config_path, agent_name)
        
        agent = create_agent(agent_name, action_size, preprocessed_obs_shape, 
                             hparams=agent_hparams, mode='train')
        if not agent: print_f(f"Failed to create agent {agent_name}. Skip."); continue

        if model_to_load_path and hasattr(agent, 'load'):
            try: agent.load(model_to_load_path) 
            except Exception as e:
                print_f(f"Load failed {os.path.basename(model_to_load_path)}: {e}. Fresh start for this agent.")
                if model_to_save_path == model_to_load_path: 
                    model_to_save_path = get_next_model_save_path(agent_name) 
                    print_f(f"Adjusted save path to new: {os.path.basename(model_to_save_path)}")
        if not model_to_save_path: model_to_save_path = get_next_model_save_path(agent_name)

        print_f(f"--- Training: {agent_name.upper()} (Save: {os.path.basename(model_to_save_path)}) ---")
        
        num_iterations = args.episodes 

        if agent_name == 'neat':
            print_f(f"NEAT: {num_iterations} generations (SERIAL EVALUATION).")
            for gen_idx in range(num_iterations): 
                if stop_training_global_ref[0]: break
                print_f(f"Agent: NEAT - Gen {agent.current_generation + 1}/{num_iterations}")
                
                current_gen_fitness_scores = []
                print_f(f"  NEAT Gen {agent.current_generation + 1}: Evaluating {len(agent.population)} genomes serially...")
                
                for genome_idx, genome_to_eval in enumerate(agent.population):
                    if stop_training_global_ref[0]: break
                    # Print progress periodically within a long serial generation
                    if (genome_idx > 0 and genome_idx % (max(1, len(agent.population)//10)) == 0) or genome_idx == len(agent.population) -1 :
                         print_f(f"    Evaluating NEAT genome {genome_idx + 1}/{len(agent.population)}...")
                    episode_info_str = f"NEAT Gen {agent.current_generation + 1} Genome {genome_idx + 1}"
                    fitness, _ = evaluate_single_genome_serially(
                        genome_to_eval, # The specific genome object
                        agent,          # The main NEATAgent instance (as agent_ref)
                        agent_hparams,
                        game_instance_main, 
                        args.max_steps_per_episode,
                        preprocessed_obs_shape, 
                        args.render, # is_render_run
                        'neat',      # agent_type_for_eval
                        args.print_interval_steps, 
                        episode_info_str 
                    )
                    current_gen_fitness_scores.append(fitness)
                    genome_to_eval.fitness = fitness 

                    if args.render: 
                        for event in pg.event.get():
                            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                                stop_training_global_ref[0] = True; print_f("NEAT training interrupted."); break
                    if stop_training_global_ref[0]: break
                if stop_training_global_ref[0]: break 

                max_fit_this_gen = -float('inf') if not current_gen_fitness_scores else max(current_gen_fitness_scores)
                
                for i in range(len(agent.population)): # Fitness already assigned
                    if agent.population[i].fitness > agent.best_fitness_overall: 
                        agent.best_fitness_overall = agent.population[i].fitness
                        agent.best_genome_overall = deepcopy(agent.population[i]) 
                                                    
                print_f(f"  NEAT Gen {agent.current_generation + 1}: Serial eval complete. MaxFit Gen: {max_fit_this_gen:.2f}, Best Overall: {agent.best_fitness_overall:.2f}")
                
                agent.current_genome_idx = agent.population_size 
                agent.learn() 
                
                if hasattr(agent, 'save') and agent.best_genome_overall:
                    agent.save(model_to_save_path)
                
                overall_episode_rewards_or_fitness.append(max_fit_this_gen if max_fit_this_gen > -float('inf') else 0)
                if not args.render: 
                    for event in pg.event.get():
                        if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                            stop_training_global_ref[0] = True; print_f("NEAT training interrupted."); break
                if stop_training_global_ref[0]: break
        
        elif agent_name == 'genetic':
            pop_size = agent_hparams.get("population_size", 20) 
            num_generations = args.episodes 
            print_f(f"Genetic Algorithm: {num_generations} generations, Pop: {pop_size} (SERIAL EVALUATION per individual).")

            for gen_idx in range(num_generations):
                if stop_training_global_ref[0]: break
                print_f(f"Agent: GENETIC - Gen {gen_idx + 1}/{num_generations}")
                current_gen_fitness_scores = []

                for indiv_idx in range(pop_size):
                    if stop_training_global_ref[0]: break
                    current_individual_network = agent.population[indiv_idx] 
                    agent.current_individual_idx = indiv_idx 

                    if (indiv_idx > 0 and indiv_idx % (max(1, pop_size//5)) == 0) or indiv_idx == pop_size -1 :
                         print_f(f"    Evaluating GA Individual {indiv_idx + 1}/{pop_size}...")
                    
                    episode_info_str = f"GA Gen {gen_idx + 1} Indiv {indiv_idx + 1}"
                    fitness, _ = evaluate_single_genome_serially( 
                        current_individual_network, # For GA, 'genome' is the network
                        agent, # The main GeneticAgent instance
                        agent_hparams,
                        game_instance_main, 
                        args.max_steps_per_episode,
                        preprocessed_obs_shape, 
                        args.render, 
                        'genetic', # agent_type_for_eval
                        args.print_interval_steps, 
                        episode_info_str
                    )
                    agent.fitness_scores[indiv_idx] = fitness 
                    current_gen_fitness_scores.append(fitness)
                    
                    if args.render:
                        for event in pg.event.get():
                            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                                stop_training_global_ref[0] = True; print_f("GA training interrupted."); break
                    if stop_training_global_ref[0]: break
                if stop_training_global_ref[0]: break
                
                max_fit_this_gen_ga = -float('inf') if not current_gen_fitness_scores else max(current_gen_fitness_scores)
                if hasattr(agent,'current_individual_idx'): agent.current_individual_idx = pop_size # Ensure it's marked as all done
                agent.learn() 

                if hasattr(agent, 'save'): agent.save(model_to_save_path)
                overall_episode_rewards_or_fitness.append(max_fit_this_gen_ga if max_fit_this_gen_ga > -float('inf') else 0)
                
                if not args.render:
                    for event in pg.event.get():
                        if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                            stop_training_global_ref[0] = True; print_f("GA training interrupted."); break
                if stop_training_global_ref[0]: break

        else: # Serial RL agents (DQN, PPO, A2C)
            completed_normally = run_serial_agent_training(
                agent, agent_name, game_instance_main, args.episodes, 
                args.max_steps_per_episode, args.render, 
                args.print_interval_steps, model_to_save_path, 
                args.save_interval, agent_hparams, overall_episode_rewards_or_fitness, 
                stop_training_global_ref
            )
            if not completed_normally: print_f(f"Serial training for {agent_name.upper()} was interrupted.")

        if stop_training_global_ref[0]: 
            print_f(f"Stopping training for subsequent agents due to interruption or error.")
            break

    avg_overall_metric = np.mean(overall_episode_rewards_or_fitness) if overall_episode_rewards_or_fitness else 0.0
    print_f(f"\n--- Overall Training Stats ---")
    print_f(f"Rolling avg reward/max_fitness (last {len(overall_episode_rewards_or_fitness)} eps/gens): {avg_overall_metric:.2f}")
    print_f(f"--- Training Finished (or Interrupted) ---")
    
    if pg.get_init(): pg.quit()
    sys.exit()

if __name__ == '__main__':
    train_all_agents()