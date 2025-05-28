# train.py
import pygame as pg
import argparse
import sys
import os
import numpy as np
from collections import deque
import json
import multiprocessing
from copy import deepcopy

try:
    from game.game_manager import Game 
    from game import config as game_config_module 
    
    from utils.model_helpers import ( 
        get_existing_model_versions, get_next_model_save_path,
        get_latest_model_path, MODELS_DIR
    )
    from utils.training_helpers import evaluate_neat_genome_worker
    # Import ALL_AGENT_TYPES_AVAILABLE from cli_args if needed for filtering args.agents
    from utils.cli_args import add_common_training_args, add_agent_config_path_args, ALL_AGENT_TYPES_AVAILABLE
    from utils.agent_factory import create_agent
except ImportError as e:
    print(f"Error importing files in train.py: {e}", flush=True)
    sys.exit(1)

# ALL_AGENT_TYPES_AVAILABLE is now imported from utils.cli_args

def print_f(*args, **kwargs): print(*args, **kwargs); sys.stdout.flush()

def load_agent_config(config_path_cli, agent_name):
    config_to_try = config_path_cli
    if not config_to_try: 
        default_config_filename = f"{agent_name}_default.json"
        # Assuming configs are in 'configs' directory at project root
        config_to_try = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", default_config_filename) if os.path.basename(os.path.dirname(__file__)) == "utils" else os.path.join(os.path.dirname(__file__), "configs", default_config_filename)

    if config_to_try and os.path.exists(config_to_try):
        try:
            with open(config_to_try, 'r') as f:
                print_f(f"  Loading config for {agent_name} from: {os.path.basename(config_to_try)}")
                return json.load(f)
        except Exception as e: print_f(f"  Warning: Error loading {os.path.basename(config_to_try)}: {e}.")
    if config_path_cli and config_path_cli != config_to_try :
         print_f(f"  Specified config '{os.path.basename(config_path_cli)}' not found or failed to load.")
    print_f(f"  Using agent's internal default hyperparameters for {agent_name}.")
    return {}


def train_all_agents():
    parser = argparse.ArgumentParser(description="Train Space Invaders AI Agents")
    # Corrected calls to cli_args functions
    parser = add_common_training_args(parser) # Now only takes parser
    parser = add_agent_config_path_args(parser) # Now only takes parser
    args = parser.parse_args()
    
    # We still use ALL_AGENT_TYPES_AVAILABLE (imported from cli_args) for validation here
    valid_agents_to_train = [a.strip() for a in args.agents.split(',') if a.strip() in ALL_AGENT_TYPES_AVAILABLE and a != 'random']
    if not valid_agents_to_train: print_f("No valid agents. Exiting."); sys.exit(1)
    
    print_f(f"--- Starting Training Session (Saving to: {MODELS_DIR}) ---")
    print_f(f"Agents: {', '.join(valid_agents_to_train)}, Episodes/Generations: {args.episodes}")
    if any(agent in ['neat', 'genetic'] for agent in valid_agents_to_train):
        print_f(f"Parallel Workers for Pop-Based Agents: {args.num_workers}")
    print_f(f"Load LATEST models: {args.load_models}")
    print_f(f"Force train (new version): {args.force_train}")
    print_f(f"Render game content: {args.render}")
    print_f(f"Max steps per episode/eval: {args.max_steps_per_episode}")
    print_f(f"Save interval (NNs): {args.save_interval} episodes")
    print_f(f"Print interval (steps): {args.print_interval_steps}")
    print_f(f"AI Training Mode (in Game): ON")
    print_f("---------------------------------")


    game_instance_main = Game(silent_mode=True, ai_training_mode=True)
    preprocessed_obs_shape = (1, 84, 84) # Standard for CNN agents
    action_size = game_instance_main.get_action_size()
    
    overall_episode_rewards = deque(maxlen=100)
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
                print_f(f"Load failed {os.path.basename(model_to_load_path)}: {e}. Fresh.")
                if model_to_save_path == model_to_load_path:
                    model_to_save_path = get_next_model_save_path(agent_name)
                    print_f(f"Adjusted save: {os.path.basename(model_to_save_path)}")
        if not model_to_save_path: model_to_save_path = get_next_model_save_path(agent_name)

        print_f(f"--- Training: {agent_name.upper()} (Save: {os.path.basename(model_to_save_path)}) ---")
        
        episode_rewards_window = deque(maxlen=20)
        num_iterations = args.episodes

        if agent_name == 'neat':
            print_f(f"NEAT: {num_iterations} generations, Workers: {args.num_workers}.")
            for gen_idx in range(num_iterations): 
                if stop_training_global_ref[0]: break
                print_f(f"Agent: NEAT - Gen {agent.current_generation + 1}/{num_iterations}")
                # Ensure deepcopy agent_hparams if they can be modified by worker (unlikely here)
                # The genome itself is more critical for deepcopy.
                tasks_for_pool = [(deepcopy(genome), agent_hparams, action_size, preprocessed_obs_shape, args.max_steps_per_episode) for genome in agent.population]
                print_f(f"  NEAT Gen {agent.current_generation + 1}: Evaluating {len(agent.population)} genomes with {args.num_workers} processes...")
                fitness_scores = []
                try:
                    # Ensure worker processes handle Pygame display initialization correctly (headless)
                    with multiprocessing.Pool(processes=args.num_workers) as pool: fitness_scores = pool.map(evaluate_neat_genome_worker, tasks_for_pool)
                    
                    max_fit_this_gen = -float('inf') if not fitness_scores else max(fitness_scores)
                    
                    # Assign fitness and update best overall
                    for i, fitness in enumerate(fitness_scores):
                        agent.population[i].fitness = fitness
                        if fitness > agent.best_fitness_overall: 
                            agent.best_fitness_overall = fitness
                            # Best genome should be a clone from the evaluated population member
                            agent.best_genome_overall = agent.population[i].clone() 
                                                        
                    print_f(f"  NEAT Gen {agent.current_generation + 1}: Eval complete. MaxFit Gen: {max_fit_this_gen:.2f}, Best Overall: {agent.best_fitness_overall:.2f}")
                    
                    agent.current_genome_idx = agent.population_size # Signal all evaluated
                    agent.learn() # NEAT evolution step
                    
                    if hasattr(agent, 'save') and agent.best_genome_overall: # Save if a best genome exists
                        agent.save(model_to_save_path)
                    
                    overall_episode_rewards.append(max_fit_this_gen if max_fit_this_gen > -float('inf') else 0) # Add valid fitness

                except Exception as e_mp: 
                    print_f(f"  NEAT: Multiprocessing or worker error: {e_mp}")
                    stop_training_global_ref[0] = True # Stop further training if MP fails
                
                # Event handling for interruption (ESC key)
                for event in pg.event.get(): # Pygame events must be pumped
                    if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                        stop_training_global_ref[0] = True
                        print_f("NEAT training interrupted by user.")
                        break
                if stop_training_global_ref[0]: break
        
        else: # Serial agents (DQN, PPO, A2C, Genetic)
            from utils.serial_training_loop import run_serial_agent_training
            completed_normally = run_serial_agent_training(
                agent, agent_name, game_instance_main, args.episodes, 
                args.max_steps_per_episode, args.render, 
                args.print_interval_steps, model_to_save_path, 
                args.save_interval, agent_hparams, overall_episode_rewards, 
                stop_training_global_ref
            )
            if not completed_normally: 
                print_f(f"Serial training for {agent_name.upper()} was interrupted.")

        if stop_training_global_ref[0]: 
            print_f(f"Stopping training for subsequent agents due to interruption or error.")
            break

    avg_overall_reward = np.mean(overall_episode_rewards) if overall_episode_rewards else 0.0
    print_f(f"\n--- Overall Training Stats ---")
    print_f(f"Rolling avg reward (last {len(overall_episode_rewards)} eps/gens): {avg_overall_reward:.2f}")
    print_f(f"--- Training Finished (or Interrupted) ---")
    
    if pg.get_init(): pg.quit()
    sys.exit()

if __name__ == '__main__':
    if sys.platform.startswith('win') or sys.platform.startswith('darwin') :
        # Ensure 'spawn' start method for multiprocessing compatibility on Windows/macOS
        # if Pygame or other non-fork-safe libraries are used in subprocesses.
        current_start_method = multiprocessing.get_start_method(allow_none=True)
        if current_start_method != 'spawn':
            try:
                multiprocessing.set_start_method('spawn', force=True)
                print_f("Set multiprocessing start method to 'spawn'.")
            except RuntimeError as e:
                # This can happen if context is already set or if force=True isn't enough.
                # It's a warning because the program might still run, or it might fail later in MP.
                print_f(f"Warning: Could not set multiprocessing start method to 'spawn'. Current: {current_start_method}. Error: {e}")
    train_all_agents()