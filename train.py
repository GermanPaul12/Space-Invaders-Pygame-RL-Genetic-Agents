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
    from game.game import Game
    from game import config as game_config_module 
    from agents.agent import Agent
    from agents.random_agent import RandomAgent
    from agents.dqn_agent import DQNAgent # Preprocess is here
    from agents.a2c_agent import A2CAgent
    from agents.ppo_agent import PPOAgent
    from agents.genetic_agent import GeneticAgent
    from agents.neat_agent import NEATAgent # GenomeNEAT not directly used here now
    
    from utils.model_helpers import ( 
        get_existing_model_versions,
        get_next_model_save_path,
        get_latest_model_path,
        MODELS_DIR
    )
    from utils.training_helpers import evaluate_neat_genome_worker # Import the worker
except ImportError as e:
    print(f"Error importing files in train.py: {e}", flush=True)
    sys.exit(1)

ALL_AGENT_TYPES_AVAILABLE = ['dqn', 'ppo', 'a2c', 'genetic', 'neat', 'random'] 

def print_f(*args, **kwargs): print(*args, **kwargs); sys.stdout.flush()

def load_agent_config(config_path_cli, agent_name): # Renamed config_path to avoid clash
    # Use config_path_cli if provided, otherwise try default path
    config_to_try = config_path_cli
    if not config_to_try: # If CLI arg was None, construct default path
        default_config_filename = f"{agent_name}_default.json"
        config_to_try = os.path.join(os.path.dirname(__file__), "configs", default_config_filename)

    if config_to_try and os.path.exists(config_to_try):
        try:
            with open(config_to_try, 'r') as f:
                print_f(f"  Loading config for {agent_name} from: {os.path.basename(config_to_try)}")
                return json.load(f)
        except Exception as e:
            print_f(f"  Warning: Error loading {os.path.basename(config_to_try)}: {e}.")
    
    # Fallback message if either specified path was bad or default was missing/bad
    if config_path_cli and config_path_cli != config_to_try : # If a custom path was given but failed
         print_f(f"  Specified config '{os.path.basename(config_path_cli)}' not found or failed to load.")
    print_f(f"  Using agent's internal default hyperparameters for {agent_name}.")
    return {}


def train_all_agents():
    parser = argparse.ArgumentParser(description="Train Space Invaders AI Agents")
    default_trainable = ",".join(filter(lambda x: x != 'random', ALL_AGENT_TYPES_AVAILABLE))
    parser.add_argument("--agents", type=str, default=default_trainable)
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes for NNs; Generations for NEAT/GA.")
    parser.add_argument("--load_models", action='store_true')
    parser.add_argument("--force_train", action='store_true')
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--max_steps_per_episode", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--print_interval_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() - 8 if os.cpu_count() and os.cpu_count() > 1 else 1),
                        help="Worker processes for NEAT/GA parallel evaluation.")

    for agent_type_cfg in filter(lambda x: x != 'random', ALL_AGENT_TYPES_AVAILABLE):
        parser.add_argument(f"--{agent_type_cfg}_config_path", type=str, default=None,
                            help=f"Path to JSON config for {agent_type_cfg} (in 'configs/' dir).")
    args = parser.parse_args()
    
    valid_agents_to_train = [a.strip() for a in args.agents.split(',') if a.strip() in ALL_AGENT_TYPES_AVAILABLE and a != 'random']
    if not valid_agents_to_train: print_f("No valid agents. Exiting."); sys.exit(1)
    
    print_f(f"--- Starting Training Session (Saving to: {MODELS_DIR}) ---")
    print_f(f"Agents: {', '.join(valid_agents_to_train)}, Episodes/Generations: {args.episodes}")
    if any(agent in ['neat', 'genetic'] for agent in valid_agents_to_train):
        print_f(f"Parallel Workers for Pop-Based Agents: {args.num_workers}")
    # ... (other initial prints)

    game_instance_main = Game(silent_mode=True, ai_training_mode=True) # For serial agents or if render needed
    preprocessed_obs_shape = (1, 84, 84)
    action_size = game_instance_main.get_action_size()
    
    overall_episode_rewards = deque(maxlen=100)
    stop_training_global = False

    for agent_name in valid_agents_to_train:
        if stop_training_global: break
        print_f(f"\n--- Preparing Agent: {agent_name.upper()} ---")
        
        # Model path and loading/skipping logic (remains mostly the same)
        existing_versions = get_existing_model_versions(agent_name) # Uses MODELS_DIR from helpers
        model_to_load_path, model_to_save_path = None, None
        if existing_versions:
            print_f(f"  Found: {', '.join([os.path.basename(p) for p in existing_versions])}")
            if args.load_models:
                model_to_load_path = existing_versions[-1]; model_to_save_path = model_to_load_path
                print_f(f"  Continue. Load: {os.path.basename(model_to_load_path)}. Save to same.")
            elif not args.force_train: print_f(f"  Skip {agent_name}. Use --load_models or --force_train."); continue
            else: model_to_save_path = get_next_model_save_path(agent_name); print_f(f"  Force. New: {os.path.basename(model_to_save_path)}")
        else: model_to_save_path = get_next_model_save_path(agent_name); print_f(f"  No models. New: {os.path.basename(model_to_save_path)}")
        if not model_to_save_path and not model_to_load_path: print_f(f"  Path error for {agent_name}. Skip."); continue
            
        agent = None
        cli_config_path = getattr(args, f"{agent_name}_config_path", None)
        agent_hparams = load_agent_config(cli_config_path, agent_name)
        
        print_f(f"  Instantiating {agent_name} with params: {agent_hparams if agent_hparams else 'agent internal defaults'}")
        # Agent Instantiation
        if agent_name == 'dqn': agent = DQNAgent(action_size, preprocessed_obs_shape, **agent_hparams)
        elif agent_name == 'a2c': agent = A2CAgent(action_size, preprocessed_obs_shape, **agent_hparams)
        elif agent_name == 'ppo': agent = PPOAgent(action_size, preprocessed_obs_shape, **agent_hparams)
        elif agent_name == 'genetic': agent = GeneticAgent(action_size, preprocessed_obs_shape, **agent_hparams)
        elif agent_name == 'neat': agent = NEATAgent(action_size, preprocessed_obs_shape, **agent_hparams)
        if not agent: print_f(f"Failed to init {agent_name}. Skip."); continue

        # Model Loading
        if model_to_load_path and hasattr(agent, 'load'):
            try: agent.load(model_to_load_path) 
            except Exception as e:
                print_f(f"Load failed {os.path.basename(model_to_load_path)}: {e}. Fresh.")
                if model_to_save_path == model_to_load_path:
                    model_to_save_path = get_next_model_save_path(agent_name)
                    print_f(f"Adjusted save: {os.path.basename(model_to_save_path)}")
        if not model_to_save_path: model_to_save_path = get_next_model_save_path(agent_name) # Fallback

        print_f(f"--- Training: {agent_name.upper()} (Save: {os.path.basename(model_to_save_path)}) ---")
        
        episode_rewards_window = deque(maxlen=20)
        num_iterations = args.episodes # For NNs: episodes; For NEAT/GA: generations

        # --- NEAT Parallel Training Logic ---
        if agent_name == 'neat':
            print_f(f"NEAT: {num_iterations} generations, Workers: {args.num_workers}.")
            for gen_idx in range(num_iterations): # Loop over generations
                if stop_training_global: break
                print_f(f"Agent: NEAT - Gen {agent.current_generation + 1}/{num_iterations}")
                
                tasks_for_pool = [(deepcopy(genome), deepcopy(agent_hparams), action_size, 
                                   preprocessed_obs_shape, args.max_steps_per_episode) 
                                  for genome in agent.population]

                print_f(f"  NEAT Gen {agent.current_generation + 1}: Evaluating {len(agent.population)} genomes with {args.num_workers} processes...")
                fitness_scores = []
                try:
                    # Using 'spawn' context can be more robust with Pygame if workers init it
                    # However, our worker now sets SDL_VIDEODRIVER to dummy before pg import
                    # ctx = multiprocessing.get_context('spawn') # Try if default 'fork' has issues
                    # with ctx.Pool(processes=args.num_workers) as pool:
                    with multiprocessing.Pool(processes=args.num_workers) as pool:
                        fitness_scores = pool.map(evaluate_neat_genome_worker, tasks_for_pool)
                    
                    max_fit_this_gen = -float('inf')
                    for i, fitness in enumerate(fitness_scores):
                        agent.population[i].fitness = fitness
                        if fitness > max_fit_this_gen: max_fit_this_gen = fitness
                        if fitness > agent.best_fitness_overall: # Update global best
                            agent.best_fitness_overall = fitness
                            agent.best_genome_overall = agent.population[i].clone()
                    
                    print_f(f"  NEAT Gen {agent.current_generation + 1}: Eval complete. MaxFit Gen: {max_fit_this_gen:.2f}, Best Overall: {agent.best_fitness_overall:.2f}")
                    
                    agent.current_genome_idx = agent.population_size # Signal all evaluated
                    agent.learn() # Evolution step (speciation, crossover, mutation)

                    if hasattr(agent, 'save'): agent.save(model_to_save_path) # Save best of gen
                    overall_episode_rewards.append(max_fit_this_gen) # Log max fitness of this "generation"

                except Exception as e_mp:
                    print_f(f"  NEAT: Multiprocessing error: {e_mp}"); stop_training_global = True
                
                for event in pg.event.get(): # Check for quit events between generations
                    if event.type == pg.QUIT or \
                       (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                        stop_training_global = True; print_f("NEAT training interrupted."); break
                if stop_training_global: break
        
        # --- Serial Training Loop for other agents (DQN, A2C, PPO, Genetic) ---
        else: # agent_name is not 'neat' or 'random'
            # For Genetic, num_iterations is also generations.
            # args.episodes for Genetic still means N individuals evaluated before learn()
            # The "episode_idx" loop in train.py will run args.episodes times.
            # GeneticAgent's learn() is called when its internal current_individual_idx >= population_size.
            
            total_steps_for_current_agent = agent.steps_done if hasattr(agent, 'steps_done') and model_to_load_path else 0

            for episode_idx in range(num_iterations):
                if stop_training_global: break
                observation = game_instance_main.reset_for_ai()
                current_episode_reward,current_episode_total_loss,current_episode_learn_steps,current_step_in_episode = 0,0,0,0
                
                pop_size_ga = agent_hparams.get("population_size", 20) if agent_name == 'genetic' else 0
                if agent_name == 'genetic': print_f(f"Agent: GENETIC - Gen {episode_idx // pop_size_ga + 1}, Indiv {agent.current_individual_idx + 1}/{pop_size_ga}")

                for step in range(args.max_steps_per_episode):
                    current_step_in_episode = step + 1
                    stop_episode_early = False
                    for event in pg.event.get(): # Handle events
                        if event.type == pg.QUIT: stop_episode_early=True; stop_training_global=True; print_f("Quit."); break
                        if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: stop_episode_early=True; stop_training_global=True; print_f("ESC."); break
                    if stop_episode_early: break
                    
                    action = agent.choose_action(observation)
                    next_observation, reward, done, info = game_instance_main.step_ai(action)
                    current_episode_reward += reward
                    if agent_name == 'dqn': total_steps_for_current_agent +=1 # DQN's internal steps_done also increments
                    
                    loss_val = None
                    if agent_name == 'dqn':
                        agent.store_transition(observation, action, next_observation, reward, done)
                        if agent.steps_done > agent_hparams.get("batch_size", 32) * 5: loss_val = agent.learn()
                    elif agent_name == 'a2c':
                        agent.store_outcome(reward, done); loss_val = agent.learn(next_observation if not done else None)
                    elif agent_name == 'ppo':
                        loss_val = agent.store_transition_outcome(reward, done, next_observation)
                    
                    if loss_val is not None: current_episode_total_loss += loss_val; current_episode_learn_steps += 1
                    observation = next_observation
                    if args.render: game_instance_main.render_for_ai()
                    if args.print_interval_steps > 0 and (step + 1) % args.print_interval_steps == 0 and current_episode_learn_steps > 0:
                        avg_step_loss = current_episode_total_loss / current_episode_learn_steps
                        print_f(f"  {agent_name.upper()} Ep {episode_idx+1}, Step {step+1}: AvgStepLoss={avg_step_loss:.4f}, CurRwd={current_episode_reward:.0f}")
                        current_episode_total_loss,current_episode_learn_steps = 0,0
                    if done: break
                if stop_episode_early or stop_training_global: break
                
                episode_rewards_window.append(current_episode_reward)
                overall_episode_rewards.append(current_episode_reward)
                avg_reward_window = np.mean(episode_rewards_window) if episode_rewards_window else 0.0
                avg_loss_seg = (current_episode_total_loss / current_episode_learn_steps) if current_episode_learn_steps > 0 else 0.0
                game_score = info.get('score',0)
                print_f(f"Agent: {agent_name.upper()} - Ep {episode_idx + 1}/{num_iterations} DONE. R={current_episode_reward:.0f}, Steps={current_step_in_episode}, AvgLoss={avg_loss_seg:.4f}, Score={game_score}, AvgR(last {len(episode_rewards_window)})={avg_reward_window:.2f}")

                if agent_name == 'genetic':
                    agent.record_fitness(game_score)
                    if agent.current_individual_idx >= pop_size_ga:
                        agent.learn(); agent.save(model_to_save_path) # GA saves per gen
                
                if args.save_interval > 0 and (episode_idx + 1) % args.save_interval == 0:
                    if agent_name not in ['genetic', 'neat', 'random'] and hasattr(agent, 'save'):
                        agent.save(model_to_save_path)
            
            # Final save for serial agents
            if not stop_training_global and hasattr(agent, 'save'):
                if agent_name not in ['genetic', 'neat', 'random']: # Genetic/NEAT save per gen
                    print_f(f"Completed training for {agent_name}. Final save to {os.path.basename(model_to_save_path)}")
                    agent.save(model_to_save_path)
                # For Genetic, if training ends mid-generation, last gen save might be it.
                elif agent_name == 'genetic' and not ((episode_idx +1) % pop_size_ga == 0):
                     print_f(f"Genetic training ended. Ensuring final save of best: {os.path.basename(model_to_save_path)}")
                     agent.save(model_to_save_path)


        if stop_training_global: print_f(f"Training for {agent_name.upper()} interrupted."); # Outer loop will break

    avg_overall_reward = np.mean(overall_episode_rewards) if overall_episode_rewards else 0.0
    print_f(f"\n--- Overall Training Stats ---"); print_f(f"Rolling avg reward (last {len(overall_episode_rewards)} eps): {avg_overall_reward:.2f}")
    print_f(f"--- Training Finished (or Interrupted) ---"); 
    if pg.get_init(): pg.quit()
    sys.exit()

if __name__ == '__main__':
    # Crucial for Pygame + multiprocessing, especially on non-Linux or if 'fork' causes issues
    # 'spawn' is generally safer as it doesn't inherit as much from the parent.
    if sys.platform.startswith('win') or sys.platform.startswith('darwin'): # Or just always use 'spawn'
        multiprocessing.set_start_method('spawn', force=True) 
    elif not multiprocessing.get_start_method(allow_none=True): # If no method set and not win/mac, try spawn
         try:
            multiprocessing.set_start_method('spawn', force=True)
         except RuntimeError: # Already set or can't change
            pass
    train_all_agents()