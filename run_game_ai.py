# run_game_ai.py
import pygame as pg
import argparse
import sys
import os
import numpy as np
from collections import deque # For evaluate_all example

try:
    from game.game import Game
    from game import config
    from agents.agent import Agent 
    from agents.random_agent import RandomAgent
    from agents.dqn_agent import DQNAgent # preprocess_observation is in dqn_agent
    from agents.a2c_agent import A2CAgent
    from agents.ppo_agent import PPOAgent
    from agents.genetic_agent import GeneticAgent
except ImportError as e:
    print(f"Error importing game or agent files in run_game_ai.py: {e}", flush=True)
    print("Ensure all files are correctly placed and PyTorch is installed if needed.", flush=True)
    sys.exit(1)

# --- Model Versioning Helper Functions (Copied from train.py or import if centralized) ---
MODELS_DIR = "trained_models" # Relative to this script's location (project root)
BASE_MODEL_FILENAME_TEMPLATE = "{agent_name}_spaceinvaders"

def print_f(*args, **kwargs): # For consistent flushing
    print(*args, **kwargs)
    sys.stdout.flush()

def get_existing_model_versions(agent_name):
    if not os.path.exists(MODELS_DIR):
        return []
    pattern_base = BASE_MODEL_FILENAME_TEMPLATE.format(agent_name=agent_name)
    versions = []
    for f_name in sorted(os.listdir(MODELS_DIR)): 
        if f_name.startswith(pattern_base) and f_name.endswith(".pth"):
            versions.append(os.path.join(MODELS_DIR, f_name)) # Store full path
    return versions

def get_next_model_save_path(agent_name): # Used for single agent training mode in this script
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    pattern_base = BASE_MODEL_FILENAME_TEMPLATE.format(agent_name=agent_name)
    base_path = os.path.join(MODELS_DIR, f"{pattern_base}.pth")
    if not os.path.exists(base_path):
        return base_path
    version = 2
    while True:
        versioned_path = os.path.join(MODELS_DIR, f"{pattern_base}_v{version}.pth")
        if not os.path.exists(versioned_path):
            return versioned_path
        version += 1

def get_latest_model_path(agent_name):
    versions = get_existing_model_versions(agent_name)
    return versions[-1] if versions else None
# --- End Model Versioning Helpers ---

# List of all known non-random agent types for evaluate_all
TRAINABLE_AGENT_TYPES = ['dqn', 'ppo', 'a2c', 'genetic']


def run_ai_operations(): # Renamed main function
    parser = argparse.ArgumentParser(description="Space Invaders AI Agent - Runner/Tester/Single-Trainer")
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'evaluate_all'], required=True,
                        help="Mode of operation: train (single agent), test, evaluate_all.")
    parser.add_argument("--agent", type=str, 
                        choices=TRAINABLE_AGENT_TYPES + ['random'], 
                        help="Agent to use (required for 'train' and 'test' modes).")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes for training/testing AI.")
    parser.add_argument("--load_model", action='store_true', help="Load the LATEST pre-trained model (used if --model_file_path not set).")
    parser.add_argument("--model_file_path", type=str, default=None, help="Specify an exact model file path to load.")
    parser.add_argument("--render", action='store_true', help="Render game during AI operations.")
    parser.add_argument("--max_steps_per_episode", type=int, default=2000, help="Max steps per episode for AI.")
    parser.add_argument("--silent", action='store_true', help="Run the game without sounds.")
    # Add args specific to single agent training if needed, e.g. --save_interval_run
    # For now, single agent training via this script will save at the end.

    args = parser.parse_args()

    if args.mode in ['train', 'test'] and not args.agent:
        parser.error(f"--agent is required for mode '{args.mode}'.")
    
    # Determine ai_training_mode for Game instance
    # For this script, 'train' mode implies ai_training_mode for speed.
    # 'test' and 'evaluate_all' might want human-like delays unless specifically overridden.
    # However, main.py now controls this. Let's assume ai_training_mode is ON for all AI ops here for speed.
    # Or, add another flag if fine-grained control is needed.
    # For now, let's make all AI ops from this script use ai_training_mode for speed.
    game_instance = Game(silent_mode=args.silent, ai_training_mode=True) 
    
    print_f(f"--- run_game_ai.py: Starting AI Operation ---")
    print_f(f"Mode: {args.mode}, Agent: {args.agent if args.agent else 'N/A for evaluate_all initial'}")
    print_f(f"Episodes: {args.episodes}, Render: {args.render}, Max Steps: {args.max_steps_per_episode}")
    print_f(f"Silent sounds: {args.silent}, AI Speedups: True")


    preprocessed_obs_shape = (1, 84, 84) 
    action_size = game_instance.get_action_size()
    agent_instance = None # Renamed from 'agent' to avoid conflict with args.agent

    # --- Agent Instantiation and Model Loading (common for train/test) ---
    if args.mode in ['train', 'test']:
        agent_name_local = args.agent # To avoid confusion with the agent object
        
        # Instantiate the chosen agent
        if agent_name_local == 'random':
            agent_instance = RandomAgent(action_size)
        elif agent_name_local == 'dqn':
            agent_instance = DQNAgent(action_size, preprocessed_obs_shape, buffer_size=50000, batch_size=32, gamma=0.99, lr=1e-4, target_update_freq=1000, eps_decay=100000)
        elif agent_name_local == 'a2c':
            agent_instance = A2CAgent(action_size, preprocessed_obs_shape, lr=7e-4, gamma=0.99)
        elif agent_name_local == 'ppo':
            agent_instance = PPOAgent(action_size, preprocessed_obs_shape, lr=2.5e-4, gamma=0.99, trajectory_n_steps=128, ppo_epochs=4, mini_batch_size=32)
        elif agent_name_local == 'genetic':
            # GA might need different episode handling if training here vs train.py
            agent_instance = GeneticAgent(action_size, preprocessed_obs_shape, population_size=20, num_elites=2, mutation_rate=0.1)
        else:
            print_f(f"Error: Agent '{agent_name_local}' not implemented in run_game_ai.py.")
            pg.quit(); sys.exit(1)

        # Determine model path to load
        path_to_load = None
        if args.model_file_path:
            if os.path.exists(args.model_file_path):
                path_to_load = args.model_file_path
            else:
                print_f(f"Warning: Specified model file not found: {args.model_file_path}")
        elif args.load_model and agent_name_local != 'random':
            path_to_load = get_latest_model_path(agent_name_local)
            if not path_to_load:
                print_f(f"Warning: --load_model specified, but no models found for agent {agent_name_local}.")

        # Load the model if a path was determined
        if path_to_load and hasattr(agent_instance, 'load') and agent_name_local != 'random':
            try:
                agent_instance.load(path_to_load) # Agent's load should print success msg
            except Exception as e:
                print_f(f"Could not load model for {agent_name_local} from {os.path.basename(path_to_load)}: {e}")
                if args.mode == 'test': print_f("Testing will proceed with an untrained agent.")
        elif (args.load_model or args.model_file_path) and agent_name_local != 'random' and not path_to_load:
             print_f(f"Model loading requested for {agent_name_local} but no valid model path determined/found.")


    # --- Mode-Specific Logic ---
    if args.mode == 'train': # Single agent training mode
        if not agent_instance or args.agent == 'random':
            print_f("Cannot train a random agent or agent instantiation failed. Exiting train mode.")
            pg.quit(); sys.exit(1)

        print_f(f"--- Training single agent: {args.agent} for {args.episodes} episodes ---")
        
        # Determine save path: if loaded, save to same path. If fresh, save to next version.
        save_path_single_train = path_to_load if path_to_load else get_next_model_save_path(args.agent)
        print_f(f"Model will be saved to: {os.path.basename(save_path_single_train)}")

        total_steps_overall = agent_instance.steps_done if hasattr(agent_instance, 'steps_done') and path_to_load else 0
        
        for episode in range(args.episodes):
            # ... (Standard training loop for a single agent, similar to train.py's inner loop) ...
            # This is a condensed version. You'd copy the detailed loop from train.py here if needed.
            observation = game_instance.reset_for_ai()
            episode_reward = 0; episode_loss = 0; num_learn_steps = 0; stop_script=False

            if args.agent == 'genetic': # Handle GA specific prints/logic if training here
                 print_f(f"  Gen {episode // agent_instance.population_size +1}, Indiv {agent_instance.current_individual_idx + 1}/{agent_instance.population_size}")

            for step in range(args.max_steps_per_episode):
                for event in pg.event.get():
                    if event.type == pg.QUIT: print_f("Quit event received."); stop_script=True; break
                if stop_script: break
                
                action = agent_instance.choose_action(observation)
                next_observation, reward, done, info = game_instance.step_ai(action)
                episode_reward += reward
                total_steps_overall +=1

                loss_val = None
                if args.agent == 'dqn':
                    agent_instance.store_transition(observation, action, next_observation, reward, done)
                    if total_steps_overall > agent_instance.batch_size * 5 : loss_val = agent_instance.learn()
                elif args.agent == 'a2c':
                    agent_instance.store_outcome(reward, done); loss_val = agent_instance.learn(next_observation if not done else None)
                elif args.agent == 'ppo':
                    loss_val = agent_instance.store_transition_outcome(reward, done, next_observation)
                
                if loss_val is not None: episode_loss += loss_val; num_learn_steps += 1
                observation = next_observation
                if args.render: game_instance.render_for_ai()
                if done: break
            if stop_script: break
            
            avg_loss = (episode_loss / num_learn_steps) if num_learn_steps > 0 else 0
            print_f(f"  Ep {episode + 1}: Rwd={episode_reward}, Steps={step+1}, AvgLoss={avg_loss:.4f}, Score={info.get('score',0)}")

            if args.agent == 'genetic':
                agent_instance.record_fitness(info.get('score', 0))
                if agent_instance.current_individual_idx >= agent_instance.population_size:
                    agent_instance.learn(); agent_instance.save(save_path_single_train)
            
            # Simplified periodic save for single agent training here
            if (episode + 1) % 50 == 0 and args.agent not in ['random', 'genetic']:
                agent_instance.save(save_path_single_train)
        
        if args.agent not in ['random', 'genetic'] and not stop_script:
            agent_instance.save(save_path_single_train)
        elif args.agent == 'genetic' and not stop_script: # Ensure final GA save
            agent_instance.save(save_path_single_train)


    elif args.mode == 'test':
        if not agent_instance:
            print_f("Agent for testing not instantiated. Exiting test mode.")
            pg.quit(); sys.exit(1)

        print_f(f"--- Testing {args.agent} for {args.episodes} episodes ---")
        all_rewards, all_scores = [], []
        for episode in range(args.episodes):
            observation = game_instance.reset_for_ai()
            episode_reward, score, stop_script = 0, 0, False
            for step in range(args.max_steps_per_episode):
                for event in pg.event.get():
                    if event.type == pg.QUIT: print_f("Quit event received."); stop_script=True; break
                if stop_script: break

                action = agent_instance.choose_action(observation) # Add is_eval=True if agent needs it
                next_observation, reward, done, info = game_instance.step_ai(action)
                episode_reward += reward
                observation = next_observation
                if args.render: game_instance.render_for_ai()
                if done: score = info.get('score',0); break
            if stop_script: break
            all_rewards.append(episode_reward); all_scores.append(score)
            print_f(f"  Test Ep {episode + 1}: Reward={episode_reward}, Score={score}")
        
        if not stop_script and all_rewards: # Only print if not interrupted and data exists
            print_f(f"--- Test Results for {args.agent} ---")
            print_f(f"  Avg Reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
            print_f(f"  Avg Score: {np.mean(all_scores):.2f} +/- {np.std(all_scores):.2f}")


    elif args.mode == 'evaluate_all':
        print_f("--- Evaluating ALL AGENTS (Latest Versions) ---")
        agents_to_evaluate = TRAINABLE_AGENT_TYPES + ['random'] # All known trainable + random
        
        for agent_to_eval_name in agents_to_evaluate:
            print_f(f"\n--- Evaluating: {agent_to_eval_name.upper()} ---")
            eval_agent_instance = None # Instantiate fresh for each eval

            if agent_to_eval_name == 'random':
                eval_agent_instance = RandomAgent(action_size)
            else: # Trainable agents
                latest_model_path = get_latest_model_path(agent_to_eval_name)
                if not latest_model_path:
                    print_f(f"  No model found for {agent_to_eval_name}. Skipping evaluation.")
                    continue
                
                if agent_to_eval_name == 'dqn': eval_agent_instance = DQNAgent(action_size, preprocessed_obs_shape, ...) # Fill params
                elif agent_to_eval_name == 'a2c': eval_agent_instance = A2CAgent(action_size, preprocessed_obs_shape, ...)
                elif agent_to_eval_name == 'ppo': eval_agent_instance = PPOAgent(action_size, preprocessed_obs_shape, ...)
                elif agent_to_eval_name == 'genetic': eval_agent_instance = GeneticAgent(action_size, preprocessed_obs_shape, ...)
                else: print_f(f"  Unknown agent {agent_to_eval_name} for eval. Skipping."); continue
                
                try:
                    eval_agent_instance.load(latest_model_path)
                except Exception as e:
                    print_f(f"  Could not load model {os.path.basename(latest_model_path)} for {agent_to_eval_name}: {e}. Skipping.")
                    continue
            
            # Evaluation loop (similar to test mode)
            eval_rewards, eval_scores = [], []
            num_eval_episodes = args.episodes # Use the --episodes arg for eval count
            print_f(f"  Running {num_eval_episodes} evaluation episodes...")
            for ep_idx in range(num_eval_episodes):
                obs = game_instance.reset_for_ai()
                ep_r, current_score, stop_script = 0,0,False
                for _ in range(args.max_steps_per_episode):
                    for event in pg.event.get():
                        if event.type == pg.QUIT: print_f("Quit event."); stop_script=True; break
                    if stop_script: break
                    
                    # For evaluation, ensure deterministic behavior if agent supports it (e.g., DQN epsilon=0)
                    action_eval = None
                    if hasattr(eval_agent_instance, 'eps_start') and agent_to_eval_name == 'dqn':
                        original_eps = eval_agent_instance.eps_start
                        eval_agent_instance.eps_start = 0.0 # Greedy for DQN eval
                        action_eval = eval_agent_instance.choose_action(obs)
                        eval_agent_instance.eps_start = original_eps
                    else:
                        action_eval = eval_agent_instance.choose_action(obs)

                    next_obs, r, d, inf = game_instance.step_ai(action_eval)
                    ep_r += r; obs = next_obs
                    if args.render: game_instance.render_for_ai()
                    if d: current_score = inf.get('score',0); break
                if stop_script: break
                eval_rewards.append(ep_r); eval_scores.append(current_score)
            if stop_script: print_f("Evaluation interrupted."); break # Break from agents loop

            if eval_rewards: # Print stats if not interrupted early
                print_f(f"  Results for {agent_to_eval_name} ({num_eval_episodes} eps):")
                print_f(f"    Avg Reward: {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")
                print_f(f"    Avg Score: {np.mean(eval_scores):.2f} +/- {np.std(eval_scores):.2f}")

    pg.quit()
    sys.exit()

if __name__ == '__main__':
    run_ai_operations()