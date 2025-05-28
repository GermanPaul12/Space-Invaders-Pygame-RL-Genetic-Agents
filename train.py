# train.py
import pygame as pg
import argparse
import sys
import os
import numpy as np
from collections import deque
<<<<<<< HEAD
import json # For loading config files

try:
    from game.game import Game
    from game import config as game_config_module 
=======
import glob # For finding model files

try:
    from game.game import Game
    from game import config
>>>>>>> c1e544472a5b20e2ac46dbdf0a24ddcbded062d8
    from agents.agent import Agent
    from agents.random_agent import RandomAgent
    from agents.dqn_agent import DQNAgent
    from agents.a2c_agent import A2CAgent
    from agents.ppo_agent import PPOAgent
    from agents.genetic_agent import GeneticAgent
    from agents.neat_agent import NEATAgent
    from utils.model_helpers import ( 
        get_existing_model_versions,
        get_next_model_save_path,
        get_latest_model_path,
        MODELS_DIR # Use MODELS_DIR from helpers ("models")
    )
except ImportError as e:
<<<<<<< HEAD
    print(f"Error importing files in train.py: {e}", flush=True)
    sys.exit(1)

# ALL_AGENT_TYPES_AVAILABLE is defined below for argparse choices
ALL_AGENT_TYPES_AVAILABLE = ['dqn', 'ppo', 'a2c', 'genetic', 'neat', 'random'] 

def print_f(*args, **kwargs): print(*args, **kwargs); sys.stdout.flush()

def load_agent_config(config_path, agent_name):
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                print_f(f"  Loading config for {agent_name} from: {os.path.basename(config_path)}")
                return json.load(f)
        except Exception as e:
            print_f(f"  Warning: Could not load/parse config {os.path.basename(config_path)} for {agent_name}: {e}. Using internal defaults.")
    else:
        # If config_path is None (meaning user selected default, but default file might be missing, or no config was passed from main.py)
        default_config_filename = f"{agent_name}_default.json"
        default_config_path = os.path.join(os.path.dirname(__file__), "configs", default_config_filename) # Correct path to configs/
        if os.path.exists(default_config_path):
            try:
                with open(default_config_path, 'r') as f:
                    print_f(f"  Loading DEFAULT config for {agent_name} from: {default_config_filename}")
                    return json.load(f)
            except Exception as e:
                 print_f(f"  Warning: Could not load/parse DEFAULT config {default_config_filename}: {e}. Using internal defaults.")
        else:
            print_f(f"  Warning: Config path for {agent_name} not specified AND default config '{default_config_filename}' not found in 'configs/'. Using agent's internal defaults.")
    return {}

def train_all_agents():
    parser = argparse.ArgumentParser(description="Train Space Invaders AI Agents")
    default_trainable = ",".join(filter(lambda x: x != 'random', ALL_AGENT_TYPES_AVAILABLE))
    parser.add_argument("--agents", type=str, default=default_trainable)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--load_models", action='store_true')
    parser.add_argument("--force_train", action='store_true')
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--max_steps_per_episode", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--print_interval_steps", type=int, default=500)
    for agent_type in filter(lambda x: x != 'random', ALL_AGENT_TYPES_AVAILABLE):
        parser.add_argument(f"--{agent_type}_config_path", type=str, default=None)
=======
    print(f"Error importing game or agent files: {e}", flush=True)
    print("Ensure all files are correctly placed and PyTorch is installed if needed for specific agents.", flush=True)
    sys.exit(1)

MODELS_DIR = "trained_models"
BASE_MODEL_FILENAME_TEMPLATE = "{agent_name}_spaceinvaders" # No extension

# This should be a constant list of all agent types your system knows about.
ALL_AGENT_TYPES_AVAILABLE = ['dqn', 'ppo', 'a2c', 'genetic', 'random']

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def print_f(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# --- Model Versioning Helper Functions ---
def get_existing_model_versions(agent_name):
    if not os.path.exists(MODELS_DIR):
        return []
    pattern_base = BASE_MODEL_FILENAME_TEMPLATE.format(agent_name=agent_name)
    versions = []
    # Ensure consistent sorting for "latest" if relying on last element
    for f_name in sorted(os.listdir(MODELS_DIR)): 
        if f_name.startswith(pattern_base) and f_name.endswith(".pth"):
            versions.append(os.path.join(MODELS_DIR, f_name))
    return versions

def get_next_model_save_path(agent_name):
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


def train_all_agents():
    parser = argparse.ArgumentParser(description="Train multiple Space Invaders AI Agents")
    parser.add_argument("--agents", type=str, 
                        default=",".join(filter(lambda x: x != 'random', ALL_AGENT_TYPES_AVAILABLE)), # Default to all trainable
                        help=f"Comma-separated list of agents to train from: {', '.join(ALL_AGENT_TYPES_AVAILABLE)}")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train each agent.")
    parser.add_argument("--load_models", action='store_true', 
                        help="Load the LATEST pre-existing model to continue training. If not set and ANY model version exists, training for that agent will be skipped unless --force_train is used.")
    parser.add_argument("--force_train", action='store_true',
                        help="Force training (creates a new versioned model) even if model(s) exist and --load_models is not set.")
    parser.add_argument("--render", action='store_true', help="Render game content visually during training.")
    parser.add_argument("--max_steps_per_episode", type=int, default=2000, help="Max steps per episode for AI.")
    parser.add_argument("--save_interval", type=int, default=50, help="Save model every N episodes (0 to disable intermediate saves).")
    parser.add_argument("--print_interval_steps", type=int, default=500, 
                        help="Print average loss and other stats every N steps within an episode (0 to disable).")
    
>>>>>>> c1e544472a5b20e2ac46dbdf0a24ddcbded062d8
    args = parser.parse_args()
    
    valid_agents_to_train = [a.strip() for a in args.agents.split(',') if a.strip() in ALL_AGENT_TYPES_AVAILABLE and a != 'random']
    if not valid_agents_to_train: print_f("No valid agents. Exiting."); sys.exit(1)
    
    print_f(f"--- Starting Training Session (MODELS_DIR: {MODELS_DIR}) ---") # Confirm MODELS_DIR
    # ... (initial prints as before) ...

<<<<<<< HEAD
=======
    # Parse the --agents argument string into a list
    agents_to_train_input = [agent.strip() for agent in args.agents.split(',') if agent.strip()]
    
    # Filter this list against ALL_AGENT_TYPES_AVAILABLE to ensure validity
    valid_agents_to_train = [agent for agent in agents_to_train_input if agent in ALL_AGENT_TYPES_AVAILABLE]

    if not valid_agents_to_train:
        print_f("No valid agents specified for training or specified agents are not in the known list. Exiting.")
        print_f(f"Available agent types: {', '.join(ALL_AGENT_TYPES_AVAILABLE)}")
        sys.exit(1)
    
    print_f(f"--- Starting Training Session ---")
    print_f(f"Agents to train: {', '.join(valid_agents_to_train)}")
    print_f(f"Episodes per agent: {args.episodes}")
    print_f(f"Load LATEST models (continue training): {args.load_models}")
    print_f(f"Force train (new version if model exists and not loading): {args.force_train}")
    print_f(f"Render game content: {args.render} (Pygame window will always be active)")
    print_f(f"Max steps per episode: {args.max_steps_per_episode}")
    print_f(f"Save interval: {args.save_interval} episodes")
    print_f(f"Print interval (steps): {args.print_interval_steps} steps")
    print_f(f"AI Training Mode: ON (reduced delays, faster simulation)")
    print_f("---------------------------------")

>>>>>>> c1e544472a5b20e2ac46dbdf0a24ddcbded062d8
    game_instance = Game(silent_mode=True, ai_training_mode=True)
    preprocessed_obs_shape = (1, 84, 84)
    action_size = game_instance.get_action_size()
    overall_episode_rewards = deque(maxlen=100)
<<<<<<< HEAD
    stop_training_global = False

    for agent_name in valid_agents_to_train:
        if stop_training_global: break
        print_f(f"\n--- Preparing Agent: {agent_name.upper()} ---")
        
        existing_versions = get_existing_model_versions(agent_name, models_base_dir_name=os.path.basename(MODELS_DIR))
        model_to_load_path = None
        model_to_save_path = None 

        if existing_versions:
            print_f(f"  Found existing model(s): {', '.join([os.path.basename(p) for p in existing_versions])}")
            if args.load_models:
                model_to_load_path = existing_versions[-1]
                model_to_save_path = model_to_load_path 
                print_f(f"  Continuing. Loading: {os.path.basename(model_to_load_path)}. Will save to same.")
            elif not args.force_train:
                print_f(f"  Skipping {agent_name}. Use --load_models or --force_train."); continue
            else: 
                model_to_save_path = get_next_model_save_path(agent_name, models_base_dir_name=os.path.basename(MODELS_DIR))
                print_f(f"  Force training. New model: {os.path.basename(model_to_save_path)}")
        else: 
            model_to_save_path = get_next_model_save_path(agent_name, models_base_dir_name=os.path.basename(MODELS_DIR))
            print_f(f"  No existing models. New model: {os.path.basename(model_to_save_path)}")

        if not model_to_save_path and not model_to_load_path:
            print_f(f"  Error: Path conflict for {agent_name}. Skipping."); continue
            
        agent = None
        config_path_for_agent = getattr(args, f"{agent_name}_config_path", None)
        agent_hparams = load_agent_config(config_path_for_agent, agent_name)
        
        print_f(f"  Instantiating {agent_name} with params: {agent_hparams if agent_hparams else 'internal defaults'}")
        # --- Agent Instantiation with loaded hyperparameters ---
        if agent_name == 'dqn':
            agent = DQNAgent(action_size, preprocessed_obs_shape, **agent_hparams)
        elif agent_name == 'a2c':
            agent = A2CAgent(action_size, preprocessed_obs_shape, **agent_hparams)
        elif agent_name == 'ppo':
            agent = PPOAgent(action_size, preprocessed_obs_shape, **agent_hparams)
        elif agent_name == 'genetic':
            agent = GeneticAgent(action_size, preprocessed_obs_shape, **agent_hparams)
        elif agent_name == 'neat':
            # NEAT agent requires its config params to be passed to __init__
            # Or, global constants in neat_agent.py need to be updated from agent_hparams
            # Assuming NEATAgent __init__ can take relevant params from agent_hparams
            agent = NEATAgent(action_size, preprocessed_obs_shape, **agent_hparams)
        
        if not agent: print_f(f"Failed to instantiate agent {agent_name}. Skipping."); continue

        if model_to_load_path and hasattr(agent, 'load'):
            try: agent.load(model_to_load_path) 
            except Exception as e:
                print_f(f"Load failed {os.path.basename(model_to_load_path)}: {e}. Training fresh.")
                if model_to_save_path == model_to_load_path:
                    model_to_save_path = get_next_model_save_path(agent_name, models_base_dir_name=os.path.basename(MODELS_DIR))
                    print_f(f"Adjusted save: {os.path.basename(model_to_save_path)}")
        
        if not model_to_save_path: model_to_save_path = get_next_model_save_path(agent_name, models_base_dir_name=os.path.basename(MODELS_DIR))

        print_f(f"--- Starting Training for Agent: {agent_name.upper()} (Save path: {os.path.basename(model_to_save_path)}) ---")
        
        # ... (The rest of your training loop: episodes, steps, learning, saving) ...
        # Make sure to use model_to_save_path in all agent.save() calls.
        # This part is largely the same as your last correct train.py loop.
        # Copy and paste the loop from "total_steps_for_current_agent = ..." down to the final save logic.
        # Ensure that the loaded agent_hparams are used or that agents' __init__ methods
        # correctly handle default values if some keys are missing from agent_hparams.
        # The **agent_hparams unpacking is a concise way if constructors match keys.
        total_steps_for_current_agent = agent.steps_done if hasattr(agent, 'steps_done') and model_to_load_path else 0
        episode_rewards_window = deque(maxlen=20)
        num_train_py_episodes = args.episodes

        for episode_idx in range(num_train_py_episodes):
            if stop_training_global: break
            observation = game_instance.reset_for_ai()
            current_episode_reward = 0; current_episode_total_loss = 0; current_episode_learn_steps = 0; current_step_in_episode = 0
            
            pop_size_current_agent = agent_hparams.get("population_size", 50) # Default if not in hparams (for GA/NEAT)
            if agent_name == 'genetic': print_f(f"Agent: GENETIC - Gen {episode_idx // pop_size_current_agent + 1}, Indiv {agent.current_individual_idx + 1}/{pop_size_current_agent}")
            elif agent_name == 'neat': print_f(f"Agent: NEAT - Gen {agent.current_generation}, Genome {agent.current_genome_idx + 1}/{pop_size_current_agent}")

            for step in range(args.max_steps_per_episode):
                current_step_in_episode = step + 1
                stop_episode_early = False
                for event in pg.event.get():
                    if event.type == pg.QUIT: stop_episode_early = True; stop_training_global = True; print_f("Quit."); break
                    if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: stop_episode_early = True; stop_training_global = True; print_f("ESC.");break
                if stop_episode_early: break
                action = agent.choose_action(observation)
                next_observation, reward, done, info = game_instance.step_ai(action)
                current_episode_reward += reward
                if agent_name == 'dqn': total_steps_for_current_agent +=1 # DQN uses global steps for eps decay in its own step counter
                
=======
    stop_training_global = False # Flag to stop all further agent training


    for agent_name in valid_agents_to_train: # Use the filtered list
        if stop_training_global: break

        print_f(f"\n--- Preparing Agent: {agent_name.upper()} ---")
        
        existing_versions = get_existing_model_versions(agent_name)
        model_to_load_path = None
        model_to_save_path = None # This will be determined based on logic below

        if agent_name == 'random':
            print_f("Random agent does not require training. Skipping.")
            continue

        if existing_versions:
            print_f(f"Found existing model(s) for {agent_name}: {', '.join([os.path.basename(p) for p in existing_versions])}")
            if args.load_models:
                model_to_load_path = existing_versions[-1] # Load the latest
                model_to_save_path = model_to_load_path # Continue training overwrites the loaded model
                print_f(f"Continuing training. Loading: {os.path.basename(model_to_load_path)}. Will save to same file.")
            elif not args.force_train:
                print_f(f"Skipping training for {agent_name}. Use --load_models to continue or --force_train to create a new version.")
                continue
            else: # force_train is True, and not loading
                model_to_save_path = get_next_model_save_path(agent_name)
                print_f(f"Force training. A new model version will be saved to: {os.path.basename(model_to_save_path)}")
        else: # No existing models
            model_to_save_path = get_next_model_save_path(agent_name) # Should be the base name
            print_f(f"No existing models found for {agent_name}. Will save new model to: {os.path.basename(model_to_save_path)}")

        if not model_to_save_path:
             # This case implies we are loading an existing model and not forcing a new one if loading fails.
             # If model_to_load_path is set, and loading fails, we might need a new save path.
             # This is handled inside the agent loading block.
             if not model_to_load_path: # Should only happen if logic is flawed or all paths taken
                print_f(f"Error: Could not determine a save path for a new/forced training of {agent_name}. Skipping.")
                continue
            
        agent = None
        # --- Agent Instantiation ---
        if agent_name == 'dqn':
            agent = DQNAgent(action_size, preprocessed_obs_shape, buffer_size=50000, batch_size=32, gamma=0.99, lr=1e-4, target_update_freq=1000, eps_decay=100000)
        elif agent_name == 'a2c':
            agent = A2CAgent(action_size, preprocessed_obs_shape, lr=7e-4, gamma=0.99)
        elif agent_name == 'ppo':
            agent = PPOAgent(action_size, preprocessed_obs_shape, lr=2.5e-4, gamma=0.99, trajectory_n_steps=128, ppo_epochs=4, mini_batch_size=32)
        elif agent_name == 'genetic':
            agent = GeneticAgent(action_size, preprocessed_obs_shape, population_size=20, num_elites=2, mutation_rate=0.1)
        # No else needed as we filtered valid_agents_to_train

        # --- Model Loading ---
        if model_to_load_path and hasattr(agent, 'load'):
            try:
                agent.load(model_to_load_path) 
            except Exception as e:
                print_f(f"Could not load model for {agent_name} from {os.path.basename(model_to_load_path)}: {e}.")
                print_f("Training fresh instead.")
                # If loading failed, and we were supposed to save to the loaded path,
                # we now need a *new* save path to avoid overwriting the problematic loaded file with a fresh model.
                if model_to_save_path == model_to_load_path:
                    model_to_save_path = get_next_model_save_path(agent_name)
                    print_f(f"Adjusted save path for fresh training to: {os.path.basename(model_to_save_path)}")
        
        # Ensure model_to_save_path is set if it wasn't (e.g. if only loading was intended but failed)
        if not model_to_save_path:
            model_to_save_path = get_next_model_save_path(agent_name)
            print_f(f"Defaulting save path for new training to: {os.path.basename(model_to_save_path)}")


        print_f(f"--- Starting Training for Agent: {agent_name.upper()} (Save path: {os.path.basename(model_to_save_path)}) ---")
        total_steps_for_current_agent = agent.steps_done if hasattr(agent, 'steps_done') and args.load_models else 0
        episode_rewards_window = deque(maxlen=20)

        for episode in range(args.episodes):
            if stop_training_global: break
            observation = game_instance.reset_for_ai()
            episode_reward = 0
            current_episode_total_loss = 0
            current_episode_learn_steps = 0

            if agent_name == 'genetic':
                print_f(f"Agent: GENETIC - Gen {episode // agent.population_size + 1}, Indiv {agent.current_individual_idx + 1}/{agent.population_size}")

            for step in range(args.max_steps_per_episode):
                stop_episode_early = False
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        print_f("Pygame window closed. Exiting training session...")
                        stop_episode_early = True; stop_training_global = True; break
                    if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                        print_f("Escape key pressed. Exiting training session..."); 
                        stop_episode_early = True; stop_training_global = True; break
                if stop_episode_early: break

                action = agent.choose_action(observation)
                next_observation, reward, done, info = game_instance.step_ai(action)
                episode_reward += reward
                
                if agent_name != 'random': total_steps_for_current_agent +=1

>>>>>>> c1e544472a5b20e2ac46dbdf0a24ddcbded062d8
                loss_val = None
                # ... (Learning logic as before) ...
                if agent_name == 'dqn':
                    agent.store_transition(observation, action, next_observation, reward, done)
<<<<<<< HEAD
                    if agent.steps_done > agent_hparams.get("batch_size", 32) * 5: loss_val = agent.learn() # Use batch_size from hparams
=======
                    if total_steps_for_current_agent > agent.batch_size * 5: loss_val = agent.learn()
>>>>>>> c1e544472a5b20e2ac46dbdf0a24ddcbded062d8
                elif agent_name == 'a2c':
                    agent.store_outcome(reward, done); loss_val = agent.learn(next_observation if not done else None)
                elif agent_name == 'ppo':
                    loss_val = agent.store_transition_outcome(reward, done, next_observation)
<<<<<<< HEAD
                if loss_val is not None: current_episode_total_loss += loss_val; current_episode_learn_steps += 1
                observation = next_observation
                if args.render: game_instance.render_for_ai()
                if args.print_interval_steps > 0 and (step + 1) % args.print_interval_steps == 0 and current_episode_learn_steps > 0:
                    avg_step_loss = current_episode_total_loss / current_episode_learn_steps
                    print_f(f"  {agent_name.upper()} Ep {episode_idx+1}, Step {step+1}: AvgStepLoss={avg_step_loss:.4f}, CurRwd={current_episode_reward:.0f}")
                    current_episode_total_loss = 0; current_episode_learn_steps = 0
                if done: break
            if stop_episode_early or stop_training_global: break
            episode_rewards_window.append(current_episode_reward)
            overall_episode_rewards.append(current_episode_reward)
            avg_reward_window = np.mean(episode_rewards_window) if episode_rewards_window else 0.0
            avg_episode_loss_segment = (current_episode_total_loss / current_episode_learn_steps) if current_episode_learn_steps > 0 else 0.0
            game_score = info.get('score',0)
            print_f(f"Agent: {agent_name.upper()} - Ep {episode_idx + 1}/{num_train_py_episodes} DONE. R={current_episode_reward:.0f}, Steps={current_step_in_episode}, AvgLoss={avg_episode_loss_segment:.4f}, Score={game_score}, AvgR(last {len(episode_rewards_window)})={avg_reward_window:.2f}")
            
            if agent_name == 'genetic' or agent_name == 'neat':
                agent.record_fitness(game_score)
                # Use pop_size_current_agent for the check
                if agent.current_genome_idx >= pop_size_current_agent if agent_name == 'neat' else agent.current_individual_idx >= pop_size_current_agent :
                    agent.learn()
                    if hasattr(agent, 'save'): agent.save(model_to_save_path)
            if args.save_interval > 0 and (episode_idx + 1) % args.save_interval == 0:
                if agent_name not in ['genetic', 'neat', 'random'] and hasattr(agent, 'save'):
                    agent.save(model_to_save_path)
        
        if not stop_training_global:
            if agent_name not in ['genetic', 'neat', 'random'] and hasattr(agent, 'save'): agent.save(model_to_save_path)
            elif agent_name in ['genetic', 'neat'] and hasattr(agent, 'save'): agent.save(model_to_save_path)
        if stop_training_global: print_f(f"Training for {agent_name.upper()} interrupted."); break

    avg_overall_reward = np.mean(overall_episode_rewards) if overall_episode_rewards else 0.0
    print_f(f"\n--- Overall Training Stats ---"); print_f(f"Rolling avg reward (last {len(overall_episode_rewards)} eps): {avg_overall_reward:.2f}")
    print_f(f"--- Training Finished (or Interrupted) ---"); pg.quit(); sys.exit()

=======
                
                if loss_val is not None:
                    current_episode_total_loss += loss_val
                    current_episode_learn_steps += 1
                
                observation = next_observation
                if args.render: game_instance.render_for_ai()

                if args.print_interval_steps > 0 and (step + 1) % args.print_interval_steps == 0 and current_episode_learn_steps > 0:
                    avg_step_loss = current_episode_total_loss / current_episode_learn_steps
                    print_f(f"  {agent_name.upper()} Ep {episode+1}, Step {step+1}: AvgStepLoss={avg_step_loss:.4f}, CurrentRwd={episode_reward}")
                    current_episode_total_loss = 0; current_episode_learn_steps = 0
                if done: break
            
            if stop_episode_early or stop_training_global: break # from episode loop

            episode_rewards_window.append(episode_reward)
            overall_episode_rewards.append(episode_reward)
            avg_reward_window = np.mean(episode_rewards_window) if episode_rewards_window else 0.0
            avg_episode_loss = (current_episode_total_loss / current_episode_learn_steps) if current_episode_learn_steps > 0 else 0.0
            current_score = info.get('score',0)
            print_f(f"Agent: {agent_name.upper()} - Ep {episode + 1}/{args.episodes} DONE. "
                  f"TotalR={episode_reward:.0f}, Steps={step+1}, AvgLoss(last_segment)={avg_episode_loss:.4f}, Score={current_score}, AvgR(last {len(episode_rewards_window)})={avg_reward_window:.2f}")

            if agent_name == 'genetic':
                agent.record_fitness(current_score)
                if agent.current_individual_idx >= agent.population_size:
                    agent.learn()
                    if hasattr(agent, 'save'): 
                        print_f(f"GA: Saving best of generation to {os.path.basename(model_to_save_path)}")
                        agent.save(model_to_save_path)

            if args.save_interval > 0 and (episode + 1) % args.save_interval == 0:
                if agent_name not in ['genetic', 'random'] and hasattr(agent, 'save'):
                    print_f(f"Saving model for {agent_name} to {os.path.basename(model_to_save_path)} at episode {episode + 1}...")
                    agent.save(model_to_save_path)
        
        # Final save for the current agent, if not interrupted before completing all its episodes
        if not stop_training_global:
            if agent_name not in ['genetic', 'random'] and hasattr(agent, 'save'):
                print_f(f"Completed training for {agent_name}. Saving final model to {os.path.basename(model_to_save_path)}")
                agent.save(model_to_save_path)
            elif agent_name == 'genetic' and hasattr(agent, 'save'):
                print_f(f"GA: Ensuring final save of best current individual to {os.path.basename(model_to_save_path)}")
                agent.save(model_to_save_path) 
        
        if stop_training_global:
            print_f(f"Training for {agent_name.upper()} was interrupted. Stopping training for subsequent agents.")
            break # Break from the loop over agents

    avg_overall_reward = np.mean(overall_episode_rewards) if overall_episode_rewards else 0.0
    print_f(f"\n--- Overall Training Stats ---")
    print_f(f"Rolling average reward over last {len(overall_episode_rewards)} episodes: {avg_overall_reward:.2f}")
    print_f(f"--- All Specified Agent Training Finished (or Interrupted) ---")
    pg.quit()
    sys.exit()
>>>>>>> c1e544472a5b20e2ac46dbdf0a24ddcbded062d8

if __name__ == '__main__':
    train_all_agents()