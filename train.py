# train.py
import pygame as pg
import argparse
import sys
import os
import numpy as np

try:
    from game import Game
    import config
    # Agent base class and specific agent implementations
    from agents.agent import Agent
    from agents.random_agent import RandomAgent # RandomAgent doesn't train, but could be included for completeness
    from agents.dqn_agent import DQNAgent
    from agents.a2c_agent import A2CAgent
    from agents.ppo_agent import PPOAgent
    from agents.genetic_agent import GeneticAgent
except ImportError as e:
    print(f"Error importing game or agent files: {e}")
    print("Ensure all files are correctly placed and PyTorch is installed if needed for specific agents.")
    sys.exit(1)

MODELS_DIR = "trained_models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

ALL_AGENT_TYPES = ['dqn', 'ppo', 'a2c', 'genetic', 'random'] # Add 'random' if you want to "test" it in a similar loop

def get_model_path(agent_name):
    return os.path.join(MODELS_DIR, f"{agent_name}_spaceinvaders.pth")

def train_all_agents():
    parser = argparse.ArgumentParser(description="Train multiple Space Invaders AI Agents")
    parser.add_argument("--agents", type=str, default="dqn,ppo,a2c,genetic",
                        help=f"Comma-separated list of agents to train from: {', '.join(ALL_AGENT_TYPES)}")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train each agent.")
    parser.add_argument("--load_models", action='store_true', help="Load pre-existing models for all selected agents before training.")
    parser.add_argument("--render", action='store_true', help="Render game during training for all agents.")
    parser.add_argument("--max_steps_per_episode", type=int, default=2000, help="Max steps per episode for AI.")
    parser.add_argument("--save_interval", type=int, default=50, help="Save model every N episodes (0 to disable intermediate saves).")


    args = parser.parse_args()

    agents_to_train = [agent.strip() for agent in args.agents.split(',') if agent.strip() in ALL_AGENT_TYPES]
    if not agents_to_train:
        print("No valid agents specified for training. Exiting.")
        sys.exit(1)

    print(f"--- Starting Training Session ---")
    print(f"Agents to train: {', '.join(agents_to_train)}")
    print(f"Episodes per agent: {args.episodes}")
    print(f"Load existing models: {args.load_models}")
    print(f"Render game: {args.render}")
    print(f"Max steps per episode: {args.max_steps_per_episode}")
    print(f"Save interval: {args.save_interval} episodes")
    print("---------------------------------")

    game_instance = Game(silent_mode=True) # Create one game instance, reset for each agent/episode

    # This shape should match what the preprocessing functions (if used externally) would output,
    # or what the agent's network is designed to accept.
    # If agents do their own preprocessing from raw pixels, this specific shape might be
    # more of an internal agent detail. For now, we define it as in main.py.
    preprocessed_obs_shape = (1, 84, 84) # (C, H, W) - Example
    action_size = game_instance.get_action_size()


    for agent_name in agents_to_train:
        print(f"\n--- Training Agent: {agent_name.upper()} ---")

        agent = None
        model_path = get_model_path(agent_name)

        if agent_name == 'random':
            print("Random agent does not require training. Skipping.")
            continue # Random agent doesn't train
        elif agent_name == 'dqn':
            agent = DQNAgent(action_size, preprocessed_obs_shape,
                             buffer_size=50000, batch_size=32, gamma=0.99, lr=1e-4,
                             target_update_freq=1000, eps_decay=100000)
        elif agent_name == 'a2c':
            agent = A2CAgent(action_size, preprocessed_obs_shape, lr=7e-4, gamma=0.99)
        elif agent_name == 'ppo':
            agent = PPOAgent(action_size, preprocessed_obs_shape, lr=2.5e-4, gamma=0.99,
                             trajectory_n_steps=256, ppo_epochs=4, mini_batch_size=64)
        elif agent_name == 'genetic':
            agent = GeneticAgent(action_size, preprocessed_obs_shape, population_size=20, num_elites=2, mutation_rate=0.1)
        else:
            print(f"Agent type '{agent_name}' is recognized but not implemented for training in this script. Skipping.")
            continue

        if args.load_models and os.path.exists(model_path):
            try:
                agent.load(model_path)
                print(f"Loaded pre-trained model for {agent_name} from {model_path}")
            except Exception as e:
                print(f"Could not load model for {agent_name} from {model_path}: {e}. Starting fresh.")
        elif args.load_models:
            print(f"Model file not found for {agent_name} at {model_path}. Starting fresh.")

        total_steps_for_current_agent = 0 # For DQN learning condition
        
        for episode in range(args.episodes):
            observation = game_instance.reset_for_ai() # Game state is raw pixels
            episode_reward = 0
            episode_loss = 0
            num_learn_steps = 0

            if agent_name == 'genetic':
                 # GA's episode count refers to evaluating one individual.
                 # A "generation" for GA is population_size episodes.
                print(f"Generation {episode // agent.population_size + 1}, Individual {agent.current_individual_idx + 1}/{agent.population_size}")

            for step in range(args.max_steps_per_episode):
                # Agents are expected to handle preprocessing of 'observation' if their NNs require it
                action = agent.choose_action(observation)
                next_observation, reward, done, info = game_instance.step_ai(action)
                episode_reward += reward
                total_steps_for_current_agent +=1

                loss_val = None
                if agent_name == 'dqn':
                    agent.store_transition(observation, action, next_observation, reward, done)
                    # Start learning after collecting some experience for DQN
                    if total_steps_for_current_agent > agent.batch_size * 5 :
                         loss_val = agent.learn()
                elif agent_name == 'a2c':
                    agent.store_outcome(reward, done)
                    loss_val = agent.learn(next_observation if not done else None)
                elif agent_name == 'ppo':
                    loss_val = agent.store_transition_outcome(reward, done, next_observation)
                # Genetic agent learns at the end of a generation (all individuals evaluated)

                if loss_val is not None:
                    episode_loss += loss_val
                    num_learn_steps += 1
                
                observation = next_observation
                if args.render:
                    game_instance.render_for_ai()
                    pg.time.wait(1) # Small delay for human eyes if rendering

                if done:
                    break
            
            avg_episode_loss = (episode_loss / num_learn_steps) if num_learn_steps > 0 else 0
            current_score = info.get('score',0)
            print(f"Agent: {agent_name.upper()} - Episode {episode + 1}/{args.episodes}: "
                  f"Reward={episode_reward}, Steps={step+1}, AvgLoss={avg_episode_loss:.4f}, Score={current_score}")

            if agent_name == 'genetic':
                agent.record_fitness(current_score) # Use game score as fitness
                if agent.current_individual_idx >= agent.population_size: # End of a generation
                    agent.learn() # Evolve population
                    print(f"Agent: GENETIC - Generation {episode // agent.population_size + 1} complete. Best score: {agent.best_fitness_overall}")
                    agent.save(model_path) # Save best model of the generation

            # Save model periodically for non-GA agents (GA saves at end of generation)
            if args.save_interval > 0 and (episode + 1) % args.save_interval == 0:
                if agent_name not in ['genetic']:
                    print(f"Saving model for {agent_name} at episode {episode + 1}...")
                    agent.save(model_path)
        
        # Final save for the agent (unless it's GA, which saves its best already)
        if agent_name not in ['genetic', 'random']:
            print(f"Completed training for {agent_name}. Saving final model to {model_path}")
            agent.save(model_path)
        elif agent_name == 'genetic' and not os.path.exists(model_path): # Ensure GA model is saved if no generation completed
            agent.save(model_path)


    print("\n--- All Specified Agent Training Finished ---")
    pg.quit()
    sys.exit()

if __name__ == '__main__':
    train_all_agents()