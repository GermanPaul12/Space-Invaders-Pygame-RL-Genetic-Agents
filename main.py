# main.py
import pygame as pg
import argparse
import sys
import os
import numpy as np # For observation shape for DQN etc.

try:
    from game import Game
    import config
    from agents.agent import Agent # Base class
    from agents.random_agent import RandomAgent
    from agents.dqn_agent import DQNAgent, preprocess_observation as dqn_preprocess
    from agents.a2c_agent import A2CAgent, preprocess_observation as ac_preprocess
    from agents.ppo_agent import PPOAgent, preprocess_observation as ppo_preprocess
    from agents.genetic_agent import GeneticAgent, preprocess_observation as ga_preprocess
except ImportError as e:
    print(f"Error importing game or agent files: {e}")
    print("Ensure all files are correctly placed and PyTorch is installed if needed.")
    sys.exit(1)

MODELS_DIR = "trained_models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def get_model_path(agent_name):
    return os.path.join(MODELS_DIR, f"{agent_name}_spaceinvaders.pth")

def main():
    parser = argparse.ArgumentParser(description="Space Invaders with Pygame and AI Agents")
    parser.add_argument("--mode", type=str, choices=['play', 'train', 'test', 'evaluate_all'], default='play',
                        help="Mode of operation.")
    parser.add_argument("--agent", type=str, 
                        choices=['human', 'random', 'dqn', 'ppo', 'a2c', 'genetic'], 
                        default='human',
                        help="Agent to use.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes for training/testing AI.")
    parser.add_argument("--load_model", action='store_true', help="Load a pre-trained model for the agent.")
    parser.add_argument("--render", action='store_true', help="Render game during AI training/testing.")
    parser.add_argument("--max_steps_per_episode", type=int, default=2000, help="Max steps per episode for AI.")


    args = parser.parse_args()

    game_instance = Game()

    if args.mode == 'play':
        if args.agent != 'human':
            print(f"Warning: 'play' mode selected, but agent is '{args.agent}'. Defaulting to human player.")
        print("Starting game in human player mode...")
        game_instance.run_player_mode()
        pg.quit()
        sys.exit()

    # --- AI Mode Setup ---
    print(f"Selected AI mode: {args.mode} with agent: {args.agent}")
    if args.agent == 'human' and args.mode != 'play':
        print("Error: 'human' agent cannot be used for AI modes.")
        pg.quit()
        sys.exit(1)

    # Determine observation shape (after preprocessing for NNs)
    # For simplicity, let's assume a fixed preprocessed shape (e.g., 1x84x84 for grayscale)
    # This should match what the preprocessing functions output.
    # Example: if dqn_preprocess outputs (1, 84, 84)
    preprocessed_obs_shape = (1, 84, 84) # (C, H, W) - adjust if your preprocess changes this!

    agent = None
    agent_name = args.agent
    model_path = get_model_path(agent_name)
    action_size = game_instance.get_action_size()

    if agent_name == 'random':
        agent = RandomAgent(action_size)
    elif agent_name == 'dqn':
        # Args for DQN. Could be moved to config or agent-specific configs
        agent = DQNAgent(action_size, preprocessed_obs_shape, 
                         buffer_size=50000, batch_size=32, gamma=0.99, lr=1e-4, 
                         target_update_freq=1000, eps_decay=100000) # Slower decay for more steps
    elif agent_name == 'a2c':
        agent = A2CAgent(action_size, preprocessed_obs_shape, lr=7e-4, gamma=0.99)
    elif agent_name == 'ppo':
        agent = PPOAgent(action_size, preprocessed_obs_shape, lr=2.5e-4, gamma=0.99, 
                         trajectory_n_steps=256, ppo_epochs=4, mini_batch_size=64)
    elif agent_name == 'genetic':
        agent = GeneticAgent(action_size, preprocessed_obs_shape, population_size=20, num_elites=2, mutation_rate=0.1)
    else:
        print(f"Agent '{agent_name}' not implemented yet.")
        pg.quit()
        sys.exit(1)

    if args.load_model and os.path.exists(model_path) and agent_name != 'random':
        try:
            agent.load(model_path)
            print(f"Loaded pre-trained model for {agent_name} from {model_path}")
        except Exception as e:
            print(f"Could not load model for {agent_name}: {e}")
    elif args.load_model and agent_name != 'random':
        print(f"Model file not found for {agent_name} at {model_path}. Starting fresh.")


    # --- Training / Testing / Evaluation Logic ---
    
    if args.mode == 'train':
        print(f"--- Training {agent_name} for {args.episodes} episodes ---")
        total_steps_overall = 0
        for episode in range(args.episodes):
            observation = game_instance.reset_for_ai()
            episode_reward = 0
            episode_loss = 0
            num_learn_steps = 0

            if agent_name == 'genetic':
                print(f"Generation {episode // agent.population_size +1}, Individual {agent.current_individual_idx + 1}/{agent.population_size}")

            for step in range(args.max_steps_per_episode):
                action = agent.choose_action(observation)
                next_observation, reward, done, info = game_instance.step_ai(action)
                episode_reward += reward
                total_steps_overall +=1

                # Learning step varies by agent
                loss_val = None
                if agent_name == 'dqn':
                    agent.store_transition(observation, action, next_observation, reward, done)
                    if total_steps_overall > agent.batch_size * 5 : # Start learning after some experience
                         loss_val = agent.learn()
                elif agent_name == 'a2c':
                    agent.store_outcome(reward, done) # Store r, d for current action
                    loss_val = agent.learn(next_observation if not done else None) # Pass next_obs for bootstrap
                elif agent_name == 'ppo':
                    # PPO store_transition_outcome will trigger learn internally when buffer is full or done
                    loss_val = agent.store_transition_outcome(reward, done, next_observation)
                # Genetic agent learns at the end of a generation (all individuals evaluated)
                # Random agent doesn't learn

                if loss_val is not None:
                    episode_loss += loss_val
                    num_learn_steps += 1
                
                observation = next_observation
                if args.render:
                    game_instance.render_for_ai()
                    pg.time.wait(1) # Small delay for human eyes

                if done:
                    break
            
            avg_episode_loss = (episode_loss / num_learn_steps) if num_learn_steps > 0 else 0
            print(f"Episode {episode + 1}: Reward={episode_reward}, Steps={step+1}, AvgLoss={avg_episode_loss:.4f}, Score={info.get('score',0)}")

            if agent_name == 'genetic':
                agent.record_fitness(info.get('score', 0)) # Use score as fitness
                if agent.current_individual_idx >= agent.population_size:
                    agent.learn() # Evolve population

            # Save model periodically
            if (episode + 1) % 50 == 0 and agent_name not in ['random', 'genetic']: # Genetic saves best at end of generation
                agent.save(model_path)
            if agent_name == 'genetic' and agent.current_individual_idx == 0 and episode > 0: # Start of new gen
                 agent.save(model_path) # Save best of previous generation

        # Final save
        if agent_name not in ['random', 'genetic']:
            agent.save(model_path)
        elif agent_name == 'genetic': # Ensure best of last gen is saved
            agent.save(model_path)


    elif args.mode == 'test':
        print(f"--- Testing {agent_name} for {args.episodes} episodes ---")
        all_rewards = []
        all_scores = []
        for episode in range(args.episodes):
            observation = game_instance.reset_for_ai()
            episode_reward = 0
            
            if agent_name == 'genetic': # Test the current best or first individual if not trained
                if not args.load_model: print("Warning: Testing GA without loading a model, using first individual.")

            for step in range(args.max_steps_per_episode):
                # For DQN/PPO/A2C in test mode, often epsilon is set to 0 or very low
                if hasattr(agent, 'eps_start') and agent_name == 'dqn': # Simple way to force greedy for DQN
                    original_eps_start = agent.eps_start
                    agent.eps_start = 0.00 # Act greedily
                    action = agent.choose_action(observation)
                    agent.eps_start = original_eps_start # Restore
                else:
                    action = agent.choose_action(observation)
                
                next_observation, reward, done, info = game_instance.step_ai(action)
                episode_reward += reward
                observation = next_observation

                if args.render:
                    game_instance.render_for_ai()
                    pg.time.wait(1)
                if done:
                    break
            all_rewards.append(episode_reward)
            all_scores.append(info.get('score',0))
            print(f"Test Episode {episode + 1}: Reward={episode_reward}, Score={info.get('score',0)}")
        print(f"--- Test Results for {agent_name} ---")
        print(f"Average Reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
        print(f"Average Score: {np.mean(all_scores):.2f} +/- {np.std(all_scores):.2f}")

    elif args.mode == 'evaluate_all':
        print("--- Evaluating all trained agents ---")
        # This would loop through a list of agent types, load their models, and run test mode
        # For now, this is a placeholder for a more robust evaluation script.
        # Example:
        # agent_types_to_eval = ['dqn', 'ppo', 'a2c', 'genetic', 'random']
        # for ag_type in agent_types_to_eval:
        #     print(f"\nEvaluating {ag_type}...")
        #     # Re-initialize or set up agent
        #     # Load model for ag_type
        #     # Run test loop (similar to above)
        # This part needs careful implementation to manage agent instances and model loading.
        print("Evaluate_all mode is not fully implemented yet. Run --mode test --agent <name> for individual tests.")


    pg.quit()
    sys.exit()

if __name__ == '__main__':
    main()