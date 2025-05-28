# evaluate.py
import pygame as pg
import argparse
import sys
import os
import numpy as np
import csv
from datetime import datetime

try:
    from game.game import Game
    from game import config
    from agents.agent import Agent 
    from agents.random_agent import RandomAgent
    from agents.dqn_agent import DQNAgent
    from agents.a2c_agent import A2CAgent
    from agents.ppo_agent import PPOAgent
    from agents.genetic_agent import GeneticAgent
    from agents.neat_agent import NEATAgent
    from utils.model_helpers import (
        get_latest_model_path,
        MODELS_DIR # Use MODELS_DIR from helpers
    )
except ImportError as e:
    print(f"Error importing files in evaluate.py: {e}", flush=True)
    sys.exit(1)

EVAL_RESULTS_DIR = "evaluation_results"
if not os.path.exists(EVAL_RESULTS_DIR):
    os.makedirs(EVAL_RESULTS_DIR)

ALL_AGENT_TYPES_AVAILABLE = ['dqn', 'ppo', 'a2c', 'genetic', 'neat', 'random']

def print_f(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate all Space Invaders AI Agents")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes per agent.")
    parser.add_argument("--max_steps_per_episode", type=int, default=3000)
    # No render option for evaluation, it should run headless for speed.
    # No specific model path, it always evaluates latest.
    args = parser.parse_args()

    print_f(f"--- Starting Evaluation of All Agents (Latest Models) ---")
    print_f(f"Evaluation Episodes: {args.episodes}, Max Steps: {args.max_steps_per_episode}")

    game_instance = Game(silent_mode=True, ai_training_mode=True) # Fast and silent
    preprocessed_obs_shape = (1, 84, 84)
    action_size = game_instance.get_action_size()
    
    evaluation_data = [] # To store dicts for CSV

    for agent_name in ALL_AGENT_TYPES_AVAILABLE:
        print_f(f"\n--- Evaluating: {agent_name.upper()} ---")
        agent_instance = None
        model_to_load = None
        
        if agent_name == 'random':
            agent_instance = RandomAgent(action_size)
            print_f("  Type: Random Agent")
        else:
            model_to_load = get_latest_model_path(agent_name)
            if not model_to_load:
                print_f(f"  No model found for {agent_name}. Skipping.")
                continue
            print_f(f"  Loading latest model: {os.path.basename(model_to_load)}")

            # Instantiate with evaluation parameters (e.g., no exploration)
            if agent_name == 'dqn': agent_instance = DQNAgent(action_size, preprocessed_obs_shape, eps_start=0.0, eps_end=0.0)
            elif agent_name == 'a2c': agent_instance = A2CAgent(action_size, preprocessed_obs_shape, entropy_coef=0.0)
            elif agent_name == 'ppo': agent_instance = PPOAgent(action_size, preprocessed_obs_shape, entropy_coef=0.0)
            elif agent_name == 'genetic': agent_instance = GeneticAgent(action_size, preprocessed_obs_shape, population_size=1, mutation_rate=0)
            elif agent_name == 'neat': 
                from agents.neat_agent import NEAT_POPULATION_SIZE # Not ideal here
                agent_instance = NEATAgent(action_size, preprocessed_obs_shape, population_size=1)
            else: print_f(f"  Unknown agent type {agent_name}. Skipping."); continue
            
            try:
                agent_instance.load(model_to_load)
            except Exception as e:
                print_f(f"  Error loading model for {agent_name}: {e}. Skipping."); continue
        
        agent_rewards, agent_scores, agent_steps = [], [], []
        stop_script_flag = False

        for ep_idx in range(args.episodes):
            if stop_script_flag: break
            obs = game_instance.reset_for_ai()
            ep_r, current_score, current_s = 0,0,0
            for step_idx in range(args.max_steps_per_episode):
                current_s = step_idx + 1
                for event in pg.event.get(): # Still allow quit
                    if event.type == pg.QUIT: print_f("Quit."); stop_script_flag=True; break
                    if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: print_f("ESC."); stop_script_flag=True; break
                if stop_script_flag: break
                
                action = agent_instance.choose_action(obs)
                next_obs, r, d, inf = game_instance.step_ai(action)
                ep_r += r; obs = next_obs
                if d: current_score = inf.get('score',0); break
            
            if stop_script_flag: break
            agent_rewards.append(ep_r); agent_scores.append(current_score); agent_steps.append(current_s)
            print_f(f"    Eval Ep {ep_idx+1}: Reward={ep_r:.0f}, Score={current_score}, Steps={current_s}")

        if stop_script_flag: print_f("Evaluation interrupted."); break # Stop evaluating other agents

        if agent_rewards: # If any episodes completed
            avg_r = np.mean(agent_rewards); std_r = np.std(agent_rewards)
            avg_s = np.mean(agent_scores); std_s = np.std(agent_scores)
            avg_st = np.mean(agent_steps)
            print_f(f"  Results for {agent_name} ({len(agent_rewards)} eps):")
            print_f(f"    Avg Reward: {avg_r:.2f} +/- {std_r:.2f}")
            print_f(f"    Avg Score:  {avg_s:.2f} +/- {std_s:.2f}")
            print_f(f"    Avg Steps:  {avg_st:.1f}")
            evaluation_data.append({
                "Agent": agent_name,
                "Model": os.path.basename(model_to_load) if model_to_load else "N/A (Random/Untrained)",
                "Episodes": len(agent_rewards),
                "Avg Reward": f"{avg_r:.2f}", "Std Reward": f"{std_r:.2f}",
                "Avg Score": f"{avg_s:.2f}", "Std Score": f"{std_s:.2f}",
                "Avg Steps": f"{avg_st:.1f}"
            })
        else:
            print_f(f"  No evaluation episodes completed for {agent_name}.")

    # Save results to CSV
    if evaluation_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(EVAL_RESULTS_DIR, f"evaluation_summary_{timestamp}.csv")
        fieldnames = evaluation_data[0].keys()
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(evaluation_data)
            print_f(f"\nEvaluation results saved to: {csv_filename}")
        except IOError:
            print_f(f"Error: Could not write evaluation results to CSV: {csv_filename}")
    else:
        print_f("\nNo evaluation data was generated.")

    print_f("\n--- Evaluation Finished ---")
    pg.quit()
    sys.exit()

if __name__ == '__main__':
    run_evaluation()