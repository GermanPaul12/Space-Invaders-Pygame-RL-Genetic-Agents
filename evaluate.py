# evaluate.py
import pygame as pg
import argparse
import sys
import os
import numpy as np

try:
    from game.game_manager import Game # MODIFIED
    from utils.model_helpers import get_latest_model_path, MODELS_DIR
    from utils.cli_args import add_common_eval_args, ALL_AGENT_TYPES_AVAILABLE
    from utils.agent_factory import create_agent
    from utils.episode_runner import run_agent_episodes_for_test_eval
    from utils.report_utils import save_evaluation_to_csv, EVAL_RESULTS_DIR
except ImportError as e:
    print(f"Error importing files in evaluate.py: {e}", flush=True); sys.exit(1)

def print_f(*args, **kwargs): print(*args, **kwargs); sys.stdout.flush()

def run_evaluation_main(): 
    parser = argparse.ArgumentParser(description="Evaluate all Space Invaders AI Agents")
    parser = add_common_eval_args(parser)
    args = parser.parse_args()

    print_f(f"--- Evaluating All Agents (Latest Models) ---")
    print_f(f"  Eval Episodes: {args.episodes}, Max Steps: {args.max_steps_per_episode}")
    print_f(f"  Results will be saved to: {EVAL_RESULTS_DIR}")


    # For evaluation, game is typically fast and silent, and headless.
    # headless_worker_mode=True ensures Pygame display is handled for non-rendering scenarios.
    game_instance = Game(silent_mode=True, ai_training_mode=True, headless_worker_mode=True) 
    preprocessed_obs_shape = (1, 84, 84) # Standard for CNN agents
    action_size = game_instance.get_action_size()
    
    evaluation_summary_data = []
    stop_script_flag = False # Flag to stop if user interrupts (e.g., Ctrl+C in episode_runner)

    # Ensure EVAL_RESULTS_DIR exists
    if not os.path.exists(EVAL_RESULTS_DIR):
        try:
            os.makedirs(EVAL_RESULTS_DIR)
            print_f(f"Created evaluation results directory: {EVAL_RESULTS_DIR}")
        except OSError as e:
            print_f(f"Error creating directory {EVAL_RESULTS_DIR}: {e}. Results might not be saved.")
            # Optionally, exit or handle differently if saving is critical.


    for agent_name in ALL_AGENT_TYPES_AVAILABLE:
        if stop_script_flag: break
        print_f(f"\n--- Evaluating: {agent_name.upper()} ---")
        
        model_to_load = None
        if agent_name != 'random': # Random agent doesn't load models
            model_to_load = get_latest_model_path(agent_name)
            if not model_to_load:
                print_f(f"  No model found for {agent_name}. Skipping evaluation for this agent.")
                continue
            print_f(f"  Loading model: {os.path.basename(model_to_load)}")

        # Create agent instance for evaluation mode
        agent_instance = create_agent(agent_name, action_size, preprocessed_obs_shape, mode='evaluate')
        if not agent_instance:
            print_f(f"  Failed to create agent {agent_name}. Skipping.")
            continue

        if model_to_load and hasattr(agent_instance, 'load') and agent_name != 'random':
            try:
                agent_instance.load(model_to_load)
            except Exception as e:
                print_f(f"  Error loading model for {agent_name}: {e}. Skipping this agent.")
                continue
        
        # run_agent_episodes_for_test_eval runs episodes and collects stats
        rewards, scores, steps_taken, interrupted_by_user = run_agent_episodes_for_test_eval(
            agent=agent_instance,
            agent_name=agent_name,
            game_instance=game_instance,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps_per_episode,
            render_flag=False, # Evaluation is typically headless
            run_mode="evaluate", # Explicitly set mode
            gif_params=None,   # No GIFs for bulk evaluation
            is_eval_mode=True  # Indicates this is for formal evaluation
        )

        if interrupted_by_user:
            print_f(f"Evaluation interrupted by user during {agent_name}'s run.")
            stop_script_flag = True # Propagate interruption to stop evaluating further agents

        if rewards: # If any episodes were completed for this agent
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            avg_steps = np.mean(steps_taken)
            
            print_f(f"  Results ({len(rewards)} episodes):")
            print_f(f"    Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
            print_f(f"    Avg Score:  {avg_score:.2f} ± {std_score:.2f}")
            print_f(f"    Avg Steps:  {avg_steps:.1f}")
            
            evaluation_summary_data.append({
                "Agent": agent_name,
                "Model": os.path.basename(model_to_load) if model_to_load else "N/A (Random or Untrained)",
                "Episodes Run": len(rewards),
                "Avg Reward": f"{avg_reward:.2f}", 
                "Std Reward": f"{std_reward:.2f}",
                "Avg Score": f"{avg_score:.2f}", 
                "Std Score": f"{std_score:.2f}",
                "Avg Steps": f"{avg_steps:.1f}"
            })
        elif not interrupted_by_user: # No rewards, but not due to interruption (e.g., agent failed early)
            print_f(f"  No evaluation episodes completed successfully for {agent_name}.")
            evaluation_summary_data.append({
                "Agent": agent_name,
                "Model": os.path.basename(model_to_load) if model_to_load else "N/A (Random or Untrained)",
                "Episodes Run": 0, "Avg Reward": "N/A", "Std Reward": "N/A",
                "Avg Score": "N/A", "Std Score": "N/A", "Avg Steps": "N/A"
            })
            
    if evaluation_summary_data:
        # Save the collected data to a CSV file
        # The filename for the CSV can be timestamped or fixed as per report_utils.py logic
        csv_filename = save_evaluation_to_csv(evaluation_summary_data, output_dir_name=EVAL_RESULTS_DIR) # Ensure EVAL_RESULTS_DIR basename is passed if function expects that
        if csv_filename:
             print_f(f"\nEvaluation summary saved to: {csv_filename}")
        else:
             print_f("\nFailed to save evaluation summary.")
    else:
        print_f("\nNo evaluation data was generated.")

    print_f("\n--- Evaluation Finished ---")
    if pg.get_init(): pg.quit() # Clean up Pygame
    sys.exit()

if __name__ == '__main__':
    run_evaluation_main()