# evaluate.py
import pygame as pg
import argparse
import sys
import os
import numpy as np

try:
    from game.game import Game
    # from game import config # Not directly needed if Game handles config
    from utils.model_helpers import get_latest_model_path, MODELS_DIR
    from utils.cli_args import add_common_eval_args, ALL_AGENT_TYPES_AVAILABLE
    from utils.agent_factory import create_agent
    from utils.episode_runner import run_agent_episodes_for_test_eval
    from utils.report_utils import save_evaluation_to_csv, EVAL_RESULTS_DIR
except ImportError as e:
    print(f"Error importing files in evaluate.py: {e}", flush=True); sys.exit(1)

def print_f(*args, **kwargs): print(*args, **kwargs); sys.stdout.flush()

def run_evaluation_main(): # Renamed
    parser = argparse.ArgumentParser(description="Evaluate all Space Invaders AI Agents")
    parser = add_common_eval_args(parser)
    args = parser.parse_args()

    print_f(f"--- Evaluating All Agents (Latest Models) ---")
    print_f(f"  Eval Episodes: {args.episodes}, Max Steps: {args.max_steps_per_episode}")

    game_instance = Game(silent_mode=True, ai_training_mode=True) # Fast and silent for eval
    preprocessed_obs_shape = (1, 84, 84)
    action_size = game_instance.get_action_size()
    
    evaluation_summary_data = []
    stop_script_flag = False

    for agent_name in ALL_AGENT_TYPES_AVAILABLE:
        if stop_script_flag: break
        print_f(f"\n--- Evaluating: {agent_name.upper()} ---")
        
        model_to_load = None
        if agent_name != 'random':
            model_to_load = get_latest_model_path(agent_name)
            if not model_to_load:
                print_f(f"  No model for {agent_name}. Skipping."); continue
            print_f(f"  Loading: {os.path.basename(model_to_load)}")

        agent_instance = create_agent(agent_name, action_size, preprocessed_obs_shape, mode='evaluate')
        if not agent_instance: print_f(f"  Failed to create agent {agent_name}. Skipping."); continue

        if model_to_load and hasattr(agent_instance, 'load') and agent_name != 'random':
            try: agent_instance.load(model_to_load)
            except Exception as e: print_f(f"  Load error for {agent_name}: {e}. Skipping."); continue
        
        rewards, scores, steps_taken, interrupted = run_agent_episodes_for_test_eval(
            agent=agent_instance,
            agent_name=agent_name,
            game_instance=game_instance,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps_per_episode,
            render_flag=False, # Evaluation is headless
            run_mode="test",
            gif_params=None,   # No GIFs for bulk evaluation
            is_eval_mode=True
        )

        if interrupted: stop_script_flag = True # Propagate interruption

        if rewards: # If any episodes completed for this agent
            avg_r, std_r = np.mean(rewards), np.std(rewards)
            avg_s, std_s = np.mean(scores), np.std(scores)
            avg_st = np.mean(steps_taken)
            print_f(f"  Results ({len(rewards)} eps): AvgR={avg_r:.2f}±{std_r:.2f}, AvgS={avg_s:.2f}±{std_s:.2f}, AvgSteps={avg_st:.1f}")
            evaluation_summary_data.append({
                "Agent": agent_name,
                "Model": os.path.basename(model_to_load) if model_to_load else "N/A",
                "Episodes": len(rewards),
                "Avg Reward": f"{avg_r:.2f}", "Std Reward": f"{std_r:.2f}",
                "Avg Score": f"{avg_s:.2f}", "Std Score": f"{std_s:.2f}",
                "Avg Steps": f"{avg_st:.1f}"
            })
        elif not interrupted:
            print_f(f"  No evaluation episodes completed for {agent_name}.")
            
    if evaluation_summary_data:
        save_evaluation_to_csv(evaluation_summary_data, output_dir_name=os.path.basename(EVAL_RESULTS_DIR))
    else:
        print_f("\nNo evaluation data generated.")

    print_f("\n--- Evaluation Finished ---")
    if pg.get_init(): pg.quit()
    sys.exit()

if __name__ == '__main__':
    run_evaluation_main()