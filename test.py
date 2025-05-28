# test.py
import pygame as pg
import argparse
import sys
import os
import numpy as np

try:
    from game.game import Game
    # from game import config # Not directly needed if Game handles config
    from utils.model_helpers import get_latest_model_path, MODELS_DIR
    from utils.cli_args import add_common_test_args, add_gif_args, ALL_AGENT_TYPES_AVAILABLE
    from utils.agent_factory import create_agent
    from utils.episode_runner import run_agent_episodes_for_test_eval, GIFS_DIR as GIF_OUTPUT_DIR
except ImportError as e:
    print(f"Error importing files in test.py: {e}", flush=True); sys.exit(1)

def print_f(*args, **kwargs): print(*args, **kwargs); sys.stdout.flush()

def run_test_main(): # Renamed from run_test to avoid conflict if imported
    parser = argparse.ArgumentParser(description="Test a Space Invaders AI Agent")
    parser = add_common_test_args(parser)
    # MAX_FRAMES_PER_GIF_SEGMENT needs to be accessible for default in add_gif_args
    # For simplicity, hardcode or pass it. Let's use a default from add_gif_args.
    parser = add_gif_args(parser) 
    args = parser.parse_args()

    game_instance = Game(silent_mode=args.silent, ai_training_mode=False) # Slower, human-like for testing
    
    print_f(f"--- Testing Agent: {args.agent.upper()} ---")
    print_f(f"  Episodes: {args.episodes}, Render: {args.render}, Max Steps: {args.max_steps_per_episode}")
    if args.gif_episodes > 0:
        print_f(f"  GIF: {args.gif_episodes} eps, FPS: {args.gif_fps}, Interval: {args.gif_capture_every_n_steps}, MaxFrames: {args.max_gif_frames}")

    preprocessed_obs_shape = (1, 84, 84) # Standard for CNN agents
    action_size = game_instance.get_action_size()
    
    path_to_load = args.model_file_path
    if not path_to_load and args.agent != 'random':
        path_to_load = get_latest_model_path(args.agent)
        if path_to_load: print_f(f"  Loading LATEST: {os.path.basename(path_to_load)}")
        else: print_f(f"  Warning: No model for {args.agent}. Running untrained.")

    agent_instance = create_agent(args.agent, action_size, preprocessed_obs_shape, mode='test') # Factory handles test params
    if not agent_instance: pg.quit(); sys.exit(1)

    if path_to_load and hasattr(agent_instance, 'load') and args.agent != 'random':
        try: agent_instance.load(path_to_load)
        except Exception as e: print_f(f"  Load failed {os.path.basename(path_to_load)}: {e}. Untrained.")
    
    gif_params_dict = None
    if args.gif_episodes > 0:
        model_name_for_gif = args.agent
        if path_to_load: model_name_for_gif = os.path.splitext(os.path.basename(path_to_load))[0]
        gif_params_dict = {
            'record_n_eps': args.gif_episodes,
            'fps': args.gif_fps,
            'capture_interval': args.gif_capture_every_n_steps,
            'max_frames_segment': args.max_gif_frames,
            'model_name_for_gif': model_name_for_gif,
            'output_dir': GIF_OUTPUT_DIR # Pass the correct GIF output directory
        }

    all_rewards, all_scores, _, stop_flag = run_agent_episodes_for_test_eval(
        agent=agent_instance,
        agent_name=args.agent,
        game_instance=game_instance,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        render_flag=args.render,
        run_mode="test",
        gif_params=gif_params_dict,
        is_eval_mode=True # Test mode is essentially evaluation mode for agent behavior
    )
    
    if not stop_flag and all_rewards:
        print_f(f"\n--- Test Results for {args.agent} ({os.path.basename(path_to_load) if path_to_load else 'Untrained'}) ---")
        print_f(f"  Avg Reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
        print_f(f"  Avg Score:  {np.mean(all_scores):.2f} +/- {np.std(all_scores):.2f}")
    elif not stop_flag:
        print(f"  No test episodes completed for {args.agent}.")

    print_f("\n--- Test Finished ---")
    if pg.get_init(): pg.quit()
    sys.exit()

if __name__ == '__main__':
    run_test_main()