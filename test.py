# test.py
import pygame as pg
import argparse
import sys
import os
import numpy as np

try:
    from game.game_manager import Game # MODIFIED
    from utils.model_helpers import get_latest_model_path, MODELS_DIR
    from utils.cli_args import add_common_test_args, add_gif_args, ALL_AGENT_TYPES_AVAILABLE
    from utils.agent_factory import create_agent
    from utils.episode_runner import run_agent_episodes_for_test_eval, GIFS_DIR as GIF_OUTPUT_DIR # Use GIFS_DIR from episode_runner
except ImportError as e:
    print(f"Error importing files in test.py: {e}", flush=True); sys.exit(1)

def print_f(*args, **kwargs): print(*args, **kwargs); sys.stdout.flush()

def run_test_main(): 
    parser = argparse.ArgumentParser(description="Test a Space Invaders AI Agent")
    parser = add_common_test_args(parser)
    parser = add_gif_args(parser) 
    args = parser.parse_args()

    # For testing, ai_training_mode=False (game runs at normal speed unless overridden by render settings)
    # headless_worker_mode depends on args.render. If rendering, not headless. If not rendering, can be headless.
    game_instance = Game(
        silent_mode=args.silent, 
        ai_training_mode=False, # Test mode is not training mode
        headless_worker_mode=not args.render # Headless if not rendering
    ) 
    
    print_f(f"--- Testing Agent: {args.agent.upper()} ---")
    print_f(f"  Episodes: {args.episodes}, Render: {args.render}, Max Steps: {args.max_steps_per_episode}, Silent: {args.silent}")
    if args.gif_episodes > 0:
        print_f(f"  GIF Recording: {args.gif_episodes} episode(s), FPS: {args.gif_fps}, Capture Interval: {args.gif_capture_every_n_steps} steps, Max Frames/Segment: {args.max_gif_frames}")
        print_f(f"  GIFs will be saved to: {GIF_OUTPUT_DIR}") # Inform user about GIF save location


    preprocessed_obs_shape = (1, 84, 84) # Standard for CNN agents (C, H, W)
    action_size = game_instance.get_action_size()
    
    path_to_load = args.model_file_path # This comes directly from CLI if specified
    
    if not path_to_load and args.agent != 'random': # If no specific path, try to get latest for the agent
        path_to_load = get_latest_model_path(args.agent)
        if path_to_load:
            print_f(f"  No specific model_file_path provided. Loading LATEST model: {os.path.basename(path_to_load)}")
        else:
            print_f(f"  Warning: No model found for {args.agent.upper()} (neither specific nor latest). Agent will run untrained/randomly initialized.")

    # Create agent instance in 'test' mode
    agent_instance = create_agent(args.agent, action_size, preprocessed_obs_shape, mode='test') 
    if not agent_instance:
        print_f(f"Error: Failed to create agent '{args.agent}'. Exiting.")
        if pg.get_init(): pg.quit()
        sys.exit(1)

    if path_to_load and hasattr(agent_instance, 'load') and args.agent != 'random':
        try:
            agent_instance.load(path_to_load)
            print_f(f"  Successfully loaded model: {os.path.basename(path_to_load)}")
        except Exception as e:
            print_f(f"  Error loading model from {os.path.basename(path_to_load)}: {e}. Agent will run untrained.")
            path_to_load = None # Mark as not loaded for display purposes
    
    gif_params_dict = None
    if args.gif_episodes > 0:
        # Ensure GIF output directory exists
        if not os.path.exists(GIF_OUTPUT_DIR):
            try:
                os.makedirs(GIF_OUTPUT_DIR)
                print_f(f"Created GIF output directory: {GIF_OUTPUT_DIR}")
            except OSError as e_gif_dir:
                print_f(f"Warning: Could not create GIF directory {GIF_OUTPUT_DIR}: {e_gif_dir}. GIFs might not be saved.")
        
        model_name_for_gif = args.agent # Default to agent type
        if path_to_load: # If a model was loaded, use its name for GIF filename
            model_name_for_gif = os.path.splitext(os.path.basename(path_to_load))[0]
        
        gif_params_dict = {
            'record_n_eps': args.gif_episodes,
            'fps': args.gif_fps,
            'capture_interval': args.gif_capture_every_n_steps,
            'max_frames_segment': args.max_gif_frames,
            'model_name_for_gif': model_name_for_gif,
            'output_dir': GIF_OUTPUT_DIR 
        }

    # Run episodes using the common runner function
    all_rewards, all_scores, _, user_interrupted = run_agent_episodes_for_test_eval(
        agent=agent_instance,
        agent_name=args.agent,
        game_instance=game_instance,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        render_flag=args.render,
        run_mode="test", 
        gif_params=gif_params_dict,
        is_eval_mode=False # Test mode is for observation, not strict performance evaluation like `evaluate.py`
                           # This flag can control aspects like detailed logging in episode_runner if needed
    )
    
    if user_interrupted:
        print_f("\n--- Test interrupted by user ---")
    elif all_rewards: # If episodes were run and not interrupted immediately
        model_display_name = os.path.basename(path_to_load) if path_to_load else ('Untrained' if args.agent != 'random' else 'N/A')
        print_f(f"\n--- Test Results for {args.agent.upper()} (Model: {model_display_name}) ---")
        print_f(f"  Episodes completed: {len(all_rewards)}")
        print_f(f"  Average Reward: {np.mean(all_rewards):.2f} (Std: {np.std(all_rewards):.2f})")
        print_f(f"  Average Score:  {np.mean(all_scores):.2f} (Std: {np.std(all_scores):.2f})")
        print_f(f"  Min Reward: {np.min(all_rewards):.2f}, Max Reward: {np.max(all_rewards):.2f}")
        print_f(f"  Min Score: {np.min(all_scores):.2f}, Max Score: {np.max(all_scores):.2f}")
    elif not user_interrupted: # No rewards, not interrupted (e.g., agent failed or 0 episodes)
        print_f(f"\n  No test episodes were completed for {args.agent.upper()}.")

    print_f("\n--- Test Finished ---")
    if pg.get_init(): pg.quit() # Clean up Pygame
    sys.exit()

if __name__ == '__main__':
    run_test_main()