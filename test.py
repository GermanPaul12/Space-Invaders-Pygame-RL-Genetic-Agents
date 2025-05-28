# test.py
import pygame as pg
import argparse
import sys
import os
import numpy as np
import imageio # For GIF creation

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
        get_existing_model_versions,
        MODELS_DIR # Use MODELS_DIR from helpers
    )
except ImportError as e:
    print(f"Error importing files in test.py: {e}", flush=True)
    sys.exit(1)

GIFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gifs")
if not os.path.exists(GIFS_DIR):
    os.makedirs(GIFS_DIR)

ALL_AGENT_TYPES_AVAILABLE = ['dqn', 'ppo', 'a2c', 'genetic', 'neat', 'random']

# Approximate max frames to keep GIF under 50MB. This is highly dependent on content and GIF settings.
# 84x84 grayscale, 15 FPS. An uncompressed frame is 84*84 bytes = ~7KB.
# GIF compression is decent. Let's try a conservative estimate.
# 50MB = 50 * 1024 * 1024 bytes.
# If each compressed frame averages 2-5KB: 50000KB / 5KB = 10000 frames. This is likely too high.
# If each compressed frame averages 0.5-1KB for simple scenes: 50000KB / 1KB = 50000 frames.
# A 15 FPS GIF for 1 minute = 15 * 60 = 900 frames. This might be around 5-15MB.
# Let's set a lower, safer limit first for testing, e.g., 600 frames (40 seconds at 15fps).
# You will need to TUNE THIS VALUE based on observed GIF sizes.
MAX_FRAMES_PER_GIF_SEGMENT = 600 # Adjust this value!

def print_f(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def run_test():
    parser = argparse.ArgumentParser(description="Test a Space Invaders AI Agent")
    parser.add_argument("--agent", type=str, choices=ALL_AGENT_TYPES_AVAILABLE, required=True, help="Agent to test.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes.")
    parser.add_argument("--model_file_path", type=str, default=None, help="Specific model file path to load. If None, loads latest.")
    parser.add_argument("--render", action='store_true', help="Render game.")
    parser.add_argument("--max_steps_per_episode", type=int, default=3000)
    parser.add_argument("--silent", action='store_true', help="No sounds.")
    parser.add_argument("--gif_episodes", type=int, default=0, help="Record N initial episodes as GIF.")
    parser.add_argument("--gif_fps", type=int, default=15)
    parser.add_argument("--gif_capture_every_n_steps", type=int, default=4)
    parser.add_argument("--max_gif_frames", type=int, default=MAX_FRAMES_PER_GIF_SEGMENT,
                        help=f"Max frames per GIF segment before splitting (default: {MAX_FRAMES_PER_GIF_SEGMENT}). Tune for size.")
    
    args = parser.parse_args()
    max_gif_frames_per_segment = args.max_gif_frames


    game_instance = Game(silent_mode=args.silent, ai_training_mode=False)
    
    print_f(f"--- Testing Agent: {args.agent.upper()} ---")
    # ... (other initial prints) ...
    if args.gif_episodes > 0:
        print_f(f"GIF Max Frames/Segment: {max_gif_frames_per_segment}")


    preprocessed_obs_shape = (1, 84, 84)
    action_size = game_instance.get_action_size()
    agent_instance = None
    path_to_load = args.model_file_path

    if not path_to_load and args.agent != 'random':
        path_to_load = get_latest_model_path(args.agent)
        if path_to_load: print_f(f"Loading LATEST: {os.path.basename(path_to_load)}")
        else: print_f(f"Warning: No model for {args.agent}. Running untrained.")

    # Agent Instantiation (with test-specific params)
    if args.agent == 'random': agent_instance = RandomAgent(action_size)
    elif args.agent == 'dqn': agent_instance = DQNAgent(action_size, preprocessed_obs_shape, eps_start=0.00, eps_end=0.00)
    elif args.agent == 'a2c': agent_instance = A2CAgent(action_size, preprocessed_obs_shape, entropy_coef=0.0)
    elif args.agent == 'ppo': agent_instance = PPOAgent(action_size, preprocessed_obs_shape, entropy_coef=0.0)
    elif args.agent == 'genetic': agent_instance = GeneticAgent(action_size, preprocessed_obs_shape, population_size=1, mutation_rate=0)
    elif args.agent == 'neat':
        from agents.neat_agent import NEAT_POPULATION_SIZE # Not used here, pop_size=1 for test
        agent_instance = NEATAgent(action_size, preprocessed_obs_shape, population_size=1)
    else: print_f(f"Unknown agent: {args.agent}"); pg.quit(); sys.exit(1)

    if path_to_load and hasattr(agent_instance, 'load') and args.agent != 'random':
        try: agent_instance.load(path_to_load)
        except Exception as e: print_f(f"Could not load model {os.path.basename(path_to_load)}: {e}. Running untrained.")
    
    all_rewards, all_scores = [], []
    stop_script_flag = False

    for episode_idx in range(args.episodes):
        if stop_script_flag: break
        observation = game_instance.reset_for_ai()
        episode_reward, current_score, current_steps_ep = 0, 0, 0
        
        model_name_for_gif = args.agent
        if path_to_load: model_name_for_gif = os.path.splitext(os.path.basename(path_to_load))[0]
        
        record_this_episode_gif = (args.gif_episodes > 0 and episode_idx < args.gif_episodes)
        gif_writer = None
        gif_segment_count = 0
        frames_in_current_gif_segment = 0
        
        def start_new_gif_segment(ep_idx_func, segment_idx_func, score_func=None):
            nonlocal gif_writer # To modify the outer scope variable
            segment_suffix = f"_part{segment_idx_func}" if segment_idx_func > 0 else ""
            score_suffix = f"_score{score_func}" if score_func is not None and segment_idx_func == 0 else "" # only for first/full if not split
            
            # For ongoing parts, don't add score yet. Add IN_PROGRESS.
            # For the very first segment, it might become the only segment.
            status_suffix = "_IN_PROGRESS"
            if done and segment_idx_func > 0 : # if this is the end of a multi-part gif.
                 status_suffix = f"_finalscore{current_score}" # Use actual current_score
            elif done and segment_idx_func == 0: # single part gif ended
                 status_suffix = f"_score{current_score}"


            gif_filename_base = f"test_{model_name_for_gif}_ep{ep_idx_func+1}"
            # Use current_score for the *final* segment filename if the episode ends.
            # For intermediate segments, just use part number.
            # This logic for naming will be handled when closing.
            
            # Temporary naming while segment is being written
            temp_gif_filename = f"{gif_filename_base}{segment_suffix}{status_suffix}.gif"
            current_gif_path = os.path.join(GIFS_DIR, temp_gif_filename)

            try:
                gif_writer = imageio.get_writer(current_gif_path, mode='I', fps=args.gif_fps, loop=0)
                print_f(f"    Recording GIF Segment {segment_idx_func+1} for episode {ep_idx_func+1} to {os.path.basename(current_gif_path)}...")
                return current_gif_path
            except Exception as e: 
                print_f(f"      GIF writer error for segment {segment_idx_func+1}: {e}. Disabled for this segment.");
                return None

        current_gif_file_path = None
        if record_this_episode_gif:
            current_gif_file_path = start_new_gif_segment(episode_idx, gif_segment_count)

        for step in range(args.max_steps_per_episode):
            current_steps_ep = step + 1
            for event in pg.event.get():
                if event.type == pg.QUIT: print_f("Quit."); stop_script_flag=True; break
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: print_f("ESC."); stop_script_flag=True; break
            if stop_script_flag: break

            action = agent_instance.choose_action(observation)
            next_observation, reward, done, info = game_instance.step_ai(action)
            episode_reward += reward
            observation = next_observation
            
            if args.render or gif_writer: game_instance.render_for_ai()
            
            if gif_writer and current_gif_file_path:
                if step % args.gif_capture_every_n_steps == 0:
                    try:
                        frame = pg.surfarray.array3d(pg.display.get_surface()).transpose(1,0,2)
                        gif_writer.append_data(frame)
                        frames_in_current_gif_segment += 1

                        if frames_in_current_gif_segment >= max_gif_frames_per_segment and not done:
                            print_f(f"    Max frames reached for GIF segment {gif_segment_count+1}. Closing and starting new one.")
                            gif_writer.close()
                            # Rename the completed segment (remove IN_PROGRESS)
                            finalized_segment_name = f"test_{model_name_for_gif}_ep{episode_idx+1}_part{gif_segment_count+1}.gif"
                            finalized_segment_path = os.path.join(GIFS_DIR, finalized_segment_name)
                            try:
                                if os.path.exists(current_gif_file_path):
                                    os.rename(current_gif_file_path, finalized_segment_path)
                                    print_f(f"      GIF Segment saved: {finalized_segment_path}")
                            except OSError as e_rename: print_f(f"    Error renaming segment: {e_rename}")

                            gif_segment_count += 1
                            frames_in_current_gif_segment = 0
                            current_gif_file_path = start_new_gif_segment(episode_idx, gif_segment_count)
                            if not current_gif_file_path: # Failed to start new segment writer
                                gif_writer = None # Disable further gif writing for this episode
                                
                    except Exception as e: 
                        print_f(f"    GIF frame/split error: {e}. GIF disabled for current segment."); 
                        gif_writer.close(); gif_writer=None
                        if current_gif_file_path and os.path.exists(current_gif_file_path):
                            try: os.remove(current_gif_file_path) # Remove partial/failed segment
                            except OSError: pass
            
            if done: current_score = info.get('score',0); break
        
        # End of episode, close any open GIF writer
        if gif_writer and current_gif_file_path: 
            gif_writer.close()
            # Finalize filename for the last/only segment
            segment_suffix = f"_part{gif_segment_count+1}" if gif_segment_count > 0 else ""
            final_gif_name = f"test_{model_name_for_gif}_ep{episode_idx+1}{segment_suffix}_score{current_score}.gif"
            final_gif_path = os.path.join(GIFS_DIR, final_gif_name)
            try:
                if os.path.exists(current_gif_file_path): # current_gif_file_path has IN_PROGRESS
                     os.rename(current_gif_file_path, final_gif_path)
                     print_f(f"    GIF saved: {final_gif_path}")
            except OSError as e_rename: print_f(f"    Error renaming final GIF segment: {e_rename}")


        if stop_script_flag: break
        all_rewards.append(episode_reward); all_scores.append(current_score)
        print_f(f"  Ep {episode_idx+1}: Reward={episode_reward:.0f}, Score={current_score}, Steps={current_steps_ep}")

    if not stop_script_flag and all_rewards:
        print_f(f"\n--- Test Results for {args.agent} ({os.path.basename(path_to_load) if path_to_load else 'Untrained'}) ---")
        print_f(f"  Avg Reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
        print_f(f"  Avg Score: {np.mean(all_scores):.2f} +/- {np.std(all_scores):.2f}")
    
    print_f("\n--- Test Finished ---")
    pg.quit()
    sys.exit()

if __name__ == '__main__':
    run_test()