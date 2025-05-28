# utils/episode_runner.py
import pygame as pg
import os
import numpy as np
import imageio # For GIF creation
import sys # For print_f_runner

# Assumes GIFS_DIR is defined in the calling script (test.py or evaluate.py)
# or passed as an argument. For simplicity, let's assume it's passed or globally accessible.
# Or, define it here based on this file's location.
GIFS_DIR_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gifs")
if not os.path.exists(GIFS_DIR_RUNNER):
    os.makedirs(GIFS_DIR_RUNNER)

def print_f_runner(*args, **kwargs): # Local print_f
    print(*args, **kwargs); sys.stdout.flush()

def run_agent_episodes_for_test_eval(
    agent, 
    agent_name, 
    game_instance, 
    num_episodes, 
    max_steps_per_episode, 
    render_flag,
    run_mode, # NEW: Pass "test" or "evaluate"
    gif_params=None, 
    is_eval_mode=False 
    ):
    """
    Runs an agent for a number of episodes, for testing or evaluation.
    Handles rendering and GIF recording.
    Returns (list_of_rewards, list_of_scores, list_of_steps_taken_per_episode, was_interrupted_flag)
    """
    all_rewards, all_scores, all_steps = [], [], []
    stop_script_flag = False

    gif_record_n_eps = 0
    gif_fps = 15
    gif_capture_interval = 4
    max_gif_frames_seg = 600
    model_name_gif = agent_name 

    if gif_params:
        gif_record_n_eps = gif_params.get('record_n_eps', 0)
        gif_fps = gif_params.get('fps', 15)
        gif_capture_interval = gif_params.get('capture_interval', 4)
        max_gif_frames_seg = gif_params.get('max_frames_segment', 600)
        model_name_gif = gif_params.get('model_name_for_gif', agent_name)
        # GIF_OUTPUT_DIR should be part of gif_params or a global/passed constant
        gif_output_directory = gif_params.get('output_dir', GIFS_DIR_RUNNER) 
    else:
        gif_output_directory = GIFS_DIR_RUNNER


    for episode_idx in range(num_episodes):
        if stop_script_flag: break
        observation = game_instance.reset_for_ai()
        episode_reward, current_score_ep, current_steps_ep = 0, 0, 0 # Renamed current_score
        
        gif_writer = None; temp_gif_path_ep = ""; gif_segment_count_ep = 0; frames_in_current_segment_ep = 0
        record_this_ep_gif = (gif_record_n_eps > 0 and episode_idx < gif_record_n_eps)
        current_gif_file_path_ep = None

        # Define start_new_gif_segment_ep inside where 'done' and 'current_score_ep' are in scope
        # or pass them as arguments. For now, it's defined where it can access them.

        def start_new_gif_segment_ep_local(ep_idx_func, segment_idx_func, episode_done_flag, final_score_func):
            nonlocal gif_writer 
            nonlocal current_gif_file_path_ep # Allow modification
            segment_suffix = f"_part{segment_idx_func+1}" if segment_idx_func > 0 else ""
            
            status_suffix = "_IN_PROGRESS"
            # Determine filename suffix based on whether the episode is done
            # This logic for naming will be handled when closing the segment.
            # For temporary name, just use IN_PROGRESS.
            
            gif_filename_base = f"{run_mode}_{model_name_gif}_ep{ep_idx_func+1}" # Use run_mode here
            temp_name = f"{gif_filename_base}{segment_suffix}{status_suffix}.gif"
            current_gif_file_path_ep = os.path.join(gif_output_directory, temp_name) # Use determined output dir

            try:
                gif_writer = imageio.get_writer(current_gif_file_path_ep, mode='I', fps=gif_fps, loop=0)
                print_f_runner(f"    Recording GIF Segment {segment_idx_func+1} to {os.path.basename(current_gif_file_path_ep)}...")
                return True # Indicates success
            except Exception as e: 
                print_f_runner(f"    GIF writer error for segment {segment_idx_func+1}: {e}. Disabled for this segment.");
                gif_writer = None # Ensure writer is None if failed
                current_gif_file_path_ep = None
                return False # Indicates failure

        if record_this_ep_gif:
            # Initial call to start the first segment. 'done' is false, 'current_score_ep' is 0.
            if not start_new_gif_segment_ep_local(episode_idx, gif_segment_count_ep, False, 0):
                gif_writer = None 

        for step in range(max_steps_per_episode):
            current_steps_ep = step + 1
            for event in pg.event.get():
                if event.type == pg.QUIT: print_f_runner("Quit."); stop_script_flag=True; break
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: print_f_runner("ESC."); stop_script_flag=True; break
            if stop_script_flag: break

            action_to_take = None
            # Agent action selection (handle eval mode)
            if agent_name == 'dqn' and is_eval_mode : 
                 original_eps_start = agent.eps_start; original_eps_end = agent.eps_end
                 agent.eps_start = 0.0; agent.eps_end = 0.0
                 action_to_take = agent.choose_action(observation)
                 agent.eps_start = original_eps_start; agent.eps_end = original_eps_end
            else: # For other agents or DQN not in strict eval mode
                 action_to_take = agent.choose_action(observation)


            next_observation, reward, done, info = game_instance.step_ai(action_to_take)
            episode_reward += reward
            observation = next_observation
            
            if render_flag or gif_writer: game_instance.render_for_ai()
            
            if gif_writer and current_gif_file_path_ep:
                if step % gif_capture_interval == 0:
                    try:
                        frame = pg.surfarray.array3d(pg.display.get_surface()).transpose(1,0,2)
                        gif_writer.append_data(frame)
                        frames_in_current_segment_ep += 1

                        if frames_in_current_segment_ep >= max_gif_frames_seg and not done:
                            gif_writer.close() # Close current segment
                            # Finalize name of completed segment (without IN_PROGRESS)
                            finalized_segment_name = f"{run_mode}_{model_name_gif}_ep{episode_idx+1}_part{gif_segment_count_ep+1}.gif"
                            finalized_segment_path = os.path.join(gif_output_directory, finalized_segment_name)
                            if os.path.exists(current_gif_file_path_ep): 
                                os.rename(current_gif_file_path_ep, finalized_segment_path)
                                print_f_runner(f"    GIF Segment saved: {finalized_segment_path}")
                            
                            gif_segment_count_ep += 1; frames_in_current_segment_ep = 0
                            if not start_new_gif_segment_ep_local(episode_idx, gif_segment_count_ep, False, 0): # Start new segment
                                gif_writer = None # Disable further GIF writing if new segment fails
                    
                    except Exception as e: 
                        print_f_runner(f"    GIF frame/split error: {e}. GIF disabled for current segment."); 
                        if gif_writer: gif_writer.close(); gif_writer=None # Ensure close on error
                        if current_gif_file_path_ep and os.path.exists(current_gif_file_path_ep):
                            try: os.remove(current_gif_file_path_ep) 
                            except OSError: pass
            
            if done: current_score_ep = info.get('score',0); break
        
        # End of episode, close any open GIF writer and finalize its name
        if gif_writer and current_gif_file_path_ep: 
            gif_writer.close()
            segment_suffix = f"_part{gif_segment_count_ep+1}" if gif_segment_count_ep > 0 or (frames_in_current_segment_ep > 0 and gif_segment_count_ep == 0 and not os.path.exists(os.path.join(gif_output_directory, f"{run_mode}_{model_name_gif}_ep{episode_idx+1}_score{current_score_ep}.gif")) ) else ""
            final_gif_name = f"{run_mode}_{model_name_gif}_ep{episode_idx+1}{segment_suffix}_score{current_score_ep}.gif"
            final_gif_path = os.path.join(gif_output_directory, final_gif_name)
            try:
                if os.path.exists(current_gif_file_path_ep): 
                     os.rename(current_gif_file_path_ep, final_gif_path)
                     print_f_runner(f"    GIF saved: {final_gif_path}")
            except OSError as e_rename: print_f_runner(f"    Error renaming final GIF segment: {e_rename}")

        if stop_script_flag: break
        all_rewards.append(episode_reward); all_scores.append(current_score_ep); all_steps.append(current_steps_ep)
        print_f_runner(f"    Ep {episode_idx+1}: Reward={episode_reward:.0f}, Score={current_score_ep}, Steps={current_steps_ep}")

    return all_rewards, all_scores, all_steps, stop_script_flag