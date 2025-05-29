# utils/episode_runner.py
import pygame as pg
import os
import numpy as np
import imageio 
import sys 

GIFS_DIR_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gifs")
if not os.path.exists(GIFS_DIR_RUNNER):
    try: 
        os.makedirs(GIFS_DIR_RUNNER)
    except OSError as e:
        print(f"Warning: Could not create GIF directory {GIFS_DIR_RUNNER}: {e}")

def print_f_runner(*args, **kwargs): 
    print(*args, **kwargs); sys.stdout.flush()

def run_agent_episodes_for_test_eval(
    agent, 
    agent_name, 
    game_instance, 
    num_episodes, 
    max_steps_per_episode, 
    render_flag,
    run_mode, 
    gif_params=None, 
    is_eval_mode=False  # This flag determines if agent should be in "evaluation" (greedy) mode
    ):
    all_rewards, all_scores, all_steps = [], [], []
    stop_script_flag = False

    # Set agent to evaluation or training mode
    original_agent_eval_state = None
    if hasattr(agent, 'is_evaluating'): # Store original state if exists
        original_agent_eval_state = agent.is_evaluating
    if hasattr(agent, 'set_eval_mode'):
        agent.set_eval_mode(is_eval_mode)

    gif_record_n_eps = 0; gif_fps = 15; gif_capture_interval = 4
    max_gif_frames_seg = 600; model_name_gif = agent_name 
    gif_output_directory = GIFS_DIR_RUNNER

    if gif_params:
        gif_record_n_eps = gif_params.get('record_n_eps', 0)
        gif_fps = gif_params.get('fps', 15)
        gif_capture_interval = gif_params.get('capture_interval', 4)
        max_gif_frames_seg = gif_params.get('max_frames_segment', 600)
        model_name_gif = gif_params.get('model_name_for_gif', agent_name)
        gif_output_directory = gif_params.get('output_dir', GIFS_DIR_RUNNER) 

    for episode_idx in range(num_episodes):
        if stop_script_flag: break
        observation = game_instance.reset_for_ai()
        episode_reward, current_score_ep, current_steps_ep = 0.0, 0, 0
        
        gif_writer = None; current_gif_file_path_ep = None
        gif_segment_count_ep = 0; frames_in_current_segment_ep = 0
        record_this_ep_gif = (gif_record_n_eps > 0 and episode_idx < gif_record_n_eps and render_flag)

        def start_new_gif_segment_ep_local(ep_idx_func, segment_idx_func):
            nonlocal gif_writer, current_gif_file_path_ep
            segment_suffix = f"_part{segment_idx_func+1}" if segment_idx_func > 0 else ""
            status_suffix = "_IN_PROGRESS"
            gif_filename_base = f"{run_mode}_{model_name_gif}_ep{ep_idx_func+1}"
            temp_name = f"{gif_filename_base}{segment_suffix}{status_suffix}.gif"
            current_gif_file_path_ep = os.path.join(gif_output_directory, temp_name)
            try:
                if not os.path.exists(gif_output_directory): os.makedirs(gif_output_directory)
                gif_writer = imageio.get_writer(current_gif_file_path_ep, mode='I', fps=gif_fps, loop=0)
                # print_f_runner(f"    Recording GIF Segment {segment_idx_func+1} to {os.path.basename(current_gif_file_path_ep)}...")
                return True
            except Exception as e: 
                print_f_runner(f"    GIF writer error for seg {segment_idx_func+1}: {e}. GIF disabled.");
                gif_writer = None; current_gif_file_path_ep = None
                return False

        if record_this_ep_gif:
            if not start_new_gif_segment_ep_local(episode_idx, gif_segment_count_ep):
                gif_writer = None 

        for step in range(max_steps_per_episode):
            current_steps_ep = step + 1
            for event in pg.event.get():
                if event.type == pg.QUIT: print_f_runner("Quit signal."); stop_script_flag=True; break
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: print_f_runner("ESC pressed."); stop_script_flag=True; break
            if stop_script_flag: break

            # Agent chooses action. It will use its internal `is_evaluating` flag.
            action_to_take = agent.choose_action(observation)
            
            if render_flag or record_this_ep_gif: 
                if hasattr(game_instance, 'set_render_for_ai_this_step'):
                    game_instance.set_render_for_ai_this_step(True)
            
            next_observation, reward, done, info = game_instance.step_ai(action_to_take)
            episode_reward += reward
            observation = next_observation
            
            if gif_writer and current_gif_file_path_ep and (render_flag or record_this_ep_gif):
                if step % gif_capture_interval == 0:
                    try:
                        display_surface = pg.display.get_surface()
                        if display_surface:
                            frame = pg.surfarray.array3d(display_surface).transpose(1,0,2)
                            gif_writer.append_data(frame)
                            frames_in_current_segment_ep += 1
                            if frames_in_current_segment_ep >= max_gif_frames_seg and not done:
                                gif_writer.close() 
                                finalized_segment_name = f"{run_mode}_{model_name_gif}_ep{episode_idx+1}_part{gif_segment_count_ep+1}.gif"
                                finalized_segment_path = os.path.join(gif_output_directory, finalized_segment_name)
                                if os.path.exists(current_gif_file_path_ep): 
                                    os.rename(current_gif_file_path_ep, finalized_segment_path)
                                    # print_f_runner(f"    GIF Segment saved: {os.path.basename(finalized_segment_path)}")
                                gif_segment_count_ep += 1; frames_in_current_segment_ep = 0
                                if not start_new_gif_segment_ep_local(episode_idx, gif_segment_count_ep):
                                    gif_writer = None 
                        elif record_this_ep_gif:
                                print_f_runner(f"    Warning: No display surface for GIF frame at step {step}.")
                    except Exception as e: 
                        print_f_runner(f"    GIF frame/split error: {e}. GIF disabled."); 
                        if gif_writer: gif_writer.close(); gif_writer=None 
                        if current_gif_file_path_ep and os.path.exists(current_gif_file_path_ep):
                            try: os.remove(current_gif_file_path_ep) 
                            except OSError: pass
            if done: 
                current_score_ep = info.get('score',0)
                break 
        
        if gif_writer and current_gif_file_path_ep: 
            gif_writer.close()
            base_name = f"{run_mode}_{model_name_gif}_ep{episode_idx+1}"
            part_suffix = f"_part{gif_segment_count_ep+1}" if gif_segment_count_ep > 0 or (gif_segment_count_ep == 0 and frames_in_current_segment_ep > 0) else ""
            final_gif_name = f"{base_name}{part_suffix}_score{current_score_ep}.gif"
            final_gif_path = os.path.join(gif_output_directory, final_gif_name)
            try:
                if os.path.exists(current_gif_file_path_ep): 
                     if current_gif_file_path_ep != final_gif_path:
                        os.rename(current_gif_file_path_ep, final_gif_path)
                     print_f_runner(f"    GIF saved: {os.path.basename(final_gif_path)}")
                elif not os.path.exists(final_gif_path): 
                    print_f_runner(f"    Warning: Temp GIF file {current_gif_file_path_ep} not found for final rename.")
            except OSError as e_rename: print_f_runner(f"    Error renaming final GIF: {e_rename}")

        if stop_script_flag: break 
        all_rewards.append(episode_reward)
        all_scores.append(current_score_ep)
        all_steps.append(current_steps_ep)
        print_f_runner(f"    Ep {episode_idx+1}/{num_episodes}: Reward={episode_reward:.0f}, Score={current_score_ep}, Steps={current_steps_ep}")

    # Restore original agent evaluation state if it was changed
    if hasattr(agent, 'set_eval_mode') and original_agent_eval_state is not None:
        agent.set_eval_mode(original_agent_eval_state)
    elif hasattr(agent, 'set_eval_mode') and original_agent_eval_state is None: # If it had the method but we didn't store original
        agent.set_eval_mode(False) # Default to training mode

    return all_rewards, all_scores, all_steps, stop_script_flag