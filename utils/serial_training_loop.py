# utils/serial_training_loop.py
import os
import sys
import pygame as pg
import numpy as np
from collections import deque

def print_f_serial(*args, **kwargs): # Local print_f or import from a common util
    print(*args, **kwargs)
    sys.stdout.flush()

def run_serial_agent_training(
    agent, 
    agent_name, # For printing
    game_instance, 
    num_episodes, 
    max_steps_per_episode, 
    render_flag, 
    print_interval_steps, 
    model_save_path, 
    save_interval,
    agent_hparams, # For pop_size of Genetic, batch_size of DQN
    overall_rewards_deque, # To append rewards for global stats
    stop_training_flag_ref # A list [False] or similar mutable to signal global stop
    ):
    """
    Runs the training loop for serial agents (DQN, PPO, A2C, Genetic).
    `stop_training_flag_ref` is a list e.g. [False], modified if user quits.
    Returns True if training completed, False if interrupted.
    """
    print_f_serial(f"--- Serial Training: {agent_name.upper()} for {num_episodes} episodes/evals ---")
    print_f_serial(f"    (Saving to: {os.path.basename(model_save_path)})")

    total_steps_for_this_agent = agent.steps_done if hasattr(agent, 'steps_done') and agent.memory else 0 # Check if memory exists for DQN
    episode_rewards_window = deque(maxlen=20)

    for episode_idx in range(num_episodes):
        if stop_training_flag_ref[0]: break # Check global stop flag

        observation = game_instance.reset_for_ai()
        current_episode_reward, current_episode_total_loss, current_episode_learn_steps, current_step_in_episode = 0,0,0,0
        
        # For Genetic Agent population size from its hparams
        pop_size_ga = agent_hparams.get("population_size", 20) if agent_name == 'genetic' else 0
        if agent_name == 'genetic':
            # Ensure current_individual_idx is an attribute of agent
            if hasattr(agent, 'current_individual_idx'):
                 print_f_serial(f"Agent: GENETIC - Gen {episode_idx // pop_size_ga + 1}, Indiv {agent.current_individual_idx + 1}/{pop_size_ga}")
            else:
                 print_f_serial(f"Agent: GENETIC - Ep {episode_idx+1} (pop size for gen: {pop_size_ga})")


        for step in range(max_steps_per_episode):
            current_step_in_episode = step + 1
            stop_episode_early = False
            for event in pg.event.get():
                if event.type == pg.QUIT: stop_episode_early=True; stop_training_flag_ref[0]=True; print_f_serial("Quit."); break
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: stop_episode_early=True; stop_training_flag_ref[0]=True; print_f_serial("ESC."); break
            if stop_episode_early: break
            
            action = agent.choose_action(observation)
            next_observation, reward, done, info = game_instance.step_ai(action)
            current_episode_reward += reward
            
            if agent_name == 'dqn': # DQN increments its own steps_done
                pass # total_steps_for_this_agent might not be needed if agent.steps_done is used directly for learning condition

            loss_val = None
            if agent_name == 'dqn':
                agent.store_transition(observation, action, next_observation, reward, done)
                # Use agent's internal steps_done for learning condition
                if agent.steps_done > agent_hparams.get("batch_size", 32) * 5: 
                    loss_val = agent.learn()
            elif agent_name == 'a2c':
                agent.store_outcome(reward, done); loss_val = agent.learn(next_observation if not done else None)
            elif agent_name == 'ppo':
                loss_val = agent.store_transition_outcome(reward, done, next_observation)
            
            if loss_val is not None: current_episode_total_loss += loss_val; current_episode_learn_steps += 1
            observation = next_observation
            if render_flag: game_instance.render_for_ai()
            
            if print_interval_steps > 0 and (step + 1) % print_interval_steps == 0 and current_episode_learn_steps > 0:
                avg_step_loss = current_episode_total_loss / current_episode_learn_steps
                print_f_serial(f"  {agent_name.upper()} Ep {episode_idx+1}, Step {step+1}: AvgStepLoss={avg_step_loss:.4f}, CurRwd={current_episode_reward:.0f}")
                current_episode_total_loss, current_episode_learn_steps = 0,0
            if done: break
        
        if stop_episode_early or stop_training_flag_ref[0]: break
        
        episode_rewards_window.append(current_episode_reward)
        overall_rewards_deque.append(current_episode_reward)
        avg_reward_window = np.mean(episode_rewards_window) if episode_rewards_window else 0.0
        avg_loss_seg = (current_episode_total_loss / current_episode_learn_steps) if current_episode_learn_steps > 0 else 0.0
        game_score = info.get('score',0)
        print_f_serial(f"Agent: {agent_name.upper()} - Ep {episode_idx + 1}/{num_episodes} DONE. R={current_episode_reward:.0f}, Steps={current_step_in_episode}, AvgLoss={avg_loss_seg:.4f}, Score={game_score}, AvgR(last {len(episode_rewards_window)})={avg_reward_window:.2f}")

        if agent_name == 'genetic':
            agent.record_fitness(game_score)
            if hasattr(agent, 'current_individual_idx') and agent.current_individual_idx >= pop_size_ga:
                agent.learn(); 
                if hasattr(agent, 'save'): agent.save(model_save_path)
        
        if save_interval > 0 and (episode_idx + 1) % save_interval == 0:
            if agent_name not in ['genetic', 'neat', 'random'] and hasattr(agent, 'save'): # NEAT saves per gen
                agent.save(model_save_path)
    
    # Final save if not interrupted
    if not stop_training_flag_ref[0] and hasattr(agent, 'save'):
        if agent_name not in ['random']: # Genetic/NEAT would have saved their last gen already
            print_f_serial(f"Completed training for {agent_name}. Final save to {os.path.basename(model_save_path)}")
            agent.save(model_save_path)
    
    return not stop_training_flag_ref[0] # Return True if completed, False if interrupted