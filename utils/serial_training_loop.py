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

    # DQN's steps_done is internal and cumulative. For others, steps are per-episode for display.
    # total_steps_for_this_agent can track steps for print_interval if agent doesn't have its own counter.
    
    episode_rewards_window = deque(maxlen=20) # For per-agent rolling average

    for episode_idx in range(num_episodes):
        if stop_training_flag_ref[0]: break 

        observation = game_instance.reset_for_ai() # Resets game and gets initial observation
        current_episode_reward = 0.0
        current_episode_total_loss = 0.0
        current_episode_learn_steps = 0 # Number of times agent.learn() was called and returned a loss
        current_step_in_episode = 0
        
        pop_size_ga = agent_hparams.get("population_size", 20) if agent_name == 'genetic' else 1 # Default to 1 for non-GA
        
        # Genetic agent specific print
        if agent_name == 'genetic':
            # Assuming GeneticAgent has current_individual_idx and it's 0-indexed
            # Episode_idx for GA refers to individual evaluations.
            current_gen_ga = (agent.current_individual_idx // pop_size_ga) +1 if hasattr(agent, 'current_individual_idx') else (episode_idx // pop_size_ga) + 1
            current_ind_ga = (agent.current_individual_idx % pop_size_ga) +1 if hasattr(agent, 'current_individual_idx') else (episode_idx % pop_size_ga) + 1
            print_f_serial(f"Agent: GENETIC - Gen {current_gen_ga}, Indiv {current_ind_ga}/{pop_size_ga}")
        else:
            print_f_serial(f"Agent: {agent_name.upper()} - Episode {episode_idx + 1}/{num_episodes}")


        for step in range(max_steps_per_episode):
            current_step_in_episode = step + 1
            
            # Pygame event handling for interruption
            stop_episode_early = False
            for event in pg.event.get(): # Must pump events, especially if rendering
                if event.type == pg.QUIT: 
                    stop_episode_early=True; stop_training_flag_ref[0]=True; print_f_serial("Quit signal received."); break
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: 
                    stop_episode_early=True; stop_training_flag_ref[0]=True; print_f_serial("Escape key pressed."); break
            if stop_episode_early: break # Break from steps loop
            
            action = agent.choose_action(observation)
            
            # Signal to game instance if rendering is needed for this AI step
            if render_flag:
                game_instance.set_render_for_ai_this_step(True) # MODIFIED
            
            next_observation, reward, done, info = game_instance.step_ai(action)
            current_episode_reward += reward
            
            loss_val = None # Initialize loss for this step

            # Agent-specific learning logic
            if agent_name == 'dqn':
                agent.store_transition(observation, action, next_observation, reward, done)
                # DQN learns based on its internal step counter and buffer size
                if len(agent.memory) > agent.batch_size : # A common condition to start learning
                    loss_val = agent.learn()
            elif agent_name == 'a2c':
                agent.store_outcome(reward, done) # Store R, D for current step
                # A2C often learns at the end of a trajectory or when its buffer is full.
                # The learn method in A2CAgent is designed to be called after each step or when buffer full.
                # If A2C agent's learn is per-step, call it. If it's per-trajectory, it handles its own buffer.
                # The provided A2CAgent learns per step based on its buffer.
                if done : # Learn on terminal state
                     loss_val = agent.learn(None) # Pass None for next_obs if terminal
                # else: # Or learn every step if designed that way (A2CAgent handles its buffer)
                #    loss_val = agent.learn(next_observation) # This might be too frequent depending on A2C variant

            elif agent_name == 'ppo':
                # PPO stores transitions and learns when its trajectory buffer is full or episode ends
                loss_val = agent.store_transition_outcome(reward, done, next_observation)
            # Genetic agent learns per generation (handled after the loop for an individual)

            if loss_val is not None: 
                current_episode_total_loss += loss_val
                current_episode_learn_steps += 1
            
            observation = next_observation # Move to next state
            
            # Print step-level stats (optional)
            if print_interval_steps > 0 and (step + 1) % print_interval_steps == 0 and current_episode_learn_steps > 0:
                avg_step_loss_interval = current_episode_total_loss / current_episode_learn_steps
                print_f_serial(f"  {agent_name.upper()} Ep {episode_idx+1}, Step {step+1}: AvgLossInterval={avg_step_loss_interval:.4f}, EpRwd={current_episode_reward:.0f}")
                # Resetting interval loss/steps here or accumulate for episode avg
            
            if done: break # Break from steps loop if episode is done
        
        if stop_episode_early or stop_training_flag_ref[0]: break # Break from episodes loop
        
        episode_rewards_window.append(current_episode_reward)
        overall_rewards_deque.append(current_episode_reward) # Add to global deque for final stats
        
        avg_reward_rolling = np.mean(episode_rewards_window) if episode_rewards_window else 0.0
        avg_loss_episode = (current_episode_total_loss / current_episode_learn_steps) if current_episode_learn_steps > 0 else 0.0
        game_score = info.get('score',0) # Get final game score from info dict

        print_f_serial(
            f"Agent: {agent_name.upper()} - Ep {episode_idx + 1}/{num_episodes} "
            f"DONE. Rwd={current_episode_reward:.0f}, Steps={current_step_in_episode}, "
            f"AvgLossEp={avg_loss_episode:.4f}, GameScore={game_score}, "
            f"AvgRwd(last {len(episode_rewards_window)})={avg_reward_rolling:.2f}"
        )

        # Genetic Agent learning (per individual / end of population evaluation)
        if agent_name == 'genetic':
            agent.record_fitness(game_score) # Record fitness (usually game_score for GA)
            if hasattr(agent, 'current_individual_idx') and agent.current_individual_idx >= pop_size_ga:
                print_f_serial(f"  GENETIC: End of generation. Evolving population...")
                agent.learn() # Evolve population
                if hasattr(agent, 'save'): agent.save(model_save_path) # Save best after evolution
        
        # A2C learning (if it learns at the end of an episode rather than per step)
        # If A2CAgent is designed to accumulate a trajectory and learn at 'done':
        if agent_name == 'a2c' and done and hasattr(agent, 'learn_trajectory'): 
            loss_val_traj = agent.learn_trajectory() # Assuming a method like this
            if loss_val_traj is not None:
                 print_f_serial(f"  A2C Ep {episode_idx+1} Trajectory Loss: {loss_val_traj:.4f}")


        # Save models at specified interval (for DQN, PPO, A2C)
        # Genetic/NEAT save per generation/evolution step.
        if agent_name not in ['genetic', 'neat', 'random'] and hasattr(agent, 'save'):
            if save_interval > 0 and (episode_idx + 1) % save_interval == 0:
                print_f_serial(f"  Saving {agent_name} model at episode {episode_idx + 1}...")
                agent.save(model_save_path)
    
    # Final save for NN-based agents if training wasn't interrupted
    if not stop_training_flag_ref[0] and hasattr(agent, 'save'):
        if agent_name not in ['genetic', 'neat', 'random']: 
            print_f_serial(f"Completed training for {agent_name}. Final save to {os.path.basename(model_save_path)}")
            agent.save(model_save_path)
    elif stop_training_flag_ref[0]:
        print_f_serial(f"Training for {agent_name} interrupted. No final save unless interval save occurred.")
    
    return not stop_training_flag_ref[0]