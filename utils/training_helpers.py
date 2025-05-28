# utils/training_helpers.py
import os
import numpy as np
# Pygame and game imports will be inside the worker function

def evaluate_neat_genome_worker(args_tuple):
    """
    Worker function to evaluate a single NEAT genome's fitness.
    This function runs in a separate process.
    """
    worker_pid = os.getpid()
    print(f"[Worker PID: {worker_pid}] Starting evaluation.", flush=True)

    # Set SDL_VIDEODRIVER to dummy *before* any pygame imports in this process
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy" 

    import pygame as pg_worker 
    from game.game import Game as WorkerGame 
    from agents.dqn_agent import preprocess_observation as worker_preprocess
    from PIL import Image # For saving debug frames

    genome_copy, agent_hparams, game_action_size, game_preprocessed_shape, eval_max_steps = args_tuple
    
    game_sim = None
    try:
        # print(f"[Worker PID: {worker_pid}] Initializing Game instance...", flush=True)
        game_sim = WorkerGame(silent_mode=True, ai_training_mode=True, headless_worker_mode=True)
        # print(f"[Worker PID: {worker_pid}] Game instance initialized.", flush=True)
    except Exception as e_game_init:
        print(f"[Worker PID: {worker_pid}] FATAL ERROR initializing Game in worker: {e_game_init}", flush=True)
        # If game can't init, this worker can't proceed. Return a very low fitness.
        if pg_worker.get_init(): pg_worker.quit()
        return -float('inf') # Or some other indicator of failure

    class MinimalNeatAgentRef:
        def __init__(self, hparams, num_in, num_out, p_h, p_w):
            self.output_activation = hparams.get("output_activation", "tanh")
            self.hidden_activation = hparams.get("hidden_activation", "tanh")
            self.ff_passes = hparams.get("ff_passes", 3)
            self.num_inputs = num_in; self.num_outputs = num_out
            self.initial_connection_type = hparams.get("initial_connection_type", "minimal") # Needed by GenomeNEAT
            # Add other params from NEAT config if GenomeNEAT or its methods need them
            self.weight_perturb_strength = hparams.get("weight_perturb_strength", 0.1)
            self.weight_random_range = hparams.get("weight_random_range", 2.0)
            self.max_weight = hparams.get("max_weight", 5.0)
            # Note: innovation_tracker is NOT part of this minimal ref for evaluation
        
        def get_node_activation_name(self, node_id):
            if node_id < self.num_inputs: return "linear" 
            if node_id < self.num_inputs + self.num_outputs: return self.output_activation
            return self.hidden_activation

    genome_copy.agent_ref = MinimalNeatAgentRef(
        agent_hparams, 
        genome_copy.num_inputs, genome_copy.num_outputs,
        game_preprocessed_shape[1], game_preprocessed_shape[2]
    )

    observation_raw = game_sim.reset_for_ai()
    game_score_for_fitness = 0
    info = {} 
    done = False

    # --- DEBUG: Identify which genome this worker is evaluating ---
    # To make this more useful, you could pass genome_idx as part of args_tuple
    # print(f"[Worker PID: {worker_pid}] Evaluating genome (details if passed).")

    for step_idx in range(eval_max_steps):
        if observation_raw is None:
            print(f"[Worker PID: {worker_pid}] Error: observation_raw is None at step {step_idx}. Resetting.", flush=True)
            observation_raw = game_sim.reset_for_ai() # Attempt to recover
            if observation_raw is None: # Still None, serious issue
                print(f"[Worker PID: {worker_pid}] FATAL: observation_raw is still None after reset. Aborting eval.", flush=True)
                game_score_for_fitness = -float('inf') # Penalize heavily
                break


        processed_obs_np = worker_preprocess(observation_raw, 
                                             new_size=(game_preprocessed_shape[1], game_preprocessed_shape[2]))
        flat_input = processed_obs_np.flatten()
        
        # --- DEBUG PRINT FOR NEAT AGENT (inside worker) ---
        # This now calls the genome's feed_forward directly
        network_outputs = genome_copy.feed_forward(flat_input)
        action = np.argmax(network_outputs)

        # Print for the first few steps of this worker's current genome evaluation
        if step_idx < 3: # Print for the first 3 steps
            print(f"  [Worker PID: {worker_pid}, Step {step_idx+1}] Genome Outputs: {np.round(network_outputs, 3)}, Action: {action}", flush=True)
        
        # --- DEBUG IMAGE SAVING (inside worker) ---
        # if step_idx < 2: # Save first 2 preprocessed frames
        #     try:
        #         img_to_save_data = (processed_obs_np.squeeze() * 255).astype(np.uint8)
        #         img_to_save = Image.fromarray(img_to_save_data, mode='L')
        #         debug_image_dir = os.path.join(os.path.dirname(__file__), "..", "debug_frames_worker") # Save in project root/debug_frames_worker
        #         if not os.path.exists(debug_image_dir): os.makedirs(debug_image_dir)
        #         img_to_save.save(os.path.join(debug_image_dir, f"worker_{worker_pid}_eval_step_{step_idx}.png"))
        #     except Exception as e_img:
        #         print(f"    [Worker PID: {worker_pid}] Error saving debug frame: {e_img}", flush=True)
        # --- END DEBUG IMAGE SAVING ---
        
        try:
            observation_raw, _, done, info = game_sim.step_ai(action)
        except Exception as e_step:
            print(f"[Worker PID: {worker_pid}] Error during game_sim.step_ai(): {e_step}", flush=True)
            done = True # Assume error means episode ends
            game_score_for_fitness = -float('inf') # Penalize
        
        if done:
            game_score_for_fitness = info.get('score', 0)
            break
    
    if not done: 
        game_score_for_fitness = info.get('score', 0)

    # print(f"[Worker PID: {worker_pid}] Evaluation finished. Fitness: {game_score_for_fitness}", flush=True)
    if pg_worker.get_init():
        pg_worker.quit()
    return game_score_for_fitness