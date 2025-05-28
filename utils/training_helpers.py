# utils/training_helpers.py
import os
import numpy as np
# No Pygame import at module level here

def evaluate_neat_genome_worker(args_tuple):
    """
    Worker function to evaluate a single NEAT genome's fitness.
    This function runs in a separate process.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy" 

    import pygame as pg_worker # Import Pygame AFTER setting env vars
    from game.game import Game as WorkerGame # Game class will handle its own Pygame init
    from agents.dqn_agent import preprocess_observation as worker_preprocess # Assuming dqn_agent.py is safe to import

    genome_copy, agent_hparams, game_action_size, game_preprocessed_shape, eval_max_steps = args_tuple
    
    # WorkerGame.__init__ will call pg_worker.init(), set display mode (dummy), and load_all_game_images()
    game_sim = WorkerGame(silent_mode=True, ai_training_mode=True, headless_worker_mode=True)
    
    class MinimalNeatAgentRef: # (As defined before)
        def __init__(self, hparams, num_in, num_out, p_h, p_w):
            self.output_activation = hparams.get("output_activation", "tanh")
            self.hidden_activation = hparams.get("hidden_activation", "tanh")
            self.ff_passes = hparams.get("ff_passes", 3); self.num_inputs = num_in; self.num_outputs = num_out
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

    for _ in range(eval_max_steps):
        processed_obs_np = worker_preprocess(observation_raw, 
                                             new_size=(game_preprocessed_shape[1], game_preprocessed_shape[2]))
        flat_input = processed_obs_np.flatten()
        network_outputs = genome_copy.feed_forward(flat_input)
        action = np.argmax(network_outputs)
        
        observation_raw, _, done, info = game_sim.step_ai(action)
        
        if done:
            game_score_for_fitness = info.get('score', 0)
            break
    
    if not done: 
        game_score_for_fitness = info.get('score', 0)

    if pg_worker.get_init():
        pg_worker.quit()
    return game_score_for_fitness