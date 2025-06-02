# agents/neat_agent.py
import os
import time
import json
import random
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import numpy as np
from PIL import Image
import pickle

try:
    import neat
except ImportError:
    print("Error: neat-python library not found. Please install it with 'pip install neat-python'")
    # Define dummy classes if neat is not available to prevent import errors if Agent is instantiated
    class neat:
        class Population: pass
        class Config: pass
        class DefaultGenome: pass
        class nn:
            class FeedForwardNetwork:
                @staticmethod
                def create(genome, config): pass
    # Raise an error or exit if neat is critical and not found
    # raise ImportError("neat-python is required for NEATAgent.")


from .agent import Agent # Your base Agent class

# --- Preprocessing ---
def preprocess_observation_neat(obs, new_size=(84, 84), flatten=True):
    if obs is None:
        flat_size = new_size[0] * new_size[1] if flatten else 1
        return np.zeros(flat_size if flatten else (1, new_size[0], new_size[1]), dtype=np.float32)
    if isinstance(obs, tuple):
        obs = obs[0]

    img = Image.fromarray(obs)
    img = img.convert('L')
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    if flatten:
        return img_array.flatten()
    else:
        return np.expand_dims(img_array, axis=0) # (1, H, W)


class NEATAgent(Agent):
    _default_neat_hparams = {
        "max_generations": 100,
        "max_steps_per_eval": 2000,
        "neat_config_file": "configs/neat_config.txt", # Default path, relative to ROOT_DIR
        "checkpoint_prefix": "neat_checkpoint_",
        "save_best_genome_filename": "best_neat_genome.pkl"
    }
    DEFAULT_ENV_ID = "ALE/SpaceInvaders-v5"

    def __init__(self, env_id, hparams, mode, models_dir_for_agent, gifs_dir_for_agent):
        super().__init__(env_id if env_id else self.DEFAULT_ENV_ID,
                         hparams, mode, models_dir_for_agent, gifs_dir_for_agent)

        if 'neat' not in globals() or not hasattr(neat, 'Config'): # Check if neat imported correctly
            raise ImportError("neat-python is required for NEATAgent but was not imported successfully.")

        self.merged_hparams = self._default_neat_hparams.copy()
        self.merged_hparams.update(self.hparams)

        # NEAT config path needs to be absolute or relative to a known root
        config_file_path = self.merged_hparams["neat_config_file"]
        if not os.path.isabs(config_file_path):
            # Assuming ROOT_DIR is defined where main.py is, and this agent is in agents/
            # So, go up one level from self.models_dir (which is ROOT_DIR/models) to get ROOT_DIR
            project_root_dir = os.path.dirname(self.models_dir) if self.models_dir else "."
            config_file_path = os.path.join(project_root_dir, config_file_path)
        
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"NEAT configuration file not found at: {config_file_path}. "
                                    "Please create it. A sample is in neat_agent.py comments.")
        
        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                       neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                       config_file_path)
        
        # Verify num_inputs and num_outputs match environment
        temp_env = gym.make(self.env_id)
        self.action_size = temp_env.action_space.n
        # Preprocessed and flattened observation size
        self.flat_obs_size = 84 * 84 
        temp_env.close()

        if self.neat_config.genome_config.num_inputs != self.flat_obs_size:
            raise ValueError(f"NEAT config num_inputs ({self.neat_config.genome_config.num_inputs}) "
                             f"does not match expected flat observation size ({self.flat_obs_size}). Adjust neat_config.txt.")
        if self.neat_config.genome_config.num_outputs != self.action_size:
            raise ValueError(f"NEAT config num_outputs ({self.neat_config.genome_config.num_outputs}) "
                             f"does not match environment action size ({self.action_size}). Adjust neat_config.txt.")

        print(f"  NEAT Agent: Env '{self.env_id}', Action Size: {self.action_size}, Flat Obs Size: {self.flat_obs_size}")

        self.population = None # Will be initialized or loaded in train/load
        self.best_genome_overall = None
        self.current_generation = 0
        self.active_net = None # For choose_action during test/eval

        self.checkpoint_dir = os.path.join(self.models_dir, "neat_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_genome_save_path = os.path.join(self.models_dir, self.merged_hparams["save_best_genome_filename"])


    def get_model_file_extension(self):
        # NEAT uses its own checkpointing. We also save the best genome as .pkl.
        return ".pkl" # For the best_genome file

    def _eval_genomes(self, genomes, config):
        """ Evaluates a list of genomes, assigning fitness. Called by NEAT. """
        eval_env = gym.make(self.env_id, render_mode="rgb_array", full_action_space=False) # Headless for speed
        max_steps = self.merged_hparams['max_steps_per_eval']

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = 0.0 # Start with zero fitness

            obs, info = eval_env.reset()
            current_obs_p_flat = preprocess_observation_neat(obs, flatten=True)
            total_reward = 0
            
            for _ in range(max_steps):
                outputs = net.activate(current_obs_p_flat)
                action = np.argmax(outputs) # Choose action with highest activation
                
                next_obs_raw, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                current_obs_p_flat = preprocess_observation_neat(next_obs_raw, flatten=True)
                
                if terminated or truncated:
                    break
            genome.fitness = total_reward
        eval_env.close()


    def train(self, episodes, max_steps_per_episode, render_mode_str, # episodes is generations
              path_to_load_model, force_new_training_if_model_exists,
              save_interval_eps, print_interval_steps, **kwargs): # print_interval for stats reporter

        num_generations = episodes if episodes > 0 else self.merged_hparams['max_generations']
        # max_steps_per_episode is used by _eval_genomes as max_steps_per_eval
        self.merged_hparams['max_steps_per_eval'] = max_steps_per_episode if max_steps_per_episode > 0 else self.merged_hparams['max_steps_per_eval']

        print(f"\n--- NEAT Training Started ---")
        print(f"  Generations: {num_generations}, Max Steps/Genome Eval: {self.merged_hparams['max_steps_per_eval']}")
        print(f"  NEAT Config: {self.merged_hparams['neat_config_file']}")

        # Load population or create new
        loaded_checkpoint = None
        if path_to_load_model and os.path.exists(path_to_load_model) and not force_new_training_if_model_exists:
            # path_to_load_model for NEAT usually means a checkpoint file, or our best_genome.pkl
            if path_to_load_model.endswith(".pkl"): # It's our best_genome.pkl, not a population checkpoint
                print(f"  Loading best genome from {path_to_load_model}, but NEAT needs a population to continue training.")
                print(f"  Will start new population unless a NEAT checkpoint is also found.")
                self.load(path_to_load_model) # Loads self.best_genome_overall
                # Now try to find a numeric checkpoint to resume population training
                checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith(self.merged_hparams["checkpoint_prefix"])]
                if checkpoints:
                    latest_checkpoint_num = max([int(f.split('_')[-1]) for f in checkpoints])
                    loaded_checkpoint = os.path.join(self.checkpoint_dir, f"{self.merged_hparams['checkpoint_prefix']}{latest_checkpoint_num}")
            elif path_to_load_model.startswith(self.merged_hparams["checkpoint_prefix"]): # It is a checkpoint
                loaded_checkpoint = os.path.join(self.checkpoint_dir, os.path.basename(path_to_load_model))
            else: # Unrecognized, assume it's a best_genome.pkl
                self.load(path_to_load_model)

        if loaded_checkpoint and os.path.exists(loaded_checkpoint):
            print(f"  Restoring NEAT population from checkpoint: {loaded_checkpoint}")
            self.population = neat.Checkpointer.restore_checkpoint(loaded_checkpoint)
            self.current_generation = self.population.generation
        else:
            if force_new_training_if_model_exists and path_to_load_model:
                 print(f"  Force new training specified. Ignoring potential load path {path_to_load_model} for population.")
            print("  Creating new NEAT population.")
            self.population = neat.Population(self.neat_config)
            self.current_generation = 0
            self.best_genome_overall = None # Reset if starting fresh

        # Add reporters for output
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        # Checkpoint interval: NEAT's Checkpointer saves every N generations if generation_interval is set.
        # save_interval_eps from main.py can map to this.
        checkpoint_interval = save_interval_eps if save_interval_eps > 0 else num_generations # Save at end if 0
        self.population.add_reporter(neat.Checkpointer(generation_interval=checkpoint_interval,
                                                       time_interval_seconds=None, # No time-based checkpointing
                                                       filename_prefix=os.path.join(self.checkpoint_dir, self.merged_hparams["checkpoint_prefix"])))
        
        start_generation_for_run = self.population.generation # Generation number from loaded pop
        generations_to_run = num_generations - start_generation_for_run

        if generations_to_run <=0:
            print(f"  Loaded population is already at or beyond target generation {num_generations}. Training complete.")
            if self.population.best_genome: self.best_genome_overall = self.population.best_genome
        else:
            print(f"  Running NEAT evolution for {generations_to_run} more generations (up to gen {num_generations}).")
            try:
                winner_genome = self.population.run(self._eval_genomes, generations_to_run)
                if winner_genome:
                    self.best_genome_overall = winner_genome # Store the overall best
                print(f"\n--- NEAT Training Finished ---")
                if self.best_genome_overall:
                    print(f"Best genome found: ID {self.best_genome_overall.key}, Fitness: {self.best_genome_overall.fitness:.2f}")
                else: # Could happen if evolution stops early or no clear winner
                    print("No single best genome returned by population.run. Checking population's best.")
                    if self.population.best_genome:
                        self.best_genome_overall = self.population.best_genome
                        print(f"Best genome from population: ID {self.best_genome_overall.key}, Fitness: {self.best_genome_overall.fitness:.2f}")


            except KeyboardInterrupt:
                print("\n  NEAT Training interrupted by user.")
            except Exception as e:
                print(f"  An error occurred during NEAT training: {e}")
                import traceback; traceback.print_exc()
            finally:
                 # Save best genome explicitly, as NEAT's checkpointer only saves population
                if self.best_genome_overall is None and self.population and self.population.best_genome:
                    self.best_genome_overall = self.population.best_genome # Ensure we have the best from the run
                self.save(self.best_genome_save_path) # Saves the best_genome_overall via .pkl

    def choose_action(self, observation_preprocessed_flat, deterministic=True):
        if self.active_net is None:
            # This can happen if test/evaluate is called without loading a model/genome
            # Or if called directly after init before any training or loading.
            print("Warning: NEAT active_net not set. Choosing random action.")
            return random.randrange(self.action_size)
        
        outputs = self.active_net.activate(observation_preprocessed_flat)
        action = np.argmax(outputs)
        return action

    def test(self, model_path_to_load, episodes, max_steps_per_episode,
             render_during_test, record_video_flag, video_fps, **kwargs):
        print(f"\n--- NEAT Testing ---")
        
        if model_path_to_load and os.path.exists(model_path_to_load):
            print(f"  Loading best NEAT genome from: {model_path_to_load}")
            self.load(model_path_to_load) # This should load self.best_genome_overall
        elif model_path_to_load:
             print(f"  Warning: NEAT test model path '{model_path_to_load}' not found.")
        
        if self.best_genome_overall is None:
            print("  Error: No best NEAT genome loaded or found for testing. Cannot proceed.")
            return

        self.active_net = neat.nn.FeedForwardNetwork.create(self.best_genome_overall, self.neat_config)
        print(f"  Testing with best genome ID {self.best_genome_overall.key}, Fitness: {getattr(self.best_genome_overall, 'fitness', 'N/A'):.2f}")

        test_env_render_mode = "human" if render_during_test else "rgb_array"
        test_env = gym.make(self.env_id, render_mode=test_env_render_mode, full_action_space=False)
        if record_video_flag:
            ts = time.strftime("%Y%m%d_%H%M%S")
            video_folder = os.path.join(self.gifs_dir if self.gifs_dir else "videos", f"neat_test_{ts}")
            test_env = RecordVideo(test_env, video_folder=video_folder, name_prefix=f"neat_test_run",
                                   episode_trigger=lambda ep_id: True, fps=video_fps)
        test_env = RecordEpisodeStatistics(test_env)

        all_rewards = []
        for i in range(episodes):
            obs, info = test_env.reset()
            current_obs_p_flat = preprocess_observation_neat(obs, flatten=True)
            episode_reward, current_steps = 0, 0
            for _ in range(max_steps_per_episode):
                action = self.choose_action(current_obs_p_flat)
                next_obs_raw, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward
                current_obs_p_flat = preprocess_observation_neat(next_obs_raw, flatten=True)
                current_steps +=1
                if render_during_test and test_env_render_mode=="human" and not record_video_flag: test_env.render()
                if terminated or truncated: break
            actual_ep_reward = info.get('episode', {}).get('r', episode_reward)
            all_rewards.append(actual_ep_reward)
            print(f"  NEAT Test Episode {i+1}/{episodes} - Score: {actual_ep_reward:.0f}, Steps: {current_steps}")
        
        test_env.close()
        avg_r = np.mean(all_rewards) if all_rewards else 0
        std_r = np.std(all_rewards) if all_rewards else 0
        print(f"\n  NEAT Test Summary: Avg Score: {avg_r:.2f} +/- {std_r:.2f}")
        print(f"--- NEAT Testing Ended ---")

    def evaluate(self, model_path_to_load, episodes, max_steps_per_episode, **kwargs):
        print(f"\n--- NEAT Evaluation ---")
        if model_path_to_load and os.path.exists(model_path_to_load):
            self.load(model_path_to_load)
        elif model_path_to_load:
            print(f"  Warning: NEAT eval model path '{model_path_to_load}' not found.")

        if self.best_genome_overall is None:
            print("  Error: No best NEAT genome available for evaluation.")
            return {}
        
        self.active_net = neat.nn.FeedForwardNetwork.create(self.best_genome_overall, self.neat_config)
        print(f"  Evaluating with best genome ID {self.best_genome_overall.key}, Fitness: {getattr(self.best_genome_overall, 'fitness', 'N/A'):.2f}")

        eval_env = gym.make(self.env_id, render_mode="rgb_array", full_action_space=False)
        eval_env = RecordEpisodeStatistics(eval_env)
        all_ep_rewards, all_ep_steps = [], []

        for i in range(episodes):
            obs, info = eval_env.reset()
            current_obs_p_flat = preprocess_observation_neat(obs, flatten=True)
            ep_r, ep_s = 0, 0
            for _ in range(max_steps_per_episode):
                action = self.choose_action(current_obs_p_flat)
                next_obs_raw, reward, terminated, truncated, info = eval_env.step(action)
                ep_r += reward; ep_s += 1
                current_obs_p_flat = preprocess_observation_neat(next_obs_raw, flatten=True)
                if terminated or truncated: break
            actual_ep_reward = info.get('episode', {}).get('r', ep_r)
            actual_ep_steps = info.get('episode', {}).get('l', ep_s)
            all_ep_rewards.append(actual_ep_reward)
            all_ep_steps.append(actual_ep_steps)
            if (i + 1) % max(1, episodes // 5) == 0:
                 print(f"    NEAT Eval Ep {i+1}/{episodes}: Score={actual_ep_reward:.0f}, Steps={actual_ep_steps}")
        eval_env.close()

        results = {}
        if all_ep_rewards:
            results = {
                "num_episodes_eval": episodes,
                "avg_score": round(np.mean(all_ep_rewards), 2),
                "std_dev_score": round(np.std(all_ep_rewards), 2),
                "min_score": round(np.min(all_ep_rewards), 2),
                "max_score": round(np.max(all_ep_rewards), 2),
                "avg_steps": round(np.mean(all_ep_steps), 1)
            }
        print(f"--- NEAT Evaluation Ended ---")
        return results

    def save(self, path_for_best_genome_pkl): # Path is for the .pkl best genome
        # NEAT population checkpointing is handled by its own Checkpointer reporter
        # This save method is primarily for saving the best genome found overall
        if self.best_genome_overall is not None and path_for_best_genome_pkl:
            os.makedirs(os.path.dirname(path_for_best_genome_pkl), exist_ok=True)
            try:
                with open(path_for_best_genome_pkl, 'wb') as f:
                    pickle.dump(self.best_genome_overall, f)
                print(f"  Best NEAT genome saved to {os.path.basename(path_for_best_genome_pkl)}")
            except Exception as e:
                print(f"  Error saving best NEAT genome to {path_for_best_genome_pkl}: {e}")
        elif not self.best_genome_overall:
            print("  NEAT save attempt: No best_genome_overall to save.")
        
        # Note: The population itself is saved by neat.Checkpointer during `population.run()`

    def load(self, path_to_best_genome_pkl): # Path is for the .pkl best genome
        # This method primarily loads the best_genome_overall.
        # Population loading for resuming training is handled by neat.Checkpointer.restore_checkpoint
        # in the train() method.
        if path_to_best_genome_pkl and os.path.exists(path_to_best_genome_pkl):
            try:
                with open(path_to_best_genome_pkl, 'rb') as f:
                    self.best_genome_overall = pickle.load(f)
                print(f"  Best NEAT genome loaded from {os.path.basename(path_to_best_genome_pkl)}")
                if self.best_genome_overall:
                    # Prepare active_net if a best genome was loaded, for immediate test/eval
                    self.active_net = neat.nn.FeedForwardNetwork.create(self.best_genome_overall, self.neat_config)
            except Exception as e:
                print(f"  Error loading best NEAT genome from {path_to_best_genome_pkl}: {e}")
                self.best_genome_overall = None
        elif path_to_best_genome_pkl:
            print(f"  Load path for best NEAT genome does not exist: {path_to_best_genome_pkl}")

        # For resuming training, main.py might pass a checkpoint path to train method,
        # or train method can look for latest checkpoint.
        # This load focuses on the best individual for testing/evaluation.