# agents/dqn_agent.py
import os
import time
import json
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import numpy as np
import torch # SB3 uses PyTorch

try:
    from stable_baselines3 import DQN as SB3_DQN
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
    from stable_baselines3.common.atari_wrappers import AtariWrapper # For single env recording
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 or its dependencies not found. DQNAgent will not be available.")
    # Provide dummy classes if SB3 is not available to avoid import errors if Agent is instantiated
    class SB3_DQN: pass
    class VecFrameStack: pass
    class BaseCallback: pass


from .agent import Agent # Your base Agent class

class PrintCallback(BaseCallback):
    """
    A custom callback that prints a message at a set frequency.
    """
    def __init__(self, print_interval_steps=10000, verbose=0):
        super(PrintCallback, self).__init__(verbose)
        self.print_interval_steps = print_interval_steps
        self.last_print_step = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_print_step) >= self.print_interval_steps:
            if self.logger: # Check if logger is available (it should be)
                # Try to get recent mean reward from the Monitor wrapper's buffer
                # This assumes the environment is wrapped with SB3's Monitor or RecordEpisodeStatistics
                # which populates info['episode']
                ep_info_buffer = self.training_env.get_attr('ep_info_buffer', indices=0) # For VecEnv
                if ep_info_buffer: # Check if the VecEnv has this attribute
                    ep_info_buffer = ep_info_buffer[0] # Get it from the first env
                    if len(ep_info_buffer) > 0:
                        recent_rewards = [ep_info['r'] for ep_info in ep_info_buffer]
                        mean_reward = np.mean(recent_rewards) if recent_rewards else float('nan')
                        self.logger.record("rollout/ep_rew_mean_custom", mean_reward)
                        print(f"Custom Log: Timesteps: {self.num_timesteps}, Mean Reward (last ~100): {mean_reward:.2f}")

                # Standard SB3 logging call to output all recorded values
                self.logger.dump(step=self.num_timesteps)
            else:
                 print(f"Timesteps: {self.num_timesteps}")
            self.last_print_step = self.num_timesteps
        return True


class DQNAgent(Agent):
    """
    A Deep Q-Network Agent using Stable Baselines 3.
    """
    # Default hyperparameters for SB3 DQN, can be overridden by JSON config
    _default_sb3_dqn_hparams = {
        "learning_rate": 0.0001,
        "buffer_size": 100_000, # Adjusted for typical Atari
        "learning_starts": 10_000, # How many steps to collect data before training starts
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4, # Train every N steps
        "gradient_steps": 1, # How many gradient steps to perform when training
        "exploration_fraction": 0.1, # Fraction of entire training period over which exploration rate is reduced
        "exploration_final_eps": 0.01,
        "target_update_interval": 1000, # Update target network every N steps
        "policy_kwargs": None, # e.g., dict(net_arch=[256, 256])
        "tensorboard_log": None, # Path to log tensorboard data, e.g., "./dqn_tensorboard/"
        "verbose": 0, # 0 for no output, 1 for info messages, 2 for debug messages
        "device": "auto"
    }
    # Environment specific params for SB3 make_atari_env
    _sb3_env_params = {
        "n_envs": 1, # Number of parallel environments. More can speed up training.
        "seed": None, # Seed for reproducibility
        "wrapper_kwargs": dict(clip_rewards=True, episodic_life=True, fire_on_reset=True),
        "n_stack": 4, # Number of frames to stack for FrameStack wrapper
    }
    # SB3 uses "SpaceInvadersNoFrameskip-v4" for Atari typically
    DEFAULT_ENV_ID = "SpaceInvadersNoFrameskip-v4"


    def __init__(self, env_id, hparams, mode, models_dir_for_agent, gifs_dir_for_agent):
        if not SB3_AVAILABLE:
            raise ImportError("Stable Baselines 3 is not available. Please install it to use DQNAgent.")

        super().__init__(env_id if env_id else self.DEFAULT_ENV_ID,
                         hparams, mode, models_dir_for_agent, gifs_dir_for_agent)
        
        self.merged_hparams = self._default_sb3_dqn_hparams.copy()
        self.merged_hparams.update(self.hparams) # User config overrides defaults

        # Env params can also be in hparams from JSON if needed
        self.env_creation_params = self._sb3_env_params.copy()
        if "n_envs" in self.merged_hparams: self.env_creation_params["n_envs"] = self.merged_hparams.pop("n_envs")
        if "seed" in self.merged_hparams: self.env_creation_params["seed"] = self.merged_hparams.pop("seed")
        if "n_stack" in self.merged_hparams: self.env_creation_params["n_stack"] = self.merged_hparams.pop("n_stack")


        # Tensorboard log path setup
        if self.merged_hparams.get("tensorboard_log") is True: # if just "true", use default path
            self.merged_hparams["tensorboard_log"] = os.path.join(self.models_dir, "tensorboard_logs", "dqn")
        elif isinstance(self.merged_hparams.get("tensorboard_log"), str): # if path string given
            self.merged_hparams["tensorboard_log"] = os.path.join(self.models_dir, self.merged_hparams["tensorboard_log"])


        # The actual SB3 model (self.model) will be created in train or when loading
        # self.action_size and self.observation_shape are set by SB3 internally based on env
        # For info:
        try:
            temp_env = make_atari_env(self.env_id, n_envs=1, wrapper_kwargs=self.env_creation_params["wrapper_kwargs"])
            temp_env = VecFrameStack(temp_env, n_stack=self.env_creation_params["n_stack"])
            self.action_size = temp_env.action_space.n
            self.observation_shape = temp_env.observation_space.shape
            print(f"  DQN Agent: Env '{self.env_id}', Action Size: {self.action_size}, Obs Shape: {self.observation_shape}")
            temp_env.close()
            del temp_env
        except Exception as e:
            print(f"  DQN Agent: Could not query temp env for spaces: {e}")
            self.action_size = 6 # Default for SpaceInvaders
            self.observation_shape = (self.env_creation_params["n_stack"], 84, 84) # Default for Atari preprocessing

    def get_model_file_extension(self):
        return ".zip"

    def _create_env(self, for_training=True, n_envs_override=None):
        """Helper to create the SB3 vectorized environment."""
        env_seed = self.env_creation_params["seed"]
        if env_seed is None and for_training: # Use a time-based seed for training if none specified
            env_seed = int(time.time())
        elif env_seed is None and not for_training: # Use a fixed seed for eval/test if none specified
            env_seed = 42

        n_actual_envs = n_envs_override if n_envs_override is not None else self.env_creation_params["n_envs"]
        
        print(f"  Creating SB3 VecEnv: {self.env_id}, N_Envs={n_actual_envs}, Seed={env_seed}, N_Stack={self.env_creation_params['n_stack']}")

        vec_env = make_atari_env(
            self.env_id,
            n_envs=n_actual_envs,
            seed=env_seed,
            wrapper_kwargs=self.env_creation_params["wrapper_kwargs"]
        )
        vec_env = VecFrameStack(vec_env, n_stack=self.env_creation_params["n_stack"])
        return vec_env

    def train(self, episodes, max_steps_per_episode, render_mode_str,
              path_to_load_model, force_new_training_if_model_exists,
              save_interval_eps, print_interval_steps, **kwargs):
        
        print(f"\n--- DQN Training Started ---")
        print(f"  Episodes: {episodes}, Max Steps/Ep: {max_steps_per_episode}")
        print(f"  Load Model Path: {path_to_load_model}")
        print(f"  Force New: {force_new_training_if_model_exists}")
        print(f"  Save Interval (eps): {save_interval_eps}, Print Interval (steps): {print_interval_steps}")

        # Rendering during SB3 VecEnv training is not straightforward and usually avoided
        if render_mode_str == "human":
            print("  Warning: Human rendering during SB3 VecEnv training is generally not supported directly by main.py's method. Training will be headless.")

        self.train_env = self._create_env(for_training=True)

        # Determine save path
        self.model_save_path = None
        if path_to_load_model and os.path.exists(path_to_load_model) and not force_new_training_if_model_exists:
            self.model_save_path = path_to_load_model # Continue training this model
            print(f"  Attempting to load and continue training model: {os.path.basename(self.model_save_path)}")
            try:
                self.model = SB3_DQN.load(self.model_save_path, env=self.train_env, device=self.merged_hparams["device"])
                print(f"  Model loaded successfully from {os.path.basename(self.model_save_path)}.")
            except Exception as e:
                print(f"  Failed to load model: {e}. Creating a new model.")
                path_to_load_model = None # Nullify to prevent re-attempt
                self.model_save_path = self.get_next_version_save_path("dqn") # Get a new path
                self.model = SB3_DQN("CnnPolicy", self.train_env, **self.merged_hparams)
        else:
            if force_new_training_if_model_exists and path_to_load_model and os.path.exists(path_to_load_model):
                print(f"  Force new training: A model exists at {os.path.basename(path_to_load_model)} but will be versioned.")
                self.model_save_path = self.get_next_version_save_path("dqn")
            elif path_to_load_model and not os.path.exists(path_to_load_model):
                print(f"  Specified model to load not found: {path_to_load_model}. Creating new model.")
                self.model_save_path = self.get_next_version_save_path("dqn")
            else: # No load path or doesn't exist, and not forcing new over existing (or no existing)
                self.model_save_path = self.get_model_save_path_for_agent("dqn") # Default name
                if os.path.exists(self.model_save_path) and not force_new_training_if_model_exists:
                     print(f"  Model {os.path.basename(self.model_save_path)} exists. Consider --force_train or loading if you want to overwrite or make new version.")
                     # Decide: either load it, or make a new version path.
                     # For simplicity, if it exists and we aren't loading/forcing, let's just get a new version path.
                     self.model_save_path = self.get_next_version_save_path("dqn")


            print(f"  Creating new DQN model. It will be saved to: {os.path.basename(self.model_save_path)}")
            self.model = SB3_DQN("CnnPolicy", self.train_env, **self.merged_hparams)

        # SB3's `learn` method uses total_timesteps.
        # We need to convert episodes to an approximate number of timesteps.
        # This is a rough estimate; actual steps can vary.
        total_timesteps_estimate = episodes * max_steps_per_episode
        print(f"  Target total timesteps for SB3 learn(): ~{total_timesteps_estimate}")

        # Callbacks
        callbacks = []
        if print_interval_steps > 0:
            callbacks.append(PrintCallback(print_interval_steps=print_interval_steps))
        
        # SB3 handles episodes internally. If save_interval_eps is used,
        # it requires a custom callback that checks episode count.
        # For simplicity with SB3, we'll save at the end of `total_timesteps_estimate`.
        # If `save_interval_eps` is very important, a more complex callback is needed.
        if save_interval_eps > 0:
            print(f"  Note: save_interval_eps ({save_interval_eps}) for SB3 DQN is best handled with a custom CheckpointCallback based on episodes, not directly supported here. Model will save at end of training.")


        try:
            self.model.learn(
                total_timesteps=total_timesteps_estimate,
                callback=callbacks if callbacks else None,
                log_interval=max(1, print_interval_steps // (self.model.n_steps if hasattr(self.model, 'n_steps') else 2048)), # Heuristic, n_steps is for on-policy
                reset_num_timesteps= (path_to_load_model is None) # Reset timesteps if it's a new model
            )
            print(f"  DQN training loop finished.")
            self.save(self.model_save_path) # Save final model
        except KeyboardInterrupt:
            print("\n  Training interrupted by user. Saving current model...")
            self.save(self.model_save_path)
        except Exception as e:
            print(f"  An error occurred during DQN training: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Attempting to save model at interruption/error to {self.model_save_path}")
            self.save(self.model_save_path) # Attempt to save on error
        finally:
            if self.train_env:
                self.train_env.close()
            print(f"--- DQN Training Ended ---")


    def test(self, model_path_to_load, episodes, max_steps_per_episode,
             render_during_test, record_video_flag, video_fps, **kwargs):
        print(f"\n--- DQN Testing ---")
        if not model_path_to_load or not os.path.exists(model_path_to_load):
            print(f"  Error: Model path '{model_path_to_load}' not found or not specified. Cannot test.")
            if self.mode == 'test' and model_path_to_load is None: # main.py might allow testing untrained
                print(f"  Attempting to test with a new, untrained model instance (if this was intended).")
                # Create a temporary environment for model instantiation structure
                temp_eval_env = self._create_env(for_training=False, n_envs_override=1)
                self.model = SB3_DQN("CnnPolicy", temp_eval_env, **self.merged_hparams)
                temp_eval_env.close() # Close immediately after
                model_path_to_load = "Untrained Model" # For display
            else:
                return

        # Create a single environment for evaluation/testing
        # SB3 evaluate_policy can use a VecEnv, but for simple testing loop, single env is fine
        # For rendering, we need a single env anyway.
        eval_env = gym.make(self.env_id, render_mode="human" if render_during_test else "rgb_array", full_action_space=False)
        # Apply necessary wrappers manually if not using make_atari_env for this single env.
        # AtariWrapper includes: NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame, ScaledFloatFrame, ClipRewardEnv, FrameStack
        # For consistency with training's VecFrameStack, we need similar preprocessing.
        eval_env = AtariWrapper(eval_env, clip_reward=False) # No reward clipping for pure evaluation
        eval_env = gym.wrappers.FrameStack(eval_env, self.env_creation_params["n_stack"]) # Manual FrameStack
        eval_env = Monitor(eval_env) # To get episode statistics like SB3's default VecEnv setup

        if model_path_to_load != "Untrained Model":
            print(f"  Loading model for testing: {os.path.basename(model_path_to_load)}")
            # SB3 load needs an env for structure sometimes, but for predict it might not be strict
            # Providing the eval_env (or a similarly structured one) is safer.
            try:
                self.model = SB3_DQN.load(model_path_to_load, env=None, device=self.merged_hparams["device"]) # Try loading without env first
            except Exception: # If it fails (e.g. custom policy needs env), try with env
                 self.model = SB3_DQN.load(model_path_to_load, env=eval_env, device=self.merged_hparams["device"])
        
        all_rewards = []
        best_reward_for_video = -float('inf')
        video_env_instance = None # To hold the env instance that recorded the video

        for i in range(episodes):
            obs, info = eval_env.reset()
            terminated, truncated = False, False
            episode_reward = 0
            current_steps = 0
            
            while not (terminated or truncated) and current_steps < max_steps_per_episode:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                current_steps += 1
                if render_during_test and eval_env.render_mode == "human":
                    eval_env.render() # Should be handled by gym_play if that was used. Here, manual.
            
            actual_ep_reward = info.get('episode', {}).get('r', episode_reward) # From Monitor wrapper
            all_rewards.append(actual_ep_reward)
            print(f"  Test Episode {i+1}/{episodes} - Score: {actual_ep_reward:.0f}, Steps: {current_steps}")

            if actual_ep_reward > best_reward_for_video:
                best_reward_for_video = actual_ep_reward
        
        eval_env.close() # Close the primary test environment

        avg_reward = np.mean(all_rewards) if all_rewards else 0
        std_reward = np.std(all_rewards) if all_rewards else 0
        print(f"\n  Test Summary: Avg Score: {avg_reward:.2f} +/- {std_reward:.2f} (over {len(all_rewards)} episodes)")

        # Record video of one run with the loaded model
        if record_video_flag and model_path_to_load != "Untrained Model":
            print(f"\n  Recording a representative run (video)...")
            
            rec_env_id = self.env_id # Use the same base env_id
            # Create a new, clean env for recording.
            video_record_env = gym.make(rec_env_id, render_mode="rgb_array", full_action_space=False)
            # Apply the same wrappers for consistency as the eval_env
            video_record_env = AtariWrapper(video_record_env, clip_reward=False)
            video_record_env = gym.wrappers.FrameStack(video_record_env, self.env_creation_params["n_stack"])

            ts = time.strftime("%Y%m%d_%H%M%S")
            video_folder = os.path.join(self.gifs_dir if self.gifs_dir else "videos", f"dqn_test_{ts}")
            
            # Wrap with RecordVideo
            # episode_trigger lambda x: True means record all episodes (we only run one)
            video_env_instance = RecordVideo(
                video_record_env,
                video_folder=video_folder,
                name_prefix=f"dqn_run_{os.path.basename(model_path_to_load).replace(self.get_model_file_extension(),'')}",
                episode_trigger=lambda episode_id: episode_id == 0, # Record the first episode
                video_length=max_steps_per_episode, # Max frames for the video
                fps=video_fps
            )
            
            try:
                obs, info = video_env_instance.reset()
                terminated, truncated = False, False
                vid_ep_reward = 0
                vid_steps = 0
                while not (terminated or truncated) and vid_steps < max_steps_per_episode:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = video_env_instance.step(action)
                    vid_ep_reward += reward
                    vid_steps += 1
                print(f"  Video recorded for one run. Score: {vid_ep_reward:.0f}. Saved in: {video_folder}")
            except Exception as e_vid:
                print(f"  Error during video recording: {e_vid}")
            finally:
                if video_env_instance:
                    video_env_instance.close() # This finalizes the video saving process.
        print(f"--- DQN Testing Ended ---")


    def evaluate(self, model_path_to_load, episodes, max_steps_per_episode, **kwargs):
        print(f"\n--- DQN Evaluation ---")
        if not model_path_to_load or not os.path.exists(model_path_to_load):
            print(f"  Error: Model path '{model_path_to_load}' not found. Cannot evaluate.")
            return {}
        
        # Create a single environment for evaluation, similar to test but headless
        # We can use SB3's evaluate_policy for robustness if desired, or a manual loop.
        # evaluate_policy handles multiple episodes and returns mean_reward, std_reward.
        
        # For evaluate_policy, it's better to use a VecEnv structure it expects.
        eval_vec_env = self._create_env(for_training=False, n_envs_override=1) # Single env in a VecEnv

        try:
            self.model = SB3_DQN.load(model_path_to_load, env=eval_vec_env, device=self.merged_hparams["device"]) # Pass env for structure
            print(f"  Model loaded for evaluation: {os.path.basename(model_path_to_load)}")

            mean_reward, std_reward = evaluate_policy(
                self.model,
                eval_vec_env,
                n_eval_episodes=episodes,
                deterministic=True,
                render=False,
                warn=False # Suppress warnings about non-VecEnv if it occurs
            )
            # evaluate_policy doesn't directly give avg_steps, min/max score per episode.
            # If those are needed, a manual loop like in test() is better.
            # For now, let's use this and augment if necessary.

            # To get more detailed stats, we run a manual loop as well (or instead of evaluate_policy)
            all_ep_rewards = []
            all_ep_steps = []
            
            # Re-create or reset env for manual stat collection if needed after evaluate_policy
            # For simplicity, let's assume evaluate_policy leaves the env in a usable state for more interaction
            # or just run a manual loop entirely.
            # eval_vec_env.reset() # Reset before manual loop if using same env
            
            print(f"  Running manual loop for detailed stats over {episodes} episodes...")
            for i in range(episodes):
                obs = eval_vec_env.reset() # VecEnv obs is a list/array
                terminated, truncated = np.array([False]), np.array([False]) # For VecEnv
                episode_reward = 0
                episode_steps = 0
                while not (terminated[0] or truncated[0]) and episode_steps < max_steps_per_episode:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_vec_env.step(action)
                    episode_reward += reward[0] # Reward from VecEnv
                    episode_steps += 1
                
                # Get stats from Monitor wrapper if used by make_atari_env
                # info is a list of dicts for VecEnv
                actual_ep_reward = info[0].get('episode', {}).get('r', episode_reward)
                actual_ep_steps = info[0].get('episode', {}).get('l', episode_steps)
                all_ep_rewards.append(actual_ep_reward)
                all_ep_steps.append(actual_ep_steps)
                if (i + 1) % max(1, episodes // 5) == 0:
                    print(f"    Eval Ep {i+1}/{episodes}: Score={actual_ep_reward:.0f}, Steps={actual_ep_steps}")


            avg_score_manual = np.mean(all_ep_rewards) if all_ep_rewards else 0
            std_score_manual = np.std(all_ep_rewards) if all_ep_rewards else 0
            min_score_manual = np.min(all_ep_rewards) if all_ep_rewards else 0
            max_score_manual = np.max(all_ep_rewards) if all_ep_rewards else 0
            avg_steps_manual = np.mean(all_ep_steps) if all_ep_steps else 0

            print(f"  Evaluation Summary (from evaluate_policy): Mean Score: {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"  Evaluation Summary (manual loop): Avg Score: {avg_score_manual:.2f} +/- {std_score_manual:.2f}")
            print(f"    Min Score: {min_score_manual:.2f}, Max Score: {max_score_manual:.2f}, Avg Steps: {avg_steps_manual:.1f}")

            results = {
                "num_episodes_eval": episodes,
                "avg_score": round(avg_score_manual, 2), # Prioritize manual detailed stats
                "std_dev_score": round(std_score_manual, 2),
                "min_score": round(min_score_manual, 2),
                "max_score": round(max_score_manual, 2),
                "avg_steps": round(avg_steps_manual, 1),
                # "sb3_eval_mean_reward": round(mean_reward, 2), # Optional: include SB3's direct result
                # "sb3_eval_std_reward": round(std_reward, 2)
            }
        except Exception as e:
            print(f"  Error during DQN evaluation: {e}")
            import traceback; traceback.print_exc()
            results = {}
        finally:
            if eval_vec_env:
                eval_vec_env.close()
        
        print(f"--- DQN Evaluation Ended ---")
        return results


    def choose_action(self, observation, deterministic=False):
        # This method is more for agents with custom loops.
        # For SB3, prediction is usually done within its own context or test/eval loops.
        if self.model is None:
            # print("Warning: DQN model not loaded or trained. Returning random action.")
            # Requires an action space. If called standalone, this is an issue.
            # Assuming it's called after model is set up:
            if self.action_size is None: # Fallback if action_size wasn't determined
                 # Create a temp env to get action_size if absolutely necessary and not set
                try:
                    temp_env = gym.make(self.env_id)
                    action_s = temp_env.action_space.n
                    temp_env.close()
                    return np.random.choice(action_s)
                except: return 0 # Absolute fallback
            return np.random.choice(self.action_size)


        # SB3 model expects observation from its VecEnv (potentially batched or preprocessed)
        # `observation` here should match what the model expects.
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action


    def save(self, path):
        if self.model is not None and path is not None:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.model.save(path)
                print(f"  DQN Agent (SB3) model saved to {os.path.basename(path)}")
            except Exception as e:
                print(f"  Error saving DQN model to {path}: {e}")
        elif self.model is None:
            print("  Attempted to save DQN agent, but model is not initialized.")
        elif path is None:
            print("  Attempted to save DQN agent, but no path was provided.")

    def load(self, path):
        # For SB3, loading usually re-instantiates the model.
        # The `env` argument is crucial for SB3 load if the policy/model needs env specs.
        # In our structure, main.py calls test/evaluate which might load.
        # The train method also loads. They should provide the correct env.
        # This standalone load might be less used if model is set via other methods.
        if path is not None and os.path.exists(path):
            try:
                # We need an environment structure for loading.
                # If called standalone, create a temporary one.
                # Better: ensure load is called from a context where an env (or VecEnv) is available.
                print(f"  DQN Agent (SB3) attempting to load model from {os.path.basename(path)}.")
                print(f"    Note: For SB3, an environment (or its structure) is often needed for loading.")
                print(f"    If this load fails, ensure it's called from train/test/evaluate which set up an env.")
                
                # A placeholder env for structure if no current env is set by a running process (train/test)
                # This is a bit of a hack for a standalone load() call.
                # Ideally, self.model is loaded within train() or test() which prepare their own envs.
                temp_env_for_load = self._create_env(for_training=False, n_envs_override=1)

                self.model = SB3_DQN.load(path, env=temp_env_for_load, device=self.merged_hparams["device"])
                print(f"  DQN Agent (SB3) model loaded successfully from {os.path.basename(path)}.")
                
                # Update action_size and observation_shape from the loaded model's env if possible
                if self.model.env:
                    self.action_size = self.model.env.action_space.n
                    self.observation_shape = self.model.env.observation_space.shape
                
                temp_env_for_load.close() # Close the temp env

            except Exception as e:
                print(f"  Error loading DQN model from {path}: {e}")
                import traceback; traceback.print_exc()
                self.model = None # Ensure model is None if load failed
        elif path is None:
            print("  Load path for DQN agent is None.")
        else: # path doesn't exist
            print(f"  Load path for DQN agent does not exist: {path}")