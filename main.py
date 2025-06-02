# main.py
import sys
import os
import json
import csv
import time # For GIF naming, etc.
import gymnasium as gym
from gymnasium.utils.play import play as gym_play # For human play
import ale_py

gym.register_envs(ale_py)

# --- Path Setup ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR_MAIN = os.path.join(ROOT_DIR, "models")
AGENT_CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
GIF_DIR_MAIN = os.path.join(ROOT_DIR, "gifs") # For MP4 videos recorded by agents
EVAL_DIR_MAIN = os.path.join(ROOT_DIR, "evaluation_results")

sys.path.insert(0, ROOT_DIR) 
sys.path.insert(0, os.path.join(ROOT_DIR, "agents"))

# --- AGENT IMPORTS ---
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent
from agents.genetic_agent import GeneticAgent
from agents.neat_agent import NEATAgent

# === HELPER FUNCTIONS (mostly UI and File System) ===

def print_f(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_choice(prompt, options, allow_cancel=False, default_on_enter=None):
    print(prompt)
    for i, option in enumerate(options): print(f"{i+1}. {option}")
    default_prompt_part = ""
    if default_on_enter is not None and 1 <= default_on_enter <= len(options):
        default_prompt_part = f" (default: {options[default_on_enter-1]})"
    cancel_option_num = 0
    if allow_cancel:
        cancel_option_num = len(options) + 1
        print(f"{cancel_option_num}. Cancel / Go Back")
    while True:
        try:
            choice_str = input(f"Enter your choice (number){default_prompt_part}: ").strip()
            if not choice_str and default_on_enter is not None: choice = default_on_enter
            elif not choice_str and allow_cancel: return "cancel"
            elif not choice_str and not allow_cancel and default_on_enter is None:
                 print("Input required."); continue
            else: choice = int(choice_str)
            if 1 <= choice <= len(options): return choice - 1
            elif allow_cancel and choice == cancel_option_num: return "cancel"
            else:
                upper_bound = cancel_option_num if allow_cancel else len(options)
                print(f"Invalid choice. Please enter between 1 and {upper_bound}.")
        except ValueError: print("Invalid input. Please enter a number.")
        except Exception as e: print(f"Unexpected input error: {e}")

def get_yes_no(prompt, default_yes=None):
    options_str = " (yes/no)"
    if default_yes is True: options_str = " (YES/no)"
    elif default_yes is False: options_str = " (yes/NO)"
    while True:
        answer = input(f"{prompt}{options_str}: ").strip().lower()
        if not answer and default_yes is not None: return default_yes
        if answer in ['yes', 'y']: return True
        elif answer in ['no', 'n']: return False
        else: print("Invalid input. Enter 'yes' or 'no'.")

# --- Model File Path Helpers ---
BASE_MODEL_FILENAME_TEMPLATE = "{agent_name}_spaceinvaders"

def get_model_extension(agent_name):
    # This will be ideally fetched from agent's own property after instantiation
    # For now, a heuristic before agent is created.
    # This should match what agent's get_model_file_extension() would return.
    if agent_name in ['dqn', 'ppo', 'a2c']: return ".zip" # Default if they use SB3
    if agent_name == "neat": return ".pkl"
    if agent_name == "genetic": return ".pth" # Or .pkl, depends on GeneticAgent impl.
    return ".pth" # Fallback

def get_existing_model_versions(agent_name, models_base_dir=MODELS_DIR_MAIN):
    if not os.path.exists(models_base_dir): return []
    pattern_base = BASE_MODEL_FILENAME_TEMPLATE.format(agent_name=agent_name)
    versions = []
    all_extensions = (".zip", ".pkl", ".pth") # Check all known types
    for f_name in sorted(os.listdir(models_base_dir)):
        if f_name.startswith(pattern_base) and f_name.endswith(all_extensions):
            versions.append(os.path.join(models_base_dir, f_name))
    return versions

def get_next_model_save_path(agent_name, models_base_dir=MODELS_DIR_MAIN):
    # This path generation is for user display; agent will confirm/use its own internal logic.
    if not os.path.exists(models_base_dir): os.makedirs(models_base_dir)
    pattern_base = BASE_MODEL_FILENAME_TEMPLATE.format(agent_name=agent_name)
    extension = get_model_extension(agent_name)
    base_path = os.path.join(models_base_dir, f"{pattern_base}{extension}")
    if not os.path.exists(base_path): return base_path
    version = 2
    while True:
        versioned_path = os.path.join(models_base_dir, f"{pattern_base}_v{version}{extension}")
        if not os.path.exists(versioned_path): return versioned_path
        version += 1

def get_latest_model_path(agent_name, models_base_dir=MODELS_DIR_MAIN):
    versions = get_existing_model_versions(agent_name, models_base_dir)
    expected_extension = get_model_extension(agent_name)
    specific_versions = [v for v in versions if v.endswith(expected_extension)]
    if specific_versions:
        return specific_versions[-1]
    return versions[-1] if versions else None

def get_model_filenames_for_display(agent_name, models_base_dir=MODELS_DIR_MAIN):
    if not os.path.exists(models_base_dir): return []
    pattern_base = BASE_MODEL_FILENAME_TEMPLATE.format(agent_name=agent_name)
    filenames = []
    all_extensions = (".zip", ".pkl", ".pth")
    for f_name in sorted(os.listdir(models_base_dir)):
        if f_name.startswith(pattern_base) and f_name.endswith(all_extensions):
            filenames.append(f_name)
    return filenames

# --- Agent Config Loading ---
def load_agent_config(config_path_cli, agent_name, configs_dir=AGENT_CONFIGS_DIR):
    config_to_try = config_path_cli
    if not config_to_try:
        default_config_filename = f"{agent_name}_default.json"
        config_to_try = os.path.join(configs_dir, default_config_filename)

    if config_to_try and os.path.exists(config_to_try):
        try:
            with open(config_to_try, 'r') as f:
                print_f(f"  Loading config for {agent_name} from: {os.path.basename(config_to_try)}")
                return json.load(f)
        except Exception as e:
            print_f(f"  Warning: Error loading config '{os.path.basename(config_to_try)}': {e}.")
    if config_path_cli and (not os.path.exists(config_path_cli) or \
                            (config_path_cli == config_to_try and not _did_config_load_successfully(config_to_try))):
         print_f(f"  Specified config '{os.path.basename(config_path_cli)}' not found or failed to load.")
    print_f(f"  Using agent's internal default hyperparameters for {agent_name}.")
    return {}

def _did_config_load_successfully(path):
    if path and os.path.exists(path):
        try:
            with open(path, 'r') as f: json.load(f)
            return True
        except: return False
    return False

# --- Agent Factory ---
ALL_AGENT_TYPES_AVAILABLE = ["random", "dqn", "ppo", "a2c", "genetic", "neat"]
TRAINABLE_AGENTS_PROMPT = [agent for agent in ALL_AGENT_TYPES_AVAILABLE if agent != 'random']

def create_agent_gym(agent_name, env_id, agent_hparams, mode='train',
                     models_dir=MODELS_DIR_MAIN, gifs_dir=GIF_DIR_MAIN):
    print_f(f"  Factory: Creating {agent_name} for env '{env_id}' in mode '{mode}'.")
    agent_params = {
        'env_id': env_id, 'hparams': agent_hparams, 'mode': mode,
        'models_dir_for_agent': models_dir, 'gifs_dir_for_agent': gifs_dir
    }
    if agent_name == 'random': return RandomAgent(env_id=env_id, hparams=agent_hparams, mode=mode)
    elif agent_name == 'dqn': return DQNAgent(**agent_params)
    elif agent_name == 'a2c': return A2CAgent(**agent_params)
    elif agent_name == 'ppo': return PPOAgent(**agent_params)
    elif agent_name == 'genetic': return GeneticAgent(**agent_params)
    elif agent_name == 'neat': return NEATAgent(**agent_params)
    else: raise ValueError(f"Unknown agent type: {agent_name}")

# --- CSV Reporting ---
def save_evaluation_to_csv(evaluation_data_list, output_dir=EVAL_DIR_MAIN):
    if not evaluation_data_list: print_f("No evaluation data to save."); return None
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f"evaluation_summary_{timestamp}.csv")
    try:
        fieldnames = list(evaluation_data_list[0].keys())
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(evaluation_data_list)
        print_f(f"\nEvaluation results saved to: {csv_filename}")
        return csv_filename
    except Exception as e: print_f(f"Error writing CSV: {e}"); return None

# === CORE ROUTINES ===

ENV_ID_SPINVADERS = "ALE/SpaceInvaders-v5"

def play_human_gym():
    print_f("\n--- Human Player Mode (Gymnasium) ---")
    print_f(f"Environment: {ENV_ID_SPINVADERS}")
    print_f("Controls: Use arrow keys to move, Space to fire. Press Q or ESC to quit game.")
    print_f("The game window needs to be focused to capture key presses.")

    try:
        # For ALE/SpaceInvaders-v5, full_action_space=False:
        # 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT, 4: RIGHTFIRE, 5: LEFTFIRE
        # Pygame keys are used by gymnasium.utils.play.play
        # We need to import pygame for key constants if not already imported globally
        import pygame.constants as pg_keys

        keys_to_action_map = {
            (pg_keys.K_LEFT,): 3,  # LEFT
            (pg_keys.K_RIGHT,): 2, # RIGHT
            (pg_keys.K_SPACE,): 1, # FIRE
            # Combined actions for Space Invaders (if supported by full_action_space=True, or if desired for False)
            # For full_action_space=False, these are distinct actions:
            (pg_keys.K_LEFT, pg_keys.K_SPACE): 5,  # LEFTFIRE
            (pg_keys.K_RIGHT, pg_keys.K_SPACE): 4, # RIGHTFIRE
        }
        # For full_action_space=False, single key presses for combined actions might be more intuitive
        # For example, map 'a' to LEFTFIRE, 'd' to RIGHTFIRE if desired,
        # or rely on the two-key tuple for simultaneous press detection by gym_play.
        # The gymnasium.utils.play function handles simultaneous key presses if defined as tuples.

        # Create the environment. `gym_play` will use its render_mode.
        env = gym.make(ENV_ID_SPINVADERS, render_mode="rgb_array", full_action_space=False)
        
        print_f("Starting human play session...")
        gym_play(env, keys_to_action=keys_to_action_map, zoom=3, fps=30)
        # `gym_play` handles the game loop, rendering, and input based on keys_to_action.
        # It also handles quitting (usually ESC or closing the window).
        
        print_f("Human play session finished.")

    except ImportError:
        print_f("Error: Pygame is required for human play controls. Please install it ('pip install pygame').")
    except Exception as e:
        print_f(f"Error during human play setup or execution: {e}")
        import traceback
        traceback.print_exc()
    input("Press Enter to return to menu...")


def train_agents_main(agents_to_train_names, episodes_per_agent, render_mode_str, max_steps_per_episode,
                      configs_paths_map, load_models_flag, force_train_flag,
                      save_interval_eps, print_interval_train, silent_audio_flag_ignored):
    print_f("\n--- Training Session ---")
    for agent_name in agents_to_train_names:
        print_f(f"\n-- Preparing Agent for Training: {agent_name.upper()} --")
        agent_hparams = load_agent_config(configs_paths_map.get(agent_name), agent_name)
        try:
            agent = create_agent_gym(agent_name, ENV_ID_SPINVADERS, agent_hparams, mode='train')
            path_to_load = None
            if load_models_flag:
                path_to_load = get_latest_model_path(agent_name)
                if not path_to_load: print_f(f"  No existing model found to load for {agent_name}.")
            agent.train(
                episodes=episodes_per_agent, max_steps_per_episode=max_steps_per_episode,
                render_mode_str=render_mode_str, path_to_load_model=path_to_load,
                force_new_training_if_model_exists=force_train_flag,
                save_interval_eps=save_interval_eps, print_interval_steps=print_interval_train
            )
        except Exception as e:
            print_f(f"Error during training setup or execution for {agent_name}: {e}")
            import traceback; traceback.print_exc()
    print_f("\n--- Training session finished for selected agents. ---")

def test_agent_main(agent_name, model_path_selected, episodes, max_steps,
                    record_video_flag, video_fps, video_capture_interval_ignored):
    print_f(f"\n--- Testing Agent: {agent_name.upper()} ---")
    agent_hparams = {}
    if not model_path_selected and agent_name != "random":
        print_f(f"  Warning: No model selected for {agent_name}. Agent will be tested untrained.")
    try:
        agent = create_agent_gym(agent_name, ENV_ID_SPINVADERS, agent_hparams, mode='test')
        agent.test(
            model_path_to_load=model_path_selected, episodes=episodes,
            max_steps_per_episode=max_steps, render_during_test=True,
            record_video_flag=record_video_flag, video_fps=video_fps
        )
    except Exception as e:
        print_f(f"Error during testing for {agent_name}: {e}")
        import traceback; traceback.print_exc()

def evaluate_all_agents_main(episodes_per_agent_eval, max_steps_per_episode_eval):
    print_f("\n--- Evaluating All Agents (Latest Models) ---")
    all_evaluation_results = []
    agents_for_evaluation = sorted(list(set(TRAINABLE_AGENTS_PROMPT + ["random"])))
    for agent_name in agents_for_evaluation:
        print_f(f"\n-- Evaluating: {agent_name.upper()} --")
        model_path_to_eval = None
        if agent_name != "random":
            model_path_to_eval = get_latest_model_path(agent_name)
            if not model_path_to_eval:
                print_f(f"  No model found for {agent_name}. Skipping evaluation.")
                continue
        agent_hparams = {}
        try:
            agent = create_agent_gym(agent_name, ENV_ID_SPINVADERS, agent_hparams, mode='evaluate')
            eval_metrics = agent.evaluate(
                model_path_to_load=model_path_to_eval, episodes=episodes_per_agent_eval,
                max_steps_per_episode=max_steps_per_episode_eval
            )
            if eval_metrics:
                eval_metrics["agent_name"] = agent_name
                eval_metrics["model_path"] = os.path.basename(model_path_to_eval) if model_path_to_eval else "N/A"
                all_evaluation_results.append(eval_metrics)
            else: print_f(f"  No evaluation metrics returned by {agent_name}.")
        except Exception as e:
            print_f(f"  Error during evaluation for {agent_name}: {e}")
            import traceback; traceback.print_exc()
    if all_evaluation_results: save_evaluation_to_csv(all_evaluation_results)
    else: print_f("No evaluation results were collected.")
    print_f("\n--- Evaluation of all agents complete. ---")

# --- Main Interactive Loop ---
def main_interactive():
    for d in [MODELS_DIR_MAIN, AGENT_CONFIGS_DIR, GIF_DIR_MAIN, EVAL_DIR_MAIN]:
        if not os.path.exists(d):
            try: os.makedirs(d); print_f(f"Created directory: {d}")
            except OSError as e: print_f(f"Error creating directory '{d}': {e}")
    
    while True:
        clear_screen()
        print_f("Space Invaders AI Gym Environment Launcher")
        print_f(f"Environment: {ENV_ID_SPINVADERS}")
        print_f("==========================================")
        mode_options = ["Play as Human", "Train AI Agent(s)", "Test AI Agent", "Evaluate All Agents", "Exit Launcher"]
        mode_choice_idx = get_choice("Select mode:", mode_options, default_on_enter=1)
        
        selected_mode = "Exit Launcher"
        if mode_choice_idx != "cancel" and mode_choice_idx is not None:
            selected_mode = mode_options[mode_choice_idx]

        if selected_mode == "Exit Launcher": break
        elif selected_mode == "Play as Human": play_human_gym()
        elif selected_mode == "Train AI Agent(s)":
            print_f("\n--- Configure Training ---")
            if not TRAINABLE_AGENTS_PROMPT: print_f("No trainable agents defined."); input("Press Enter..."); continue
            agents_to_train_input = input(f"Agents to train (csv, or 'all' for [{', '.join(TRAINABLE_AGENTS_PROMPT)}]): ").strip().lower()
            if not agents_to_train_input: print_f("No agents. Menu."); input(); continue
            agents_list_requested = TRAINABLE_AGENTS_PROMPT if agents_to_train_input == 'all' else \
                                  [s.strip() for s in agents_to_train_input.split(',')]
            valid_agents_to_train = [req for req in agents_list_requested if req in TRAINABLE_AGENTS_PROMPT]
            if not valid_agents_to_train: print_f("No valid agents selected. Menu."); input(); continue
            print_f(f"Selected for Training: {', '.join(valid_agents_to_train)}")

            is_pop_agent_selected = any(a in ['neat','genetic'] for a in valid_agents_to_train)
            default_eps = 100 if is_pop_agent_selected else 1000
            default_max_s = 2000 if is_pop_agent_selected else 10000
            default_save_int = 50 if is_pop_agent_selected else 0
            default_print_int = 1 if is_pop_agent_selected else 5000

            episodes = int(input(f"Episodes/Generations (def {default_eps}): ") or str(default_eps))
            max_s = int(input(f"Max steps per ep/eval (def {default_max_s}): ") or str(default_max_s))
            render_opt = get_yes_no("Render training (human view, if agent supports)?", default_yes=False)
            render_m = "human" if render_opt else "rgb_array"
            load_m = get_yes_no("Load LATEST models to continue training (if available)?", default_yes=False)
            force_t = False if load_m else get_yes_no("Force new training (agent creates new version if model exists)?", default_yes=False)
            save_int = int(input(f"Save model every N eps/gens (0=agent_default/end, def {default_save_int}): ") or str(default_save_int))
            print_int = int(input(f"Suggest print/log interval (steps/gens, def {default_print_int}): ") or str(default_print_int))
            
            cfgs = {}
            for ag_name in valid_agents_to_train:
                def_cfg_path = os.path.join(AGENT_CONFIGS_DIR, f"{ag_name}_default.json")
                use_def = True
                if os.path.exists(def_cfg_path):
                    use_def = get_yes_no(f"Use default config for {ag_name.upper()} ('{os.path.basename(def_cfg_path)}')?", default_yes=True)
                custom_path_input = ""
                if not use_def or not os.path.exists(def_cfg_path):
                    if not os.path.exists(def_cfg_path) and use_def: print_f(f"Default config for {ag_name} not found.")
                    custom_path_input = input(f"  Path to {ag_name} JSON config (blank for internal defaults): ").strip()
                if custom_path_input:
                    path_to_check1 = os.path.join(AGENT_CONFIGS_DIR, custom_path_input)
                    path_to_check2 = custom_path_input
                    if os.path.exists(path_to_check1): cfgs[ag_name] = path_to_check1
                    elif os.path.exists(path_to_check2): cfgs[ag_name] = path_to_check2
                    else: print_f(f"  Custom config for {ag_name} not found. Using agent's internal defaults.")
                elif use_def and os.path.exists(def_cfg_path): cfgs[ag_name] = def_cfg_path
            
            train_agents_main(valid_agents_to_train, episodes, render_m, max_s, cfgs, load_m, force_t, save_int, print_int, False)
            input("\nTraining finished or interrupted. Press Enter...")

        elif selected_mode == "Test AI Agent":
            print_f("\n--- Test Specific AI Agent ---")
            agent_idx = get_choice("Agent type:", ALL_AGENT_TYPES_AVAILABLE, allow_cancel=True)
            if agent_idx == "cancel": continue
            agent_n = ALL_AGENT_TYPES_AVAILABLE[agent_idx]
            model_p = None
            if agent_n != "random":
                avail_models = get_model_filenames_for_display(agent_n)
                if not avail_models:
                    if not get_yes_no(f"No models for {agent_n}. Test untrained?", default_yes=False): continue
                else:
                    print_f(f"\nModels for {agent_n.upper()}:")
                    model_i = get_choice("Select model (or Cancel to test untrained):", avail_models, allow_cancel=True, default_on_enter=len(avail_models))
                    if model_i == "cancel":
                        if not get_yes_no(f"Test {agent_n} untrained?", default_yes=False): continue
                    else: model_p = os.path.join(MODELS_DIR_MAIN, avail_models[model_i])
            
            test_eps = int(input("Test episodes (def 5): ") or "5")
            test_max_s = int(input("Max steps per ep (def 3000): ") or "3000")
            rec_vid_flag = get_yes_no("Record video (MP4) of test runs (if agent supports)?", default_yes=True)
            vid_fps_val = 15
            if rec_vid_flag: vid_fps_val = int(input("  Video FPS (def 15): ") or "15")
            test_agent_main(agent_n, model_p, test_eps, test_max_s, rec_vid_flag, vid_fps_val, 0)
            input("\nTesting finished. Press Enter...")

        elif selected_mode == "Evaluate All Agents":
            print_f("\n--- Evaluate All Agents (Latest Models) ---")
            eval_eps = int(input("Evaluation episodes per agent (def 10): ") or "10")
            eval_max_s = int(input("Max steps per episode (def 3000): ") or "3000")
            print_f("Evaluation will run headlessly. Results will be saved to CSV.")
            evaluate_all_agents_main(eval_eps, eval_max_s)
            input("\nEvaluation finished. Press Enter...")

    print_f("\nExiting Launcher. Goodbye!")

if __name__ == "__main__":
    main_interactive()