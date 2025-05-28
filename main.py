# main.py
import multiprocessing
import subprocess
import sys
import os

# --- Path Setup ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GAME_PACKAGE_DIR = os.path.join(ROOT_DIR, "game")
MODELS_DIR_MAIN = os.path.join(ROOT_DIR, "models")
AGENT_CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
BASE_MODEL_FILENAME_TEMPLATE_MAIN = "{agent_name}_spaceinvaders"

sys.path.insert(0, ROOT_DIR) # Add project root to path

# --- Helper Functions ---
def get_python_executable():
    """Gets the currently running Python executable."""
    return sys.executable

def clear_screen():
    """Clears the terminal screen."""
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
            elif not choice_str and allow_cancel: return "cancel" # if empty and cancel allowed, treat as cancel
            elif not choice_str and not allow_cancel and default_on_enter is None: # if empty, no default, no cancel
                 print("Input required.")
                 continue
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

def build_command(script_name, args_dict):
    command = [get_python_executable(), "-u", script_name]
    for arg, value in args_dict.items():
        if isinstance(value, bool):
            if value: command.append(arg)
        elif value is not None:
            command.append(arg); command.append(str(value))
    return command

def run_command(command):
    print(f"\nExecuting: {' '.join(command)}\n")
    try:
        # Use Popen for real-time output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in process.stdout: # Live output
            print(line, end='', flush=True)
        process.wait() # Wait for the process to complete
        if process.returncode != 0:
            print(f"\n--- CMD Error (Return Code: {process.returncode}) ---")
        else:
            print(f"\n--- CMD Success ---")
    except FileNotFoundError:
        print(f"Error: Script '{command[1]}' not found. Ensure it's in the project root.")
    except Exception as e:
        print(f"An error occurred while trying to run the command: {e}")


def get_existing_model_filenames_ui(agent_name):
    # This function now relies on utils.model_helpers which uses the correct "models" dir name.
    from utils.model_helpers import get_model_filenames_for_display # model_helpers knows MODELS_DIR
    return get_model_filenames_for_display(agent_name) 

# --- Main Interactive Loop ---
def main_interactive():
    if not os.path.exists(MODELS_DIR_MAIN):
        print(f"Models directory '{MODELS_DIR_MAIN}' not found. Creating it.")
        try: os.makedirs(MODELS_DIR_MAIN)
        except OSError as e: print(f"Error creating '{MODELS_DIR_MAIN}': {e}.")

    if not os.path.exists(AGENT_CONFIGS_DIR):
        print(f"Warning: Agent configuration directory '{AGENT_CONFIGS_DIR}' not found. Creating it.")
        try:
            os.makedirs(AGENT_CONFIGS_DIR)
            print(f"Please place default JSON config files (e.g., dqn_default.json) in '{AGENT_CONFIGS_DIR}'.")
        except OSError as e:
            print(f"Error creating '{AGENT_CONFIGS_DIR}': {e}. Config loading might fail.")
            input("Press Enter to continue anyway...")

    while True:
        clear_screen()
        print("Welcome to Space Invaders AI Launcher!")
        print("======================================")
        mode_options = ["Play as Human", "Run AI Agent", "Exit Launcher"]
        mode_choice_result = get_choice("Select mode:", mode_options, default_on_enter=1) # Default to Play
        
        selected_mode_idx = mode_choice_result
        if selected_mode_idx == "cancel" or selected_mode_idx is None : # Should not happen with default
            selected_mode = "Exit Launcher" 
        else:
            selected_mode = mode_options[selected_mode_idx]


        if selected_mode == "Exit Launcher": break
        elif selected_mode == "Play as Human":
            print("\n--- Player Mode ---")
            try:
                from game.game_manager import Game as GameClass # MODIFIED
                game_args = {
                    'silent_mode': get_yes_no("Run game without sounds?", default_yes=False),
                    'ai_training_mode': False,
                    'headless_worker_mode': False # Player mode is never headless
                }
                print("\nStarting game in player mode...")
                game_instance = GameClass(**game_args)
                game_instance.run_player_mode()
            except ImportError as e: print(f"Error: Could not import Game class: {e}")
            except Exception as e: print(f"An error occurred in player mode: {e}")
            input("\nPress Enter to return to the main menu...")

        elif selected_mode == "Run AI Agent":
            clear_screen()
            print("--- AI Agent Mode ---")
            ai_op_options = ["Train Agent(s)", "Test/Run Specific Agent Version", "Evaluate All Agents (Latest Versions)", "Back to Main Menu"]
            ai_op_choice_result = get_choice("\nSelect AI operation:", ai_op_options, allow_cancel=True, default_on_enter=1) 
            
            if ai_op_choice_result == "cancel" or ai_op_choice_result is None: continue # Back to main menu if cancelled
            selected_ai_operation = ai_op_options[ai_op_choice_result]
            if selected_ai_operation == "Back to Main Menu": continue


            script_to_run = None # Initialize
            cmd_args = {}
            # ALL_AGENT_TYPES_AVAILABLE should be imported from utils.cli_args for consistency
            try:
                from utils.cli_args import ALL_AGENT_TYPES_AVAILABLE
            except ImportError:
                print("Warning: Could not import ALL_AGENT_TYPES_AVAILABLE. Using hardcoded list.")
                ALL_AGENT_TYPES_AVAILABLE = ["dqn", "ppo", "a2c", "genetic", "neat", "random"]
            
            trainable_agents_for_prompt = [agent for agent in ALL_AGENT_TYPES_AVAILABLE if agent != 'random']

            if selected_ai_operation == "Train Agent(s)":
                script_to_run = "train.py"
                print("\n--- Training Configuration ---")
                agents_to_train_this_session = []
                train_all_q = get_yes_no(f"Train all non-random agents ({', '.join(trainable_agents_for_prompt)})?", default_yes=True)
                if train_all_q: agents_to_train_this_session = trainable_agents_for_prompt
                else:
                    selected_agents_str = input(f"Enter comma-separated agent names to train (e.g., dqn,neat) from [{', '.join(trainable_agents_for_prompt)}]: ").strip()
                    if selected_agents_str:
                        agents_to_train_this_session = [s.strip().lower() for s in selected_agents_str.split(',') if s.strip().lower() in trainable_agents_for_prompt]
                    if not agents_to_train_this_session: print("No valid agents selected for training. Aborting operation."); input("\nPress Enter..."); continue
                cmd_args["--agents"] = ",".join(agents_to_train_this_session)

                for agent_name_config in agents_to_train_this_session:
                    default_config_name = f"{agent_name_config}_default.json"
                    default_config_path = os.path.join(AGENT_CONFIGS_DIR, default_config_name)
                    print(f"\nConfig for {agent_name_config.upper()}:")
                    config_to_pass_for_agent = None 
                    if os.path.exists(default_config_path):
                        use_default = get_yes_no(f"Use default config '{default_config_name}'?", default_yes=True)
                        if use_default: config_to_pass_for_agent = default_config_path
                        else:
                            custom_config_name = input(f"Enter filename for {agent_name_config} config (in '{AGENT_CONFIGS_DIR}/', or full path): ").strip()
                            if custom_config_name:
                                # Check if it's a full path first
                                if os.path.exists(custom_config_name):
                                    config_to_pass_for_agent = custom_config_name
                                else: # Assume it's a filename in AGENT_CONFIGS_DIR
                                    custom_config_path_local = os.path.join(AGENT_CONFIGS_DIR, custom_config_name)
                                    if os.path.exists(custom_config_path_local): config_to_pass_for_agent = custom_config_path_local
                                    else: print(f"Warning: Custom config '{custom_config_name}' not found. {agent_name_config.upper()} will use internal defaults or no config file if default was also skipped.")
                        if config_to_pass_for_agent: # Only add arg if a path is chosen
                             cmd_args[f"--{agent_name_config}_config_path"] = config_to_pass_for_agent
                    else:
                        print(f"Note: Default config '{default_config_name}' not found. {agent_name_config.upper()} will use internal defaults if no custom path is specified next.")
                        custom_config_name = input(f"Optional: Enter filename for {agent_name_config} config (in '{AGENT_CONFIGS_DIR}/', or full path), or leave blank for internal defaults: ").strip()
                        if custom_config_name:
                            if os.path.exists(custom_config_name): config_to_pass_for_agent = custom_config_name
                            else:
                                custom_config_path_local = os.path.join(AGENT_CONFIGS_DIR, custom_config_name)
                                if os.path.exists(custom_config_path_local): config_to_pass_for_agent = custom_config_path_local
                                else: print(f"Warning: Custom config '{custom_config_name}' not found. Using internal defaults for {agent_name_config.upper()}.")
                        if config_to_pass_for_agent: # Only add arg if a path is chosen
                            cmd_args[f"--{agent_name_config}_config_path"] = config_to_pass_for_agent


                cmd_args["--episodes"] = int(input("Episodes/Generations (default 1000): ") or "1000")
                cmd_args["--num_workers"] = int(input("Num workers for NEAT/GA (default cpu_count-1, min 1): ") or max(1, os.cpu_count() -1 if os.cpu_count() else 1))
                cmd_args["--load_models"] = get_yes_no("Load LATEST models to continue training (if exist)?", default_yes=False)
                if not cmd_args.get("--load_models", False): # Only ask force_train if not loading
                    cmd_args["--force_train"] = get_yes_no("Force training (create new version if model exists and not loading)?", default_yes=False)
                cmd_args["--render"] = get_yes_no("Render game content during training?", default_yes=False)
                cmd_args["--max_steps_per_episode"] = int(input("Max steps per episode/evaluation (default 2000): ") or "2000")
                cmd_args["--save_interval"] = int(input("Save NN models every N episodes (GA/NEAT save per gen, default 50): ") or "50")
                cmd_args["--print_interval_steps"] = int(input("Print training stats every N steps (default 500): ") or "500")


            elif selected_ai_operation == "Test/Run Specific Agent Version":
                script_to_run = "test.py"
                agent_choice_result = get_choice("\nSelect agent type:", ALL_AGENT_TYPES_AVAILABLE, allow_cancel=True, default_on_enter=1)
                if agent_choice_result == "cancel" or agent_choice_result is None: continue
                cmd_args["--agent"] = ALL_AGENT_TYPES_AVAILABLE[agent_choice_result]
                
                model_file_to_load = None
                if cmd_args["--agent"] != "random":
                    available_versions = get_existing_model_filenames_ui(cmd_args["--agent"])
                    if not available_versions:
                        print(f"No trained models found for {cmd_args['--agent'].upper()}.")
                        if not get_yes_no(f"Run {cmd_args['--agent'].upper()} untrained (randomly initialized)?", default_yes=False): 
                            input("\nPress Enter to return to AI Menu..."); continue
                        # No --model_file_path to pass, test.py will handle agent without loaded model
                    else:
                        print(f"\nAvailable models for {cmd_args['--agent'].upper()}:")
                        # Default to latest model (last in list)
                        default_model_choice_ui = len(available_versions) 
                        chosen_model_idx = get_choice("Select model to test:", available_versions, allow_cancel=True, default_on_enter=default_model_choice_ui)
                        
                        if chosen_model_idx == "cancel" or chosen_model_idx is None: 
                            input("\nPress Enter to return to AI Menu..."); continue
                        
                        # Construct full path to model. model_helpers.get_model_filenames_for_display returns basenames.
                        # MODELS_DIR_MAIN is the correct base path for models.
                        model_file_to_load = os.path.join(MODELS_DIR_MAIN, available_versions[chosen_model_idx])
                        cmd_args["--model_file_path"] = model_file_to_load 
                
                cmd_args["--episodes"] = int(input(f"Number of episodes to run (default {'5' if model_file_to_load or cmd_args['--agent'] == 'random' else '1'}): ") or ("5" if model_file_to_load or cmd_args['--agent'] == 'random' else "1"))
                cmd_args["--render"] = get_yes_no("Render game during testing?", default_yes=True)
                cmd_args["--max_steps_per_episode"] = int(input("Max steps per episode (default 3000): ") or "3000")
                cmd_args["--silent"] = get_yes_no("Run game without sounds?", default_yes=True)
                
                record_gif = get_yes_no("Record GIFs of initial episodes?", default_yes=False)
                if record_gif:
                    cmd_args["--gif_episodes"] = int(input("Number of initial episodes to record as GIF (default 1, 0 to disable): ") or "1")
                    if cmd_args.get("--gif_episodes", 0) > 0:
                        cmd_args["--gif_fps"] = int(input("GIF FPS (default 15): ") or "15")
                        cmd_args["--gif_capture_every_n_steps"] = int(input("GIF frame capture interval (steps, default 4): ") or "4")
                        cmd_args["--max_gif_frames"] = int(input("Max frames per GIF segment (default 500): ") or "500")
                else:
                    cmd_args["--gif_episodes"] = 0


            elif selected_ai_operation == "Evaluate All Agents (Latest Versions)":
                script_to_run = "evaluate.py"
                print("\n--- Evaluate All Latest Agents Configuration ---")
                cmd_args["--episodes"] = int(input("Evaluation episodes per agent (default 20): ") or "20")
                cmd_args["--max_steps_per_episode"] = int(input("Max steps per episode (default 3000): ") or "3000")
                # Evaluation is typically silent and headless for speed.
                cmd_args["--silent"] = True 
                cmd_args["--render"] = False 
                # No GIF recording for bulk evaluation.

            if script_to_run: # Check if a script was actually selected
                command_list = build_command(script_to_run, cmd_args)
                run_command(command_list)
            # No else needed, as "Back to Main Menu" is handled by 'continue'
            
            input("\nPress Enter to return to the AI Agent Menu...") 
    
    print("\nExiting Space Invaders AI Launcher. Goodbye!")

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' if on Windows or macOS,
    # as Pygame and other libraries might not be fork-safe.
    # This is better done in the main entry point (here) than in train.py's __main__
    # to cover all potential subprocess uses if other scripts also used multiprocessing.
    if sys.platform.startswith('win') or sys.platform.startswith('darwin'):
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            try:
                multiprocessing.set_start_method('spawn', force=True)
                # print("INFO: Multiprocessing start method set to 'spawn'.") # Optional info
            except RuntimeError:
                # This can happen if context is already set or if force=True isn't enough.
                pass # Silently pass if it fails, hoping for the best or relying on script-specific settings.
    main_interactive()