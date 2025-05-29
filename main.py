# main.py
# REMOVED: import multiprocessing 
import subprocess
import sys
import os
import json # For checking config files

# --- Path Setup ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GAME_PACKAGE_DIR = os.path.join(ROOT_DIR, "game")
MODELS_DIR_MAIN = os.path.join(ROOT_DIR, "models")
AGENT_CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
BASE_MODEL_FILENAME_TEMPLATE_MAIN = "{agent_name}_spaceinvaders"

sys.path.insert(0, ROOT_DIR) # Add project root to path

# --- Helper Functions (no changes needed here) ---
def get_python_executable():
    return sys.executable

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
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in process.stdout: print(line, end='', flush=True)
        process.wait()
        if process.returncode != 0: print(f"\n--- CMD Error (Return Code: {process.returncode}) ---")
        else: print(f"\n--- CMD Success ---")
    except FileNotFoundError: print(f"Error: Script '{command[1]}' not found. Ensure it's in the project root.")
    except Exception as e: print(f"An error occurred while trying to run the command: {e}")

def get_existing_model_filenames_ui(agent_name):
    from utils.model_helpers import get_model_filenames_for_display
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
        mode_choice_result = get_choice("Select mode:", mode_options, default_on_enter=1)
        
        selected_mode_idx = mode_choice_result
        if selected_mode_idx == "cancel" or selected_mode_idx is None : 
            selected_mode = "Exit Launcher" 
        else:
            selected_mode = mode_options[selected_mode_idx]

        if selected_mode == "Exit Launcher": break
        elif selected_mode == "Play as Human":
            print("\n--- Player Mode ---")
            try:
                from game.game_manager import Game as GameClass
                game_args = {
                    'silent_mode': get_yes_no("Run game without sounds?", default_yes=False),
                    'ai_training_mode': False,
                    'headless_worker_mode': False 
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
            
            if ai_op_choice_result == "cancel" or ai_op_choice_result is None: continue
            selected_ai_operation = ai_op_options[ai_op_choice_result]
            if selected_ai_operation == "Back to Main Menu": continue

            script_to_run = None 
            cmd_args = {}
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
                    if not agents_to_train_this_session: print("No valid agents selected. Aborting."); input("\nPress Enter..."); continue
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
                                if os.path.exists(custom_config_name): config_to_pass_for_agent = custom_config_name
                                else:
                                    custom_config_path_local = os.path.join(AGENT_CONFIGS_DIR, custom_config_name)
                                    if os.path.exists(custom_config_path_local): config_to_pass_for_agent = custom_config_path_local
                                    else: print(f"Warning: Custom config '{custom_config_name}' not found. {agent_name_config.upper()} using internal defaults.")
                        if config_to_pass_for_agent:
                             cmd_args[f"--{agent_name_config}_config_path"] = config_to_pass_for_agent
                    else: 
                        print(f"Note: Default config '{default_config_name}' not found.")
                        custom_config_name = input(f"Optional: Enter custom config for {agent_name_config} (in '{AGENT_CONFIGS_DIR}/', or full path), or blank for internal defaults: ").strip()
                        if custom_config_name: 
                            if os.path.exists(custom_config_name): config_to_pass_for_agent = custom_config_name
                            else:
                                custom_config_path_local = os.path.join(AGENT_CONFIGS_DIR, custom_config_name)
                                if os.path.exists(custom_config_path_local): config_to_pass_for_agent = custom_config_path_local
                                else: print(f"Warning: Custom config '{custom_config_name}' not found. Using internal defaults for {agent_name_config.upper()}.")
                        if config_to_pass_for_agent:
                            cmd_args[f"--{agent_name_config}_config_path"] = config_to_pass_for_agent
                        else:
                             print(f"{agent_name_config.upper()} will use internal defaults.")

                cmd_args["--episodes"] = int(input("Episodes/Generations (default 50): ") or "50")
                # REMOVED --num_workers prompt here, as it's removed from cli_args for training
                
                cmd_args["--load_models"] = get_yes_no("Load LATEST models to continue training (if exist)?", default_yes=True)
                if not cmd_args.get("--load_models", False):
                    cmd_args["--force_train"] = get_yes_no("Force training (new version if model exists and not loading)?", default_yes=False)
                cmd_args["--render"] = get_yes_no("Render game content during training?", default_yes=False) 
                cmd_args["--max_steps_per_episode"] = int(input("Max steps per episode/evaluation (default 20000): ") or "20000") 
                cmd_args["--save_interval"] = int(input("Save NN models every N episodes (GA/NEAT save per gen, default 10): ") or "10") 
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
                        if not get_yes_no(f"Run {cmd_args['--agent'].upper()} untrained?", default_yes=False): 
                            input("\nPress Enter..."); continue
                    else:
                        print(f"\nModels for {cmd_args['--agent'].upper()}:")
                        default_model_choice_ui = len(available_versions) 
                        chosen_model_idx = get_choice("Select model:", available_versions, allow_cancel=True, default_on_enter=default_model_choice_ui)
                        if chosen_model_idx == "cancel" or chosen_model_idx is None: input("\nPress Enter..."); continue
                        model_file_to_load = os.path.join(MODELS_DIR_MAIN, available_versions[chosen_model_idx])
                        cmd_args["--model_file_path"] = model_file_to_load 
                
                cmd_args["--episodes"] = int(input(f"Episodes (default {'5' if model_file_to_load or cmd_args['--agent'] == 'random' else '1'}): ") or ("5" if model_file_to_load or cmd_args['--agent'] == 'random' else "1"))
                cmd_args["--render"] = get_yes_no("Render game?", default_yes=True)
                cmd_args["--max_steps_per_episode"] = int(input("Max steps (default 100000): ") or "100000")
                cmd_args["--silent"] = get_yes_no("No sounds?", default_yes=True)
                record_gif = get_yes_no("Record GIFs?", default_yes=False)
                if record_gif:
                    cmd_args["--gif_episodes"] = int(input("Record N initial episodes as GIF (0=disable, default 1): ") or "1")
                    if cmd_args.get("--gif_episodes", 0) > 0:
                        cmd_args["--gif_fps"] = int(input("GIF FPS (default 15): ") or "15")
                        cmd_args["--gif_capture_every_n_steps"] = int(input("GIF capture interval (default 4): ") or "4")
                        cmd_args["--max_gif_frames"] = int(input("Max frames per GIF segment (default 500): ") or "500")
                else: cmd_args["--gif_episodes"] = 0

            elif selected_ai_operation == "Evaluate All Agents (Latest Versions)":
                script_to_run = "evaluate.py"
                print("\n--- Evaluate All Configuration ---")
                cmd_args["--episodes"] = int(input("Eval episodes per agent (default 20): ") or "20")
                cmd_args["--max_steps_per_episode"] = int(input("Max steps (default 3000): ") or "3000")
                cmd_args["--silent"] = True 
                cmd_args["--render"] = False 

            if script_to_run: 
                command_list = build_command(script_to_run, cmd_args)
                run_command(command_list)
            
            input("\nPress Enter to return to the AI Agent Menu...") 
    
    print("\nExiting Space Invaders AI Launcher. Goodbye!")

if __name__ == "__main__":
    # REMOVED multiprocessing.set_start_method block as multiprocessing is no longer used by train.py
    main_interactive()