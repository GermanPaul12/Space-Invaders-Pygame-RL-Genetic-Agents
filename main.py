# main.py (NEW interactive launcher)
import subprocess
# main.py (NEW interactive launcher)
import subprocess
import sys
import os
import json # For checking config files (optional, can be done by train.py)

# --- Path Setup ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GAME_PACKAGE_DIR = os.path.join(ROOT_DIR, "game")
MODELS_DIR_MAIN = os.path.join(ROOT_DIR, "models") # CORRECTED to "models"
AGENT_CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
BASE_MODEL_FILENAME_TEMPLATE_MAIN = "{agent_name}_spaceinvaders"

sys.path.insert(0, ROOT_DIR) # Add project root to path for 'from game.game import Game'

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
            elif not choice_str and allow_cancel: return "cancel"
            else: choice = int(choice_str)
            if 1 <= choice <= len(options): return choice - 1
            elif allow_cancel and choice == cancel_option_num: return "cancel"
            else:
                upper_bound = cancel_option_num if allow_cancel else len(options)
                print(f"Invalid choice. Please enter between 1 and {upper_bound}.")
        except ValueError: print("Invalid input. Please enter a number.")
        except Exception: print("Unexpected input error.")

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
        if process.returncode != 0: print(f"\n--- CMD Error (code: {process.returncode}) ---")
        else: print(f"\n--- CMD Success ---")
    except FileNotFoundError: print(f"Error: Script '{command[1]}' not found in project root.")
    except Exception as e: print(f"Subprocess error: {e}")

def get_existing_model_filenames_ui(agent_name):
    # This function now relies on utils.model_helpers which uses the correct "models" dir name.
    from utils.model_helpers import get_model_filenames_for_display
    return get_model_filenames_for_display(agent_name) # model_helpers knows MODELS_DIR is "models"

# --- Main Interactive Loop ---
def main_interactive():
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
        mode_choice_result = get_choice("Select mode:", mode_options)
        
        if mode_choice_result is None or mode_choice_result == "cancel": # Handle potential None from get_choice
            selected_mode = "Exit Launcher" # Default to exit if choice is strange
        else:
            selected_mode = mode_options[mode_choice_result]

        if selected_mode == "Exit Launcher": break
        elif selected_mode == "Play as Human":
            print("\n--- Player Mode ---")
            try:
                from game.game import Game as GameClass
                game_args = {'silent_mode': get_yes_no("Run game without sounds?", default_yes=False), 'ai_training_mode': False}
                print("\nStarting game in player mode...")
                game_instance = GameClass(**game_args)
                game_instance.run_player_mode()
            except ImportError: print("Error: Could not import Game class.")
            except Exception as e: print(f"An error in player mode: {e}")
            input("\nPress Enter to return to the main menu...")

        elif selected_mode == "Run AI Agent":
            clear_screen()
            print("--- AI Agent Mode ---")
            ai_op_options = ["Train Agent(s)", "Test/Run Specific Agent Version", "Evaluate All Agents (Latest Versions)", "Back to Main Menu"]
            ai_op_choice_result = get_choice("\nSelect AI operation:", ai_op_options, allow_cancel=True) # Allow cancel here
            if ai_op_choice_result is None or ai_op_choice_result == "cancel": continue
            selected_ai_operation = ai_op_options[ai_op_choice_result]

            # No need to set script_to_run default here, set it inside conditions
            cmd_args = {}
            agent_options_list = ["dqn", "ppo", "a2c", "genetic", "neat", "random"]
            trainable_agents_for_prompt = [agent for agent in agent_options_list if agent != 'random']

            if selected_ai_operation == "Train Agent(s)":
                script_to_run = "train.py"
                print("\n--- Training Configuration ---")
                agents_to_train_this_session = []
                train_all_q = get_yes_no(f"Train all non-random agents ({', '.join(trainable_agents_for_prompt)})?", default_yes=True)
                if train_all_q: agents_to_train_this_session = trainable_agents_for_prompt
                else:
                    selected_agents_str = input(f"Enter comma-separated agent names to train (e.g., dqn,neat): ").strip()
                    if selected_agents_str:
                        agents_to_train_this_session = [s.strip().lower() for s in selected_agents_str.split(',') if s.strip().lower() in trainable_agents_for_prompt]
                    if not agents_to_train_this_session: print("No valid agents. Aborting."); input("\nPress Enter..."); continue
                cmd_args["--agents"] = ",".join(agents_to_train_this_session)

                for agent_name_config in agents_to_train_this_session:
                    default_config_name = f"{agent_name_config}_default.json"
                    default_config_path = os.path.join(AGENT_CONFIGS_DIR, default_config_name)
                    print(f"\nConfig for {agent_name_config.upper()}:")
                    config_to_pass_for_agent = None # Initialize
                    if os.path.exists(default_config_path):
                        use_default = get_yes_no(f"Use default config '{default_config_name}'?", default_yes=True)
                        config_to_pass_for_agent = default_config_path # Default to default
                        if not use_default:
                            custom_config_name = input(f"Enter filename for {agent_name_config} config (in '{AGENT_CONFIGS_DIR}/'): ").strip()
                            if custom_config_name:
                                custom_config_path = os.path.join(AGENT_CONFIGS_DIR, custom_config_name)
                                if os.path.exists(custom_config_path): config_to_pass_for_agent = custom_config_path
                                else: print(f"Warning: Custom config '{custom_config_name}' not found. Using default.")
                        cmd_args[f"--{agent_name_config}_config_path"] = config_to_pass_for_agent
                    else:
                        print(f"Warning: Default config '{default_config_name}' not found. {agent_name_config.upper()} will use internal defaults.")
                
                cmd_args["--episodes"] = int(input("Episodes per agent/individual (default 1000): ") or "1000")
                cmd_args["--load_models"] = get_yes_no("Load LATEST models to continue training?", default_yes=False)
                if not cmd_args.get("--load_models", False):
                    cmd_args["--force_train"] = get_yes_no("Force training (new version if model exists and not loading)?", default_yes=False)
                cmd_args["--render"] = get_yes_no("Render game content?", default_yes=False)
                cmd_args["--max_steps_per_episode"] = int(input("Max steps per episode (default 2000): ") or "2000")
                cmd_args["--save_interval"] = int(input("Save NN models every N episodes (default 50, GA/NEAT per gen): ") or "50")
                cmd_args["--print_interval_steps"] = int(input("Print stats every N steps (default 500): ") or "500")

            elif selected_ai_operation == "Test/Run Specific Agent Version":
                script_to_run = "test.py"
                agent_choice_result = get_choice("\nSelect agent type:", agent_options_list, allow_cancel=True)
                if agent_choice_result == "cancel" or agent_choice_result is None: continue
                cmd_args["--agent"] = agent_options_list[agent_choice_result]
                
                model_file_to_load = None
                if cmd_args["--agent"] != "random":
                    available_versions = get_existing_model_filenames_ui(cmd_args["--agent"])
                    if not available_versions:
                        print(f"No models for {cmd_args['--agent']}.")
                        if not get_yes_no(f"Run {cmd_args['--agent']} untrained?", default_yes=False): input("\nPress Enter..."); continue
                        # No --model_file_path to pass, test.py will handle agent without loaded model
                    else:
                        print(f"\nModels for {cmd_args['--agent']}:")
                        chosen_model_idx = get_choice("Select model:", available_versions, allow_cancel=True)
                        if chosen_model_idx == "cancel" or chosen_model_idx is None: input("\nPress Enter..."); continue
                        model_file_to_load = os.path.join(MODELS_DIR_MAIN, available_versions[chosen_model_idx])
                        cmd_args["--model_file_path"] = model_file_to_load # test.py uses this
                
                cmd_args["--episodes"] = int(input(f"Episodes (default {'5' if model_file_to_load else '1'}): ") or ("5" if model_file_to_load else "1"))
                cmd_args["--render"] = get_yes_no("Render game?", default_yes=True)
                cmd_args["--max_steps_per_episode"] = int(input("Max steps (default 3000): ") or "3000")
                cmd_args["--silent"] = get_yes_no("No sounds?", default_yes=True)
                cmd_args["--gif_episodes"] = int(input("Record N initial episodes as GIF (0=disable, default 0): ") or "0")
                if cmd_args.get("--gif_episodes", 0) > 0:
                    cmd_args["--gif_fps"] = int(input("GIF FPS (default 15): ") or "15")
                    cmd_args["--gif_capture_every_n_steps"] = int(input("GIF capture interval (default 4): ") or "4")

            elif selected_ai_operation == "Evaluate All Agents (Latest Versions)":
                script_to_run = "evaluate.py"
                print("\n--- Evaluate All Configuration ---")
                cmd_args["--episodes"] = int(input("Eval episodes per agent (default 20): ") or "20")
                cmd_args["--max_steps_per_episode"] = int(input("Max steps (default 3000): ") or "3000")
                cmd_args["--silent"] = True # Evaluation usually silent
                cmd_args["--render"] = False # Evaluation usually headless

            if script_to_run and cmd_args: # Ensure script_to_run is set
                command_list = build_command(script_to_run, cmd_args)
                run_command(command_list)
            elif script_to_run: # No cmd_args but script selected (e.g. evaluate all with defaults)
                 command_list = build_command(script_to_run, {})
                 run_command(command_list)
            
            input("\nPress Enter to return to the AI Agent Menu...") # For all AI ops
    print("\nExiting launcher. Goodbye!")

if __name__ == "__main__":
    main_interactive()
if __name__ == "__main__":
    main_interactive()