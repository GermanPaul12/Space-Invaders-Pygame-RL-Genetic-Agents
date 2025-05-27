# main.py (NEW interactive launcher)
import subprocess
import sys
import os
import glob # For finding model files

# --- Path Setup ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GAME_PACKAGE_DIR = os.path.join(ROOT_DIR, "game")
MODELS_DIR_MAIN = os.path.join(ROOT_DIR, "trained_models")
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
    """Gets a valid choice from the user.
       `default_on_enter` is the 1-based index of the default choice if user just presses Enter.
    """
    print(prompt)
    for i, option in enumerate(options):
        print(f"{i+1}. {option}")
    
    default_prompt_part = ""
    if default_on_enter is not None and 1 <= default_on_enter <= len(options):
        default_prompt_part = f" (default: {default_on_enter})"
    
    cancel_option_num = 0
    if allow_cancel:
        cancel_option_num = len(options) + 1
        print(f"{cancel_option_num}. Cancel / Go Back")

    while True:
        try:
            choice_str = input(f"Enter your choice (number){default_prompt_part}: ").strip()
            if not choice_str and default_on_enter is not None:
                choice = default_on_enter
            elif not choice_str and allow_cancel: # Empty input with cancel allowed could mean cancel
                 return "cancel"
            else:
                choice = int(choice_str)

            if 1 <= choice <= len(options):
                return choice - 1 # Return 0-based index
            elif allow_cancel and choice == cancel_option_num:
                return "cancel"
            else:
                upper_bound = cancel_option_num if allow_cancel else len(options)
                print(f"Invalid choice. Please enter a number between 1 and {upper_bound}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception: # Catch other potential errors with input
            print("An unexpected error occurred with your input.")


def get_yes_no(prompt, default_yes=None):
    """Gets a yes/no answer from the user.
       `default_yes`=True means 'yes' is default, False means 'no', None means no default.
    """
    options = " (yes/no)"
    if default_yes is True:
        options = " (YES/no)"
    elif default_yes is False:
        options = " (yes/NO)"
    
    while True:
        answer = input(f"{prompt}{options}: ").strip().lower()
        if not answer and default_yes is not None: # User pressed Enter
            return default_yes
        if answer in ['yes', 'y']:
            return True
        elif answer in ['no', 'n']:
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def build_command(script_name, args_dict):
    """Builds a command list for subprocess."""
    command = [get_python_executable(), "-u", script_name] # -u for unbuffered output
    for arg, value in args_dict.items():
        if isinstance(value, bool):
            if value: # Add flag if true
                command.append(arg)
        elif value is not None: # Add argument with value
            command.append(arg)
            command.append(str(value))
    return command

def run_command(command):
    """Runs a command using subprocess and prints output."""
    print(f"\nExecuting: {' '.join(command)}\n")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in process.stdout: # Stream output
            print(line, end='', flush=True)
        process.wait() # Wait for the process to complete
        if process.returncode != 0:
            print(f"\n--- Command finished with error (code: {process.returncode}) ---")
        else:
            print(f"\n--- Command finished successfully ---")
    except FileNotFoundError:
        print(f"Error: Script '{command[1]}' not found. Make sure it's in the project root.")
    except Exception as e:
        print(f"An error occurred while running the command: {e}")

def get_existing_model_filenames(agent_name):
    """Finds all existing model filenames (not full paths) for a given agent."""
    if not os.path.exists(MODELS_DIR_MAIN):
        return []
    pattern_base = BASE_MODEL_FILENAME_TEMPLATE_MAIN.format(agent_name=agent_name)
    versions = []
    # Sort to have a somewhat predictable order, e.g., base, _v2, _v3
    for f_name in sorted(os.listdir(MODELS_DIR_MAIN)):
        if f_name.startswith(pattern_base) and f_name.endswith(".pth"):
            versions.append(f_name)
    return versions

def main_interactive():
    while True: # Main loop for the launcher
        clear_screen()
        print("Welcome to Space Invaders AI Launcher!")
        print("======================================")

        mode_options = ["Play as Human", "Run AI Agent", "Exit Launcher"]
        mode_choice_idx = get_choice("Select mode:", mode_options)

        if mode_options[mode_choice_idx] == "Exit Launcher":
            break

        elif mode_options[mode_choice_idx] == "Play as Human":
            print("\n--- Player Mode ---")
            try:
                from game.game import Game as GameClass
                game_args = {}
                game_args['silent_mode'] = get_yes_no("Run game without sounds?", default_yes=False)
                game_args['ai_training_mode'] = False # Player mode is not AI training
                
                print("\nStarting game in player mode...")
                game_instance = GameClass(**game_args)
                game_instance.run_player_mode()
            except ImportError:
                print("Error: Could not import the Game class. Ensure 'game' package and files are correct.")
            except Exception as e:
                print(f"An error occurred while trying to start player mode: {e}")
            input("\nPress Enter to return to the main menu...")


        elif mode_options[mode_choice_idx] == "Run AI Agent":
            clear_screen()
            print("--- AI Agent Mode ---")
            ai_op_options = ["Train Agent(s)", "Test/Run Specific Agent Version", "Evaluate All Agents (Latest Versions)", "Back to Main Menu"]
            ai_op_choice_idx = get_choice("\nSelect AI operation:", ai_op_options)
            selected_ai_operation = ai_op_options[ai_op_choice_idx]

            if selected_ai_operation == "Back to Main Menu":
                continue

            script_to_run = "run_game_ai.py" # Default, used by Test and Evaluate
            cmd_args = {}
            agent_options_list = ["dqn", "ppo", "a2c", "genetic", "random"]


            if selected_ai_operation == "Train Agent(s)":
                script_to_run = "train.py"
                print("\n--- Training Configuration ---")
                
                train_all_q = get_yes_no("Train all available non-random agents (dqn, ppo, a2c, genetic)?", default_yes=True)
                if train_all_q:
                    cmd_args["--agents"] = "dqn,ppo,a2c,genetic"
                else:
                    selected_agents_str = input(f"Enter comma-separated agent names to train (from {agent_options_list[:-1]}): ").strip()
                    if selected_agents_str:
                        cmd_args["--agents"] = selected_agents_str
                    else:
                        print("No agents selected for training. Aborting.")
                        input("\nPress Enter to continue..."); continue
                
                cmd_args["--episodes"] = int(input("Number of episodes per agent (default 1000): ") or "1000")
                cmd_args["--load_models"] = get_yes_no("Load LATEST models to continue training?", default_yes=False)
                if not cmd_args.get("--load_models", False):
                    cmd_args["--force_train"] = get_yes_no("Force training (new version if model exists and not loading)?", default_yes=False)
                cmd_args["--render"] = get_yes_no("Render game content during training?", default_yes=False)
                cmd_args["--max_steps_per_episode"] = int(input("Max steps per episode (default 2000): ") or "2000")
                cmd_args["--save_interval"] = int(input("Save model every N episodes (default 50): ") or "50")
                cmd_args["--print_interval_steps"] = int(input("Print stats every N steps within episode (default 500): ") or "500")

            elif selected_ai_operation == "Test/Run Specific Agent Version":
                cmd_args["--mode"] = "test" # Both use 'test' mode in run_game_ai.py
                
                agent_choice_idx = get_choice("\nSelect agent type to test/run:", agent_options_list)
                cmd_args["--agent"] = agent_options_list[agent_choice_idx]

                model_file_to_load = None
                if cmd_args["--agent"] != "random":
                    available_versions = get_existing_model_filenames(cmd_args["--agent"])
                    if not available_versions:
                        print(f"Warning: No trained models found for {cmd_args['--agent']}.")
                        if not get_yes_no(f"Run {cmd_args['--agent']} untrained (random behavior)?", default_yes=False):
                            input("\nPress Enter to continue..."); continue # Back to AI op menu
                        cmd_args["--load_model"] = False # Explicitly don't load
                    else:
                        print(f"\nAvailable model versions for {cmd_args['--agent']}:")
                        chosen_model_idx = get_choice("Select model version:", available_versions, allow_cancel=True)
                        if chosen_model_idx == "cancel" or chosen_model_idx is None:
                             input("\nPress Enter to continue..."); continue
                        model_file_to_load = os.path.join(MODELS_DIR_MAIN, available_versions[chosen_model_idx])
                        cmd_args["--model_file_path"] = model_file_to_load # Pass specific path
                        cmd_args["--load_model"] = True # Indicate model loading is intended
                
                cmd_args["--episodes"] = int(input(f"Number of episodes to run/test (default {'10' if model_file_to_load else '1'}): ") or ("10" if model_file_to_load else "1"))
                cmd_args["--render"] = get_yes_no("Render game during this run?", default_yes=True)
                cmd_args["--max_steps_per_episode"] = int(input("Max steps per episode (default 2000): ") or "2000")
                cmd_args["--silent"] = get_yes_no("Run game without sounds (for this run)?", default_yes=True)


            elif selected_ai_operation == "Evaluate All Agents (Latest Versions)":
                cmd_args["--mode"] = "evaluate_all"
                print("\n--- Evaluate All Configuration ---")
                cmd_args["--episodes"] = int(input("Number of evaluation episodes per agent (default 10): ") or "10")
                cmd_args["--render"] = get_yes_no("Render game during evaluation?", default_yes=False)
                cmd_args["--max_steps_per_episode"] = int(input("Max steps per episode (default 2000): ") or "2000")
                cmd_args["--silent"] = get_yes_no("Run game without sounds during evaluation?", default_yes=True)

            # Build and run the command for AI operations
            if cmd_args: # Ensure some arguments were set (e.g., agent wasn't cancelled)
                command_list = build_command(script_to_run, cmd_args)
                run_command(command_list)
            input("\nPress Enter to return to the AI Agent Menu...")


    print("\nExiting launcher. Goodbye!")

if __name__ == "__main__":
    main_interactive()