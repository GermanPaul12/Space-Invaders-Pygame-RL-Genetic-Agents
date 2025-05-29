# utils/cli_args.py
import argparse
import os

ALL_AGENT_TYPES_AVAILABLE = ['dqn', 'ppo', 'a2c', 'genetic', 'neat', 'random']
TRAINABLE_AGENTS_DEFAULT_LIST = [agent for agent in ALL_AGENT_TYPES_AVAILABLE if agent not in ['random']]


def add_common_training_args(parser):
    """Adds common arguments for the main training script (train.py)."""
    parser.add_argument("--agents", type=str, 
                        default=",".join(TRAINABLE_AGENTS_DEFAULT_LIST),
                        help=f"Comma-separated list of agents to train from: {', '.join(ALL_AGENT_TYPES_AVAILABLE)}")
    parser.add_argument("--episodes", type=int, default=1000, 
                        help="Episodes for NN agents; Generations for NEAT/Genetic.")
    parser.add_argument("--load_models", action='store_true', 
                        help="Load LATEST model to continue. Skips if no model unless --force_train.")
    parser.add_argument("--force_train", action='store_true',
                        help="Force new training (new versioned model) even if model(s) exist.")
    parser.add_argument("--render", action='store_true', help="Render game content during training.")
    parser.add_argument("--max_steps_per_episode", type=int, default=20000,
                        help="Max game steps per evaluation/episode.")
    parser.add_argument("--save_interval", type=int, default=10, 
                        help="Save NN models every N eps. Pop-based save per generation.")
    parser.add_argument("--print_interval_steps", type=int, default=500, 
                        help="Print stats every N steps within an episode.")
    parser.add_argument("--silent_training", action='store_true', help="Run training without sounds (even if rendering).")
    return parser

def add_agent_config_path_args(parser):
    """Adds --<agent_name>_config_path arguments for all trainable agents."""
    for agent_type in TRAINABLE_AGENTS_DEFAULT_LIST: 
        parser.add_argument(f"--{agent_type}_config_path", type=str, default=None,
                            help=f"Path to JSON config for {agent_type} (in 'configs/' dir).")
    return parser

def add_common_test_args(parser):
    """Adds common arguments for the testing script (test.py)."""
    parser.add_argument("--agent", type=str, choices=ALL_AGENT_TYPES_AVAILABLE, required=True, help="Agent to test.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes.")
    parser.add_argument("--model_file_path", type=str, default=None, 
                        help="Specific model file path to load. If None, loads latest for the agent.")
    parser.add_argument("--render", action='store_true', help="Render game during test.")
    parser.add_argument("--max_steps_per_episode", type=int, default=100000)
    parser.add_argument("--silent", action='store_true', help="Run game without sounds.")
    return parser

def add_gif_args(parser, default_max_frames=600): 
    """Adds GIF recording arguments."""
    parser.add_argument("--gif_episodes", type=int, default=0, help="Record N initial episodes as GIF (0=disable).")
    parser.add_argument("--gif_fps", type=int, default=15, help="Playback FPS for GIFs.")
    parser.add_argument("--gif_capture_every_n_steps", type=int, default=4, help="Capture frame every N game steps.")
    parser.add_argument("--max_gif_frames", type=int, default=default_max_frames,
                        help=f"Max frames per GIF segment (default: {default_max_frames}). Tune for size.")
    return parser

def add_common_eval_args(parser):
    """Adds common arguments for the evaluation script (evaluate.py)."""
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes per agent.")
    parser.add_argument("--max_steps_per_episode", type=int, default=3000)
    return parser