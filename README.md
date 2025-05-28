# Space Invaders with Pygame, Reinforcement Learning, and Genetic/NEAT Agents

This project implements the classic Space Invaders game using Pygame and features a variety of AI agents that can learn to play the game. These agents include standard Reinforcement Learning algorithms like DQN, PPO, A2C, as well as population-based methods like a Genetic Algorithm and NEAT (NeuroEvolution of Augmenting Topologies).

The base game mechanics are inspired by [leerob/space-invaders](https://github.com/leerob/space-invaders), while all AI agent implementations and the training/evaluation framework are custom-built for this project.

**GitHub Repository:** [https://github.com/GermanPaul12/Space-Invaders-Pygame-RL-Genetic-Agents](https://github.com/GermanPaul12/Space-Invaders-Pygame-RL-Genetic-Agents)

## Table of Contents

- [Space Invaders with Pygame, Reinforcement Learning, and Genetic/NEAT Agents](#space-invaders-with-pygame-reinforcement-learning-and-geneticneat-agents)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Folder Structure](#folder-structure)
  - [Setup Instructions](#setup-instructions)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [How to Run](#how-to-run)
    - [Interactive Launcher (`main.py`)](#interactive-launcher-mainpy)
    - [Direct Script Execution (Advanced)](#direct-script-execution-advanced)
      - [`train.py`](#trainpy)
      - [`test.py`](#testpy)
      - [`evaluate.py`](#evaluatepy)
  - [Agent Implementations](#agent-implementations)
    - [DQN (Deep Q-Network)](#dqn-deep-q-network)
    - [PPO (Proximal Policy Optimization)](#ppo-proximal-policy-optimization)
    - [A2C (Advantage Actor-Critic)](#a2c-advantage-actor-critic)
    - [Genetic Algorithm (GA)](#genetic-algorithm-ga)
    - [NEAT (NeuroEvolution of Augmenting Topologies)](#neat-neuroevolution-of-augmenting-topologies)
    - [Random Agent](#random-agent)
  - [Configuration](#configuration)
    - [Game Configuration (`game/config.py`)](#game-configuration-gameconfigpy)
    - [Agent Hyperparameters (`configs/`)](#agent-hyperparameters-configs)
  - [Project Functionality Overview](#project-functionality-overview)
    - [Game Core](#game-core)
    - [AI Agent Interface](#ai-agent-interface)
    - [Training Pipeline](#training-pipeline)
    - [Testing \& Evaluation](#testing--evaluation)
    - [Model Versioning](#model-versioning)
  - [Potential Future Work](#potential-future-work)
  - [Sources and Libraries](#sources-and-libraries)
  - [Contributing](#contributing)
  - [License](#license)

## Features

*   Classic Space Invaders gameplay implemented with Pygame.
*   Interactive launcher (`main.py`) for easy selection of game modes and AI operations.
*   Multiple AI agent implementations:
    *   Deep Q-Network (DQN)
    *   Proximal Policy Optimization (PPO)
    *   Advantage Actor-Critic (A2C)
    *   Genetic Algorithm (GA)
    *   NeuroEvolution of Augmenting Topologies (NEAT)
    *   Random baseline agent.
*   Training mode for all agents, with support for:
    *   Loading pre-trained models to continue training.
    *   Forcing retraining and creating new model versions.
    *   Agent-specific hyperparameter configuration via JSON files.
    *   Optional rendering during training.
*   Testing mode to run specific pre-trained agent model versions.
*   Evaluation mode to quantitatively assess the performance of the latest models for all agents and save results to CSV.
*   GIF recording functionality during testing/evaluation.
*   Optimized for faster training (headless mode, reduced delays for AI).
*   Model versioning system (`agentname_vX.pth/.pkl`).

## Folder Structure

```
project_root/
├── main.py              # Interactive launcher application
├── train.py             # Script for training multiple AI agents
├── test.py              # Script for testing/running specific agent models
├── evaluate.py          # Script for evaluating all agents and generating reports
│
├── game/                  # Core game logic and assets
│   ├── __init__.py
│   ├── game.py            # Main game class and sprite logic
│   └── config.py          # Game settings (screen, FPS, paths, etc.)
│
├── agents/                # AI agent implementations
│   ├── __init__.py
│   ├── agent.py           # Abstract base class for agents
│   ├── dqn_agent.py
│   ├── ppo_agent.py
│   ├── a2c_agent.py
│   ├── genetic_agent.py
│   ├── neat_agent.py
│   └── random_agent.py
│
├── configs/               # Agent-specific hyperparameter configuration files
│   ├── dqn_default.json
│   ├── ppo_default.json
│   ├── a2c_default.json
│   ├── genetic_default.json
│   └── neat_default.json
│
├── models/                # Saved trained agent models (.pth for PyTorch, .pkl for NEAT)
│   └── (e.g., dqn_spaceinvaders.pth, neat_spaceinvaders_v2.pkl)
│
├── evaluation_results/    # CSV files from the evaluation script
│   └── (e.g., evaluation_summary_20231027_103000.csv)
│
├── gifs/                  # Saved gameplay GIFs from test/evaluation mode
│   └── (e.g., test_dqn_spaceinvaders_ep1_score120.gif)
│
├── utils/                 # Utility scripts and helper functions
│   ├── __init__.py
│   └── model_helpers.py   # Functions for model path and version management
│
├── fonts/
│   └── space_invaders.ttf
├── images/
│   └── (game image assets - .png files)
├── sounds/
│   └── (game sound assets - .wav files)
│
├── LICENSE          # MIT License
├── requirements.txt          # Requirements for the project
├── uv.lock          # used by uv to sync dependencies
└── README.md              # This file
```

## Setup Instructions

### Prerequisites

*   Python 3.8 or higher.
*   [uv](https://github.com/astral-sh/uv) (recommended for environment and package management, but `pip` can also be used).
*   A GPU with CUDA support is highly recommended for training neural network based agents (DQN, PPO, A2C) for reasonable performance. CPU training will be very slow.

### Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/GermanPaul12/Space-Invaders-Pygame-RL-Genetic-Agents.git
    cd Space-Invaders-Pygame-RL-Genetic-Agents
    ```

2.  **Create a virtual environment (recommended):**
    Using `uv`:
    ```
    uv venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate  # On Windows
    ```
    Alternatively, using standard `venv`:
    ```
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    ```
    # Using uv (recommended for speed)
    uv sync
    # or
    # uv add -r requirements.txt

    # Or using pip
    # pip install -r requirements.txt
    ```
    *   **PyTorch:** Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) for the correct installation command based on your OS, package manager (pip/conda), and CUDA version (if applicable).
    *   **Pillow:** Used for image preprocessing.
    *   **Imageio:** Used for GIF creation. `ffmpeg` or `imageimagick` might be needed as backends for `imageio` for certain formats/features, install them via your system's package manager if `imageio` reports missing backends.

## How to Run

The primary way to interact with the project is through the `main.py` interactive launcher.

### Interactive Launcher (`main.py`)

Run the launcher from the project root directory:
```
uv run main.py 
# Or if not using uv's run command:
# python3 main.py
```

This will present a series of menus to:
1.  **Play as Human:** Start a normal game session.
2.  **Run AI Agent:**
    *   **Train Agent(s):**
        *   Select which agents to train (all default trainable or a custom list).
        *   Configure agent-specific hyperparameters by choosing default JSON configs from the `configs/` directory or providing filenames for custom configs (also in `configs/`).
        *   Set general training parameters like total episodes, loading existing models (latest version), forcing retraining (creates a new versioned model), rendering, max steps, save intervals.
        *   This mode calls `train.py` with the constructed arguments.
    *   **Test/Run Specific Agent Version:**
        *   Select an agent type.
        *   If models exist for that agent, a list of available model versions (e.g., `dqn_spaceinvaders.pth`, `dqn_spaceinvaders_v2.pth`) will be shown for selection.
        *   Configure test parameters like number of episodes, rendering, GIF recording, etc.
        *   This mode calls `test.py` with the constructed arguments, including the specific model file path.
    *   **Evaluate All Agents (Latest Versions):**
        *   Configure evaluation parameters (number of episodes per agent, etc.).
        *   This mode calls `evaluate.py` which will test the *latest* saved model for each known agent type and save results to a CSV file.

### Direct Script Execution (Advanced)

You can also run the individual scripts (`train.py`, `test.py`, `evaluate.py`) directly from the command line if you prefer to pass all arguments manually. Each script supports a `--help` flag to show its available arguments.

All scripts should be run from the **project root directory**.

#### `train.py`
Used for training one or more agents.
```
# Example: Train DQN and NEAT, load existing models if available, render
uv run python train.py --agents dqn,neat --load_models --render --episodes 500 --dqn_config_path configs/dqn_custom_params.json 

# Example: Force retrain PPO (creates new version), use default PPO config
uv run python train.py --agents ppo --force_train --episodes 2000
```
*   Accepts agent-specific config paths (e.g., `--dqn_config_path configs/dqn_params.json`). If not provided, it attempts to load `configs/<agent_name>_default.json`.
*   Handles model versioning for saving.

#### `test.py`
Used for testing or watching a specific pre-trained agent model.
```
# Example: Test a specific version of a DQN model, render, and record 1 GIF
uv run python test.py --agent dqn --model_file_path models/dqn_spaceinvaders_v2.pth --render --gif_episodes 1 --episodes 3

# Example: Test the latest PPO model
uv run python test.py --agent ppo --render --episodes 5 
# (test.py will load the latest ppo model if model_file_path is not given)
```
*   `--model_file_path`: Specifies the exact model file to test.
*   If `--model_file_path` is omitted, it attempts to load the *latest* version for the given `--agent`.
*   Includes GIF recording options with frame-based splitting.

#### `evaluate.py`
Used for headless evaluation of the latest models for all agent types.
```
# Example: Evaluate all agents for 20 episodes each
uv run python evaluate.py --episodes 20
```
*   Results are saved to a CSV file in the `evaluation_results/` directory.

## Agent Implementations

All agents receive preprocessed game observations (typically 1x84x84 grayscale images) and output an action index.

### DQN (Deep Q-Network)
*   **File:** `agents/dqn_agent.py`
*   **Description:** Uses a Q-Network to approximate the action-value function. Employs a replay buffer and a target network for stable learning. Epsilon-greedy exploration.
*   **Config:** `configs/dqn_default.json` (buffer size, batch size, learning rate, gamma, epsilon decay, etc.)
*   **Model format:** `.pth` (PyTorch state dictionary)

### PPO (Proximal Policy Optimization)
*   **File:** `agents/ppo_agent.py`
*   **Description:** An actor-critic method that uses a clipped surrogate objective function for more stable policy updates. Collects trajectories of experience.
*   **Config:** `configs/ppo_default.json` (learning rate, gamma, GAE lambda, clip ratio, epochs, entropy coefficient, etc.)
*   **Model format:** `.pth`

### A2C (Advantage Actor-Critic)
*   **File:** `agents/a2c_agent.py`
*   **Description:** A synchronous actor-critic method. The actor learns the policy, and the critic learns a value function to estimate returns.
*   **Config:** `configs/a2c_default.json` (learning rate, gamma, entropy coefficient, value loss coefficient.)
*   **Model format:** `.pth`

### Genetic Algorithm (GA)
*   **File:** `agents/genetic_agent.py`
*   **Description:** A population-based approach where neural network weights are treated as "genes." Evolution proceeds through selection, crossover, and mutation based on game score as fitness.
*   **Config:** `configs/genetic_default.json` (population size, mutation rates, crossover rate, number of elites.)
*   **Model format:** `.pth` (saves the state dict of the best network in the population).

### NEAT (NeuroEvolution of Augmenting Topologies)
*   **File:** `agents/neat_agent.py`
*   **Description:** Evolves both network weights and topologies (structure). Features speciation, innovation tracking, and complexification of networks over generations.
*   **Config:** `configs/neat_default.json` (population size, compatibility thresholds, mutation rates for structure and weights, speciation parameters, etc.)
*   **Model format:** `.pkl` (saves the Python `GenomeNEAT` object of the best individual using `pickle`).

### Random Agent
*   **File:** `agents/random_agent.py`
*   **Description:** Chooses actions randomly. Serves as a baseline for comparison. No training or saving/loading.

## Configuration

### Game Configuration (`game/config.py`)
This file contains settings for the core game visuals and mechanics:
*   Screen dimensions, colors.
*   Paths to assets (fonts, images, sounds).
*   Game FPS (`FPS` for human play, `AI_TRAIN_RENDER_FPS` for visually sped-up AI training).
*   Player and enemy speeds, laser speeds, positions.
*   Score values for different enemy types.

### Agent Hyperparameters (`configs/`)
The `configs/` directory holds JSON files for configuring agent-specific hyperparameters. For each agent type (e.g., `dqn`), a `dqn_default.json` file provides default parameters. Users can create custom JSON files (e.g., `dqn_custom_run1.json`) in this directory and specify them via the interactive launcher or command-line arguments to `train.py`.

If a config file is not found or cannot be parsed, agents will fall back to internal default values defined in their respective class constructors.

## Project Functionality Overview

### Game Core
*   Located in `game/game.py`.
*   Manages game state, sprite groups (player, enemies, bullets, blockers, mystery ship), rendering, collision detection, and scoring.
*   Provides modes for human play (`run_player_mode()`) and AI interaction (`reset_for_ai()`, `step_ai()`).
*   Supports `silent_mode` (no sounds) and `ai_training_mode` (reduced delays for faster simulation).

### AI Agent Interface
*   The abstract base class `agents/agent.py` defines the common interface (`choose_action`, `learn`, `save`, `load`).
*   Each specific agent inherits from this and implements the methods.
*   Agents generally take a preprocessed game screen as input. Preprocessing (grayscale, resize, normalize) is handled by a shared function typically in `agents/dqn_agent.py` and used by other vision-based agents.

### Training Pipeline
*   **Interactive Launcher (`main.py`):** Allows users to select agents and configure training parameters, including agent-specific config files.
*   **`train.py`:**
    *   Receives a list of agents to train and their respective config file paths.
    *   Iterates through each agent, instantiating it with loaded hyperparameters.
    *   Manages loading existing models (latest version) for continued training or skipping/forcing retraining based on user flags.
    *   Runs the training loop: episodes, steps, agent actions, game steps, learning updates.
    *   Handles model saving with versioning (e.g., `agent_v2.pth`). Population-based agents like NEAT and GA save their best individual/genome at the end of each generation.

### Testing & Evaluation
*   **Interactive Launcher (`main.py`):**
    *   For "Test/Run," allows selection of an agent and a specific saved model version.
    *   For "Evaluate All," triggers evaluation of the latest model for all known agent types.
*   **`test.py`:**
    *   Loads a specified agent model (or latest if none specified).
    *   Runs the agent in the game for a set number of episodes.
    *   Supports rendering and GIF recording (with frame-based splitting).
    *   Prints performance metrics (rewards, scores) to the console.
*   **`evaluate.py`:**
    *   Runs headless evaluations of the latest model for each agent.
    *   Collects performance metrics.
    *   Saves a summary of results to a CSV file in `evaluation_results/`.

### Model Versioning
*   Implemented in `utils/model_helpers.py`.
*   When training an agent fresh or with `--force_train`, models are saved with an incrementing version suffix (e.g., `_v2`, `_v3`) if a base model or previous versions exist.
    *   Example: `dqn_spaceinvaders.pth`, then `dqn_spaceinvaders_v2.pth`.
*   NEAT models are saved as `.pkl` files, others as `.pth`.
*   Loading for continued training (`--load_models` in `train.py`) loads the latest available version.

## Potential Future Work

*   **Advanced NEAT Resume:** Implement full saving/loading of NEAT's evolutionary state (population, species, innovation numbers) for perfect resumption of training.
*   **Hyperparameter Optimization:** Integrate tools like Optuna or Ray Tune for automatic hyperparameter searching.
*   **More Sophisticated Preprocessing:** Implement frame stacking for DQN/PPO/A2C to give agents a sense of motion.
*   **Curriculum Learning:** Start with simpler game versions or objectives and gradually increase complexity.
*   **GUI for Launcher:** Develop a simple graphical user interface instead of the terminal-based one.
*   **Expanded Agent Zoo:** Implement more RL algorithms (e.g., SAC, TD3 for continuous control if game actions were adapted).
*   **Detailed Performance Logging:** Use tools like TensorBoard or Weights & Biases for logging training progress.
*   **Unit Tests:** Add unit tests for game logic and agent components.

## Sources and Libraries

*   **Base Game Inspiration:** [github.com/leerob/space-invaders](https://github.com/leerob/space-invaders) by Lee Robinson.
*   **Pygame:**
    *   Official Website: [pygame.org](https://www.pygame.org/)
    *   Documentation: [pygame.org/docs/](https://www.pygame.org/docs/)
    *   *Use:* Core game development, rendering, event handling, sound.
*   **NumPy:**
    *   Official Website: [numpy.org](https://numpy.org/)
    *   Documentation: [numpy.org/doc/](https://numpy.org/doc/)
    *   *Use:* Numerical operations, array manipulation for observations and agent calculations.
*   **PyTorch (for DQN, PPO, A2C):**
    *   Official Website: [pytorch.org](https://pytorch.org/)
    *   Documentation: [pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
    *   Tutorials (RL specific): [PyTorch Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
    *   *Use:* Building and training neural networks for value-based and policy-based RL agents.
*   **Pillow (PIL Fork):**
    *   Documentation: [pillow.readthedocs.io](https://pillow.readthedocs.io/en/stable/)
    *   *Use:* Image preprocessing (resizing, grayscaling) of game frames for AI agents.
*   **Imageio:**
    *   Documentation: [imageio.readthedocs.io](https://imageio.readthedocs.io/en/stable/)
    *   *Use:* Recording gameplay GIFs.
*   **NEAT Algorithm:**
    *   Original Paper: [Evolving Neural Networks through Augmenting Topologies (Stanley & Miikkulainen, 2002)](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
    *   Helpful Resource: [NEAT Users Page (historical but good concepts)](http://www.cs.ucf.edu/~kstanley/neat.html)
    *   Many online resources and implementations exist that explain NEAT concepts.
*   **Genetic Algorithms:**
    *   A broad field with many introductory texts and tutorials. Example: [Wikipedia - Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm)
*   **Reinforcement Learning (General Concepts for DQN, PPO, A2C):**
    *   Book: "Reinforcement Learning: An Introduction" by Sutton and Barto (Highly recommended foundational text). [Link](http://incompleteideas.net/book/the-book-2nd.html)
    *   OpenAI Spinning Up: [spinningup.openai.com](https://spinningup.openai.com/en/latest/) (Excellent explanations of RL algorithms).

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to fork the repository, make changes, and open a pull request. If you encounter any bugs or have suggestions, please open an issue on the GitHub repository.

## License

This project is open-source. Please check the `LICENSE` file for more details.
