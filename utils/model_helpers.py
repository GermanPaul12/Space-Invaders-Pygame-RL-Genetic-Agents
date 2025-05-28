# utils/model_helpers.py
import os
import glob

MODELS_DIR = "models"
BASE_MODEL_FILENAME_TEMPLATE = "{agent_name}_spaceinvaders"

project_root_for_models = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
full_models_dir_path = os.path.join(project_root_for_models, MODELS_DIR)
if not os.path.exists(full_models_dir_path):
    os.makedirs(full_models_dir_path)

def get_existing_model_versions(agent_name, models_base_dir_name=MODELS_DIR):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    target_dir = os.path.join(project_root, models_base_dir_name)
    # ... rest of the function remains the same
    if not os.path.exists(target_dir):
        return []
    pattern_base = BASE_MODEL_FILENAME_TEMPLATE.format(agent_name=agent_name)
    versions = []
    for f_name in sorted(os.listdir(target_dir)):
        if f_name.startswith(pattern_base) and (f_name.endswith(".pth") or f_name.endswith(".pkl")):
            versions.append(os.path.join(target_dir, f_name))
    return versions


def get_next_model_save_path(agent_name, models_base_dir_name=MODELS_DIR):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    target_dir = os.path.join(project_root, models_base_dir_name)
    # ... rest of the function remains the same
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    pattern_base = BASE_MODEL_FILENAME_TEMPLATE.format(agent_name=agent_name)
    extension = ".pkl" if agent_name == "neat" else ".pth"
    base_path = os.path.join(target_dir, f"{pattern_base}{extension}")
    if not os.path.exists(base_path):
        return base_path
    version = 2
    while True:
        versioned_path = os.path.join(target_dir, f"{pattern_base}_v{version}{extension}")
        if not os.path.exists(versioned_path):
            return versioned_path
        version += 1

def get_latest_model_path(agent_name, models_base_dir_name=MODELS_DIR):
    versions = get_existing_model_versions(agent_name, models_base_dir_name)
    return versions[-1] if versions else None

def get_model_filenames_for_display(agent_name, models_base_dir_name=MODELS_DIR):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    target_dir = os.path.join(project_root, models_base_dir_name)
    # ... rest of the function remains the same
    if not os.path.exists(target_dir): return []
    pattern_base = BASE_MODEL_FILENAME_TEMPLATE.format(agent_name=agent_name)
    filenames = []
    for f_name in sorted(os.listdir(target_dir)):
        if f_name.startswith(pattern_base) and (f_name.endswith(".pth") or f_name.endswith(".pkl")):
            filenames.append(f_name)
    return filenames