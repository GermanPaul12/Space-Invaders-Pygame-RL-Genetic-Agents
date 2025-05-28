# utils/report_utils.py
import csv
import os
from datetime import datetime

EVAL_RESULTS_DIR_REPORT = "evaluation_results" # Relative to project root
if not os.path.exists(EVAL_RESULTS_DIR_REPORT):
    os.makedirs(EVAL_RESULTS_DIR_REPORT)

def save_evaluation_to_csv(evaluation_data_list, output_dir_name=None):
    """
    Saves a list of evaluation data (dictionaries) to a CSV file.
    `output_dir_name` is the name of the directory like "evaluation_results".
    """
    if not evaluation_data_list:
        print("No evaluation data to save.")
        return

    target_dir = output_dir_name
    if not target_dir: # Construct full path if only name given or None
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        target_dir = os.path.join(project_root, EVAL_RESULTS_DIR_REPORT)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(target_dir, f"evaluation_summary_{timestamp}.csv")
    
    # Ensure all dicts have the same keys for fieldnames, or get all unique keys
    fieldnames = list(evaluation_data_list[0].keys()) # Assumes first item has all keys
    # For more robustness if keys vary:
    # all_keys = set()
    # for item in evaluation_data_list: all_keys.update(item.keys())
    # fieldnames = sorted(list(all_keys))


    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(evaluation_data_list)
        print(f"\nEvaluation results saved to: {csv_filename}")
    except IOError as e:
        print(f"Error: Could not write evaluation results to CSV '{csv_filename}': {e}")
    except Exception as e_gen:
        print(f"An unexpected error occurred during CSV writing: {e_gen}")