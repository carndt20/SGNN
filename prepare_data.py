import os
import sys
import time
import subprocess
from tqdm import tqdm

def run_command(command, cwd=None):
    """Run a command and return True if successful, False otherwise."""
    try:
        # Set PYTHONPATH environment variable to include the project root
        env = os.environ.copy()
        project_root = os.path.dirname(os.path.abspath(__file__))
        env['PYTHONPATH'] = project_root
        
        result = subprocess.run(command, shell=True, cwd=cwd, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=env)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    # Get the absolute path of the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Maximum number of retries for each step
    max_retries = 3
    retry_delay = 5  # seconds
    
    # Test mode settings
    test_mode = True  # Set to False for full data processing
    test_size = 1000  # Number of transactions to process in test mode
    
    # Total steps in the process
    total_steps = 4
    current_step = 0
    
    print(f"Starting data preparation process... (0%)")
    print(f"Running in {'TEST' if test_mode else 'FULL'} mode")
    if test_mode:
        print(f"Processing {test_size} transactions")
    
    # Step 1: Process address data
    current_step += 1
    print(f"\nStep {current_step}/{total_steps}: Processing address data... ({int(current_step/total_steps*100)}%)")
    preprocess_dir = os.path.join(project_root, 'dataloader', 'preprocess')
    for attempt in range(max_retries):
        if run_command(f"python process_address.py {'--test' if test_mode else ''} {test_size if test_mode else ''}", cwd=preprocess_dir):
            break
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
            time.sleep(retry_delay)
        else:
            print("Max retries reached. Exiting.")
            sys.exit(1)
    
    # Step 2: Process transaction data
    current_step += 1
    print(f"\nStep {current_step}/{total_steps}: Processing transaction data... ({int(current_step/total_steps*100)}%)")
    for attempt in range(max_retries):
        if run_command(f"python process_tx.py {'--test' if test_mode else ''} {test_size if test_mode else ''}", cwd=preprocess_dir):
            break
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
            time.sleep(retry_delay)
        else:
            print("Max retries reached. Exiting.")
            sys.exit(1)
    
    # Step 3: Process value data
    current_step += 1
    print(f"\nStep {current_step}/{total_steps}: Processing value data... ({int(current_step/total_steps*100)}%)")
    for attempt in range(max_retries):
        if run_command(f"python process_value.py {'--test' if test_mode else ''} {test_size if test_mode else ''}", cwd=preprocess_dir):
            break
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
            time.sleep(retry_delay)
        else:
            print("Max retries reached. Exiting.")
            sys.exit(1)
    
    # Step 4: Generate .npz files
    current_step += 1
    print(f"\nStep {current_step}/{total_steps}: Generating .npz files... ({int(current_step/total_steps*100)}%)")
    for attempt in range(max_retries):
        if run_command(f"python dataset_utils.py {'--test' if test_mode else ''} {test_size if test_mode else ''}", cwd=os.path.join(project_root, 'utils')):
            break
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
            time.sleep(retry_delay)
        else:
            print("Max retries reached. Exiting.")
            sys.exit(1)
    
    print("\nData preparation completed successfully! (100%)")

if __name__ == "__main__":
    main() 