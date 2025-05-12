import os
import sys
import subprocess
import time

def run_command(command, cwd=None, timeout=3600):  # 1 hour timeout
    print(f"Running: {command}")
    # Set PYTHONPATH environment variable
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Run the command with a timeout
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            env=env, 
            cwd=cwd,
            timeout=timeout
        )
        
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error output: {result.stderr}")
            sys.exit(1)
            
        print(result.stdout)
        return True
        
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds: {command}")
        return False
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Create necessary directories
os.makedirs('data/process', exist_ok=True)
os.makedirs('data/np/addr0_tx_mean', exist_ok=True)

# Run preprocessing scripts with retries
max_retries = 3
retry_delay = 60  # 1 minute delay between retries

print("Step 1: Processing address data...")
preprocess_dir = os.path.join(project_root, 'dataloader', 'preprocess')
for attempt in range(max_retries):
    if run_command("python process_address.py", cwd=preprocess_dir):
        break
    if attempt < max_retries - 1:
        print(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
        time.sleep(retry_delay)
    else:
        print("Max retries reached. Exiting.")
        sys.exit(1)

print("\nStep 2: Processing transaction data...")
for attempt in range(max_retries):
    if run_command("python process_tx.py", cwd=preprocess_dir):
        break
    if attempt < max_retries - 1:
        print(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
        time.sleep(retry_delay)
    else:
        print("Max retries reached. Exiting.")
        sys.exit(1)

print("\nStep 3: Processing value data...")
for attempt in range(max_retries):
    if run_command("python process_value.py", cwd=preprocess_dir):
        break
    if attempt < max_retries - 1:
        print(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
        time.sleep(retry_delay)
    else:
        print("Max retries reached. Exiting.")
        sys.exit(1)

print("\nStep 4: Generating .npz files...")
for attempt in range(max_retries):
    if run_command("python generate_data.py", cwd=project_root):
        break
    if attempt < max_retries - 1:
        print(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
        time.sleep(retry_delay)
    else:
        print("Max retries reached. Exiting.")
        sys.exit(1)

print("\nAll preprocessing steps completed!") 