import os
import sys

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Create necessary directories
os.makedirs('data/np/addr0_tx_mean', exist_ok=True)

try:
    # Import the function
    from utils.dataset_utils import get_all_np_list
    
    # Run the function
    print("Generating .npz files...")
    get_all_np_list()
    print("Done!")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current Python path: {sys.path}")
    print(f"Project root: {project_root}")
    print("Make sure you're running this script from the project root directory.") 