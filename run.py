import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
 
# Now import and run the training script
from train.train_address import * 