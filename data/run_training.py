#!/usr/bin/env python3
"""
3NGIN3 Training Runner

A simple script to run the 3NGIN3 training pipeline from the /data/ directory.
This script provides an easy way to execute training with the command:

    python data/run_training.py

or from the data directory:

    cd data && python run_training.py
"""

import sys
import os

# Ensure we're working from the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Change to project root directory if not already there
if os.getcwd() != project_root:
    os.chdir(project_root)
    print(f"📁 Changed working directory to: {project_root}")

# Add the project root to the path so we can import from the project root
sys.path.insert(0, project_root)

def main():
    """Run the 3NGIN3 training pipeline."""
    print("🚀 Starting 3NGIN3 Training from /data/ directory...")
    print("=" * 60)
    
    try:
        # Import and run the training pipeline
        from training.train_3ngin3 import main as run_training
        
        # Execute the training
        run_training()
        
        print("\n✅ Training completed successfully!")
        print("📊 Check the training reports in the training/ directory.")
        
    except ImportError as e:
        print(f"❌ Error importing training module: {e}")
        print("Make sure you're running this from the project root or have the correct path setup.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()