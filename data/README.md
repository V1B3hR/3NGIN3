# Data Directory

This directory contains data-related scripts and utilities for the 3NGIN3 project.

## Running Training

You can now easily run the 3NGIN3 training pipeline from this directory using the `run_training.py` script:

### Usage Options

1. **From project root:**
   ```bash
   python data/run_training.py
   ```

2. **From data directory:**
   ```bash
   cd data
   python run_training.py
   ```

3. **Direct execution (if executable):**
   ```bash
   ./data/run_training.py
   ```

### What it does

The `run_training.py` script:
- Automatically sets up the correct working directory and Python path
- Runs the complete 3NGIN3 training and evaluation pipeline
- Tests the engine on multiple datasets (UCI, Kaggle, and image datasets)
- Generates comprehensive training reports
- Provides clear feedback on training progress and results

### Output

After running, you'll find detailed training reports in the `training/` directory:
- `training_report_results.md` - Comprehensive training results and analysis

## Other Scripts

- `train_alzheimer.py` - PyTorch-based Alzheimer's disease classification training
- `split_train_val.py` - Utility to split data into training and validation sets
- `val.py` - Validation script

## Dependencies

Make sure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

The training pipeline gracefully handles missing optional dependencies by using synthetic data when needed.