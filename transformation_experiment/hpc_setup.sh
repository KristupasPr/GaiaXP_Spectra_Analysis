#!/bin/bash
# Run this ONCE on HPC to set up the Python environment
# Usage: bash hpc_setup.sh

set -e

echo "=== Setting up Python environment on HPC ==="

# Create a virtual environment in scratch
WORK_DIR="/scratch/lustre/home/$(whoami)/experiment"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install numpy pandas scikit-learn xgboost optuna joblib scipy

echo ""
echo "=== Verifying imports ==="
python3 -c "
import numpy, pandas, sklearn, xgboost, optuna, scipy
print('numpy:', numpy.__version__)
print('pandas:', pandas.__version__)
print('sklearn:', sklearn.__version__)
print('xgboost:', xgboost.__version__)
print('optuna:', optuna.__version__)
print('All OK!')
"

echo ""
echo "=== Setup complete ==="
echo "Work directory: $WORK_DIR"
echo "Now upload your data and script to: $WORK_DIR"
