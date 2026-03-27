#!/bin/bash
# Run this FROM YOUR LAPTOP to upload files to HPC
# Usage: bash hpc_upload.sh YOUR_USERNAME
#
# Example: bash hpc_upload.sh erikak

USERNAME="${1:?Usage: bash hpc_upload.sh YOUR_MIF_USERNAME}"
HPC_HOST="hpc.mif.vu.lt"
REMOTE_DIR="/scratch/lustre/home/$USERNAME/experiment"

echo "=== Uploading to $USERNAME@$HPC_HOST:$REMOTE_DIR ==="

# Create remote directories
ssh "$USERNAME@$HPC_HOST" "mkdir -p $REMOTE_DIR/data $REMOTE_DIR/results $REMOTE_DIR/models_focused"

# Upload script
scp 05_classify_focused.py "$USERNAME@$HPC_HOST:$REMOTE_DIR/"
scp hpc_run.sh "$USERNAME@$HPC_HOST:$REMOTE_DIR/"

# Upload data (Legendre + OG baseline + splits)
scp data/legendre_*_L2.csv "$USERNAME@$HPC_HOST:$REMOTE_DIR/data/"
scp data/og_xp.csv "$USERNAME@$HPC_HOST:$REMOTE_DIR/data/"
scp data/splits_rskf.json "$USERNAME@$HPC_HOST:$REMOTE_DIR/data/"

# Upload existing results so resume skips chebyshev + og_xp
scp results/focused_experiment_results.csv "$USERNAME@$HPC_HOST:$REMOTE_DIR/results/"

echo ""
echo "=== Upload complete ==="
echo "Now SSH in and run:"
echo "  ssh $USERNAME@$HPC_HOST"
echo "  cd $REMOTE_DIR"
echo "  sbatch hpc_run.sh"
