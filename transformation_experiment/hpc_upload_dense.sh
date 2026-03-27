#!/bin/bash
# Run this FROM YOUR LAPTOP to upload files to HPC
# Usage: bash hpc_upload_dense.sh YOUR_USERNAME
#
# Example: bash hpc_upload_dense.sh erikak

USERNAME="${1:?Usage: bash hpc_upload_dense.sh YOUR_MIF_USERNAME}"
HPC_HOST="hpc.mif.vu.lt"
REMOTE_DIR="/scratch/lustre/home/$USERNAME/experiment_dense"

echo "=== Uploading to $USERNAME@$HPC_HOST:$REMOTE_DIR ==="

# Create remote directories
ssh "$USERNAME@$HPC_HOST" "mkdir -p $REMOTE_DIR/data $REMOTE_DIR/results $REMOTE_DIR/models_dense"

# Upload scripts
scp 09_classify_dense.py "$USERNAME@$HPC_HOST:$REMOTE_DIR/"
scp hpc_run_dense.sh "$USERNAME@$HPC_HOST:$REMOTE_DIR/"

# Upload data: Chebyshev + Legendre L2 (n=1..25) + OG baseline + splits
for n in $(seq 1 25); do
    scp "data/chebyshev_${n}_L2.csv" "$USERNAME@$HPC_HOST:$REMOTE_DIR/data/"
    scp "data/legendre_${n}_L2.csv" "$USERNAME@$HPC_HOST:$REMOTE_DIR/data/"
done
scp data/og_xp.csv "$USERNAME@$HPC_HOST:$REMOTE_DIR/data/"
scp data/splits_rskf.json "$USERNAME@$HPC_HOST:$REMOTE_DIR/data/"

# Optionally upload existing results for resume
if [ -f results/dense_experiment_results.csv ]; then
    scp results/dense_experiment_results.csv "$USERNAME@$HPC_HOST:$REMOTE_DIR/results/"
    echo "Uploaded existing results for resume."
fi

echo ""
echo "=== Upload complete ==="
echo "Now SSH in and run:"
echo "  ssh $USERNAME@$HPC_HOST"
echo "  cd $REMOTE_DIR"
echo "  sbatch hpc_run_dense.sh"
