#!/bin/bash
#SBATCH -p main
#SBATCH -A alloc_90f5f_s2015201vult
#SBATCH -n 8
#SBATCH --time=12:00:00
#SBATCH --mem=16000
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.out
#SBATCH -J legendre_exp

WORK_DIR="/scratch/lustre/home/$(whoami)/experiment"
cd "$WORK_DIR"

source venv/bin/activate

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
echo "Working dir: $(pwd)"
echo ""

python3 05_classify_focused.py

echo ""
echo "=== Job finished at $(date) ==="
