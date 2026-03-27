#!/bin/bash
#SBATCH -p main
#SBATCH -n 8
#SBATCH --time=12:00:00
#SBATCH --mem=16000
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.out
#SBATCH -J dense_exp

WORK_DIR="/scratch/lustre/home/$(whoami)/experiment_dense"
cd "$WORK_DIR"

source venv/bin/activate

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
echo "Working dir: $(pwd)"
echo ""

python3 09_classify_dense.py \
  --only \
  chebyshev_1_L2 chebyshev_2_L2 chebyshev_3_L2 chebyshev_4_L2 chebyshev_5_L2 \
  chebyshev_6_L2 chebyshev_7_L2 chebyshev_8_L2 chebyshev_9_L2 chebyshev_10_L2 \
  chebyshev_11_L2 chebyshev_12_L2 chebyshev_13_L2 chebyshev_14_L2 chebyshev_15_L2 \
  chebyshev_16_L2 chebyshev_17_L2 chebyshev_18_L2 chebyshev_19_L2 chebyshev_20_L2 \
  chebyshev_21_L2 chebyshev_22_L2 chebyshev_23_L2 chebyshev_24_L2 chebyshev_25_L2 \
  legendre_1_L2 legendre_2_L2 legendre_3_L2 legendre_4_L2 legendre_5_L2 \
  legendre_6_L2 legendre_7_L2 legendre_8_L2 legendre_9_L2 legendre_10_L2 \
  legendre_11_L2 legendre_12_L2 legendre_13_L2 legendre_14_L2 legendre_15_L2 \
  legendre_16_L2 legendre_17_L2 legendre_18_L2 legendre_19_L2 legendre_20_L2 \
  legendre_21_L2 legendre_22_L2 legendre_23_L2 legendre_24_L2 legendre_25_L2

echo ""
echo "=== Job finished at $(date) ==="
