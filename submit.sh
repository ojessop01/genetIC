#!/bin/bash
set -euo pipefail

# Default output directory (can be overridden by first argument)
OUTDIR="${1:-/cosma7/data/dp004/dc-jess1/EDGE/GenetIC/iso_512/}"

# Create the output directory if it doesn't exist
mkdir -p "$OUTDIR"

# Clean environment
module purge

# Load COSMA 2024 stack and required dependencies
module load cosma/2024
module load intel_comp/2024.2.0
module load compiler-rt/latest
module load tbb/latest
module load compiler/latest
module load mpi/latest
module load hdf5/1.14.4
module load fftw/3.3.10
module load parmetis/4.0.3
module load gsl/2.8
module load cfitsio/4.4.1
module load Healpix/3.82
module load jemalloc/5.1.0

# OpenMP configuration
export OMP_NUM_THREADS=32

# Edit paramfile to set the output directory
PARAM=/cosma/apps/durham/dc-jess1/genetIC/example/paramfile.txt
sed -i "s|^outdir .*|outdir $OUTDIR|" "$PARAM"

# Run genetIC
EXEC=/cosma/apps/durham/dc-jess1/genetIC/genetIC/genetIC
"$EXEC" "$PARAM"