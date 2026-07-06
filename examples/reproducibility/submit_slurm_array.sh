#!/bin/bash
# =====================================================================================
# Submit the robustness grid as a SLURM job ARRAY (one config per task) so all configs
# run in parallel; wall-clock ~= a single run. A dependent merge job concatenates the
# per-task CSVs into <MODELTAG>_all_runs.csv.
#
# Run from the repo root (or set REPRO_DIR). Required: MODEL, OUTDIR.
#   MODEL=/path/iJR904.xml OUTDIR=/work/schaffran1/COCOA_revision/results \
#     CPUS=16 MEM=64G TIME=02:00:00 bash examples/reproducibility/submit_slurm_array.sh
# Optional: MODELTAG, SEEDS, EXPERIMENTS, LOGDIR.
# =====================================================================================
set -euo pipefail

REPRO_DIR="${REPRO_DIR:-$(pwd)/examples/reproducibility}"
[ -f "$REPRO_DIR/robustness_experiments.jl" ] || { echo "Set REPRO_DIR to examples/reproducibility"; exit 1; }
: "${MODEL:?set MODEL}"
: "${OUTDIR:?set OUTDIR}"

DEFAULT_SEEDS="1234,42,7,101,202,303,404,505,606,707"
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"
EXPERIMENTS="${EXPERIMENTS:-A,B,C,D}"
MODELTAG="${MODELTAG:-$(basename "${MODEL%.*}")}"
CPUS="${CPUS:-16}"; MEM="${MEM:-64G}"; TIME="${TIME:-06:00:00}"
THROTTLE="${THROTTLE:-}"   # max concurrent array tasks, e.g. 20; empty = unlimited
LOGDIR="${LOGDIR:-$(dirname "$OUTDIR")/slurm_logs}"
mkdir -p "$LOGDIR" "$OUTDIR"
MAIL="schaffran1@uni-potsdam.de"

echo "Instantiating environment ..."
julia --project="$REPRO_DIR" -e 'import Pkg; Pkg.instantiate()'

N=$(MODEL="$MODEL" SEEDS="$SEEDS" EXPERIMENTS="$EXPERIMENTS" COUNT_ONLY=1 \
    julia --project="$REPRO_DIR" "$REPRO_DIR/robustness_experiments.jl")
ARRAYSPEC="1-$N"; [ -n "$THROTTLE" ] && ARRAYSPEC="1-$N%$THROTTLE"
echo "MODELTAG=$MODELTAG  configs=$N  array=$ARRAYSPEC  cpus=$CPUS mem=$MEM time=$TIME"

AID=$(sbatch --parsable \
  --job-name="cocoa-${MODELTAG}" \
  --output="$LOGDIR/%x-%A_%a.out" --error="$LOGDIR/%x-%A_%a.err" \
  --time="$TIME" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM" \
  --array="$ARRAYSPEC" \
  --mail-type=END,FAIL --mail-user="$MAIL" \
  --export=ALL,MODEL="$MODEL",MODELTAG="$MODELTAG",OUTDIR="$OUTDIR",NPROCS=$((CPUS-1)),SEEDS="$SEEDS",EXPERIMENTS="$EXPERIMENTS" \
  --wrap="julia --project='$REPRO_DIR' '$REPRO_DIR/robustness_experiments.jl'")
echo "array job: $AID  (tasks 1-$N)"

sbatch --parsable --dependency=afterany:"$AID" \
  --job-name="cocoa-${MODELTAG}-merge" \
  --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
  --time=00:15:00 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
  --mail-type=END,FAIL --mail-user="$MAIL" \
  --wrap="awk 'FNR==1 && NR!=1 {next} {print}' $OUTDIR/${MODELTAG}_task_*.csv > $OUTDIR/${MODELTAG}_all_runs.csv && echo merged \$(wc -l < $OUTDIR/${MODELTAG}_all_runs.csv) lines"
echo "merge job submitted (afterany)"
