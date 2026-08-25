#!/bin/bash
# =====================================================================================
# Submit the robustness grid as a SLURM job ARRAY (one config per task) so all configs
# run in parallel; wall-clock ~= a single run. A dependent merge job concatenates the
# per-task CSVs into <MODELTAG>_all_runs.csv.
#
# Run from the repo root (or set REPRO_DIR). Required: MODEL, OUTDIR.
#   MODEL=path/to/iJR904.xml OUTDIR=results \
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
QOS="${QOS:-}"            # optional SLURM QOS, e.g. "long" for walltime > 2 days
PARTITION="${PARTITION:-}" # optional partition override
LOGDIR="${LOGDIR:-$(dirname "$OUTDIR")/slurm_logs}"
mkdir -p "$LOGDIR" "$OUTDIR"
MAIL="${MAIL:-your-email@example.com}"  # override via: MAIL=you@example.com

echo "Instantiating environment ..."
julia --project="$REPRO_DIR" -e 'import Pkg; Pkg.instantiate()'

N=$(MODEL="$MODEL" SEEDS="$SEEDS" EXPERIMENTS="$EXPERIMENTS" COUNT_ONLY=1 \
    julia --project="$REPRO_DIR" "$REPRO_DIR/robustness_experiments.jl")
ARRAYSPEC="1-$N"; [ -n "$THROTTLE" ] && ARRAYSPEC="1-$N%$THROTTLE"
echo "MODELTAG=$MODELTAG  configs=$N  array=$ARRAYSPEC  cpus=$CPUS mem=$MEM time=$TIME"

# NOTE: sbatch --export treats commas as variable delimiters, which corrupts
# comma-containing values like SEEDS="1234,42,..." and EXPERIMENTS="A,B,C,D".
# So we export everything into the environment and pass a bare --export=ALL instead
# of an inline VAR=val list.
export MODEL MODELTAG OUTDIR SEEDS EXPERIMENTS
export NPROCS=$((CPUS - 1))

# Optional QOS / partition (e.g. QOS=long for walltime beyond the default 2-day cap).
QOS_ARG=(); [ -n "$QOS" ] && QOS_ARG=(--qos="$QOS")
PART_ARG=(); [ -n "$PARTITION" ] && PART_ARG=(--partition="$PARTITION")

AID=$(sbatch --parsable \
  --job-name="cocoa-${MODELTAG}" \
  --output="$LOGDIR/%x-%A_%a.out" --error="$LOGDIR/%x-%A_%a.err" \
  --time="$TIME" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM" \
  --array="$ARRAYSPEC" "${QOS_ARG[@]}" "${PART_ARG[@]}" \
  --mail-type=END,FAIL --mail-user="$MAIL" \
  --export=ALL \
  --wrap="julia --project='$REPRO_DIR' '$REPRO_DIR/robustness_experiments.jl'")
echo "array job: $AID  (tasks 1-$N)"

sbatch --parsable --dependency=afterany:"$AID" \
  --job-name="cocoa-${MODELTAG}-merge" \
  --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
  --time=00:15:00 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
  --mail-type=END,FAIL --mail-user="$MAIL" \
  --wrap="awk 'FNR==1 && NR!=1 {next} {print}' $OUTDIR/${MODELTAG}_task_*.csv > $OUTDIR/${MODELTAG}_all_runs.csv && echo merged \$(wc -l < $OUTDIR/${MODELTAG}_all_runs.csv) lines"
echo "merge job submitted (afterany)"
