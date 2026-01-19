#!/bin/bash
# Utility script to decode master job task IDs and check progress

# Define parameters (must match master script)
VARIANTS=("random_25" "random_50" "random_75" "random_100")
SEEDS=(44 45 46 47 48 49 50 51 52)
RECOMMENDED_MODELS=(
    "Lipomyces_starkeyi"
    "Tortispora_caseinolytica"
    "Yarrowia_deformans"
    "Alloascoidea_hylecoeti"
    "Sporopachydermia_quercuum"
    "Pachysolen_tannophilus"
    "Komagataella_pastoris"
    "Debaryomyces_hansenii"
    "Saccharomycopsis_malanga"
    "Wickerhamomyces_ciferrii"
    "Hanseniaspora_vinae"
    "Torulaspora_delbrueckii"
    "Neurospora_crassa"
)

N_MODELS=${#RECOMMENDED_MODELS[@]}
N_VARIANTS=${#VARIANTS[@]}
N_SEEDS=${#SEEDS[@]}

# Function to decode task ID
decode_task_id() {
    local task_id=$((${1} - 1))  # Convert to 0-indexed
    
    local seed_idx=$((task_id / (N_VARIANTS * N_MODELS)))
    local remaining=$((task_id % (N_VARIANTS * N_MODELS)))
    local variant_idx=$((remaining / N_MODELS))
    local model_idx=$((remaining % N_MODELS))
    
    echo "Task ID: $1"
    echo "  Seed: ${SEEDS[$seed_idx]}"
    echo "  Variant: ${VARIANTS[$variant_idx]}"
    echo "  Model: ${RECOMMENDED_MODELS[$model_idx]}"
}

# Function to check progress
check_progress() {
    local job_id=$1
    
    echo "Checking progress for job $job_id..."
    echo ""
    
    for seed in "${SEEDS[@]}"; do
        echo "Seed $seed:"
        for variant in "${VARIANTS[@]}"; do
            local results_dir="/work/schaffran1/jobresults/${seed}/${variant}"
            local n_results=$(find "$results_dir" -name "kinetic_results_*_tol_10.jld2" 2>/dev/null | wc -l)
            echo "  $variant: $n_results / $N_MODELS results"
        done
    done
    
    echo ""
    echo "Total expected results: $((N_MODELS * N_VARIANTS * N_SEEDS))"
    echo "Total actual results: $(find /work/schaffran1/jobresults/*/random_* -name "kinetic_results_*_tol_10.jld2" 2>/dev/null | wc -l)"
}

# Function to list all tasks for a specific seed/variant/model
list_tasks() {
    local filter_type=$1
    local filter_value=$2
    
    case $filter_type in
        "seed")
            local seed_idx=-1
            for i in "${!SEEDS[@]}"; do
                if [ "${SEEDS[$i]}" = "$filter_value" ]; then
                    seed_idx=$i
                    break
                fi
            done
            if [ $seed_idx -eq -1 ]; then
                echo "Error: Seed $filter_value not found"
                return 1
            fi
            local start_task=$((seed_idx * N_VARIANTS * N_MODELS + 1))
            local end_task=$((start_task + N_VARIANTS * N_MODELS - 1))
            echo "Tasks for seed $filter_value: $start_task-$end_task"
            ;;
        "variant")
            local variant_idx=-1
            for i in "${!VARIANTS[@]}"; do
                if [ "${VARIANTS[$i]}" = "$filter_value" ]; then
                    variant_idx=$i
                    break
                fi
            done
            if [ $variant_idx -eq -1 ]; then
                echo "Error: Variant $filter_value not found"
                return 1
            fi
            echo "Tasks for variant $filter_value:"
            for ((seed_idx=0; seed_idx<N_SEEDS; seed_idx++)); do
                local start_task=$((seed_idx * N_VARIANTS * N_MODELS + variant_idx * N_MODELS + 1))
                local end_task=$((start_task + N_MODELS - 1))
                echo "  Seed ${SEEDS[$seed_idx]}: $start_task-$end_task"
            done
            ;;
        "model")
            local model_idx=-1
            for i in "${!RECOMMENDED_MODELS[@]}"; do
                if [ "${RECOMMENDED_MODELS[$i]}" = "$filter_value" ]; then
                    model_idx=$i
                    break
                fi
            done
            if [ $model_idx -eq -1 ]; then
                echo "Error: Model $filter_value not found"
                return 1
            fi
            echo "Tasks for model $filter_value:"
            for ((seed_idx=0; seed_idx<N_SEEDS; seed_idx++)); do
                for ((variant_idx=0; variant_idx<N_VARIANTS; variant_idx++)); do
                    local task=$((seed_idx * N_VARIANTS * N_MODELS + variant_idx * N_MODELS + model_idx + 1))
                    echo "  Seed ${SEEDS[$seed_idx]}, ${VARIANTS[$variant_idx]}: $task"
                done
            done
            ;;
        *)
            echo "Error: Unknown filter type. Use: seed, variant, or model"
            return 1
            ;;
    esac
}

# Main script
case "${1}" in
    "decode")
        if [ -z "${2}" ]; then
            echo "Usage: $0 decode <task_id>"
            exit 1
        fi
        decode_task_id "${2}"
        ;;
    "progress")
        if [ -z "${2}" ]; then
            echo "Usage: $0 progress <job_id>"
            exit 1
        fi
        check_progress "${2}"
        ;;
    "list")
        if [ -z "${2}" ] || [ -z "${3}" ]; then
            echo "Usage: $0 list <seed|variant|model> <value>"
            echo "Examples:"
            echo "  $0 list seed 44"
            echo "  $0 list variant random_50"
            echo "  $0 list model Komagataella_pastoris"
            exit 1
        fi
        list_tasks "${2}" "${3}"
        ;;
    "summary")
        echo "Master Job Configuration:"
        echo "  Models: ${N_MODELS}"
        echo "  Variants: ${N_VARIANTS}"
        echo "  Seeds: ${N_SEEDS}"
        echo "  Total tasks: $((N_MODELS * N_VARIANTS * N_SEEDS))"
        echo ""
        echo "Task ID ranges:"
        for ((seed_idx=0; seed_idx<N_SEEDS; seed_idx++)); do
            local start=$((seed_idx * N_VARIANTS * N_MODELS + 1))
            local end=$((start + N_VARIANTS * N_MODELS - 1))
            echo "  Seed ${SEEDS[$seed_idx]}: tasks $start-$end"
        done
        ;;
    *)
        echo "Usage: $0 <command> [args]"
        echo ""
        echo "Commands:"
        echo "  decode <task_id>              - Show what seed/variant/model a task ID represents"
        echo "  progress <job_id>             - Check progress of a master job"
        echo "  list <type> <value>           - List all task IDs for a seed/variant/model"
        echo "  summary                       - Show configuration summary"
        echo ""
        echo "Examples:"
        echo "  $0 decode 1"
        echo "  $0 decode 468"
        echo "  $0 progress 62693"
        echo "  $0 list seed 45"
        echo "  $0 list variant random_75"
        echo "  $0 list model Komagataella_pastoris"
        echo "  $0 summary"
        exit 1
        ;;
esac
