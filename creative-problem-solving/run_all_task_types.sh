#!/bin/bash

# List of all task types defined in the Python script
TASK_TYPES=(
    "creative"
    "nominal"
    "creative-obj"
    "creative-task"
    "creative-task-obj"
    "creative-chain"
    "nominal-chain"
    "creative-obj-chain"
    "creative-task-chain"
    "creative-task-obj-chain"
)

echo "Starting evaluations for all task types..."
echo "=========================================="

# Loop through the array for each task type
for TASK in "${TASK_TYPES[@]}"; do
    
    
    if [[ "$TASK" == *-chain* ]]; then
        VISION_FLAGS=("" "--bypass-vision")
        DYN_FLAGS=("" "--dynamic-descriptions")
    else
        VISION_FLAGS=("")
        DYN_FLAGS=("")
    fi

    # Loop through combinations
    for DYN_FLAG in "${DYN_FLAGS[@]}"; do
        for VIS_FLAG in "${VISION_FLAGS[@]}"; do
            
            DYN_STATUS=${DYN_FLAG:-"None"}
            VIS_STATUS=${VIS_FLAG:-"None"}
            
            echo "▶ Running task: $TASK"
            echo "  Flags -> Dynamic: $DYN_STATUS | Vision: $VIS_STATUS"
            
            # build the command
            CMD=(python eval_task.py --task-type "$TASK" --save-reasoning)
            
            if [[ -n "$DYN_FLAG" ]]; then
                CMD+=("$DYN_FLAG")
            fi
            
            if [[ -n "$VIS_FLAG" ]]; then
                CMD+=("$VIS_FLAG")
            fi
            
            # Execute the command
            "${CMD[@]}"
            
            # Check if the command was successful
            if [ $? -eq 0 ]; then
                echo "✓ Successfully completed"
            else
                echo "✗ Error running command"
            fi
            echo "------------------------------------------"
            
        done
    done
done

echo "All evaluations finished!"