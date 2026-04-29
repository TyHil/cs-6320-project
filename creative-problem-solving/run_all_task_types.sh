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

# Loop through the array and execute the python script for each task type
for TASK in "${TASK_TYPES[@]}"; do
    echo "▶ Running task: $TASK"
    python eval_task.py --task-type "$TASK" --save-reasoning
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $TASK"
    else
        echo "✗ Error running: $TASK"
    fi
    echo "------------------------------------------"
done

echo "All evaluations finished!"