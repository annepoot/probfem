#!/bin/bash

# Submit initial job array
INITIAL_JOB_ID=$(sbatch --array=0-14 --parsable fem.job)
CURRENT_JOB_ID=$INITIAL_JOB_ID

echo "Submitted initial job array with Job ID: $INITIAL_JOB_ID"

while true; do
    # Wait 15 minutes minutes to avoid hammering SLURM
    sleep 900

    echo "Checking job array status for Job ID: $CURRENT_JOB_ID"

    # Query sacct for task states
    STATES=$(sacct -j $CURRENT_JOB_ID --format=JobID,State --parsable2 | grep -E "^${CURRENT_JOB_ID}_[0-9]+\|")

    if [ -z "$STATES" ]; then
        echo "Job ID $CURRENT_JOB_ID not yet registered in SLURM accounting. Waiting..."
        continue
    fi

    PENDING=$(echo "$STATES" | grep -E 'PENDING')
    RUNNING=$(echo "$STATES" | grep -E 'RUNNING')
    FAILED=$(echo "$STATES" | grep -E 'FAILED|CANCELLED|NODE_FAIL|TIMEOUT')
    SUCCEEDED=$(echo "$STATES" | grep -E 'COMPLETED')

    echo "Pending:"
    echo "$PENDING"
    echo "Running:"
    echo "$RUNNING"
    echo "Failed:"
    echo "$FAILED"
    echo "Succeeded:"
    echo "$SUCCEEDED"

    if [ -n "$PENDING" ] || [ -n "$RUNNING" ]; then
        echo "Some tasks are still pending or running. Waiting..."
        continue
    fi

    if [ -n "$FAILED" ]; then
	FAILED_TASKS=$(echo "$FAILED" | cut -d_ -f2 | cut -d'|' -f1 | sort -n | uniq | paste -sd,)

        echo "Detected failed tasks: $FAILED_TASKS"

        if [ -z "$FAILED_TASKS" ]; then
            echo "No valid task IDs found. Exiting."
            break
        fi

        echo "Resubmitting failed tasks..."
        CURRENT_JOB_ID=$(sbatch --array=$FAILED_TASKS --parsable fem.job)
        echo "Resubmitted failed tasks with Job ID: $CURRENT_JOB_ID"
    else
        echo "All tasks completed successfully."
        break
    fi
done
