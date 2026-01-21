#!/bin/bash

# Enable strict error checking for safety in loop operations
set -e

echo "Starting batch submission for all 'or*' subfolders..."
echo "========================================================"

# Loop through all directories starting with "or" in the current directory
for dir in or*/; do
    # Check if it is actually a directory
    if [ -d "$dir" ]; then
        # Remove trailing slash for cleaner printing
        dirname=${dir%/}
        echo "Processing folder: $dirname"
        
        target_script="$dirname/submit_all.sh"
        
        if [ -f "$target_script" ]; then
            echo "  -> Found submit_all.sh, executing in subshell..."
            
            # Use a subshell (parentheses) to cd into directory, run script, and automatically return
            # This ensures we don't get lost in directory navigation
            (
                cd "$dirname" || exit
                bash submit_all.sh
            )
            
            echo "  -> Done with $dirname"
        else
            echo "  -> WARNING: No 'submit_all.sh' found in $dirname, skipping."
        fi
        echo "--------------------------------------------------------"
    fi
done

echo "All subfolder scripts have been executed."
