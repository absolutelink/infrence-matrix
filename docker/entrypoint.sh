#!/bin/sh
# Entrypoint script - runs all scripts in /etc/entrypoint.d/ in order

set -e

echo "Running entrypoint scripts..."

if [ -d "/etc/entrypoint.d" ]; then
    # Run all .sh scripts in alphabetical order
    for script in $(ls /etc/entrypoint.d/*.sh 2>/dev/null | sort); do
        echo ">>> Running: $(basename $script)"
        . "$script"
    done
fi

echo "Entrypoint scripts complete."

# Execute the main CMD
exec "$@"
