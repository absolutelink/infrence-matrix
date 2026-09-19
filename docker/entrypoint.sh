#!/bin/sh
# Entrypoint script - runs all scripts in /etc/entrypoint.d/ in order
# 
# Naming convention:
# - 0XX = delivered with the application (010, 020, 030, etc.)
# - 1XX+ = user-added custom scripts (100, 110, 120, etc.)
#
# Scripts execute in alphabetical order

set -e

echo "Running entrypoint scripts..."

if [ -d "/etc/entrypoint.d" ]; then
    # Run all .sh scripts in alphabetical order
    for script in $(ls /etc/entrypoint.d/*.sh 2>/dev/null | sort); do
        echo ">>> Running: $(basename $script)"
        . "$script"
    done
else
    echo "No entrypoint scripts found in /etc/entrypoint.d"
fi

echo "Entrypoint scripts complete."

# Execute the main CMD
exec "$@"
