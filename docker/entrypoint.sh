#!/bin/sh
# Entrypoint script - executes scripts from /etc/entrypoint.d in numeric order
# then executes the CMD passed in from the Dockerfile.
#
# Naming convention:
# - 0XX = delivered with the application (010, 020, 030, etc.)
# - 1XX+ = user-added custom scripts (100, 110, 120, etc.)

set -e

###############################################
# Copy optional rootfs overlay
###############################################
if [ -d "/opt/rootfs" ] && [ "$(ls -A /opt/rootfs)" ]; then
    echo "Copying /opt/rootfs to root filesystem..."
    cp -rf /opt/rootfs/* /
    echo "Copy done!"
else
    echo "/opt/rootfs doesn't exist or is empty. Skipping overlay copy."
fi

###############################################
# Ensure entrypoint directories exist
###############################################
mkdir -p /etc/entrypoint.d
mkdir -p /opt/rootfs

###############################################
# Execute scripts from /etc/entrypoint.d/ in numeric order
###############################################
echo "Running entrypoint scripts..."

find /etc/entrypoint.d/ -type f -name '*.sh' | sort -V | while IFS= read -r f; do
    if [ -e "$f" ]; then
        echo "Executing $f"
        if ! . "$f"; then
            echo "Error executing $f" 
            exit 1
        fi
    else
        echo "Warning: $f not found"
    fi
done

echo "Entrypoint scripts complete."

###############################################
# Execute the CMD passed in from the Dockerfile
###############################################
# first arg is `-f` or `--some-option`
if [ "${1#-}" != "$1" ]; then
    set -- "$@"
fi

# Some scripts may need to change the CMD based on runtime conditions.
# If this file is set, execute the contents of that file instead of the Dockerfile CMD.
if [ -f /tmp/docker_cmd_override ]; then
    echo "CMD override detected, executing: $(cat /tmp/docker_cmd_override)"
    docker_cmd_override=$(cat /tmp/docker_cmd_override)
    rm /tmp/docker_cmd_override
    set -- $docker_cmd_override
    exec "$@"
else
    # Execute the CMD passed in from the Dockerfile
    exec "$@"
fi
