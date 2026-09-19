#!/bin/sh
# Generate runtime config from environment variables

if [ -n "$API_URL" ]; then
    echo "window.APP_CONFIG = { API_URL: '$API_URL' }" > /app/backend/app/frontend/config.js
    echo "Generated config.js with API_URL=$API_URL"
fi
