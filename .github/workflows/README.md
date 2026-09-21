# GitHub Workflows

This directory contains GitHub Actions workflows for building and deploying Inference Matrix components.

## Workflows

- `build-and-push.yml` - Builds and pushes the main application and agent images, and recipes

## Building Recipes

The Vulkan llama.cpp recipe is built as a separate Docker image to provide optimized GPU acceleration for AMD and Intel GPUs.

The workflow is structured to:
1. First build and push the base agent image
2. Then build the Vulkan recipe using the built agent image as a base

This ensures that the recipe uses the latest agent image and incorporates all the latest changes.