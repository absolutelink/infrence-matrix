# GitHub Workflows

This directory contains GitHub Actions workflows for building and deploying Inference Matrix components.

## Workflows

- `build-and-push.yml` - Builds and pushes the main application and agent images
- `build-vulkan-recipe.yml` - Builds the Vulkan llama.cpp recipe image
- `build-base-agent.yml` - Builds the base agent image used by recipes

## Building Recipes

The Vulkan llama.cpp recipe is built as a separate Docker image to provide optimized GPU acceleration for AMD and Intel GPUs.

To build the recipe, the workflow:
1. Uses the base agent image as a foundation
2. Installs Vulkan dependencies
3. Compiles llama.cpp with Vulkan support
4. Builds the necessary binaries
5. Packages everything into a Docker image

The resulting image can be used with the agent service for GPU-accelerated inference.