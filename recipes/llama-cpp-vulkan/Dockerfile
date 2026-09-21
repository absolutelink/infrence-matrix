# Build Inference Matrix Agent with llama.cpp Vulkan support
FROM ghcr.io/absolutelink/agent:main

USER root

# Install Vulkan dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    vulkan-tools \
    libvulkan1 \
    libvulkan-dev \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp with Vulkan
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp && \
    cd /tmp/llama.cpp && \
    cmake -B build \
        -DLLAMA_VULKAN=ON \
        -DLLAMA_CUDA=OFF \
        -DLLAMA_METAL=OFF \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc) && \
    cp build/bin/llama-server /usr/local/bin/llama-server && \
    cp build/bin/llama-cli /usr/local/bin/llama-cli && \
    cp build/bin/llama-bench /usr/local/bin/llama-bench && \
    rm -rf /tmp/llama.cpp

# Verify installation
RUN llama-server --version || true

# Switch back to non-root user
USER appuser

# Set Vulkan as default GPU backend
ENV GPU_BACKEND=vulkan

EXPOSE 8080

CMD ["uv", "run", "python", "-m", "app.main"]
