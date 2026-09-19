FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ggerganov/llama.cpp.git \
    && cd llama.cpp \
    && cmake -B build \
    && cmake --build build --config Release \
    && cp build/bin/llama-server /usr/local/bin/

COPY pyproject.toml .

RUN pip install uv && uv sync --frozen

COPY app/ ./app/

EXPOSE 8080

CMD ["uv", "run", "python", "-m", "app.main"]
