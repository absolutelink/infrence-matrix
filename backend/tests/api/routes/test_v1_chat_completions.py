"""Tests for V1 Chat Completions API endpoint."""


from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import Model


class TestCreateChatCompletion:
    """Test POST /v1/chat/completions endpoint."""

    def test_chat_completion_non_streaming(self, client: TestClient, db: Session) -> None:
        """Test non-streaming chat completion."""
        model = Model(
            name="test-model.Q4_K_M.gguf",
            path="/models/test-model.Q4_K_M.gguf",
            size_bytes=1234567890,
            architecture="llama",
            parameter_count=3000000000,
            quantization="Q4_K_M",
            supports_embeddings=False,
            supports_vision=False,
            context_length=4096,
            source="huggingface",
        )
        db.add(model)
        db.commit()

        payload = {
            "model": "test-model.Q4_K_M.gguf",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 100,
        }

        response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "id" in data
            assert data["object"] == "chat.completion"
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert "message" in data["choices"][0]
            assert "usage" in data

    def test_chat_completion_streaming(self, client: TestClient, db: Session) -> None:
        """Test streaming chat completion."""
        model = Model(
            name="test-model.Q4_K_M.gguf",
            path="/models/test-model.Q4_K_M.gguf",
            size_bytes=1234567890,
            architecture="llama",
            parameter_count=3000000000,
            quantization="Q4_K_M",
            supports_embeddings=False,
            supports_vision=False,
            context_length=4096,
            source="huggingface",
        )
        db.add(model)
        db.commit()

        payload = {
            "model": "test-model.Q4_K_M.gguf",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "stream": True,
            "temperature": 0.7,
        }

        response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

            content = response.text
            lines = content.strip().split("\n\n")

            assert len(lines) > 0

            first_line = lines[0]
            assert first_line.startswith("data: ")

            has_done = False
            for line in lines:
                if "[DONE]" in line:
                    has_done = True
                    break

            assert has_done, "Streaming response should end with [DONE]"

    def test_chat_completion_model_not_found(self, client: TestClient, db: Session) -> None:
        """Test chat completion with non-existent model."""
        payload = {
            "model": "nonexistent-model.gguf",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 404

    def test_chat_completion_with_parameters(self, client: TestClient, db: Session) -> None:
        """Test chat completion with various parameters."""
        model = Model(
            name="test-model.Q4_K_M.gguf",
            path="/models/test-model.Q4_K_M.gguf",
            size_bytes=1234567890,
            architecture="llama",
            parameter_count=3000000000,
            quantization="Q4_K_M",
            supports_embeddings=False,
            supports_vision=False,
            context_length=4096,
            source="huggingface",
        )
        db.add(model)
        db.commit()

        payload = {
            "model": "test-model.Q4_K_M.gguf",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi there!"}
            ],
            "stream": False,
            "temperature": 0.5,
            "max_tokens": 50,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stop": ["END"],
        }

        response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code in [200, 503]

    def test_chat_completion_multiple_messages(self, client: TestClient, db: Session) -> None:
        """Test chat completion with conversation history."""
        model = Model(
            name="test-model.Q4_K_M.gguf",
            path="/models/test-model.Q4_K_M.gguf",
            size_bytes=1234567890,
            architecture="llama",
            parameter_count=3000000000,
            quantization="Q4_K_M",
            supports_embeddings=False,
            supports_vision=False,
            context_length=4096,
            source="huggingface",
        )
        db.add(model)
        db.commit()

        payload = {
            "model": "test-model.Q4_K_M.gguf",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "Tell me a joke."}
            ],
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=payload)

        assert response.status_code in [200, 503]
