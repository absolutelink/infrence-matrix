import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from sqlmodel import Session, select

from app.core.config import settings
from app.models import PromptCache

logger = logging.getLogger(__name__)


class PromptCacheManager:
    """Manages prompt caching with hybrid tracking."""

    def __init__(self) -> None:
        self.cache_dir = Path(settings.CACHE_PATH)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_cache_key(
        self,
        model_id: UUID,
        conversation_id: UUID | None = None,
        cache_type: Literal["conversation", "hierarchical"] = "conversation",
    ) -> str:
        """Generate a unique cache key."""
        if cache_type == "conversation" and conversation_id:
            key_data = f"conv:{conversation_id}:{model_id}"
        else:
            key_data = f"hierarchical:{model_id}"

        return hashlib.sha256(key_data.encode()).hexdigest()

    def compute_content_hash(self, content: str | bytes) -> str:
        """Compute SHA256 hash of content."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def create_cache_entry(
        self,
        session: Session,
        model_id: UUID,
        cache_key: str,
        cache_type: Literal["conversation", "hierarchical"],
        content_hash: str,
        token_count: int,
        ttl_seconds: int = 3600,
        conversation_id: UUID | None = None,
        llama_cache_id: str | None = None,
    ) -> PromptCache:
        """Create a new cache entry in the database."""
        now = datetime.now(UTC)
        cache_path = str(self.cache_dir / f"{cache_key}.cache")

        cache_entry = PromptCache(
            cache_key=cache_key,
            cache_type=cache_type,
            model_id=model_id,
            content_hash=content_hash,
            llama_cache_id=llama_cache_id,
            hits=0,
            size_bytes=0,
            token_count=token_count,
            ttl_seconds=ttl_seconds,
            created_at=now,
            expires_at=now.replace(second=now.second + ttl_seconds),
            last_accessed_at=now,
            conversation_id=conversation_id,
            cache_path=cache_path,
        )

        session.add(cache_entry)
        session.commit()
        session.refresh(cache_entry)

        logger.info(f"Created cache entry {cache_key} for model {model_id}")
        return cache_entry

    def get_cache_entry(
        self,
        session: Session,
        cache_key: str,
    ) -> PromptCache | None:
        """Retrieve a cache entry by key."""
        statement = select(PromptCache).where(PromptCache.cache_key == cache_key)
        cache_entry = session.exec(statement).first()

        if cache_entry:
            now = datetime.now(UTC)
            if cache_entry.expires_at > now:
                cache_entry.hits += 1
                cache_entry.last_accessed_at = now
                session.add(cache_entry)
                session.commit()
                logger.debug(f"Cache hit for {cache_key}")
                return cache_entry
            else:
                logger.debug(f"Cache entry {cache_key} expired")
                self.delete_cache_entry(session, cache_entry)
                return None

        return None

    def delete_cache_entry(
        self,
        session: Session,
        cache_entry: PromptCache,
    ) -> bool:
        """Delete a cache entry and its file."""
        try:
            cache_file = Path(cache_entry.cache_path)
            if cache_file.exists():
                cache_file.unlink()

            session.delete(cache_entry)
            session.commit()
            logger.info(f"Deleted cache entry {cache_entry.cache_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete cache entry: {e}")
            session.rollback()
            return False

    def cleanup_expired(self, session: Session) -> int:
        """Remove expired cache entries."""
        now = datetime.now(UTC)
        statement = select(PromptCache).where(PromptCache.expires_at < now)
        expired_entries = session.exec(statement).all()

        deleted_count = 0
        for entry in expired_entries:
            if self.delete_cache_entry(session, entry):
                deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} expired cache entries")
        return deleted_count

    def update_cache_size(
        self,
        session: Session,
        cache_entry: PromptCache,
        size_bytes: int,
    ) -> None:
        """Update the size of a cache entry."""
        cache_entry.size_bytes = size_bytes
        session.add(cache_entry)
        session.commit()
        logger.debug(f"Updated cache size for {cache_entry.cache_key}: {size_bytes} bytes")

    def get_cache_stats(self, session: Session) -> dict[str, int | float]:
        """Get cache statistics."""
        statement = select(PromptCache)
        all_entries = session.exec(statement).all()

        now = datetime.now(UTC)
        active_entries = [e for e in all_entries if e.expires_at > now]
        expired_entries = [e for e in all_entries if e.expires_at <= now]

        total_size = sum(e.size_bytes for e in active_entries)
        total_hits = sum(e.hits for e in active_entries)
        total_tokens = sum(e.token_count for e in active_entries)

        return {
            "total_entries": len(all_entries),
            "active_entries": len(active_entries),
            "expired_entries": len(expired_entries),
            "total_size_bytes": total_size,
            "total_hits": total_hits,
            "total_tokens": total_tokens,
            "hit_rate": total_hits / len(active_entries) if active_entries else 0.0,
        }

    def get_conversation_cache(
        self,
        session: Session,
        conversation_id: UUID,
    ) -> PromptCache | None:
        """Get cache entry for a specific conversation."""
        statement = select(PromptCache).where(
            PromptCache.conversation_id == conversation_id,
            PromptCache.cache_type == "conversation",
        )
        return session.exec(statement).first()

    def invalidate_conversation_cache(
        self,
        session: Session,
        conversation_id: UUID,
    ) -> int:
        """Invalidate all cache entries for a conversation."""
        statement = select(PromptCache).where(
            PromptCache.conversation_id == conversation_id
        )
        entries = session.exec(statement).all()

        deleted_count = 0
        for entry in entries:
            if self.delete_cache_entry(session, entry):
                deleted_count += 1

        logger.info(f"Invalidated {deleted_count} cache entries for conversation {conversation_id}")
        return deleted_count


prompt_cache_manager = PromptCacheManager()
