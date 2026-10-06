"""HuggingFace file classification helpers for the admin HF proxy (Phase 12).

Ported from the legacy ``app.services.models`` (removed in the final
cleanup) and generalized beyond GGUF-only: the ``hf-file`` picker widget
(provider/README.md) feeds ``provider_lib.downloader.ensure_artifact``
descriptors for llama-cpp ``.gguf`` files, halogen ``.hgn`` checkpoints,
``.hnpw`` NPU weights, and tokenizer artifacts. The admin classifies
repo siblings so the UI can group/filter them; it never downloads.
"""

import re
from pathlib import Path
from typing import Any, Literal

ModelType = Literal["llm", "mtp", "mmproj", "dflash"]

FileKind = Literal["gguf", "hgn", "hnpw", "tokenizer", "other", "irrelevant"]

# Matches a quantization tag token such as UD-Q4_K_XL, Q8_0, IQ2_XXS, BF16, F16.
_QUANT_RE = re.compile(
    r"(?:UD-)?(?:IQ[0-9]+(?:_[A-Z]+)*|Q[2-8](?:_[A-Z0-9]+)*|BF16|F16|F32)"
)
# Strips a split-file suffix like "-00001-of-00003" from a basename.
_SPLIT_SUFFIX_RE = re.compile(r"-\d+-of-\d+$")
# Keys inside cardData.gguf entries that may carry a quantization label.
_QUANT_LABEL_KEYS = ("quantization", "quant", "type", "quant_type", "quantized")

# Repo housekeeping files that are never selectable artifacts.
_IRRELEVANT_BASENAMES = frozenset(
    {
        ".gitattributes",
        ".gitignore",
        "readme",
        "readme.md",
        "readme.rst",
        "license",
        "license.md",
        "licence.md",
        "notices.md",
        "contributing.md",
        "security.md",
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
    }
)
_IRRELEVANT_SUFFIXES = (".py", ".sh", ".png", ".jpg", ".jpeg", ".gif", ".svg")

# Tokenizer artifact names (checked on the lowercased basename).
_TOKENIZER_BASENAMES = frozenset(
    {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer.ggml.model",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
    }
)


def classify_file(rfilename: str | None) -> FileKind:
    """Classify a repo sibling by extension/path into a picker kind."""
    if not rfilename:
        return "other"
    lower = rfilename.lower()
    basename = lower.rsplit("/", 1)[-1]
    if lower.endswith(".gguf"):
        return "gguf"
    if lower.endswith(".hgn"):
        return "hgn"
    if lower.endswith(".hnpw"):
        return "hnpw"
    if lower.startswith("tokenizer/") or basename in _TOKENIZER_BASENAMES:
        return "tokenizer"
    if basename in _IRRELEVANT_BASENAMES or lower.endswith(_IRRELEVANT_SUFFIXES):
        return "irrelevant"
    return "other"


def guess_model_type(path: str | None, gguf_type: str | None = None) -> ModelType:
    """Guess the GGUF model type from the general.type and filename.

    mmproj / mtp / dflash files are auxiliary; everything else is a llm.
    """
    haystack_parts = []
    if gguf_type:
        haystack_parts.append(gguf_type.lower())
    if path:
        haystack_parts.append(Path(path).name.lower())
    haystack = " ".join(haystack_parts)

    for candidate in ("mmproj", "dflash", "mtp"):
        if candidate in haystack:
            return candidate  # type: ignore[return-value]
    return "llm"


def is_aux_type(model_type: str | None) -> bool:
    """Return True when the given model type is an auxiliary (non-llm) file."""
    return bool(model_type) and model_type != "llm"


def _label_from_quant_string(value: str) -> str | None:
    """Extract a clean quantization tag from an arbitrary string, if present."""
    match = _QUANT_RE.search(value)
    return match.group(0) if match else None


def _extract_from_card_gguf(card_gguf: Any, path: str, basename: str) -> str | None:
    """Look for an explicit quantization mapping in cardData.gguf.

    Tolerates several shapes: dict keyed by filename/path, dict keyed by
    quantization label (with a list of files), or a list of entries with
    name/filename + quantization/type fields. Returns None on any miss or
    unexpected shape (never raises).
    """
    try:
        if isinstance(card_gguf, dict):
            # Direct mapping: key is the file path or basename.
            for key in (path, basename):
                if key in card_gguf:
                    entry = card_gguf[key]
                    if isinstance(entry, str):
                        tag = _label_from_quant_string(entry)
                        if tag:
                            return tag
                    elif isinstance(entry, dict):
                        for label_key in _QUANT_LABEL_KEYS:
                            label = entry.get(label_key)
                            if isinstance(label, str):
                                tag = _label_from_quant_string(label)
                                if tag:
                                    return tag
            # Inverse mapping: key is the quantization label, value lists files.
            for label, files in card_gguf.items():
                if not isinstance(label, str):
                    continue
                candidates = files if isinstance(files, list) else [files]
                for candidate in candidates:
                    name = None
                    if isinstance(candidate, str):
                        name = candidate
                    elif isinstance(candidate, dict):
                        name = (
                            candidate.get("rfilename")
                            or candidate.get("path")
                            or (candidate.get("filename"))
                        )
                    if isinstance(name, str) and (
                        name == path or Path(name).name == basename
                    ):
                        tag = _label_from_quant_string(label)
                        if tag:
                            return tag
        elif isinstance(card_gguf, list):
            for entry in card_gguf:
                if not isinstance(entry, dict):
                    continue
                name = (
                    entry.get("rfilename") or entry.get("path") or entry.get("filename")
                )
                if isinstance(name, str) and (
                    name == path or Path(name).name == basename
                ):
                    for label_key in _QUANT_LABEL_KEYS:
                        label = entry.get(label_key)
                        if isinstance(label, str):
                            tag = _label_from_quant_string(label)
                            if tag:
                                return tag
    except Exception:
        # cardData.gguf is inconsistent across repos; never let it crash listing.
        return None
    return None


def extract_quantization(path: str | None, card_gguf: Any = None) -> str | None:
    """Determine a quantization tag for a model file path.

    Resolution order (defensive):
      1. Explicit mapping in cardData.gguf for this path/basename.
      2. Parent directory name when it looks like a quantization tag.
      3. File basename after stripping the split suffix.
      4. None (uncategorized).
    """
    if not path:
        return None

    p = Path(path)
    basename = p.name
    stem = p.stem
    parent = p.parent.name if str(p.parent) not in ("", ".") else ""

    tag = _extract_from_card_gguf(card_gguf, path, basename)
    if tag:
        return tag

    if parent:
        full = _QUANT_RE.fullmatch(parent)
        if full:
            return full.group(0)

    stripped = _SPLIT_SUFFIX_RE.sub("", stem)
    found = _QUANT_RE.search(stripped)
    if found:
        return found.group(0)

    return None
