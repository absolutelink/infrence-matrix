"""Shared knowledge of the halogen-flash NPU small models.

The engine only recognises the upstream model ids in a request's ``model``
field. The broker exposes each enabled NPU model as ``<instance-alias>-<suffix>``
so names are scoped per server instance and never collide across instances.
``engine_options["npu_models"]`` stores the upstream ids; the suffix map turns
them into client-facing aliases and back.
"""

# upstream model id -> client-facing suffix
NPU_SUFFIXES: dict[str, str] = {
    "qwen3-embedding-0.6b": "embed",
    "qwen3-reranker-0.6b": "rerank",
    "qwen3.5-2b": "nano",
    "decider-0.8b": "decide",
    "qwen3guard-gen-0.6b": "guard",
}

# suffix -> upstream model id (inverse of NPU_SUFFIXES)
UPSTREAM_BY_SUFFIX: dict[str, str] = {v: k for k, v in NPU_SUFFIXES.items()}

STOCK_NPU_MODEL_IDS: tuple[str, ...] = tuple(NPU_SUFFIXES)

# upstream model id -> capability key used by broker routes
NPU_CAPABILITIES: dict[str, str] = {
    "qwen3-embedding-0.6b": "embeddings",
    "qwen3-reranker-0.6b": "rerank",
    "qwen3.5-2b": "chat",
    "decider-0.8b": "decisions",
    "qwen3guard-gen-0.6b": "moderations",
}


def client_name(alias: str, upstream_id: str) -> str:
    """Return the ``<alias>-<suffix>`` public name for an enabled NPU model."""
    suffix = NPU_SUFFIXES[upstream_id]
    return f"{alias}-{suffix}"


def split_client_name(alias: str, model_ref: str) -> str | None:
    """If ``model_ref`` is ``<alias>-<suffix>`` for a known suffix, return the
    upstream model id, else ``None``."""
    prefix = f"{alias}-"
    if not model_ref.startswith(prefix):
        return None
    suffix = model_ref[len(prefix) :]
    return UPSTREAM_BY_SUFFIX.get(suffix)


class NpuPinRecord:
    """One NPU model's download record parsed from the image's pins file."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.repo: str | None = None
        self.revision: str | None = None
        self.devices_of: str = model_id
        # list of (relative_path, size, sha256)
        self.files: list[tuple[str, int, str]] = []

    @property
    def devices_files(self) -> list[tuple[str, int, str]]:
        return [f for f in self.files if f[0].startswith("devices/")]

    @property
    def own_files(self) -> list[tuple[str, int, str]]:
        return [f for f in self.files if not f[0].startswith("devices/")]


def parse_npu_pins(text: str) -> dict[str, NpuPinRecord]:
    """Parse ``/opt/halogen/npu/models.txt``.

    Mirrors the awk readers in the engine's entrypoint:

    - ``model <id> repo=<r> revision=<rev> devices=<other> ...``
    - ``file <id> <path> <size> <sha256>``
    """
    records: dict[str, NpuPinRecord] = {}

    def record(mid: str) -> NpuPinRecord:
        return records.setdefault(mid, NpuPinRecord(mid))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "model" and len(parts) >= 2:
            mid = parts[1]
            rec = record(mid)
            for token in parts[2:]:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                if key == "repo":
                    rec.repo = value
                elif key == "revision":
                    rec.revision = value
                elif key == "devices":
                    rec.devices_of = value
        elif kind == "file" and len(parts) >= 5:
            mid, path, size, sha = parts[1], parts[2], parts[3], parts[4]
            try:
                size_int = int(size)
            except ValueError:
                continue
            record(mid).files.append((path, size_int, sha))
    return records


def resolve_npu_download_set(
    records: dict[str, NpuPinRecord], upstream_ids: list[str]
) -> dict[str, tuple[str, str | None, list[tuple[str, int, str]]]]:
    """Map each target ``MODELS_PATH/npu/<id>/`` directory to its download spec.

    A model that runs on another's ``devices/`` program contributes only its
    own files; the shared ``devices/`` files come from the devices-owner's
    directory. Returns ``{dir_id: (repo_id, revision, [(path, size, sha256),
    ...])}``.
    """
    plan_files: dict[str, list[tuple[str, int, str]]] = {}
    plan_source: dict[str, tuple[str, str | None]] = {}
    seen: dict[str, set[str]] = {}

    def add(rec: NpuPinRecord, files: list[tuple[str, int, str]]) -> None:
        bucket = plan_files.setdefault(rec.model_id, [])
        plan_source.setdefault(rec.model_id, (rec.repo, rec.revision))
        paths = seen.setdefault(rec.model_id, set())
        for entry in files:
            if entry[0] in paths:
                continue
            paths.add(entry[0])
            bucket.append(entry)

    for uid in upstream_ids:
        rec = records.get(uid)
        if rec is None:
            raise ValueError(f"no NPU pin record for '{uid}'")
        if not rec.repo:
            raise ValueError(f"NPU pin record for '{uid}' names no download repo")
        # Every file this model's record lists, including its own devices/
        # program when it owns one.
        add(rec, rec.files)
        owner = rec.devices_of
        if owner != uid:
            # It runs on another model's devices/ program: fetch that too.
            owner_rec = records.get(owner)
            if owner_rec is None:
                raise ValueError(f"no NPU pin record for devices owner '{owner}'")
            if not owner_rec.repo:
                raise ValueError(f"NPU pin record for '{owner}' names no download repo")
            add(owner_rec, owner_rec.devices_files)
    return {
        dir_id: (
            plan_source[dir_id][0],
            plan_source[dir_id][1],
            files,
        )
        for dir_id, files in plan_files.items()
    }


def load_npu_pins(pins_file: str) -> dict[str, NpuPinRecord]:
    """Read and parse the image's NPU pins file. Raises OSError if unreadable."""
    with open(pins_file) as fh:
        return parse_npu_pins(fh.read())
