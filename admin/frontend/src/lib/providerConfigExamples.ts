import type { ProviderType } from "@/types/admin"

// backend_config examples per provider type, mirroring provider/README.md.
// Loaded into the JSON editor via the "Load example" button; the admin only
// validates JSON-serializability — deep validation is provider-side.
export const BACKEND_CONFIG_EXAMPLES: Record<
  ProviderType,
  { hint: string; example: Record<string, unknown> }
> = {
  mock: {
    hint: "Mock provider stream knobs: model (served name), delta_count (tokens streamed), delta_delay (s between deltas).",
    example: {
      model: "mock-model",
      delta_count: 5,
      delta_delay: 0.05,
    },
  },
  "llama-cpp": {
    hint: "main GGUF + optional mmproj/draft artifacts (hf {source,repo,file} or local {path}), llama-server args. Binary path comes from env LLAMA_SERVER_PATH, never here. flash_attn is 'on'|'off'.",
    example: {
      model: {
        source: "hf",
        repo: "ggml-org/models",
        file: "gemma/ggml-model.gguf",
      },
      mmproj: {
        source: "hf",
        repo: "ggml-org/models",
        file: "gemma/mmproj-model-f16.gguf",
      },
      args: {
        ctx: 8192,
        gpu_layers: 35,
        flash_attn: "on",
        parallel: 1,
        threads: 8,
      },
      backend_port: 9999,
    },
  },
  halogen: {
    hint: "Env-configured ROCm engine: .hgn checkpoint + tokenizer dir, api/engine ports (defaults PROVIDER_PORT+1/+2), options map to HALOGEN_* vars. Capacity = options.kv_slots.",
    example: {
      model: {
        source: "hf",
        repo: "peonist-ai/halogen-qwen3.8-27b",
        file: "qwen3.8-27b-p1w4d-d2.hgn",
      },
      tokenizer: {
        source: "hf",
        repo: "peonist-ai/halogen-qwen3.8-27b",
        file: "tokenizer",
      },
      api_port: 8082,
      engine_port: 8083,
      options: {
        kv_slots: 4,
        drafter: "mtp",
        cache_mb: 2048,
        slot_ctx: 4096,
      },
    },
  },
  "halogen-flash": {
    hint: "Native OpenResponses NPU engine: static api/engine ports keep the disk-cache fingerprint stable (else SHA-256(MACHINE_UID) into 8200-8289). options map to the 43-key HALOGEN_* env set; npu_models pinning gated by host probe.",
    example: {
      model: {
        source: "hf",
        repo: "peonist-ai/halogen-qwen3.8-flash-next",
        file: "qwen38-flash-next-w4b.hgn",
      },
      tokenizer: {
        source: "hf",
        repo: "peonist-ai/halogen-qwen3.8-flash-next",
        file: "tokenizer",
      },
      api_port: 8200,
      engine_port: 8201,
      options: {
        kv_slots: 8,
        ctx: 131072,
        max_tok: 16384,
        cache_dir_enabled: true,
        cache_disk_gib: 64,
        npu_models: ["qwen3-embedding-0.6b", "qwen3.5-2b"],
      },
    },
  },
  gufo: {
    hint: "Multi-model native-OpenResponses engine: main model artifact + options passed as `gufo serve llm` flags. Capacity = options.sessions. Binary from env GUFO_SERVER_PATH. cache_disk:true uses CACHE_DIR/<instance_id>.",
    example: {
      model: {
        source: "hf",
        repo: "ggml-org/models",
        file: "model.gguf",
      },
      backend_port: 8082,
      options: {
        context: 131072,
        served_model_name: "my-alias",
        sessions: 4,
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        cache_disk: true,
        cache_disk_bytes: 1073741824,
      },
    },
  },
}
