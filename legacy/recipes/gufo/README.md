# Gufo Recipe

This recipe publishes the Matrix agent with the
[Gufo](https://github.com/gufo-org/gufo) Strix Halo inference engine
(`ghcr.io/gufo-org/toolboxes/gufo-runtime:0.3.0`) and serves text LLM
models through `gufo serve llm`.

Gufo is built and optimized for AMD Strix Halo (`gfx1151`, Radeon 8060S)
with unified memory. It requires `/dev/kfd` and `/dev/dri`; rootless Podman
needs `crun` for `--group-add keep-groups` and the host user must be in the
`render` and `video` groups. On SELinux-enforcing hosts, GPU mapping may
require `sudo setsebool -P container_use_devices true`.

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --ipc=host \
  --ulimit memlock=-1 \
  -e AGENT_ID=agent-gufo \
  -e FRONTEND_URL=http://host.docker.internal:8000 \
  -v ./models:/models \
  -p 8080:8080 \
  ghcr.io/your-org/agent-gufo:main
```

Models are GGUF files selected from the Matrix model library. Speculative
decoding companions (DFlash2 for Qwen3.8-27B, DSpark for DeepSeek V4
Flash, MTP for Flash-Next) are also picked from the library and passed as
`--dflash-model` / `--dspark-model` / `--mtp-model`.

The agent registers with `AGENT_PLATFORM=gufo` and `AGENT_TYPE=gufo` and
runs up to `GUFO_MAX_INSTANCES` (default 2) concurrent `gufo serve llm`
processes. All server settings are exposed in the Start/Edit Server forms
and map one-to-one to gufo CLI flags; unset fields keep the gufo defaults.
