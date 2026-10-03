"""Probe whether this host can serve halogen-flash NPU models.

Mirrors the checks the engine's own entrypoint performs before starting the
NPU (``deploy/entrypoint.sh`` ``npu_preflight``): the device node, the
host's XRT with its NPU plugin (catching mounts that carry dangling
symlinks), the image's pins file, the NPU engine binary, and whether the
GPU's fabric clock is held at its top speed. The result is reported in the
agent's registration ``gpu_info["npu"]`` so the UI only offers NPU options
on capable agents.
"""

import glob
import os
import subprocess

from app.core.config import settings

XRT_LIBS = (
    "libxrt_coreutil.so.2",
    "libxrt_core.so.2",
    "libxrt_driver_xdna.so.2",
)

# Roots under which the host's /sys may be visible: -v /sys:/host/sys puts it
# at /host/sys; a privileged container reads its own /sys.
_SYS_ROOTS = ("/host/sys", "/sys")


def _check_device(reasons: list[str]) -> bool:
    node = settings.NPU_DEVICE_PATH
    if not os.path.exists(node):
        if not os.path.isdir("/sys/module/amdxdna"):
            reasons.append("the NPU driver (amdxdna) is not loaded on this host")
        else:
            reasons.append(f"{node} is not present (pass --device {node})")
        return False
    if not os.access(node, os.R_OK | os.W_OK):
        reasons.append(f"{node} is not openable by this process")
        return False
    return True


def _check_xrt(reasons: list[str]) -> bool:
    lib_dir = settings.NPU_XRT_LIB_DIR
    for name in XRT_LIBS:
        path = os.path.join(lib_dir, name)
        if os.path.islink(path) and not os.path.exists(path):
            target = os.readlink(path)
            reasons.append(
                f"{path} is a dangling link to {target}: the mount carries "
                "the host's links, not the files they point at"
            )
            return False
    coreutil = os.path.join(lib_dir, "libxrt_coreutil.so.2")
    if not os.path.exists(coreutil):
        reasons.append(f"no XRT found at {lib_dir}")
        return False
    plugin = os.path.join(lib_dir, "libxrt_driver_xdna.so.2")
    if not os.path.exists(plugin):
        reasons.append("the host's XRT has no NPU plugin (libxrt_driver_xdna.so.2)")
        return False
    return True


def _check_binary(reasons: list[str]) -> bool:
    binary = settings.NPU_BINARY_PATH
    if not os.path.exists(binary):
        reasons.append(f"{binary} is not present")
        return False
    try:
        proc = subprocess.run(  # noqa: S603
            [binary],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        reasons.append(f"the NPU engine could not run: {exc}")
        return False
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 1 or not output.startswith("usage: halogen-npu"):
        first = output.splitlines()[:1]
        reasons.append(
            "the NPU engine does not load against the host's XRT "
            f"(exit {proc.returncode}{': ' + first[0] if first else ''})"
        )
        return False
    return True


def _fabric_clock_state() -> tuple[bool | None, str]:
    """Read the GPU's fabric clock over every amdgpu device with the control.

    Returns ``(held, detail)``. ``held`` is ``None`` when no control is
    readable at all (the probe cannot tell).
    """
    for root in _SYS_ROOTS:
        devices = sorted(glob.glob(os.path.join(root, "class/drm/card*/device")))
        seen = False
        all_held = True
        detail = ""
        for d in devices:
            fclk = os.path.join(d, "pp_dpm_fclk")
            perf = os.path.join(d, "power_dpm_force_performance_level")
            if not (os.access(fclk, os.R_OK) and os.access(perf, os.R_OK)):
                continue
            seen = True
            try:
                with open(perf) as fh:
                    level = fh.read().strip()
                with open(fclk) as fh:
                    lines = [ln for ln in fh.read().splitlines() if ln.strip()]
            except OSError:
                all_held = False
                continue
            if not lines:
                all_held = False
                continue
            top = lines[-1]
            top_is_max = "*" in top
            if level == "high":
                continue
            if level == "manual" and sum("*" in ln for ln in lines) == 1 and top_is_max:
                continue
            all_held = False
            detail = f"{level} at {top}"
            break
        if seen:
            if all_held:
                return (True, "held")
            return (False, detail)
    return (None, "no fabric clock control readable from this container")


def probe_npu() -> dict:
    """Probe NPU availability. Safe to call on any platform."""
    reasons: list[str] = []
    checks = {
        "device": _check_device(reasons),
        "xrt": _check_xrt(reasons),
    }
    if checks["device"] and checks["xrt"]:
        checks["engine"] = _check_binary(reasons)
    else:
        checks["engine"] = False
    clock_held, clock_detail = _fabric_clock_state()
    if clock_held is False:
        reasons.append(
            f"the GPU's fabric clock is not held at its top speed ({clock_detail})"
        )
    # The clock is a start-time requirement, not a capability: a root
    # container with /host/sys mounted holds it itself, and the host unit
    # holds it at boot. Report it so the UI can warn, but gate only on the
    # device, XRT and engine checks.
    available = all(checks.values())
    return {
        "available": available,
        "checks": checks,
        "fabric_clock_held": clock_held,
        "fabric_clock_detail": clock_detail,
        "reasons": reasons,
    }
