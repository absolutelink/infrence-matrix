import logging
import subprocess
from dataclasses import dataclass
from typing import Literal

import psutil

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU information."""

    index: int
    name: str
    total_memory_bytes: int
    used_memory_bytes: int
    free_memory_bytes: int
    utilization_percent: float
    temperature_celsius: float | None = None


@dataclass
class GPUConfig:
    """GPU configuration for llama.cpp."""

    enabled: bool = True
    gpu_layers: int = 35
    split_mode: Literal["layer", "row", "none"] = "layer"
    main_gpu: int = 0
    tensor_split: list[float] | None = None


class GPUManager:
    """Manages GPU detection and configuration."""

    def __init__(self) -> None:
        self._gpu_info: list[GPUInfo] = []
        self._gpu_available = False
        self._gpu_type: Literal["cuda", "rocm", "metal", "vulkan", "none"] = "none"
        self._detect_gpu()

    def _detect_gpu(self) -> None:
        """Detect available GPUs."""
        try:
            if self._detect_nvidia():
                self._gpu_available = True
                self._gpu_type = "cuda"
                logger.info(f"Detected NVIDIA GPU(s): {len(self._gpu_info)} device(s)")
                return

            if self._detect_amd():
                self._gpu_available = True
                self._gpu_type = "rocm"
                logger.info(f"Detected AMD GPU(s): {len(self._gpu_info)} device(s)")
                return

            if self._detect_apple_metal():
                self._gpu_available = True
                self._gpu_type = "metal"
                logger.info("Detected Apple Metal GPU")
                return

            logger.info("No dedicated GPU detected, will use CPU")

        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            self._gpu_available = False
            self._gpu_type = "none"

    def _detect_nvidia(self) -> bool:
        """Detect NVIDIA GPUs using nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 5:
                            index = int(parts[0])
                            name = parts[1]
                            total_mem = int(parts[2]) * 1024 * 1024
                            used_mem = int(parts[3]) * 1024 * 1024
                            utilization = float(parts[4])
                            temp = float(parts[5]) if len(parts) > 5 else None

                            self._gpu_info.append(
                                GPUInfo(
                                    index=index,
                                    name=name,
                                    total_memory_bytes=total_mem,
                                    used_memory_bytes=used_mem,
                                    free_memory_bytes=total_mem - used_mem,
                                    utilization_percent=utilization,
                                    temperature_celsius=temp,
                                )
                            )

                return len(self._gpu_info) > 0

        except subprocess.TimeoutExpired, FileNotFoundError, ValueError:
            pass

        return False

    def _detect_amd(self) -> bool:
        """Detect AMD GPUs using rocm-smi."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for i, line in enumerate(lines):
                    if "GPU" in line or "Card" in line:
                        self._gpu_info.append(
                            GPUInfo(
                                index=i,
                                name=line.strip(),
                                total_memory_bytes=8 * 1024 * 1024 * 1024,
                                used_memory_bytes=0,
                                free_memory_bytes=8 * 1024 * 1024 * 1024,
                                utilization_percent=0.0,
                            )
                        )

                return len(self._gpu_info) > 0

        except subprocess.TimeoutExpired, FileNotFoundError:
            pass

        try:
            import pyamdgpuinfo

            gpus = pyamdgpuinfo.detect_gpus()
            for i, gpu in enumerate(gpus):
                self._gpu_info.append(
                    GPUInfo(
                        index=i,
                        name=gpu.name,
                        total_memory_bytes=gpu.memory_info["vram_size"],
                        used_memory_bytes=0,
                        free_memory_bytes=gpu.memory_info["vram_size"],
                        utilization_percent=0.0,
                    )
                )
            return len(self._gpu_info) > 0

        except ImportError, Exception:
            pass

        return False

    def _detect_apple_metal(self) -> bool:
        """Detect Apple Metal GPU."""
        import platform

        if platform.system() != "Darwin":
            return False

        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                output = result.stdout
                if (
                    "Apple" in output
                    or "M1" in output
                    or "M2" in output
                    or "M3" in output
                ):
                    self._gpu_info.append(
                        GPUInfo(
                            index=0,
                            name="Apple Metal",
                            total_memory_bytes=psutil.virtual_memory().total,
                            used_memory_bytes=psutil.virtual_memory().used,
                            free_memory_bytes=psutil.virtual_memory().available,
                            utilization_percent=0.0,
                        )
                    )
                    return True

        except subprocess.TimeoutExpired, FileNotFoundError:
            pass

        return False

    def get_gpu_info(self) -> list[GPUInfo]:
        """Get information about detected GPUs."""
        return self._gpu_info.copy()

    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self._gpu_available

    def get_gpu_type(self) -> Literal["cuda", "rocm", "metal", "vulkan", "none"]:
        """Get the type of GPU detected."""
        return self._gpu_type

    def get_recommended_gpu_layers(self, model_size_gb: float) -> int:
        """
        Get recommended number of GPU layers based on available VRAM.

        Args:
            model_size_gb: Model size in gigabytes

        Returns:
            Recommended number of layers to offload to GPU
        """
        if not self._gpu_available or not self._gpu_info:
            return 0

        total_vram = sum(gpu.free_memory_bytes for gpu in self._gpu_info)
        total_vram_gb = total_vram / (1024**3)

        if total_vram_gb < model_size_gb * 0.5:
            return 0
        elif total_vram_gb < model_size_gb:
            return int(35 * (total_vram_gb / model_size_gb))
        elif total_vram_gb < model_size_gb * 1.5:
            return 35
        else:
            return 99

    def get_optimal_config(
        self,
        model_size_gb: float,
        context_size: int = 4096,
    ) -> GPUConfig:
        """Get optimal GPU configuration for a model."""
        if not self._gpu_available:
            return GPUConfig(enabled=False)

        gpu_layers = self.get_recommended_gpu_layers(model_size_gb)

        if len(self._gpu_info) > 1:
            tensor_split_list: list[float] = [
                float(gpu.free_memory_bytes) for gpu in self._gpu_info
            ]
            total = sum(tensor_split_list)
            tensor_split = [v / total for v in tensor_split_list]
        else:
            tensor_split = None

        return GPUConfig(
            enabled=True,
            gpu_layers=gpu_layers,
            split_mode="layer",
            main_gpu=0,
            tensor_split=tensor_split,
        )

    def get_vram_usage(
        self,
    ) -> dict[str, int | list[dict[str, int | float | str | None]]]:
        """Get VRAM usage across all GPUs."""
        if not self._gpu_info:
            return {"total_used": 0, "total_free": 0, "gpus": []}

        total_used = sum(gpu.used_memory_bytes for gpu in self._gpu_info)
        total_free = sum(gpu.free_memory_bytes for gpu in self._gpu_info)

        return {
            "total_used": total_used,
            "total_free": total_free,
            "total_memory": total_used + total_free,
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "used": gpu.used_memory_bytes,
                    "free": gpu.free_memory_bytes,
                    "utilization": gpu.utilization_percent,
                    "temperature": gpu.temperature_celsius,
                }
                for gpu in self._gpu_info
            ],
        }

    def refresh_gpu_info(self) -> None:
        """Refresh GPU information."""
        self._gpu_info.clear()
        self._detect_gpu()
        logger.debug("GPU info refreshed")


gpu_manager = GPUManager()
