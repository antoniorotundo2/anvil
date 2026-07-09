"""Execution environment detection.

Abstracts over CUDA (NVIDIA), MPS (Apple Silicon) and CPU, and reports honestly
what is available: 4-bit quantization requires bitsandbytes + CUDA and does NOT
work on MPS; FP8 requires Hopper (sm_90+), so it is unavailable on Ampere
(RTX 30xx) and on Apple Silicon.

torch is never imported at module level: the harness must stay usable without it.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    device: str            # "cuda" | "mps" | "cpu"
    dtype: str             # "float16" | "bfloat16" | "float32"
    name: str = ""
    vram_gb: float | None = None
    supports_4bit: bool = False
    supports_bf16: bool = False
    supports_fp8: bool = False
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401,PLC0415
    except Exception:
        return False
    return True


def detect() -> DeviceInfo:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return DeviceInfo(
            device="cpu",
            dtype="float32",
            name="torch not installed",
            notes=['install the extras: pip install -e ".[models]"'],
        )

    # --- NVIDIA -----------------------------------------------------------
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        major = props.major
        vram = round(props.total_memory / 1024**3, 1)
        bf16 = major >= 8                     # Ampere and later
        fp8 = major >= 9                      # Hopper and later
        has_bnb = _bitsandbytes_available()

        notes: list[str] = []
        if not fp8:
            notes.append(
                f"FP8/Transformer Engine unavailable (compute capability {major}.x, "
                "needs Hopper sm_90+). Expected on Ampere."
            )
        if not has_bnb:
            notes.append('4-bit quantization: pip install -e ".[quant]"')
        if vram < 10:
            notes.append(f"{vram}GB VRAM: use models <=3B in fp16, or 7B in 4-bit")
        elif vram < 16:
            notes.append(f"{vram}GB VRAM: 7B in 4-bit fits; 7B in fp16 does NOT")

        return DeviceInfo(
            device="cuda",
            dtype="bfloat16" if bf16 else "float16",
            name=props.name,
            vram_gb=vram,
            supports_4bit=has_bnb,
            supports_bf16=bf16,
            supports_fp8=fp8,
            notes=notes,
        )

    # --- Apple Silicon ----------------------------------------------------
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo(
            device="mps",
            dtype="float16",
            name=f"Apple Silicon ({platform.machine()})",
            supports_4bit=False,
            supports_bf16=False,
            supports_fp8=False,
            notes=[
                "MPS: bitsandbytes unsupported, no 4-bit. Use small models (<=3B) in "
                "fp16, or llama.cpp/MLX outside this harness.",
                "Prefer a CUDA machine for the final experiments.",
            ],
        )

    # --- CPU --------------------------------------------------------------
    return DeviceInfo(
        device="cpu",
        dtype="float32",
        name=platform.processor() or platform.machine(),
        notes=["no accelerator: suitable only for very small models"],
    )


def torch_dtype(info: DeviceInfo):
    import torch  # noqa: PLC0415

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        info.dtype
    ]


def _first_line(cmd: list[str]) -> str:
    """First line of `cmd --version`, or the empty string."""
    import subprocess  # noqa: PLC0415

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return (p.stdout or p.stderr).splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        return ""


def classify_coreutils(ls_version_line: str) -> str:
    """Classify the coreutils flavour from the first line of `ls --version`.

    Pure function: testable without having all four implementations at hand.
    On BSD/macOS `ls --version` does not exist and returns the empty string.
    """
    if not ls_version_line:
        return "BSD (no --version)"
    if "uutils" in ls_version_line or "uu_ls" in ls_version_line:
        return "uutils (Rust) - NOT GNU"
    if "BusyBox" in ls_version_line:
        return "BusyBox - NOT GNU"
    if "GNU coreutils" in ls_version_line:
        return ls_version_line
    return "unknown"


def shell_environment() -> dict[str, str]:
    """Identify *which* bash and *which* coreutils will execute the scripts.

    Not a detail: the `functional` level executes the generated artifacts, and the
    outcome depends on the implementation. Three flavours diverge from a cluster:
      * BSD (macOS)      - bash 3.2, BSD coreutils
      * BusyBox (Alpine) - musl, minimal applets
      * uutils (Ubuntu >= 25.10 / 26.04) - coreutils rewritten in Rust
    HPC centres run glibc + GNU coreutils. Declaring the flavour is mandatory in
    the paper's setup section.
    """
    bash_v = _first_line(["bash", "--version"])
    flavour = classify_coreutils(_first_line(["ls", "--version"]))

    m = re.search(r"version (\d+\.\d+)", bash_v)
    bash_major = m.group(1) if m else "?"

    return {
        "bash_version": bash_major,
        "coreutils": flavour,
        "base_image": os.environ.get("ANVIL_BASE_IMAGE", "n/a (outside the container)"),
    }


def environment_report() -> dict[str, object]:
    """Summary for `anvil doctor` and for the paper's setup section."""
    info = detect()
    shell = shell_environment()
    faithful = "GNU coreutils" in shell["coreutils"]
    notes = list(info.notes)
    if not faithful:
        notes.append(
            f"non-GNU coreutils ({shell['coreutils']}): the `functional` level may "
            "diverge from the cluster. Run the verifier inside the container."
        )
    return {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "base_image": shell["base_image"],
        "bash": shell["bash_version"],
        "coreutils": shell["coreutils"],
        "gnu_faithful": faithful,
        "device": info.device,
        "device_name": info.name,
        "dtype": info.dtype,
        "vram_gb": info.vram_gb,
        "supports_4bit": info.supports_4bit,
        "supports_bf16": info.supports_bf16,
        "supports_fp8": info.supports_fp8,
        "sbatch": shutil.which("sbatch") or "not found",
        "functional_executor": "bash",   # not sbatch: see check_functional()
        "apptainer": shutil.which("apptainer") or shutil.which("singularity") or "not found",
        "notes": notes,
    }
