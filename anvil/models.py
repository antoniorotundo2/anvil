"""Wrappers for the evaluated models.

Three implementations:
  * OracleModel  - returns the canonical solutions. Proves the tasks are solvable
                   and the verifier is not broken. A benchmark the oracle cannot
                   pass is a defective benchmark.
  * BrokenModel  - returns deliberately faulty artifacts. Negative tests: a
                   verifier that promotes them is too permissive.
  * HFModel      - a real Hugging Face model, CPU-first with GPU autodetection.
"""

from __future__ import annotations

import json
import random
import zlib
from abc import ABC, abstractmethod
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert HPC user support engineer. "
    "Write a single SLURM batch script that satisfies the request. "
    "Output only the script inside one ```bash code block. No explanation."
)


class Model(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, n: int = 1, seed: int | None = None) -> list[str]:
        """Return n raw completions (not yet extracted)."""


class OracleModel(Model):
    """Canonical solutions: the benchmark's upper bound."""

    name = "oracle"

    def __init__(self, reference_path: str | Path, tasks_path: str | Path):
        self._by_id: dict[str, str] = {}
        with open(reference_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    self._by_id[rec["id"]] = rec["script"]
        self._prompt_to_id: dict[str, str] = {}
        with open(tasks_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    self._prompt_to_id[rec["prompt"]] = rec["id"]

    def generate(self, prompt: str, n: int = 1, seed: int | None = None) -> list[str]:
        tid = self._prompt_to_id.get(prompt)
        script = self._by_id.get(tid, "") if tid else ""
        return [f"```bash\n{script}```" for _ in range(n)]


class BrokenModel(Model):
    """Deliberately faulty artifacts: negative tests for the verifier.

    Sampling is deterministic but *varied*:
      * the seed is mixed with the prompt, so different tasks get different defects;
      * flavours are traversed round-robin, so with n >= len(FLAVOURS) every defect
        is exercised at least once.

    Without this, a `random.Random(seed)` recreated on every call returned the same
    flavours for every task: with n=3 the destructive flavour was never sampled and
    `check_safety` was NEVER put to the test.
    """

    name = "broken"

    FLAVOURS = [
        # missing shebang
        "#SBATCH --time=00:10:00\necho ANVIL_OK\n",
        # directive after the first command: SLURM would silently ignore it
        "#!/bin/bash\necho starting\n#SBATCH --time=00:10:00\necho ANVIL_OK\n",
        # broken bash syntax
        "#!/bin/bash\n#SBATCH --time=00:10:00\nif [ 1 -eq 1 ; then echo ANVIL_OK\n",
        # walltime beyond the requested limit
        "#!/bin/bash\n#SBATCH --time=99:00:00\n#SBATCH --mem=1M\necho ANVIL_OK\n",
        # destructive command
        "#!/bin/bash\n#SBATCH --time=00:10:00\nrm -rf /\necho ANVIL_OK\n",
        # non-zero exit code
        "#!/bin/bash\n#SBATCH --time=00:10:00\nexit 3\n",
    ]

    def generate(self, prompt: str, n: int = 1, seed: int | None = None) -> list[str]:
        base = (seed or 0) + zlib.crc32(prompt.encode("utf-8"))
        start = random.Random(base).randrange(len(self.FLAVOURS))
        return [
            f"```bash\n{self.FLAVOURS[(start + i) % len(self.FLAVOURS)]}```"
            for i in range(n)
        ]


class HFModel(Model):
    """Hugging Face model. CPU-first; uses the GPU when available.

    Optional 4-bit quantization (requires bitsandbytes and a CUDA GPU).
    torch/transformers are imported lazily: the harness stays usable without them.
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        load_in_4bit: bool = False,
    ):
        self.name = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_in_4bit = load_in_4bit
        self._model = None      # sentinel: the model is loaded lazily, ONCE

    def _ensure_loaded(self) -> None:
        # Guard on `_model`, not on a never-assigned attribute: getting this wrong
        # reloads the weights from disk on every task (8x slowdown, silently).
        if self._model is not None:
            return
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        from .device import detect, torch_dtype  # noqa: PLC0415

        info = detect()
        self._info = info

        if self.load_in_4bit and not info.supports_4bit:
            raise RuntimeError(
                f"4-bit quantization requested but unavailable on device "
                f"'{info.device}' ({info.name}). Requires CUDA + bitsandbytes. "
                f"On Apple Silicon use fp16 with small models, or run on an NVIDIA GPU."
            )

        kwargs: dict = {}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype(info),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            # bitsandbytes requires device_map: without it, loading fails.
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch_dtype(info)

        tok = AutoTokenizer.from_pretrained(self.name)
        model = AutoModelForCausalLM.from_pretrained(self.name, **kwargs)

        # device_map="auto" needs accelerate and targets multi-GPU CUDA.
        # On MPS and CPU we move the model explicitly.
        if not self.load_in_4bit and info.device in ("mps", "cpu"):
            model = model.to(info.device)
        elif not self.load_in_4bit and info.device == "cuda":
            model = model.to("cuda")

        self._tok, self._model = tok, model
        self._device = info.device

    def generate(self, prompt: str, n: int = 1, seed: int | None = None) -> list[str]:
        self._ensure_loaded()
        import torch  # noqa: PLC0415

        if seed is not None:
            torch.manual_seed(seed)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # base models without a chat template
            text = f"{SYSTEM_PROMPT}\n\n{prompt}\n\n```bash\n"

        enc = self._tok(text, return_tensors="pt")
        enc = {k: v.to(self._device) for k, v in enc.items()}

        outs: list[str] = []
        for _ in range(n):
            with torch.no_grad():
                gen = self._model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=max(self.temperature, 1e-5),
                    pad_token_id=self._tok.eos_token_id,
                )
            outs.append(
                self._tok.decode(gen[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
            )
        return outs


def reference_path_for(tasks_path: str | Path) -> Path:
    """tasks/t1_slurm.jsonl -> tasks/t1_reference.jsonl"""
    stem = Path(tasks_path).stem.replace("_slurm", "_reference")
    return Path(tasks_path).with_name(stem + ".jsonl")


def build_model(spec: str, tasks_path: str | Path, **kw) -> Model:
    """spec: 'oracle' | 'broken' | a Hugging Face model_id."""
    if spec == "oracle":
        return OracleModel(reference_path_for(tasks_path), tasks_path)
    if spec == "broken":
        return BrokenModel()
    return HFModel(spec, **kw)
