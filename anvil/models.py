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
import sys
import zlib
from abc import ABC, abstractmethod
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert HPC user support engineer. "
    "Write a single SLURM batch script that satisfies the request. "
    "Output only the script inside one ```bash code block. No explanation."
)

RECIPE_SYSTEM_PROMPT = (
    "You are an expert HPC user support engineer. "
    "Write a single Apptainer definition file (.def) that satisfies the request. "
    "Output only the recipe inside one ```singularity code block. No explanation."
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
        # startswith, not exact equality: the retrieval ablation appends
        # reference material after the task prompt (see
        # retrieval.build_prompt_with_context), so the prompt the oracle
        # actually receives may be longer than the one it was indexed under.
        tid = next((i for p, i in self._prompt_to_id.items() if prompt.startswith(p)), None)
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


DEFAULT_HF_ENDPOINT = "https://huggingface.co"


def _load_with_fallback(load):
    """Load from the default hub, and on failure retry against a declared mirror.

    An unauthenticated sweep gets rate limited by the hub, and a download failure ends a
    cell for a reason the benchmark is not measuring. This retries once against a mirror
    the caller names in `ANVIL_HF_ENDPOINT`, which is unset by default: nothing here
    points model weights at a third-party host that whoever runs it has not chosen.

    The mirror has to speak the hub's own protocol, since `huggingface_hub` is what
    performs the download. A catalogue with its own API is not a drop-in: it would need
    its manifest read, its files fetched and its bytes checked against the hub's hashes
    before `transformers` is allowed near them, which is a different piece of work and
    not this one.

    Returns (tokenizer, model, endpoint), so the caller can say where the weights came
    from rather than leave it implicit.
    """
    import os  # noqa: PLC0415

    try:
        tok, model = load()
        return tok, model, os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    except Exception as first:  # noqa: BLE001
        mirror = os.environ.get("ANVIL_HF_ENDPOINT")
        if not mirror:
            raise
        # Scoped to this load: leaving HF_ENDPOINT set would silently redirect every
        # later download in the process, including ones that would have succeeded.
        previous = os.environ.get("HF_ENDPOINT")
        os.environ["HF_ENDPOINT"] = mirror
        print(
            f"[warning] {DEFAULT_HF_ENDPOINT} failed ({type(first).__name__}: "
            f"{str(first)[:120]}), retrying from {mirror}",
            file=sys.stderr,
        )
        try:
            tok, model = load()
            print(f"[info] weights served by {mirror}", file=sys.stderr)
            return tok, model, mirror
        finally:
            if previous is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = previous


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
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.name = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_in_4bit = load_in_4bit
        self.system_prompt = system_prompt
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

        tok, model, self.weights_from = _load_with_fallback(
            lambda: (
                AutoTokenizer.from_pretrained(self.name),
                AutoModelForCausalLM.from_pretrained(self.name, **kwargs),
            )
        )

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
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # base models without a chat template
            text = f"{self.system_prompt}\n\n{prompt}\n\n```bash\n"

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
    """tasks/t1_slurm.jsonl -> tasks/t1_reference.jsonl; tasks/t3_apptainer.jsonl
    -> tasks/t3_reference.jsonl. The "tN" prefix is the only part that matters.

    A task file may also carry its own solutions beside it, which
    tasks/t1_exec.jsonl does in tasks/t1_exec_reference.jsonl. A set that only the
    sbatch executor can grade has no business in the reference file the rest of the
    benchmark shares: adding it there would change the digest every published T1
    number was measured against.
    """
    path = Path(tasks_path)
    own = path.with_name(f"{path.stem}_reference.jsonl")
    if own.exists():
        return own
    prefix = path.stem.split("_", 1)[0]
    return path.with_name(f"{prefix}_reference.jsonl")


def build_model(spec: str, tasks_path: str | Path, **kw) -> Model:
    """spec: 'oracle' | 'broken' | a Hugging Face model_id."""
    if spec == "oracle":
        return OracleModel(reference_path_for(tasks_path), tasks_path)
    if spec == "broken":
        return BrokenModel()
    return HFModel(spec, **kw)


class RecipeOracleModel(Model):
    """Canonical Apptainer recipes: the T3 benchmark's upper bound."""

    name = "oracle"

    def __init__(self, reference_path: str | Path, tasks_path: str | Path):
        self._by_id: dict[str, str] = {}
        with open(reference_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    self._by_id[rec["id"]] = rec["recipe"]
        self._prompt_to_id: dict[str, str] = {}
        with open(tasks_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    self._prompt_to_id[rec["prompt"]] = rec["id"]

    def generate(self, prompt: str, n: int = 1, seed: int | None = None) -> list[str]:
        tid = self._prompt_to_id.get(prompt)
        recipe = self._by_id.get(tid, "") if tid else ""
        return [f"```singularity\n{recipe}```" for _ in range(n)]


class RecipeBrokenModel(Model):
    """Deliberately faulty Apptainer recipes: negative tests for the T3 verifier."""

    name = "broken"

    FLAVOURS = [
        # no Bootstrap/From header at all
        "%post\n    echo hi\n%runscript\n    echo ANVIL_OK\n",
        # no %runscript or %startscript: nothing to execute
        "Bootstrap: docker\nFrom: alpine:latest\n%post\n    echo hi\n",
        # dangerous command in %post
        "Bootstrap: docker\nFrom: alpine:latest\n%post\n    rm -rf /\n%runscript\n"
        "    echo ANVIL_OK\n",
        # wrong base image (a resource_fit failure against most task specs)
        "Bootstrap: docker\nFrom: busybox:latest\n%runscript\n    echo ANVIL_OK\n",
        # runscript exits non-zero
        "Bootstrap: docker\nFrom: alpine:latest\n%runscript\n    exit 3\n",
    ]

    def generate(self, prompt: str, n: int = 1, seed: int | None = None) -> list[str]:
        base = (seed or 0) + zlib.crc32(prompt.encode("utf-8"))
        start = random.Random(base).randrange(len(self.FLAVOURS))
        return [
            f"```singularity\n{self.FLAVOURS[(start + i) % len(self.FLAVOURS)]}```"
            for i in range(n)
        ]


def build_recipe_model(spec: str, tasks_path: str | Path, **kw) -> Model:
    """spec: 'oracle' | 'broken' | a Hugging Face model_id."""
    if spec == "oracle":
        return RecipeOracleModel(reference_path_for(tasks_path), tasks_path)
    if spec == "broken":
        return RecipeBrokenModel()
    kw.setdefault("system_prompt", RECIPE_SYSTEM_PROMPT)
    return HFModel(spec, **kw)
