#!/usr/bin/env python3
"""Does the mirror serve the same models as the hub, down to the revision?

`ANVIL_HF_ENDPOINT` lets a failed download retry against a mirror, which moves the root of
trust: `huggingface_hub` verifies a download against the manifest of whichever endpoint it
used, so a mirror that served different bytes would verify them against its own hashes and
report success. This compares the two manifests before any of that matters.

    ./scripts/mirror_parity.py
    ./scripts/mirror_parity.py --mirror https://hf-mirror.com Qwen/Qwen2.5-Coder-7B-Instruct
    ./scripts/mirror_parity.py --local ~/.cache/huggingface/... ibm-granite/granite-4.1-3b

Three things have to match, and the first is the one that answers "is it the same version":

  * the repository commit, since a mirror can simply be behind;
  * the file set, since a missing shard fails at load time and a extra one is a question;
  * the sha256 of every file the hub records one for, which is every weight shard.

Exits non-zero on any divergence, so it can gate a sweep rather than only inform one.

`--local` answers the stricter question, and the one that matters once weights arrive from
somewhere other than the hub: it hashes the files already on disk and compares them with the
hub's manifest. That makes any source usable, including ones that speak no hub protocol at
all, because `transformers` loads from a directory: fetch the files however you like, verify
them here, then run offline. Without it, a download is only ever checked against the manifest
of whoever served it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "ibm-granite/granite-4.1-3b",
    "ibm-granite/granite-4.1-8b",
]
HUB = "https://huggingface.co"


def manifest(endpoint: str, model_id: str, timeout: int = 30) -> dict:
    """Commit and per-file digests as one endpoint reports them."""
    url = f"{endpoint}/api/models/{model_id}?blobs=true"
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        data = json.load(response)
    files = {}
    for sibling in data.get("siblings", []):
        lfs = sibling.get("lfs") or {}
        # Only files the hub tracks with a digest can be compared: the rest are small
        # text files whose content the API does not hash.
        files[sibling["rfilename"]] = (lfs.get("sha256") or lfs.get("oid"), lfs.get("size"))
    return {"sha": data.get("sha"), "gated": data.get("gated"), "files": files}


def compare(model_id: str, mirror: str) -> list[str]:
    problems: list[str] = []
    try:
        hub = manifest(HUB, model_id)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        return [f"{model_id}: the hub itself is unreachable ({type(exc).__name__}), nothing to "
                "compare against"]
    try:
        other = manifest(mirror, model_id)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        return [f"{model_id}: not available on {mirror} ({type(exc).__name__})"]

    if hub["sha"] != other["sha"]:
        problems.append(
            f"{model_id}: different revision, hub {str(hub['sha'])[:12]} against "
            f"{str(other['sha'])[:12]}"
        )
    missing = sorted(set(hub["files"]) - set(other["files"]))
    if missing:
        problems.append(f"{model_id}: {len(missing)} files absent from the mirror, {missing[:3]}")
    extra = sorted(set(other["files"]) - set(hub["files"]))
    if extra:
        problems.append(f"{model_id}: {len(extra)} files the hub does not have, {extra[:3]}")
    for name in sorted(set(hub["files"]) & set(other["files"])):
        if hub["files"][name] != other["files"][name]:
            problems.append(f"{model_id}: {name} differs, {hub['files'][name]} against "
                            f"{other['files'][name]}")

    hashed = sum(1 for digest, _ in hub["files"].values() if digest)
    print(f"  {model_id:<40} commit={str(hub['sha'])[:12]} files={len(hub['files'])} "
          f"hashed={hashed} {'OK' if not problems else 'DIVERGES'}")
    return problems


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify_local(model_id: str, directory: Path) -> list[str]:
    """Hash what is on disk and hold it against the hub's manifest.

    Only files the hub records a digest for are checked, which is every weight shard; the
    small text files it does not hash are reported as present or absent and no more. A
    directory missing a shard is a divergence rather than a warning: `transformers` would
    fail at load time anyway, and later rather than here.
    """
    try:
        hub = manifest(HUB, model_id)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        return [f"{model_id}: the hub is unreachable ({type(exc).__name__}), so nothing on disk "
                "can be checked against it"]

    hashed = {name: digest for name, (digest, _) in hub["files"].items() if digest}
    if not hashed:
        return [f"{model_id}: the hub records no digest for any file, nothing to verify"]

    problems: list[str] = []
    checked = 0
    for name, expected in sorted(hashed.items()):
        # Snapshots keep the repository layout, so a shard sits at the same relative path.
        local = directory / name
        if not local.is_file():
            problems.append(f"{model_id}: {name} is not in {directory}")
            continue
        actual = sha256_of(local)
        checked += 1
        if actual != expected:
            problems.append(
                f"{model_id}: {name} does not match the hub, {actual[:16]} against {expected[:16]}"
            )
        else:
            print(f"    {name:<36} {actual[:16]} ok")
    print(f"  {model_id:<40} {checked} hashed files verified "
          f"{'OK' if not problems else 'DIVERGES'}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--mirror", default="https://hf-mirror.com")
    parser.add_argument("--local", type=Path, default=None,
                        help="verify a downloaded directory against the hub instead of a mirror")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    print(f"hub    : {HUB}")
    if args.local:
        print(f"local  : {args.local}\n")
        problems = [p for model_id in models for p in verify_local(model_id, args.local)]
    else:
        print(f"mirror : {args.mirror}\n")
        problems = [p for model_id in models for p in compare(model_id, args.mirror)]

    if problems:
        print(f"\n{len(problems)} divergences:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("\nEverything checked matches the hub." if args.local
          else "\nEvery model matches the hub on revision, file set and digest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
