#!/usr/bin/env python3
"""Does the mirror serve the same models as the hub, down to the revision?

`ANVIL_HF_ENDPOINT` lets a failed download retry against a mirror, which moves the root of
trust: `huggingface_hub` verifies a download against the manifest of whichever endpoint it
used, so a mirror that served different bytes would verify them against its own hashes and
report success. This compares the two manifests before any of that matters.

    ./scripts/mirror_parity.py
    ./scripts/mirror_parity.py --mirror https://hf-mirror.com Qwen/Qwen2.5-Coder-7B-Instruct

Three things have to match, and the first is the one that answers "is it the same version":

  * the repository commit, since a mirror can simply be behind;
  * the file set, since a missing shard fails at load time and a extra one is a question;
  * the sha256 of every file the hub records one for, which is every weight shard.

Exits non-zero on any divergence, so it can gate a sweep rather than only inform one. It
compares what the two say, not what they send: verifying the bytes on disk after a download
is a stricter check and a different script.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--mirror", default="https://hf-mirror.com")
    args = parser.parse_args()

    print(f"hub    : {HUB}")
    print(f"mirror : {args.mirror}\n")
    problems = [p for model_id in (args.models or DEFAULT_MODELS)
                for p in compare(model_id, args.mirror)]

    if problems:
        print(f"\n{len(problems)} divergences:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("\nEvery model matches the hub on revision, file set and digest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
