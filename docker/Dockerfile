# A faithful verification environment: Linux + bash 5 + GNU coreutils + SLURM.
#
# Solves two problems at once:
#   1. on macOS the local bash is 3.2 with BSD coreutils, so the functional level
#      would produce outcomes that differ from the cluster;
#   2. without SLURM, the submittability level stays permanently skipped.
#
# Multi-arch: builds natively on Apple Silicon (arm64) and on x86_64.
#
#   docker build -t anvil docker/
#   docker run --rm -v "$PWD":/work -w /work anvil pytest -q
#   docker run --rm -v "$PWD":/work -w /work anvil \
#       python -m anvil.cli run --model oracle --tasks tasks/t1_slurm.jsonl -v
#
# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRIC BASE IMAGE
#
# The default is ubuntu:24.04, NOT the newest LTS. This is deliberate:
#
#   * Ubuntu 26.04 "Resolute Raccoon" replaces GNU coreutils with **uutils**
#     (a Rust reimplementation) and sudo with sudo-rs. uutils targets GNU
#     compatibility but is not GNU.
#   * No HPC centre runs uutils: clusters run RHEL/Rocky/SUSE/Ubuntu with glibc
#     and GNU coreutils.
#   * Using 26.04 would reintroduce the very problem this container solves:
#     executing generated scripts on non-GNU coreutils (as on macOS with BSD,
#     or on Alpine with BusyBox).
#
# Fidelity to the target environment beats dependency freshness: only Python and
# SLURM run inside this image, and neither needs to be current.
#
# Other bases (for the Phase 2 cross-distribution portability ablation):
#   docker build -t anvil:2604  --build-arg BASE_IMAGE=ubuntu:26.04 docker/
#   docker build -t anvil:rocky --build-arg BASE_IMAGE=rockylinux:9  docker/   # needs apt -> dnf
#
# T3 (Apptainer recipes) needs its own opt-in build:
#   docker build -t anvil:apptainer --build-arg WITH_APPTAINER=1 docker/
#   docker run --rm --security-opt seccomp=unconfined --device /dev/fuse \
#       -v "$PWD":/work -w /work anvil:apptainer python -m anvil.cli ...
# The two extra flags are required for apptainer's unprivileged build/run
# (user namespaces + squashfuse), not for anything else this image does.
# Observed to fail on Docker Desktop for Mac (nested linuxkit VM); works on
# Docker Desktop for Windows and is expected to work on a native Linux host.
# ─────────────────────────────────────────────────────────────────────────────
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
LABEL org.opencontainers.image.title="anvil-verifier" \
      org.opencontainers.image.description="Faithful Linux+SLURM environment for the Anvil verifier" \
      anvil.base_image="${BASE_IMAGE}"

# Record the base in the image: `anvil doctor` reports it, and it ends up in the
# environment report that accompanies the paper's results.
ENV ANVIL_BASE_IMAGE="${BASE_IMAGE}" \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        bash coreutils procps util-linux \
        slurm-wlm slurmd slurmctld munge \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Apptainer is opt-in: most anvil work (T1, T2) never touches it, and its PPA
# install adds real build time. Only images built with WITH_APPTAINER=1 (the
# T3 recipe verifier) pay for it.
#   docker build -t anvil:apptainer --build-arg WITH_APPTAINER=1 docker/
ARG WITH_APPTAINER=0
RUN if [ "$WITH_APPTAINER" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            software-properties-common uidmap \
        && add-apt-repository -y ppa:apptainer/ppa \
        && apt-get update && apt-get install -y --no-install-recommends apptainer \
        && rm -rf /var/lib/apt/lists/* \
        && echo "root:100000:65536" >> /etc/subuid \
        && echo "root:100000:65536" >> /etc/subgid \
        && sed -i -E 's/^[[:space:]]*mount home[[:space:]]*=[[:space:]]*yes/mount home = no/' \
            /etc/apptainer/apptainer.conf \
        && grep -qE '^mount home = no' /etc/apptainer/apptainer.conf ; \
    fi
# The PPA ships apptainer without its setuid starter, so every build and run takes the
# unprivileged path through a user namespace. That path maps root inside the namespace to
# a range of unused host ids, and refuses to start without one: `could not use fakeroot:
# no mapping entry found in /etc/subuid for root`. The range above is the mapping, and
# `uidmap` provides the newuidmap/newgidmap helpers that apply it. Neither grants any
# privilege on the host: the ids are unprivileged and belong to nobody.
#
# `mount home = no` then stops the next failure, `failed to mount /root to /root:
# permission denied`: /root belongs to the real root, which that namespace does not map.
# It is set in apptainer.conf rather than through APPTAINER_NO_MOUNT because `build`
# manages its own mounts and ignores that variable, which cost a CI run to establish.
# The grep is not decoration: without it a changed key name would leave an image that
# still binds the home and fails the same way, with nothing pointing at the cause.

# venv to work around the PEP 668 lock on Ubuntu >= 24.04
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --no-cache-dir pytest ruff

# The repo is mounted at /work at runtime: the `anvil` package is not installed
# in the image's venv. PYTHONPATH makes it importable without writing into the
# mounted directory (which may be read-only).
ENV PYTHONPATH="/work"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
