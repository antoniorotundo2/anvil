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
# The run needs several exemptions from Docker's confinement, none of them a
# capability; the Makefile's DOCKER_RUN_APPTAINER carries the current set and
# docs/DESIGN.md pairs each one with the failure that forced it. Verified on
# GitHub-hosted runners and on WSL2. Docker Desktop for Mac has not been retested.
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
# `mount home = no` was expected to stop the `failed to mount /root to /root` failure
# and, per the CI probe, is applied yet does not: the fakeroot build engine performs that
# bind regardless. The denial itself came from Docker's AppArmor profile (see the
# Makefile's apparmor=unconfined). The setting stays because this image has no use for
# host homes inside containers, and the grep still fails the build if a renamed config
# key ever makes the sed silently miss.

# Accounting is opt-in for the same reason apptainer is: nothing in T1, T2 or T3 needs
# it, and a database in a verification image is a cost nobody should pay by default.
#   docker build -t anvil:sched --build-arg WITH_SLURMDBD=1 docker/
#
# What forces it: this Ubuntu's slurm package ships accounting_storage_slurmdbd.so and no
# other accounting plugin, so with no slurmdbd the association manager stays empty and
# every job the scheduler accepts is then refused with Reason=InvalidAccount. Verified
# with a minimal stock slurm.conf in this same image, so it is not anvil's configuration.
# `sbatch --test-only` is unaffected, which is why the default image never noticed.
ARG WITH_SLURMDBD=0
RUN if [ "$WITH_SLURMDBD" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            slurmdbd mariadb-server \
        && rm -rf /var/lib/apt/lists/* \
        && mkdir -p /run/mysqld && chown mysql:mysql /run/mysqld \
        && mariadb-install-db --user=mysql --datadir=/var/lib/mysql >/dev/null ; \
    fi

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
