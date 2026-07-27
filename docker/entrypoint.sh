#!/usr/bin/env bash
# Bring up a SLURM REFERENCE CLUSTER, then exec the requested command.
#
# Why a "reference" cluster and not the real machine
# ──────────────────────────────────────────────────
# `sbatch --test-only` validates a resource request against the configured node
# topology. If that topology were inherited from the host, the benchmark would
# score differently on a laptop and on a server: a script asking for 4 GPU nodes
# would be rejected on an 8-core laptop and accepted elsewhere. Scores would
# depend on WHO runs the benchmark. Unacceptable.
#
# The topology is therefore DECLARED and fixed (v1):
#     4 nodes x 16 cores x 64000 MB x 4 GPUs, partition "debug"
# Changing it changes the results: it is part of the benchmark definition, and
# must be versioned and cited in the paper exactly like the tasks.
#
# Non-obvious details, learned empirically:
#   * cgroup.conf with IgnoreSystemd=yes. Containers have no systemd; without it
#     slurmd dies on "can't stat /sys/fs/cgroup/systemd/", never registers, nodes
#     stay "idle*" and sbatch --test-only rejects EVERY script.
#   * SlurmdParameters=config_overrides makes slurmctld trust the configuration
#     instead of interrogating the hardware.
#   * FirstJobId=12345 plus a held placeholder job: tasks that declare a
#     dependency on an existing job (afterok:12345) then validate. Without it,
#     sbatch answers "Job dependency problem".
#   * RealMemory is in MB. A task asking for --mem=16G (16384 MB) does NOT fit a
#     node declared with 16000. The reference cluster is generous on purpose.
set -euo pipefail

HOST="$(hostname -s)"
ANVIL_NODES="${ANVIL_NODES:-4}"
ANVIL_CPUS="${ANVIL_CPUS:-16}"
ANVIL_MEM_MB="${ANVIL_MEM_MB:-64000}"
ANVIL_GPUS="${ANVIL_GPUS:-4}"

mkdir -p /etc/slurm /var/spool/slurmctld /var/log/slurm /run/munge
chown -R slurm:slurm /var/spool/slurmctld /var/log/slurm 2>/dev/null || true

cat >/etc/slurm/slurm.conf <<EOF
ClusterName=anvil
SlurmctldHost=${HOST}
SlurmUser=root
AuthType=auth/munge

# Declared topology, not inherited from the hardware.
SlurmdParameters=config_overrides
GresTypes=gpu
FirstJobId=12345

ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
JobAcctGatherType=jobacct_gather/none
ReturnToService=2
SlurmdTimeout=0
InactiveLimit=0
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

SlurmdSpoolDir=/var/spool/slurmd/%n
StateSaveLocation=/var/spool/slurmctld
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd-%n.log

NodeName=node[1-${ANVIL_NODES}] NodeHostname=${HOST} NodeAddr=127.0.0.1 \
    CPUs=${ANVIL_CPUS} RealMemory=${ANVIL_MEM_MB} Gres=gpu:${ANVIL_GPUS} State=IDLE
PartitionName=debug Nodes=ALL Default=YES MaxTime=INFINITE State=UP
EOF

cat >/etc/slurm/cgroup.conf <<EOF
# Containers have no systemd. Without IgnoreSystemd, slurmd dies with
# "can't stat /sys/fs/cgroup/systemd/: No such file or directory" and never
# registers: nodes stay "idle*" (not responding) and sbatch --test-only rejects
# EVERY script with "Requested node configuration is not available".
CgroupPlugin=autodetect
IgnoreSystemd=yes
EOF

cat >/etc/slurm/gres.conf <<EOF
# GPUs declared without File: no real devices, sufficient for --test-only.
NodeName=node[1-${ANVIL_NODES}] Name=gpu Count=${ANVIL_GPUS}
EOF

if [[ ! -s /etc/munge/munge.key ]]; then
  dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key status=none
  chown munge:munge /etc/munge/munge.key 2>/dev/null || true
  chmod 400 /etc/munge/munge.key
fi
chown -R munge:munge /run/munge 2>/dev/null || true

# runuser (util-linux) is not present on every base image.
if command -v runuser >/dev/null 2>&1; then
  runuser -u munge -- /usr/sbin/munged --force >/dev/null 2>&1 || true
else
  su -s /bin/sh munge -c "/usr/sbin/munged --force" >/dev/null 2>&1 || true
fi
sleep 1

# slurmctld daemonises itself: no `&` (that would attach it to the session).
rm -rf /var/spool/slurmctld/* 2>/dev/null || true
( setsid /usr/sbin/slurmctld >/dev/null 2>&1 </dev/null & )
sleep 3

# One slurmd per virtual node (multi-slurmd). If it fails, Anvil's preflight marks
# `submittability` as SKIPPED with an explicit cause - never as a model failure.
for i in $(seq 1 "${ANVIL_NODES}"); do
  mkdir -p "/var/spool/slurmd/node${i}"
  ( setsid /usr/sbin/slurmd -N "node${i}" >/dev/null 2>&1 </dev/null & )
done
sleep 3
scontrol update nodename="node[1-${ANVIL_NODES}]" state=resume >/dev/null 2>&1 || true
sleep 1

# Held placeholder job: takes id 12345 and makes afterok dependencies valid.
cat >/tmp/anvil_placeholder.sh <<'PH'
#!/bin/bash
#SBATCH --job-name=anvil_placeholder
#SBATCH --time=00:01:00
sleep 1
PH
sbatch --hold /tmp/anvil_placeholder.sh >/dev/null 2>&1 || true

if [[ "${ANVIL_QUIET:-0}" != "1" ]]; then
  echo "==> base image: ${ANVIL_BASE_IMAGE:-unknown}"
  echo "==> reference cluster: ${ANVIL_NODES} nodes x ${ANVIL_CPUS} cores x ${ANVIL_MEM_MB} MB x ${ANVIL_GPUS} GPUs"
  sinfo -h -o "    %N %t %C %m %G" 2>/dev/null || echo "    (sinfo unavailable)"

  # `sinfo` prints the declared configuration, not what is running: SlurmdTimeout=0 keeps
  # slurmctld from ever marking an unreachable node DOWN, so nodes read `idle` with or
  # without a live slurmd behind them. Count the daemons instead of trusting that column.
  live_slurmd="$(pgrep -c slurmd 2>/dev/null || true)"
  live_slurmd="${live_slurmd:-0}"
  if [[ "${live_slurmd}" -lt "${ANVIL_NODES}" ]]; then
    echo "    NOTE: ${live_slurmd} of ${ANVIL_NODES} slurmd running, so the 'idle' above is"
    echo "    the declared state and not a live one. 'submittability' is unaffected, since"
    echo "    'sbatch --test-only' checks the configuration and needs no slurmd, but a job"
    echo "    submitted for real would stay pending. See /var/log/slurm/slurmd-*.log."
  fi

  # Canary: does the scheduler accept a minimal script?
  printf '#!/bin/bash\n#SBATCH --time=00:01:00\n#SBATCH --ntasks=1\necho canary\n' > /tmp/canary.sh
  if sbatch --test-only /tmp/canary.sh >/dev/null 2>&1; then
    echo "==> preflight OK: submittability will be ACTIVE"
  else
    echo "==> WARNING: the cluster rejects the canary. 'submittability' will be SKIPPED."
    echo "    Nodes in state 'idle*' mean slurmd did not register (see /var/log/slurm/)."
  fi
fi

exec "$@"
