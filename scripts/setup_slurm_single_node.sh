#!/usr/bin/env bash
# Install a single-node SLURM on Debian/Ubuntu (or WSL2) to unlock the
# "submittability" verification level (sbatch --test-only) and real execution.
#
#   sudo ./scripts/setup_slurm_single_node.sh
#
# Without it, Anvil still works: L2 is marked "skipped" and L3 runs under bash.
# The metrics stay honest because "skipped" != "passed".
#
# NOTE: for the reference topology used by the benchmark (4 nodes, GPUs), prefer
# the container in docker/. This script configures a single physical node and is
# meant for a WSL2 machine where you also want real GPU inference.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo." >&2
  exit 1
fi

HOSTNAME_SHORT="$(hostname -s)"
CPUS="$(nproc)"
MEM_MB="$(awk '/MemTotal/ {printf "%d", $2/1024*0.8}' /proc/meminfo)"

echo "==> Installing packages"
apt-get update -qq
apt-get install -y -qq slurmd slurmctld munge >/dev/null

echo "==> munge key"
if [[ ! -s /etc/munge/munge.key ]]; then
  /usr/sbin/mungekey -f -k /etc/munge/munge.key 2>/dev/null || \
    dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key 2>/dev/null
fi
chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key

echo "==> slurm.conf for ${HOSTNAME_SHORT} (${CPUS} cpus, ${MEM_MB} MB)"
mkdir -p /etc/slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm
chown slurm:slurm /var/spool/slurmctld /var/log/slurm 2>/dev/null || true

cat >/etc/slurm/cgroup.conf <<EOF
CgroupPlugin=autodetect
IgnoreSystemd=yes
EOF

cat >/etc/slurm/slurm.conf <<EOF
ClusterName=anvil
SlurmctldHost=${HOSTNAME_SHORT}
ProctrackType=proctrack/linuxproc
ReturnToService=2
SlurmdSpoolDir=/var/spool/slurmd
StateSaveLocation=/var/spool/slurmctld
SlurmUser=slurm
TaskPlugin=task/none
JobAcctGatherType=jobacct_gather/none
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
SlurmdParameters=config_overrides
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log

# A single node acting as both controller and compute node.
NodeName=${HOSTNAME_SHORT} CPUs=${CPUS} RealMemory=${MEM_MB} State=UNKNOWN
PartitionName=debug Nodes=ALL Default=YES MaxTime=INFINITE State=UP
EOF

echo "==> Starting services"
systemctl enable --now munge   >/dev/null 2>&1 || service munge start
systemctl restart slurmctld    >/dev/null 2>&1 || service slurmctld restart
systemctl restart slurmd       >/dev/null 2>&1 || service slurmd restart

sleep 2
echo "==> Check"
sinfo || true
echo
echo "anvil run will now enable the submittability level."
echo "If the node shows as 'drain':  scontrol update nodename=${HOSTNAME_SHORT} state=resume"
