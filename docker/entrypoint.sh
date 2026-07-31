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
#
# Three more, found the day a slurmd was asked to actually run a job rather than
# only to make `sbatch --test-only` answer. Each one kept every daemon down, and
# none of them could show up while nothing ever registered:
#   * slurmd creates its own stepd scope directory but not the parent slice, and
#     under Docker that parent does not exist ("Could not create scope directory
#     .../system.slice/nodeN_slurmstepd.scope: No such file or directory").
#   * multi-slurmd on one host needs one port per virtual node, otherwise the
#     second daemon dies on "Error binding slurm stream socket: Address already
#     in use" and only one of the four survives.
#   * a registering slurmd reports the GPUs it can see. With none, the controller
#     answers "gres/gpu count reported lower than configured (0 < 4)" and drains
#     the node, which would take `submittability` down with it.
set -euo pipefail

HOST="$(hostname -s)"
ANVIL_NODES="${ANVIL_NODES:-4}"
ANVIL_CPUS="${ANVIL_CPUS:-16}"
ANVIL_MEM_MB="${ANVIL_MEM_MB:-64000}"
ANVIL_GPUS="${ANVIL_GPUS:-4}"

mkdir -p /etc/slurm /var/spool/slurmctld /var/log/slurm /run/munge
chown -R slurm:slurm /var/spool/slurmctld /var/log/slurm 2>/dev/null || true

# Accounting is present only in the image built with WITH_SLURMDBD=1, and it exists for
# one reason: without it this SLURM refuses every job it accepts (Reason=InvalidAccount),
# so nothing can be executed for real. Detected rather than configured, so the default
# image behaves exactly as before.
ACCOUNTING=0
if command -v slurmdbd >/dev/null 2>&1 && command -v mariadbd >/dev/null 2>&1; then
  ACCOUNTING=1
fi
# Not slurmdbd's default 6819: the virtual nodes take 6818 upward and would collide.
DBD_PORT=6899

# cgroup v2 delegation, done before any daemon starts so that all of them inherit the
# leaf below and the container's own cgroup stays free of processes. That is the kernel's
# "no internal processes" rule, and it is why the move comes first: a cgroup holding
# processes cannot delegate controllers to its children.
#
# What it buys is the whole of stage 2. With the controllers reaching the scope slurmd
# creates, task/cgroup holds a job to the memory it asked for, so a script that
# under-requests comes back OUT_OF_MEMORY (ExitCode 0:125) instead of quietly succeeding.
# Swap has to be constrained too: with memory.max alone the excess is paged out and the
# job completes, which is what the first measurement showed.
#
# Detected, never assumed: without a writable /sys/fs/cgroup the delegation fails, and
# then the cgroup plugins would keep slurmd from starting at all, which is worse than no
# enforcement. In that case the configuration below stays on the plugins that need none.
ENFORCEMENT=0
container_cgroup="$(awk -F: '$1 == "0" { print $3 }' /proc/self/cgroup 2>/dev/null || true)"
cgbase="/sys/fs/cgroup${container_cgroup}"
if mkdir -p "${cgbase}/init" "${cgbase}/system.slice" 2>/dev/null; then
  while read -r pid; do
    echo "${pid}" >"${cgbase}/init/cgroup.procs" 2>/dev/null || true
  done <"${cgbase}/cgroup.procs"
  echo "+cpuset +cpu +memory" >"${cgbase}/cgroup.subtree_control" 2>/dev/null || true
  echo "+cpuset +cpu +memory" >"${cgbase}/system.slice/cgroup.subtree_control" 2>/dev/null || true
  case "$(cat "${cgbase}/system.slice/cgroup.subtree_control" 2>/dev/null)" in
    *memory*) ENFORCEMENT=1 ;;
  esac
fi
# The slice slurmd looks for sits one level above its own cgroup. That is
# ${cgbase}/system.slice once the move above succeeded, and the sibling path when it did
# not, so both are created and the daemon finds one either way.
mkdir -p "/sys/fs/cgroup${container_cgroup%/*}/system.slice" 2>/dev/null || true

if [[ "${ENFORCEMENT}" -eq 1 ]]; then
  PROCTRACK=proctrack/cgroup
  TASKPLUGIN=task/cgroup
else
  PROCTRACK=proctrack/linuxproc
  TASKPLUGIN=task/none
fi

cat >/etc/slurm/slurm.conf <<EOF
ClusterName=anvil
SlurmctldHost=${HOST}
SlurmUser=root
AuthType=auth/munge

# Declared topology, not inherited from the hardware.
SlurmdParameters=config_overrides
GresTypes=gpu
FirstJobId=12345

ProctrackType=${PROCTRACK}
TaskPlugin=${TASKPLUGIN}
JobAcctGatherType=jobacct_gather/none
ReturnToService=2
SlurmdTimeout=0
InactiveLimit=0
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

SlurmdSpoolDir=/var/spool/slurmd/%n
# Per node, like the spool directory above: the four daemons otherwise contend for one
# lock on /run/slurmd.pid, and a CI run came back with three of them alive.
SlurmdPidFile=/run/slurmd-%n.pid
StateSaveLocation=/var/spool/slurmctld
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd-%n.log

EOF

if [[ "${ACCOUNTING}" -eq 1 ]]; then
  cat >>/etc/slurm/slurm.conf <<EOF
# Enforcement stays off: the point is to give the association manager something to find,
# not to police anything. Limits would turn a job's fate into a property of this file.
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=localhost
AccountingStoragePort=${DBD_PORT}
EOF
fi

# One line per virtual node instead of the range form, because each needs its own port:
# several slurmd on the same host cannot share 6818, and the ones that lose the race die
# on "Address already in use". slurmctld holds 6817, so the nodes take 6818 upward.
for i in $(seq 1 "${ANVIL_NODES}"); do
  cat >>/etc/slurm/slurm.conf <<EOF
NodeName=node${i} NodeHostname=${HOST} NodeAddr=127.0.0.1 Port=$((6817 + i)) \
    CPUs=${ANVIL_CPUS} RealMemory=${ANVIL_MEM_MB} Gres=gpu:${ANVIL_GPUS} State=IDLE
EOF
done
echo "PartitionName=debug Nodes=ALL Default=YES MaxTime=INFINITE State=UP" \
  >>/etc/slurm/slurm.conf

cat >/etc/slurm/cgroup.conf <<EOF
# Containers have no systemd. Without IgnoreSystemd, slurmd dies with
# "can't stat /sys/fs/cgroup/systemd/: No such file or directory" and never
# registers: nodes stay "idle*" (not responding) and sbatch --test-only rejects
# EVERY script with "Requested node configuration is not available".
CgroupPlugin=autodetect
IgnoreSystemd=yes
EOF

if [[ "${ENFORCEMENT}" -eq 1 ]]; then
  cat >>/etc/slurm/cgroup.conf <<EOF
# Swap is constrained alongside RAM on purpose: with memory.max alone the pages over the
# limit are simply paged out and an under-requesting job finishes as if nothing happened.
ConstrainRAMSpace=yes
ConstrainSwapSpace=yes
ConstrainCores=yes
EOF
fi

# The declared GPUs need something on disk behind them the moment a slurmd registers:
# with nothing, it reports zero and the controller drains the node with "gres/gpu count
# reported lower than configured". Character devices where the container may create them
# (the same privilege real submission needs anyway), and the count-only form otherwise,
# which is what --test-only has always run on.
gpu_devices=0
for i in $(seq 0 $((ANVIL_GPUS - 1))); do
  dev="/dev/anvilgpu${i}"
  if [[ -e "${dev}" ]] || mknod "${dev}" c 195 "${i}" 2>/dev/null; then
    gpu_devices=$((gpu_devices + 1))
  fi
done

if [[ "${gpu_devices}" -eq "${ANVIL_GPUS}" ]]; then
  if [[ "${ANVIL_GPUS}" -eq 1 ]]; then
    gres_spec="File=/dev/anvilgpu0"
  else
    gres_spec="File=/dev/anvilgpu[0-$((ANVIL_GPUS - 1))]"
  fi
else
  gres_spec="Count=${ANVIL_GPUS}"
fi

cat >/etc/slurm/gres.conf <<EOF
NodeName=node[1-${ANVIL_NODES}] Name=gpu ${gres_spec}
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

# The database, then slurmdbd, then the cluster and the association: all of it has to be
# in place before slurmctld starts, because the controller registers with the cluster it
# finds there. The password is a local secret in a throwaway container reachable only
# over a unix socket; treating it as one would be theatre.
if [[ "${ACCOUNTING}" -eq 1 ]]; then
  mkdir -p /run/mysqld && chown mysql:mysql /run/mysqld
  ( setsid mariadbd --user=mysql >/var/log/slurm/mariadb.log 2>&1 </dev/null & )
  for _ in $(seq 1 30); do
    mariadb-admin ping >/dev/null 2>&1 && break
    sleep 1
  done
  mariadb -e "CREATE DATABASE IF NOT EXISTS slurm_acct_db;
              CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED BY 'slurm';
              GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';
              FLUSH PRIVILEGES;" >/dev/null 2>&1 || true

  # slurmdbd refuses to start on a world-readable config, since it holds the password.
  cat >/etc/slurm/slurmdbd.conf <<EOF
AuthType=auth/munge
DbdHost=localhost
DbdPort=${DBD_PORT}
SlurmUser=root
StorageType=accounting_storage/mysql
StorageHost=localhost
StorageUser=slurm
StoragePass=slurm
StorageLoc=slurm_acct_db
LogFile=/var/log/slurm/slurmdbd.log
PidFile=/run/slurmdbd.pid
EOF
  chmod 600 /etc/slurm/slurmdbd.conf
  ( setsid /usr/sbin/slurmdbd >/dev/null 2>&1 </dev/null & )
  for _ in $(seq 1 30); do
    sacctmgr -i show cluster >/dev/null 2>&1 && break
    sleep 1
  done
  sacctmgr -i add cluster anvil >/dev/null 2>&1 || true
  sacctmgr -i add account anvil Description=anvil Organization=anvil >/dev/null 2>&1 || true
  sacctmgr -i add user root DefaultAccount=anvil >/dev/null 2>&1 || true
fi

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
  if [[ "${ENFORCEMENT}" -eq 1 ]]; then
    echo "==> resource enforcement: cgroup (RAM, swap, cores), so a job that exceeds what"
    echo "    it requested comes back OUT_OF_MEMORY instead of completing"
  else
    echo "==> resource enforcement: none (${TASKPLUGIN}), so a job may use more than it"
    echo "    requested and still complete. Needs a writable /sys/fs/cgroup."
  fi

  # `sinfo` prints the declared configuration, not what is running: SlurmdTimeout=0 keeps
  # slurmctld from ever marking an unreachable node DOWN, so nodes read `idle` with or
  # without a live slurmd behind them. Count the daemons instead of trusting that column.
  # -x: an unanchored match counts slurmdbd too, and would hide four dead slurmd behind
  # the one daemon that is not a node.
  live_slurmd="$(pgrep -xc slurmd 2>/dev/null || true)"
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
