#!/usr/bin/env bash
# Single-node kubeadm + GPU Operator (host driver unchanged) + KAI NvFractions
# + embed-index pods. Run in a terminal on this host:
#   cd /home/ubuntu/embedding-indexing-poc && bash scripts/install-k8s-runai.sh
# The NVIDIA driver must stay 615.71.10. This script does not install or upgrade it.
# It does not commit, and it does not run the 1M benchmark.
set -euo pipefail
set +o xtrace
set +o verbose

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER_REQUIRED="615.71.10"
POD_CIDR="10.244.0.0/16"
FLANNEL_MANIFEST="https://github.com/flannel-io/flannel/releases/download/v0.28.10/kube-flannel.yml"
CONTAINERD_CONFIG="/etc/containerd/config.toml"
CONTAINERD_BACKUP="/etc/containerd/config.toml.pre-k8s"
KUBECONFIG_DEST="/home/ubuntu/.kube/config"
LOCAL_MANIFEST="${ROOT}/deploy/k8s/embed-and-index.local.yaml"
SUDO_KEEPALIVE=""

step() {
  echo
  echo "==> $*"
}

driver_version() {
  nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]'
}

assert_driver() {
  local version
  version="$(driver_version)"
  if [[ "${version}" != "${DRIVER_REQUIRED}" ]]; then
    echo "NVIDIA driver is ${version}; it must stay ${DRIVER_REQUIRED}. Stopping." >&2
    exit 1
  fi
  echo "NVIDIA driver ${version}"
}

hold_driver_packages() {
  local pkgs=()
  local package version
  while IFS=$'\t' read -r package version; do
    if [[ "${version}" == "${DRIVER_REQUIRED}"* ]]; then
      pkgs+=("${package}")
    fi
  done < <(dpkg-query -W -f '${Package}\t${Version}\n')
  if ((${#pkgs[@]})); then
    sudo apt-mark hold "${pkgs[@]}" >/dev/null
    echo "Held ${#pkgs[@]} NVIDIA driver packages at ${DRIVER_REQUIRED}"
  fi
}

load_env() {
  set +o xtrace
  if [[ ! -f "${ROOT}/.env" ]]; then
    echo "Missing ${ROOT}/.env" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
  local key missing=()
  for key in NGC_API_KEY AWS_REGION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY S3_BUCKET; do
    if [[ -z "${!key:-}" ]]; then
      missing+=("${key}")
    fi
  done
  if ((${#missing[@]})); then
    echo "Missing or empty .env keys: ${missing[*]}" >&2
    exit 1
  fi
}

wait_containerd() {
  local i
  for i in $(seq 1 30); do
    if sudo systemctl is-active --quiet containerd && sudo test -S /run/containerd/containerd.sock; then
      return 0
    fi
    sleep 1
  done
  echo "containerd did not become active" >&2
  return 1
}

docker_health() {
  local name="$1"
  docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "${name}" 2>/dev/null || echo missing
}

recover_docker() {
  local name state
  for name in opensearch localstack; do
    state="$(docker inspect -f '{{.State.Status}}' "${name}" 2>/dev/null || echo missing)"
    if [[ "${state}" == "missing" ]]; then
      echo "${name} container is missing" >&2
      return 1
    fi
    if [[ "${state}" != "running" ]]; then
      echo "Starting ${name}"
      docker start "${name}" >/dev/null
    fi
  done
  local i ok st
  for i in $(seq 1 90); do
    ok=1
    for name in opensearch localstack; do
      st="$(docker_health "${name}")"
      if [[ "${st}" != "healthy" ]]; then
        ok=0
      fi
    done
    if [[ "${ok}" -eq 1 ]]; then
      echo "opensearch $(docker_health opensearch); localstack $(docker_health localstack)"
      return 0
    fi
    sleep 5
  done
  echo "opensearch ($(docker_health opensearch)) or localstack ($(docker_health localstack)) is not healthy" >&2
  docker ps -a --filter name=opensearch --filter name=localstack
  return 1
}

on_docker_unhealthy() {
  docker start opensearch localstack >/dev/null 2>&1 || true
  if [[ ! -f /etc/kubernetes/admin.conf && -f "${CONTAINERD_BACKUP}" ]]; then
    echo "Kubernetes is not initialized. Restoring ${CONTAINERD_BACKUP}." >&2
    sudo cp -a "${CONTAINERD_BACKUP}" "${CONTAINERD_CONFIG}"
    sudo systemctl restart containerd || true
    docker start opensearch localstack >/dev/null 2>&1 || true
  else
    echo "Leaving containerd config in place so an existing kubelet can keep using it." >&2
  fi
  return 1
}

# Patch a copy of the live containerd config. Docker's file disables CRI and
# keeps the SOCI snapshotter; containerd 2.3 migrates that file to version 4
# at startup. Do not replace it with `containerd config default`.
write_containerd_config() {
  local src="$1" backup="$2" dest="$3"
  python3 - "${src}" "${backup}" "${dest}" <<'PY'
import re
import sys
from pathlib import Path

src, backup, dest = sys.argv[1:]
text = Path(src).read_text()
backup_text = Path(backup).read_text() if Path(backup).is_file() else text

def enable_cri(raw: str) -> str:
    match = re.search(r"(?m)^disabled_plugins\s*=\s*\[([^\]]*)\]\s*$", raw)
    if not match:
        raise SystemExit("containerd config has no disabled_plugins line; refusing to rewrite it")
    kept = []
    for item in match.group(1).split(","):
        name = item.strip().strip("'\"")
        if name and name not in ("cri", "io.containerd.grpc.v1.cri"):
            kept.append(name)
    if kept:
        rendered = "disabled_plugins = [" + ", ".join(f"'{name}'" for name in kept) + "]"
    else:
        rendered = "disabled_plugins = []"
    return raw[: match.start()] + rendered + raw[match.end() :]

def soci_slice(raw: str) -> str:
    for marker in ("# SOCI snapshotter", "[proxy_plugins]"):
        index = raw.find(marker)
        if index >= 0:
            return raw[index:].rstrip() + "\n"
    return ""

text = enable_cri(text)
text = re.sub(
    r"(?m)^([ \t]*)SystemdCgroup[ \t]*=[ \t]*false[ \t]*$",
    r"\1SystemdCgroup = true",
    text,
)

needs_systemd = "SystemdCgroup = true" not in text
needs_nri = "io.containerd.nri.v1.nri" not in text
if needs_systemd or needs_nri:
    text = text.rstrip() + "\n\n# embedding-indexing-poc: kubelet CRI + NRI. Leave the SOCI snapshotter blocks above intact.\n"
    if needs_systemd:
        text += """[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
  SystemdCgroup = true

"""
    if needs_nri:
        text += """[plugins."io.containerd.nri.v1.nri"]
  disable = false
  disable_connections = false
  socket_path = "/var/run/nri/nri.sock"
"""

has_soci = "enable_remote_snapshot_annotations" in text and (
    "[proxy_plugins.soci]" in text or "proxy_plugins.soci" in text
)
if not has_soci:
    extra = soci_slice(backup_text)
    if "enable_remote_snapshot_annotations" not in extra:
        raise SystemExit("SOCI snapshotter settings are missing from the live config and its backup")
    text = text.rstrip() + "\n\n" + extra

if not text.endswith("\n"):
    text += "\n"
Path(dest).write_text(text)
PY
}

assert_containerd_dump() {
  local dump="$1"
  python3 - "${dump}" <<'PY'
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
problems = []
if "disabled_plugins = []" not in text:
    problems.append("CRI is still disabled")
if "SystemdCgroup = true" not in text:
    problems.append("SystemdCgroup is not true")
if "/var/run/nri/nri.sock" not in text:
    problems.append("NRI socket path is missing")
nri = text.find("io.containerd.nri.v1.nri")
if nri < 0:
    problems.append("NRI plugin is missing")
else:
    block = text[nri : nri + 500]
    if "disable = true" in block or "disable_connections = true" in block:
        problems.append("NRI plugin is disabled")
if "enable_remote_snapshot_annotations" not in text or "soci" not in text:
    problems.append("SOCI snapshotter settings were dropped")
if problems:
    raise SystemExit("containerd config check failed: " + "; ".join(problems))
PY
}

ensure_nri_socket() {
  local i
  for i in $(seq 1 20); do
    if sudo test -S /var/run/nri/nri.sock || sudo test -S /run/nri/nri.sock; then
      echo "NRI socket present"
      return 0
    fi
    sleep 1
  done
  echo "NRI socket /var/run/nri/nri.sock does not exist" >&2
  sudo journalctl -u containerd -n 60 --no-pager >&2 || true
  return 1
}

ensure_containerd() {
  if [[ ! -f "${CONTAINERD_BACKUP}" ]]; then
    sudo cp -a "${CONTAINERD_CONFIG}" "${CONTAINERD_BACKUP}"
  fi
  local tmp dump
  tmp="$(mktemp)"
  dump="$(mktemp)"
  write_containerd_config "${CONTAINERD_CONFIG}" "${CONTAINERD_BACKUP}" "${tmp}"
  if ! containerd --config "${tmp}" config dump >"${dump}" 2>"${dump}.err"; then
    echo "containerd rejected the patched config" >&2
    cat "${dump}.err" >&2
    exit 1
  fi
  assert_containerd_dump "${dump}"
  local changed=0
  if cmp -s "${tmp}" "${CONTAINERD_CONFIG}"; then
    echo "containerd config already enables CRI, SystemdCgroup, and NRI"
  else
    diff_lines="$(diff -u "${CONTAINERD_CONFIG}" "${tmp}" | wc -l || true)"
    if [[ "${diff_lines}" -lt 80 ]]; then
      diff -u "${CONTAINERD_CONFIG}" "${tmp}" || true
    fi
    sudo cp "${tmp}" "${CONTAINERD_CONFIG}"
    sudo chmod 644 "${CONTAINERD_CONFIG}"
    changed=1
  fi
  sudo mkdir -p /etc/nri/conf.d /opt/nri/plugins /run/nri
  restart_containerd_if_needed() {
    sudo systemctl restart containerd
    wait_containerd || true
    recover_docker || on_docker_unhealthy
  }
  if [[ "${changed}" -eq 1 ]]; then
    echo "Restarting containerd"
    restart_containerd_if_needed
  fi
  if ! sudo test -S /var/run/nri/nri.sock && ! sudo test -S /run/nri/nri.sock; then
    echo "Restarting containerd so the NRI socket is created"
    restart_containerd_if_needed
  fi
  ensure_nri_socket
  rm -f "${tmp}" "${dump}" "${dump}.err"
}

prepare_host_network() {
  sudo modprobe overlay
  sudo modprobe br_netfilter
  sudo tee /etc/modules-load.d/k8s.conf >/dev/null <<'EOF'
overlay
br_netfilter
EOF
  # Do not change vm.max_map_count. OpenSearch is already above 262144.
  sudo tee /etc/sysctl.d/99-kubernetes-cri.conf >/dev/null <<'EOF'
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF
  sudo sysctl --system >/dev/null
}

install_kubeconfig() {
  sudo install -d -m 700 -o ubuntu -g ubuntu /home/ubuntu/.kube
  sudo cp /etc/kubernetes/admin.conf "${KUBECONFIG_DEST}"
  sudo chown ubuntu:ubuntu "${KUBECONFIG_DEST}"
  sudo chmod 600 "${KUBECONFIG_DEST}"
  export KUBECONFIG="${KUBECONFIG_DEST}"
}

init_kubernetes() {
  prepare_host_network
  if [[ ! -f /etc/kubernetes/admin.conf ]]; then
    sudo kubeadm init \
      --pod-network-cidr="${POD_CIDR}" \
      --cri-socket=unix:///run/containerd/containerd.sock
  else
    echo "kubeadm already initialized"
  fi
  install_kubeconfig
  kubectl apply -f "${FLANNEL_MANIFEST}"
  kubectl -n kube-flannel rollout status daemonset/kube-flannel-ds --timeout=180s
  kubectl taint nodes --all node-role.kubernetes.io/control-plane:NoSchedule- || true
  kubectl taint nodes --all node-role.kubernetes.io/master:NoSchedule- || true
  kubectl wait --for=condition=Ready node --all --timeout=300s
}

label_privileged() {
  local ns
  for ns in "$@"; do
    kubectl create namespace "${ns}" --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace "${ns}" \
      pod-security.kubernetes.io/enforce=privileged \
      pod-security.kubernetes.io/warn=privileged \
      --overwrite >/dev/null
  done
}

install_gpu_operator() {
  label_privileged gpu-operator
  if ! helm repo add nvidia https://helm.ngc.nvidia.com/nvidia --force-update >/dev/null; then
    # Some NGC helm endpoints want the API key. Do not print it.
    printf '%s' "${NGC_API_KEY}" | helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
      --username '$oauthtoken' \
      --password-stdin \
      --force-update >/dev/null
  fi
  helm repo update nvidia >/dev/null
  if helm -n gpu-operator status gpu-operator >/dev/null 2>&1; then
    echo "gpu-operator release already installed"
  else
    helm install gpu-operator nvidia/gpu-operator \
      --version v26.7.1 \
      -n gpu-operator --create-namespace \
      --set driver.enabled=false \
      --set cdi.nriPluginEnabled=false
  fi
  assert_driver
  local i running alloc
  for i in $(seq 1 150); do
    assert_driver
    if kubectl -n gpu-operator get daemonset -o name 2>/dev/null | grep -q 'nvidia-driver'; then
      echo "GPU Operator created a driver DaemonSet. driver.enabled=false is required." >&2
      kubectl -n gpu-operator get daemonset >&2
      exit 1
    fi
    running="$(kubectl -n gpu-operator get pods --no-headers 2>/dev/null | awk '/device-plugin/ && $3=="Running" {c++} END {print c+0}')"
    alloc="$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || true)"
    echo "device-plugin Running=${running:-0} nvidia.com/gpu=${alloc:-none}"
    if [[ "${running:-0}" -ge 1 && -n "${alloc:-}" && "${alloc}" != "0" ]]; then
      return 0
    fi
    sleep 10
  done
  echo "GPU device plugin did not become Running with nvidia.com/gpu" >&2
  kubectl -n gpu-operator get pods >&2 || true
  return 1
}

install_kai() {
  label_privileged kai-scheduler
  local log rc
  log="$(mktemp)"
  set +e
  helm upgrade -i kai-scheduler \
    oci://ghcr.io/kai-scheduler/kai-scheduler/kai-scheduler \
    --version v0.18.5 \
    -n kai-scheduler --create-namespace \
    --set global.nvFractions.set=true \
    >"${log}" 2>&1
  rc=$?
  set -e
  cat "${log}"
  if [[ "${rc}" -ne 0 ]]; then
    if grep -Eqi 'gpuSharingMode' "${log}"; then
      echo "The chart rejected gpuSharingMode. Not setting it." >&2
      rm -f "${log}"
      exit 1
    fi
    if grep -Eqi 'not found|unknown version|improper constraint|failed to (pull|get)|no chart version|manifest unknown|could not (be )?found' "${log}"; then
      echo "Helm tag v0.18.5 was rejected; trying 0.18.5"
      helm upgrade -i kai-scheduler \
        oci://ghcr.io/kai-scheduler/kai-scheduler/kai-scheduler \
        --version 0.18.5 \
        -n kai-scheduler --create-namespace \
        --set global.nvFractions.set=true
    else
      rm -f "${log}"
      exit "${rc}"
    fi
  fi
  rm -f "${log}"
  local deploy
  for deploy in $(kubectl -n kai-scheduler get deploy -o name); do
    kubectl -n kai-scheduler rollout status "${deploy}" --timeout=15m
  done
}

wait_fractioning() {
  local i cfg node
  for i in $(seq 1 120); do
    cfg="$(kubectl get gpufractioningconfig default -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
    node="$(kubectl get nodes -o jsonpath='{.items[0].status.conditions[?(@.type=="gpu-fractioning.nvidia.com/Ready")].status}' 2>/dev/null || true)"
    echo "GpuFractioningConfig Ready=${cfg:-missing} node gpu-fractioning.nvidia.com/Ready=${node:-missing}"
    if [[ "${cfg}" == "True" && "${node}" == "True" ]]; then
      return 0
    fi
    sleep 10
  done
  echo "GPU fractioning did not become Ready" >&2
  kubectl get gpufractioningconfig default -o yaml >&2 || true
  kubectl api-resources | awk 'BEGIN{IGNORECASE=1} /fraction/ {print}' >&2 || true
  return 1
}

ensure_default_queue() {
  if kubectl get queue.scheduling.run.ai default-queue >/dev/null 2>&1; then
    echo "queue default-queue exists"
    return 0
  fi
  kubectl apply -f - <<'EOF'
apiVersion: scheduling.run.ai/v2
kind: Queue
metadata:
  name: default-parent-queue
spec:
  resources:
    cpu:
      quota: 0
      limit: -1
      overQuotaWeight: 1
    gpu:
      quota: 0
      limit: -1
      overQuotaWeight: 1
    memory:
      quota: 0
      limit: -1
      overQuotaWeight: 1
---
apiVersion: scheduling.run.ai/v2
kind: Queue
metadata:
  name: default-queue
spec:
  parentQueue: default-parent-queue
  resources:
    cpu:
      quota: 0
      limit: -1
      overQuotaWeight: 1
    gpu:
      quota: 0
      limit: -1
      overQuotaWeight: 1
    memory:
      quota: 0
      limit: -1
      overQuotaWeight: 1
EOF
}

node_internal_ip() {
  kubectl get nodes -o json | python3 -c '
import json, subprocess, sys
node = json.load(sys.stdin)["items"][0]
addrs = [item["address"] for item in node["status"]["addresses"] if item["type"] == "InternalIP"]
route = subprocess.check_output(["ip", "-4", "route", "get", "1.1.1.1"], text=True).split()
src = route[route.index("src") + 1] if "src" in route else ""

def dockerish(ip: str) -> bool:
    return ip.startswith("172.17.") or ip.startswith("172.18.") or ip.startswith("127.")

if src in addrs and not dockerish(src):
    print(src)
elif any(not dockerish(ip) for ip in addrs):
    print(next(ip for ip in addrs if not dockerish(ip)))
elif addrs:
    print(addrs[0])
else:
    raise SystemExit("node has no InternalIP")
'
}

edit_manifest_file() {
  local path="$1" node_ip="$2"
  if [[ "${path}" != *.local.yaml ]]; then
    echo "Refusing to edit ${path}" >&2
    exit 1
  fi
  python3 - "${path}" "${node_ip}" <<'PY'
import re
import sys
from pathlib import Path

path, node_ip = sys.argv[1:]
if not re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", node_ip):
    raise SystemExit(f"refusing node IP {node_ip!r}")
text = Path(path).read_text()
before = re.findall(r"(?m)^\s*(?:gpu-fraction:|nvidia.com/container\.).*$", text)
endpoint = f"http://{node_ip}:4566"

def replace_once(doc: str, old: str, new: str, label: str) -> str:
    count = doc.count(old)
    if count != 1:
        raise SystemExit(f"{label} matched {count} times")
    return doc.replace(old, new, 1)

def edit_nim(doc: str) -> str:
    doc = replace_once(
        doc,
        "      schedulerName: kai-scheduler\n      containers:\n        - name: nim\n",
        "      schedulerName: kai-scheduler\n      imagePullSecrets:\n        - name: nvcr-cred\n      containers:\n        - name: nim\n",
        "nim scheduler",
    )
    doc = replace_once(
        doc,
        "          ports:\n            - containerPort: 8000\n",
        "          ports:\n            - containerPort: 8000\n              hostPort: 8000\n",
        "nim port",
    )
    doc = replace_once(
        doc,
        "          resources:\n            requests:\n              cpu: \"8\"\n              memory: 32Gi\n            limits:\n              memory: 64Gi\n",
        "          resources:\n            requests:\n              cpu: \"1\"\n              memory: 8Gi\n            limits:\n              cpu: \"4\"\n              memory: 24Gi\n",
        "nim resources",
    )
    return doc

def edit_builder(doc: str) -> str:
    doc = replace_once(
        doc,
        "          ports:\n            - containerPort: 1025\n",
        "          ports:\n            - containerPort: 1025\n              hostPort: 1025\n",
        "builder port",
    )
    needle = (
        "            - name: S3_BUCKET\n"
        "              valueFrom:\n"
        "                secretKeyRef:\n"
        "                  name: s3-creds\n"
        "                  key: S3_BUCKET\n"
    )
    doc = replace_once(
        doc,
        needle,
        needle
        + "            - name: AWS_ENDPOINT_URL\n"
        + f'              value: "{endpoint}"\n'
        + "            - name: S3_ENDPOINT\n"
        + f'              value: "{endpoint}"\n'
        + "            - name: S3_ENDPOINT_URL\n"
        + f'              value: "{endpoint}"\n'
        + "            - name: AWS_EC2_METADATA_DISABLED\n"
        + '              value: "true"\n'
        + "            - name: AWS_REQUEST_CHECKSUM_CALCULATION\n"
        + '              value: "WHEN_REQUIRED"\n'
        + "            - name: AWS_RESPONSE_CHECKSUM_VALIDATION\n"
        + '              value: "WHEN_REQUIRED"\n',
        "builder env",
    )
    doc = replace_once(
        doc,
        "          resources:\n            requests:\n              cpu: \"4\"\n              memory: 8Gi\n            limits:\n              memory: 16Gi\n",
        "          resources:\n            requests:\n              cpu: 500m\n              memory: 2Gi\n            limits:\n              cpu: \"2\"\n              memory: 8Gi\n",
        "builder resources",
    )
    return doc

def is_container(doc: str, container: str) -> bool:
    return "kind: Deployment" in doc and bool(
        re.search(rf"(?m)^[ \t]*- name: {re.escape(container)}[ \t]*$", doc)
    )

parts = text.split("\n---\n")
found = {"nim": 0, "builder": 0}
edited = []
for part in parts:
    if is_container(part, "nim"):
        part = edit_nim(part)
        found["nim"] += 1
    elif is_container(part, "builder"):
        part = edit_builder(part)
        found["builder"] += 1
    edited.append(part)
if found != {"nim": 1, "builder": 1}:
    raise SystemExit(f"unexpected deployments: {found}")
updated = "\n---\n".join(edited)
after = re.findall(r"(?m)^\s*(?:gpu-fraction:|nvidia.com/container\.).*$", updated)
if before != after:
    raise SystemExit("GPU fraction annotations changed")
for required in (
    'cpu: "1"',
    'cpu: "4"',
    "memory: 8Gi",
    "memory: 24Gi",
    "cpu: 500m",
    'cpu: "2"',
    "memory: 2Gi",
    "hostPort: 8000",
    "hostPort: 1025",
    "name: nvcr-cred",
    f'value: "{endpoint}"',
    'AWS_EC2_METADATA_DISABLED',
):
    if required not in updated:
        raise SystemExit(f"manifest is missing {required}")
Path(path).write_text(updated)
print(f"edited {path}")
PY
}

write_local_manifest() {
  local node_ip="$1"
  if [[ "${LOCAL_MANIFEST}" == "${ROOT}/deploy/k8s/embed-and-index.yaml" ]]; then
    echo "Refusing to write the committed manifest" >&2
    exit 1
  fi
  (
    cd "${ROOT}"
    .venv/bin/python scripts/pipeline_config.py manifest --detect --write "${LOCAL_MANIFEST}"
  )
  edit_manifest_file "${LOCAL_MANIFEST}" "${node_ip}"
}

apply_secrets() {
  set +o xtrace
  label_privileged embed-index
  "${ROOT}/.venv/bin/python" - <<'PY'
import base64
import json
import os
import subprocess

namespace = "embed-index"

def apply(document: str) -> None:
    completed = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=document,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

def generic(name: str, data: dict) -> None:
    lines = [
        "apiVersion: v1",
        "kind: Secret",
        "type: Opaque",
        "metadata:",
        f"  name: {name}",
        f"  namespace: {namespace}",
        "stringData:",
    ]
    for key, value in data.items():
        lines.append(f"  {key}: {json.dumps(value)}")
    apply("\n".join(lines) + "\n")

key = os.environ["NGC_API_KEY"]
generic("ngc-api", {"NGC_API_KEY": key})
generic(
    "s3-creds",
    {
        "AWS_REGION": os.environ["AWS_REGION"],
        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
        "S3_BUCKET": os.environ["S3_BUCKET"],
    },
)
auth = base64.b64encode(f"$oauthtoken:{key}".encode()).decode()
dockerconfig = {
    "auths": {
        "nvcr.io": {
            "username": "$oauthtoken",
            "password": key,
            "auth": auth,
        }
    }
}
apply(
    "\n".join(
        [
            "apiVersion: v1",
            "kind: Secret",
            "type: kubernetes.io/dockerconfigjson",
            "metadata:",
            "  name: nvcr-cred",
            f"  namespace: {namespace}",
            "stringData:",
            "  .dockerconfigjson: " + json.dumps(json.dumps(dockerconfig)),
        ]
    )
    + "\n"
)
PY
}

free_host_ports() {
  if docker ps -a --format '{{.Names}}' | grep -qx 'llama-nemotron-embed-vl-1b-v2'; then
    timeout 30 docker rm -f llama-nemotron-embed-vl-1b-v2
  fi
  if docker ps -a --format '{{.Names}}' | grep -qx 'remote-index-builder'; then
    timeout 30 docker rm -f remote-index-builder
  fi
}

wait_workloads() {
  local deadline=$((SECONDS + 2400))
  local nim builder code
  while ((SECONDS < deadline)); do
    nim="$(kubectl -n embed-index get pods -l app=nim-embed -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)"
    builder="$(kubectl -n embed-index get pods -l app=remote-index-builder -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)"
    code="$(curl --noproxy '*' -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 5 http://127.0.0.1:8000/v1/health/ready || true)"
    echo "nim=${nim:-none} builder=${builder:-none} :8000=${code:-none}"
    if [[ "${nim}" == "Running" && "${builder}" == "Running" && "${code}" == "200" ]]; then
      return 0
    fi
    sleep 20
  done
  echo "embed-index pods or NIM health did not become ready within 40 minutes" >&2
  kubectl -n embed-index get pods -o wide >&2 || true
  kubectl -n embed-index get events --sort-by='.lastTimestamp' >&2 || true
  return 1
}

wait_tcp() {
  local port="$1"
  local tries="$2"
  local i
  for i in $(seq 1 "${tries}"); do
    if python3 -c 'import socket, sys; socket.create_connection(("127.0.0.1", int(sys.argv[1])), 2).close()' "${port}" 2>/dev/null; then
      return 0
    fi
    sleep 5
  done
  echo "timed out waiting for 127.0.0.1:${port}" >&2
  return 1
}

builder_url_for_opensearch() {
  local bridge_gw os_gw candidate
  bridge_gw="$(docker network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}')"
  os_gw="$(docker inspect opensearch --format '{{range $name, $net := .NetworkSettings.Networks}}{{$net.Gateway}}{{"\n"}}{{end}}' | awk 'NF {print; exit}')"
  for candidate in "${bridge_gw}" "${os_gw}"; do
    [[ -n "${candidate}" ]] || continue
    if timeout 5 docker exec opensearch bash -c "echo >/dev/tcp/${candidate}/1025" >/dev/null 2>&1; then
      echo "http://${candidate}:1025"
      return 0
    fi
  done
  echo "OpenSearch cannot open port 1025 via docker bridge gateway ${bridge_gw} or its network gateway ${os_gw}" >&2
  return 1
}

configure_opensearch_endpoint() {
  local builder_url="$1"
  set +o xtrace
  (
    set -a
    # shellcheck disable=SC1091
    source "${ROOT}/.env"
    set +a
    export OPENSEARCH_URL="${OPENSEARCH_URL:-http://127.0.0.1:9200}"
    export REMOTE_INDEX_BUILDER_URL="${builder_url}"
    # Loopback is the OpenSearch container itself. LocalStack is on the host.
    case "${S3_ENDPOINT:-}${AWS_ENDPOINT_URL:-}" in
      *127.0.0.1*|*localhost*)
        host="${builder_url#http://}"
        host="${host%%:*}"
        export S3_ENDPOINT="http://${host}:4566"
        export AWS_ENDPOINT_URL="http://${host}:4566"
        ;;
    esac
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/configure_opensearch.py"
  )
}

print_gpu_memory_limits() {
  echo "pod spec NVIDIA_GPU_MEMORY_LIMIT:"
  kubectl -n embed-index get pods -l app=nim-embed -o jsonpath='nim {.items[0].spec.containers[0].env[?(@.name=="NVIDIA_GPU_MEMORY_LIMIT")].value}{"\n"}'
  kubectl -n embed-index get pods -l app=remote-index-builder -o jsonpath='builder {.items[0].spec.containers[0].env[?(@.name=="NVIDIA_GPU_MEMORY_LIMIT")].value}{"\n"}'
  "${ROOT}/.venv/bin/python" - <<'PY' || echo "runtime NVIDIA_GPU_MEMORY_LIMIT: unavailable" >&2
import json
import subprocess

raw = subprocess.check_output(
    ["sudo", "ctr", "-n", "k8s.io", "containers", "ls", "-q"],
    text=True,
)
for container_id in [line.strip() for line in raw.splitlines() if line.strip()]:
    raw = subprocess.check_output(
        ["sudo", "ctr", "-n", "k8s.io", "containers", "info", container_id],
        text=True,
    )
    data = json.loads(raw)
    labels = data.get("Labels") or {}
    if labels.get("io.kubernetes.pod.namespace") != "embed-index":
        continue
    name = labels.get("io.kubernetes.container.name", "")
    if name not in ("nim", "builder"):
        continue
    spec = data.get("Spec") or {}
    if isinstance(spec, str):
        spec = json.loads(spec)
    process = spec.get("process") or spec.get("Process") or {}
    env = process.get("env") or process.get("Env") or []
    value = next((item.split("=", 1)[1] for item in env if item.startswith("NVIDIA_GPU_MEMORY_LIMIT=")), "")
    print(f"runtime {name} NVIDIA_GPU_MEMORY_LIMIT={value}")
PY
}

http_code() {
  local code
  code="$(curl --noproxy '*' -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 5 "$1" 2>/dev/null || true)"
  if [[ -z "${code}" || "${code}" == "000" ]]; then
    echo down
  else
    echo "${code}"
  fi
}

report() {
  set +e
  echo
  echo "======== install report ========"
  echo "sudo: ok"
  echo "driver: $(driver_version 2>/dev/null || echo unread)"
  echo "containerd: $(containerd --version 2>/dev/null || echo unread)"
  if sudo test -S /var/run/nri/nri.sock || sudo test -S /run/nri/nri.sock; then
    echo "NRI socket: present"
  else
    echo "NRI socket: missing"
  fi
  if [[ -f "${KUBECONFIG_DEST}" ]]; then
    export KUBECONFIG="${KUBECONFIG_DEST}"
    echo "nodes:"
    kubectl get nodes -o wide 2>/dev/null || true
    echo "helm releases:"
    helm list -A 2>/dev/null || true
    echo -n "GpuFractioningConfig Ready="
    kubectl get gpufractioningconfig default -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true
    echo
    echo -n "node gpu-fractioning.nvidia.com/Ready="
    kubectl get nodes -o jsonpath='{.items[0].status.conditions[?(@.type=="gpu-fractioning.nvidia.com/Ready")].status}' 2>/dev/null || true
    echo
    echo "embed-index pods:"
    kubectl -n embed-index get pods -o wide 2>/dev/null || true
  else
    echo "kubernetes: not initialized"
  fi
  echo ":8000 $(http_code http://127.0.0.1:8000/v1/health/ready)"
  echo ":9200 $(http_code http://127.0.0.1:9200)"
  python3 -c 'import socket; socket.create_connection(("127.0.0.1",1025),3).close(); print(":1025 open")' 2>/dev/null || echo ":1025 down"
  echo "opensearch container: $(docker_health opensearch)"
  echo "localstack container: $(docker_health localstack)"
  echo "======== end report ========"
  return 0
}

cleanup() {
  local rc=$?
  trap - EXIT
  if [[ -n "${SUDO_KEEPALIVE}" ]]; then
    kill "${SUDO_KEEPALIVE}" 2>/dev/null || true
  fi
  report || true
  exit "${rc}"
}

main() {
  trap cleanup EXIT
  cd "${ROOT}"
  if [[ "$(id -un)" != "ubuntu" ]]; then
    echo "Run this as ubuntu, not $(id -un)." >&2
    exit 1
  fi
  step "Checking sudo and the NVIDIA driver"
  if ! sudo -n true 2>/dev/null; then
    echo "sudo needs your password once. The script keeps that credential refreshed."
    sudo -v
  fi
  (while true; do sudo -n true; sleep 50; done) &
  SUDO_KEEPALIVE=$!
  assert_driver
  hold_driver_packages
  load_env
  if [[ "$(docker_health opensearch)" != "healthy" || "$(docker_health localstack)" != "healthy" ]]; then
    echo "opensearch and localstack must be healthy before containerd is restarted." >&2
    exit 1
  fi

  step "Configuring containerd for kubelet"
  ensure_containerd
  assert_driver

  step "Initializing Kubernetes and flannel"
  init_kubernetes

  step "Installing the GPU Operator without a driver"
  install_gpu_operator
  step "Rechecking containerd after the GPU Operator"
  ensure_containerd
  assert_driver

  step "Installing KAI Scheduler NvFractions"
  install_kai
  wait_fractioning

  step "Writing the local embed-index manifest"
  local node_ip
  node_ip="$(node_internal_ip)"
  echo "node InternalIP ${node_ip}"
  write_local_manifest "${node_ip}"

  step "Creating secrets and the KAI queue"
  apply_secrets
  ensure_default_queue
  step "Releasing host port 1025 and applying the manifest"
  free_host_ports
  kubectl apply -f "${LOCAL_MANIFEST}"

  step "Waiting for NIM and the builder"
  wait_workloads
  wait_tcp 1025 36
  local builder_url
  builder_url="$(builder_url_for_opensearch)"
  echo "OpenSearch builder endpoint ${builder_url}"
  configure_opensearch_endpoint "${builder_url}"
  print_gpu_memory_limits
  assert_driver
  step "Install finished"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
