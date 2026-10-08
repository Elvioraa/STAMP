#!/usr/bin/env bash
# STAMP x DOMA V2R2: fail-closed single-GPU training and evaluation runner.
#
# This script intentionally does not change source YAML files.  Runtime copies,
# checkpoints, logs, hashes, and state live below opencood/logs/STAMP_DOMA.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
RUNNER_PATH="${SCRIPT_DIR}/run_stamp_doma_full.sh"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${STAMP_DOMA_GPU:-0}"
CONFIG_ROOT="opencood/hypes_yaml/opv2v/doma_protocol_stamp_v2r2"
LOCK_FILE="${CONFIG_ROOT}/protocol_lock.yaml"
OUTPUT_ROOT="opencood/logs/STAMP_DOMA"
STATE_ROOT="${OUTPUT_ROOT}/state"
ARCHIVE_ROOT="${OUTPUT_ROOT}/archive"
RESULT_ROOT="${OUTPUT_ROOT}/results"
META_FILE="${OUTPUT_ROOT}/experiment.meta"

TRAINING_ASSIGNMENT="opencood/modality_assign/opv2v_4modality.json"
FINAL_ASSIGNMENT="opencood/modality_assign/opv2v_4modality_in_order.json"

STAGES=(L1 L2 L3 L4 P0 A1 A2 A3 A4 MERGE CAV1 CAV2 CAV3 CAV4)
TRAIN_STAGES=(L1 L2 L3 L4 P0 A1 A2 A3 A4)

declare -A STAGE_CONFIG=(
    [L1]="${CONFIG_ROOT}/local/m1/config.yaml"
    [L2]="${CONFIG_ROOT}/local/m2/config.yaml"
    [L3]="${CONFIG_ROOT}/local/m3/config.yaml"
    [L4]="${CONFIG_ROOT}/local/m4/config.yaml"
    [P0]="${CONFIG_ROOT}/protocol/config.yaml"
    [A1]="${CONFIG_ROOT}/local_adapter/convnext/m1/config.yaml"
    [A2]="${CONFIG_ROOT}/local_adapter/convnext/m2/config.yaml"
    [A3]="${CONFIG_ROOT}/local_adapter/convnext/m3/config.yaml"
    [A4]="${CONFIG_ROOT}/local_adapter/convnext/m4/config.yaml"
    [MERGE]="${CONFIG_ROOT}/final_infer/convnext/config.yaml"
    [CAV1]="${CONFIG_ROOT}/final_infer/convnext/config.yaml"
    [CAV2]="${CONFIG_ROOT}/final_infer/convnext/config.yaml"
    [CAV3]="${CONFIG_ROOT}/final_infer/convnext/config.yaml"
    [CAV4]="${CONFIG_ROOT}/final_infer/convnext/config.yaml"
)

declare -A STAGE_DIR=(
    [L1]="${OUTPUT_ROOT}/local/m1"
    [L2]="${OUTPUT_ROOT}/local/m2"
    [L3]="${OUTPUT_ROOT}/local/m3"
    [L4]="${OUTPUT_ROOT}/local/m4"
    [P0]="${OUTPUT_ROOT}/protocol"
    [A1]="${OUTPUT_ROOT}/local_adapter/convnext/m1"
    [A2]="${OUTPUT_ROOT}/local_adapter/convnext/m2"
    [A3]="${OUTPUT_ROOT}/local_adapter/convnext/m3"
    [A4]="${OUTPUT_ROOT}/local_adapter/convnext/m4"
    [MERGE]="${OUTPUT_ROOT}/final_infer/convnext"
    [CAV1]="${OUTPUT_ROOT}/final_infer/convnext/CAV1"
    [CAV2]="${OUTPUT_ROOT}/final_infer/convnext/CAV2"
    [CAV3]="${OUTPUT_ROOT}/final_infer/convnext/CAV3"
    [CAV4]="${OUTPUT_ROOT}/final_infer/convnext/CAV4"
)

log() { printf '[INFO] %s\n' "$*"; }
pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
fail() { printf '[FAIL] %s\n' "$*" >&2; }
blocker() { printf '[BLOCKER] %s\n' "$*" >&2; }

usage() {
    cat <<'EOF'
Usage:
  scripts/run_stamp_doma_full.sh check
  scripts/run_stamp_doma_full.sh run
  scripts/run_stamp_doma_full.sh resume
  scripts/run_stamp_doma_full.sh status

Environment:
  STAMP_DOMA_GPU=N   one visible physical GPU index (default: 0)
  PYTHON_BIN=python  Python executable (default: python)

Safety:
  - run never overwrites an existing experiment root.
  - resume skips only cryptographically and structurally validated COMPLETE stages.
  - interrupted/FAILED stages are archived and restarted from their template;
    optimizer/scheduler state is not claimed to be losslessly resumable.
EOF
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        fail "required command is unavailable: $1"
        return 1
    }
}

sha256_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

repo_sha() {
    git rev-parse HEAD
}

utc_now() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

state_file() {
    printf '%s/%s.state\n' "${STATE_ROOT}" "$1"
}

state_value() {
    local stage="$1" key="$2" file
    file="$(state_file "${stage}")"
    [[ -f "${file}" ]] || return 1
    awk -F= -v wanted="${key}" '$1 == wanted {sub(/^[^=]*=/, ""); print; exit}' "${file}"
}

write_state() {
    local stage="$1" status="$2" detail="${3:-}"
    mkdir -p -- "${STATE_ROOT}"
    local tmp
    tmp="$(mktemp "${STATE_ROOT}/.${stage}.XXXXXX")"
    {
        printf 'STAGE=%s\n' "${stage}"
        printf 'STATUS=%s\n' "${status}"
        printf 'UPDATED_AT=%s\n' "$(utc_now)"
        printf 'GIT_SHA=%s\n' "$(repo_sha)"
        printf 'DETAIL=%s\n' "${detail//$'\n'/ }"
    } >"${tmp}"
    mv -- "${tmp}" "$(state_file "${stage}")"
}

append_state_fields() {
    local stage="$1"
    shift
    local file tmp entry
    file="$(state_file "${stage}")"
    tmp="$(mktemp "${STATE_ROOT}/.${stage}.XXXXXX")"
    cat -- "${file}" >"${tmp}"
    for entry in "$@"; do
        printf '%s\n' "${entry}" >>"${tmp}"
    done
    mv -- "${tmp}" "${file}"
}

static_repository_preflight() {
    require_command git
    require_command "${PYTHON_BIN}"
    require_command sha256sum
    require_command tee
    require_command find
    require_command sort

    [[ -f "${LOCK_FILE}" ]] || { fail "missing protocol lock: ${LOCK_FILE}"; return 1; }
    [[ -f "opencood/tools/train.py" ]] || { fail "missing train.py"; return 1; }
    [[ -f "opencood/tools/train_adapter.py" ]] || { fail "missing train_adapter.py"; return 1; }
    [[ -f "opencood/tools/merge_model_w_adapter.py" ]] || { fail "missing merge script"; return 1; }
    [[ -f "opencood/tools/inference_heter_in_order.py" ]] || { fail "missing inference script"; return 1; }

    local tracked_dirty
    tracked_dirty="$(git status --porcelain=v1 --untracked-files=no)"
    if [[ -n "${tracked_dirty}" ]]; then
        fail "tracked worktree is dirty; formal execution requires a committed source state"
        printf '%s\n' "${tracked_dirty}" >&2
        return 1
    fi

    local config
    for config in "${STAGE_CONFIG[L1]}" "${STAGE_CONFIG[L2]}" "${STAGE_CONFIG[L3]}" \
                  "${STAGE_CONFIG[L4]}" "${STAGE_CONFIG[P0]}" "${STAGE_CONFIG[A1]}" \
                  "${STAGE_CONFIG[A2]}" "${STAGE_CONFIG[A3]}" "${STAGE_CONFIG[A4]}" \
                  "${STAGE_CONFIG[MERGE]}"; do
        [[ -f "${config}" ]] || { fail "missing stage template: ${config}"; return 1; }
    done
    pass "repository entrypoints and ten immutable YAML templates are present"
}

protocol_preflight() {
    "${PYTHON_BIN}" - "${LOCK_FILE}" <<'PY'
import math
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as stream:
    lock = yaml.safe_load(stream)

errors = []
if lock.get("schema_version") != 2:
    errors.append("protocol_lock schema_version must be 2")

def confirmed(name):
    spec = lock.get(name)
    if not isinstance(spec, dict):
        errors.append(f"{name} is missing or not a mapping")
        return None
    status = str(spec.get("status", "")).lower()
    source = str(spec.get("source", "")).lower()
    value = spec.get("value")
    if not status.startswith("confirmed"):
        errors.append(f"{name}.status is {status!r}, not confirmed from formal server evidence")
    if not source or "pending" in source or "unresolved" in source:
        errors.append(f"{name}.source does not identify resolved server command/log evidence")
    if value is None:
        errors.append(f"{name}.value is unresolved")
    return value

detection = confirmed("effective_detection_range")
communication = confirmed("effective_communication_range")
expected_detection = [-204.8, -102.4, -3.0, 204.8, 102.4, 1.0]
expected_communication = 180.0

range_arg = None
if detection is not None:
    if not isinstance(detection, (list, tuple)) or len(detection) != 6:
        errors.append("effective_detection_range.value must be [xmin,ymin,zmin,xmax,ymax,zmax]")
    else:
        try:
            values = [float(v) for v in detection]
            xmin, ymin, _zmin, xmax, ymax, _zmax = values
            if not all(math.isfinite(v) for v in values):
                raise ValueError("nonfinite")
            if not (xmin < 0 < xmax and ymin < 0 < ymax):
                errors.append("effective_detection_range must straddle zero in x and y")
            if not (math.isclose(-xmin, xmax) and math.isclose(-ymin, ymax)):
                errors.append("inference_heter_in_order.py only supports symmetric x/y ranges")
            if any(not math.isclose(actual, expected) for actual, expected in zip(values, expected_detection)):
                errors.append(
                    "effective_detection_range does not match the confirmed DOMA V2 R2 "
                    "range [-204.8,-102.4,-3,204.8,102.4,1]"
                )
            range_arg = f"{xmax:g},{ymax:g}"
        except (TypeError, ValueError):
            errors.append("effective_detection_range contains nonnumeric values")

comm_value = None
if communication is not None:
    try:
        comm_value = float(communication)
        if not math.isfinite(comm_value) or comm_value <= 0:
            raise ValueError("invalid")
        if not math.isclose(comm_value, expected_communication):
            errors.append("effective_communication_range does not match the confirmed 180 m")
    except (TypeError, ValueError):
        errors.append("effective_communication_range.value must be a positive number")

if errors:
    for error in errors:
        print(f"[BLOCKER] {error}", file=sys.stderr)
    raise SystemExit(1)

print(f"DETECTION_RANGE_ARG={range_arg}")
print(f"COMMUNICATION_RANGE={comm_value:g}")
PY
}

inference_source_preflight() {
    "${PYTHON_BIN}" - "opencood/tools/inference_heter_in_order.py" \
        "${STAGE_CONFIG[MERGE]}" "${LOCK_FILE}" <<'PY'
import re
import sys
import yaml

source_path, config_path, lock_path = sys.argv[1:]
source = open(source_path, "r", encoding="utf-8").read()
config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
lock = yaml.safe_load(open(lock_path, "r", encoding="utf-8"))
errors = []

core = config.get("fusion", {}).get("core_method")
assignment = config.get("heter", {}).get("assignment_path")
if core != "intermediateheteradapter":
    errors.append(f"FINAL fusion core is {core!r}, expected 'intermediateheteradapter'")
if assignment != "opencood/modality_assign/opv2v_4modality_in_order.json":
    errors.append(f"FINAL assignment is {assignment!r}, expected test-only in-order assignment")
if config.get("model", {}).get("core_method") != "collab_w_adapter":
    errors.append("FINAL model.core_method is not collab_w_adapter")
if config.get("model", {}).get("args", {}).get("stage") != "infer":
    errors.append("FINAL model.args.stage is not infer")

if re.search(r"\[['\"]core_method['\"]\]\s*\+=\s*['\"]infer['\"]", source):
    errors.append(
        "formal inference still appends 'infer' generically and would create "
        "intermediateheteradapterinfer"
    )
if re.search(r"assignment_path.*\.replace\(\s*['\"]\.json['\"]\s*,\s*['\"]_in_order\.json['\"]", source):
    errors.append(
        "formal inference still rewrites an already in-order assignment to "
        "*_in_order_in_order.json"
    )
if "if core_method == 'intermediateheter':" not in source or "'intermediateheterinfer'" not in source:
    errors.append("formal inference is missing the exact intermediateheter -> intermediateheterinfer mapping")
if "if not assignment_root.endswith('_in_order'):" not in source:
    errors.append("formal inference is missing the idempotent in-order assignment guard")
if "hypes, new_cav_range = _apply_inference_range(hypes, opt.range)" not in source:
    errors.append("formal inference does not apply the CLI range through resolved runtime hypes")
if "elif 'yaml_parsers' in hypes:" not in source:
    errors.append("formal inference does not refresh modality-specific runtime YAML parsers")
if "build_dataset(hypes, visualize=True, train=False)" not in source:
    errors.append("formal inference dataset construction no longer uses visualize=True, train=False")
if "collate_fn=opencood_dataset.collate_batch_test" not in source:
    errors.append("formal inference no longer uses collate_batch_test")

comm_spec = lock.get("effective_communication_range", {})
comm_value = comm_spec.get("value")
constant = re.search(r"FORMAL_COMM_RANGE\s*=\s*([0-9]+(?:\.[0-9]+)?)", source)
if comm_value is not None:
    if constant is None or "hypes['comm_range'] = FORMAL_COMM_RANGE" not in source:
        errors.append("cannot prove how formal inference applies the locked communication range")
    elif float(constant.group(1)) != float(comm_value):
        errors.append(
            f"formal inference sets comm_range={constant.group(1)}, "
            f"but protocol_lock confirms {comm_value}"
        )

snapshot_call = source.rfind("_runtime_protocol_snapshot(")
dataset_build = source.find("build_dataset(hypes, visualize=True, train=False)")
dataloader_build = source.find("data_loader = DataLoader(")
if (
    snapshot_call < 0
    or dataset_build < 0
    or dataloader_build < 0
    or not (dataset_build < snapshot_call < dataloader_build)
):
    errors.append("initialized dataset protocol is not validated before DataLoader creation")
if 'default="204.8,102.4"' not in source:
    errors.append("formal inference default range is not 204.8,102.4")

for threshold in ("0.3", "0.5", "0.7"):
    if threshold not in source:
        errors.append(f"formal inference no longer evaluates IoU {threshold}")

if errors:
    for error in errors:
        print(f"[BLOCKER] {error}", file=sys.stderr)
    raise SystemExit(1)
print("[PASS] FINAL adapter fusion, assignment, test collation, and evaluator source contracts")
PY
}

dataset_preflight() {
    "${PYTHON_BIN}" - "${REPO_ROOT}" "${TRAINING_ASSIGNMENT}" "${FINAL_ASSIGNMENT}" <<'PY'
import copy
import json
import os
import random
import sys
from pathlib import Path

try:
    import h5py
    from PIL import Image
except Exception as exc:
    raise SystemExit(f"[FAIL] dataset preflight dependency import failed: {exc}")

repo = Path(sys.argv[1]).resolve()
training_assignment_path = repo / sys.argv[2]
final_assignment_path = repo / sys.argv[3]
base_root = repo / "dataset" / "OPV2V"
hetero_root = repo / "dataset" / "OPV2V_Hetero"
errors = []
counts = {"yaml": 0, "lidar64": 0, "lidar32": 0, "camera_frames": 0, "visibility": 0}

def error(message):
    if len(errors) < 200:
        errors.append(message)

def real_file(path, label):
    path = Path(path)
    try:
        if not path.is_file():
            error(f"missing {label}: {path}")
            return False
        if path.stat().st_size <= 0:
            error(f"empty {label}: {path}")
            return False
        return True
    except OSError as exc:
        error(f"cannot stat {label} {path}: {exc}")
        return False

if not base_root.is_dir():
    error(f"base dataset directory is missing: {base_root}")
if not hetero_root.is_dir():
    error(f"heterogeneous supplement directory is missing: {hetero_root}")
if base_root.exists() and hetero_root.exists() and base_root.resolve() == hetero_root.resolve():
    error("OPV2V and OPV2V_Hetero resolve to the same directory")

try:
    training_assignment = json.load(open(training_assignment_path, "r", encoding="utf-8"))
    final_assignment = json.load(open(final_assignment_path, "r", encoding="utf-8"))
except Exception as exc:
    print(f"[FAIL] cannot load assignment JSON: {exc}", file=sys.stderr)
    raise SystemExit(1)

def check_image(path, label):
    if not real_file(path, label):
        return
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        error(f"unreadable {label} {path}: {exc}")

def check_camera_depth(cav_dir, hetero_cav_dir, timestamp):
    hdf5_path = cav_dir / f"{timestamp}_imgs.hdf5"
    if hdf5_path.is_file():
        if not real_file(hdf5_path, "camera/depth HDF5"):
            return
        try:
            with h5py.File(hdf5_path, "r") as handle:
                missing = [key for key in [f"camera{i}" for i in range(4)] + [f"depth{i}" for i in range(4)] if key not in handle]
                if missing:
                    error(f"HDF5 {hdf5_path} is missing datasets: {missing}")
        except Exception as exc:
            error(f"cannot read HDF5 {hdf5_path}: {exc}")
    else:
        for index in range(4):
            check_image(cav_dir / f"{timestamp}_camera{index}.png", "OPV2V camera")
            check_image(hetero_cav_dir / f"{timestamp}_depth{index}.png", "OPV2V_Hetero depth")
    counts["camera_frames"] += 1

def visibility_path(cav_dir, split, scenario, cav_id, timestamp):
    direct = cav_dir / f"{timestamp}_bev_visibility.png"
    if direct.is_file():
        return direct
    return base_root / "additional" / split / scenario / cav_id / f"{timestamp}_bev_visibility.png"

def scan_split(split, assignment, all_modalities):
    split_root = base_root / split
    hetero_split_root = hetero_root / split
    if not split_root.is_dir():
        error(f"missing OPV2V split: {split_root}")
        return
    if not hetero_split_root.is_dir():
        error(f"missing OPV2V_Hetero split: {hetero_split_root}")
        return
    scenarios = sorted(path.name for path in split_root.iterdir() if path.is_dir() and path.name != "additional")
    missing_assignment = sorted(set(scenarios) - set(assignment))
    if missing_assignment:
        error(f"{split} scenarios absent from its assignment: {missing_assignment}")
    used_scenarios = sorted(set(scenarios) & set(assignment))
    if not used_scenarios:
        error(f"{split} has no scenarios selected by its assignment")
        return

    for scenario in used_scenarios:
        mapping = assignment[scenario]
        for cav_id, assigned_modality in mapping.items():
            cav_dir = split_root / scenario / str(cav_id)
            hetero_cav_dir = hetero_split_root / scenario / str(cav_id)
            if not cav_dir.is_dir():
                error(f"assignment CAV directory missing: {cav_dir}")
                continue
            yaml_files = sorted(
                path for path in cav_dir.glob("*.yaml")
                if "additional" not in path.name
                and not (scenario == "2021_08_20_21_10_24" and path.stem == "000265")
            )
            if not yaml_files:
                error(f"no loader-enumerated YAML timestamps in {cav_dir}")
                continue
            for yaml_path in yaml_files:
                timestamp = yaml_path.stem
                counts["yaml"] += 1
                real_file(yaml_path, "OPV2V metadata YAML")

                modalities = {"m1", "m2", "m3", "m4"} if all_modalities else {assigned_modality}
                if "m1" in modalities:
                    lidar64 = cav_dir / f"{timestamp}.pcd"
                    if real_file(lidar64, "OPV2V 64-line LiDAR"):
                        counts["lidar64"] += 1
                if "m3" in modalities:
                    lidar32 = hetero_cav_dir / f"{timestamp}_32.pcd"
                    if real_file(lidar32, "OPV2V_Hetero 32-line LiDAR"):
                        counts["lidar32"] += 1
                if modalities.intersection({"m2", "m4"}):
                    check_camera_depth(cav_dir, hetero_cav_dir, timestamp)
                if all_modalities:
                    visibility = visibility_path(cav_dir, split, scenario, str(cav_id), timestamp)
                    if real_file(visibility, "bev_visibility"):
                        check_image(visibility, "bev_visibility")
                        counts["visibility"] += 1

    print(f"[PASS] {split} assignment-to-directory mapping: scenarios={len(used_scenarios)}")

# Training and validation can randomly reassign every selected CAV to any stage
# modality across L1-L4/P0/A1-A4, so all sensor resources are required there.
scan_split("train", training_assignment, all_modalities=True)
scan_split("validate", training_assignment, all_modalities=True)
# FINAL test uses the fixed in-order assignment; only the assigned sensor source
# is required for each CAV.
scan_split("test", final_assignment, all_modalities=False)

if errors:
    for item in errors:
        print(f"[FAIL] {item}", file=sys.stderr)
    if len(errors) >= 200:
        print("[FAIL] error output truncated after 200 entries", file=sys.stderr)
    raise SystemExit(1)

# Verify the paths produced by the actual official loader, not merely the
# directory layout or symlink names.
os.chdir(repo)
sys.path.insert(0, str(repo))
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils

config_root = repo / "opencood" / "hypes_yaml" / "opv2v" / "doma_protocol_stamp_v2r2"

def iter_records(dataset):
    for scenario in dataset.scenario_database.values():
        for cav_id, cav in scenario.items():
            for timestamp, record in cav.items():
                if timestamp == "ego":
                    continue
                yield str(cav_id), str(timestamp), record

def assert_loader_paths(label, dataset, expected_root):
    actual_root = Path(dataset.root_dir).resolve()
    if actual_root != expected_root.resolve():
        raise RuntimeError(f"{label} loader root {actual_root} != {expected_root.resolve()}")
    checked = 0
    for cav_id, timestamp, record in iter_records(dataset):
        modality = record.get("modality_name")
        lidar = Path(record["lidar"])
        normalized = lidar.as_posix()
        if modality == "m1":
            if "OPV2V_Hetero" in normalized or normalized.endswith("_32.pcd"):
                raise RuntimeError(f"m1 loader path is not base 64-line LiDAR: {lidar}")
        elif modality == "m3":
            if "OPV2V_Hetero" not in normalized or not normalized.endswith("_32.pcd"):
                raise RuntimeError(f"m3 loader path is not OPV2V_Hetero *_32.pcd: {lidar}")
        elif modality in ("m2", "m4"):
            cameras = [Path(path).as_posix() for path in record["cameras"]]
            depths = [Path(path).as_posix() for path in record["depths"]]
            if len(cameras) != 4 or any("OPV2V_Hetero" in path for path in cameras):
                raise RuntimeError(f"{modality} camera loader paths are invalid: {cameras}")
            if len(depths) != 4 or any("OPV2V_Hetero" not in path for path in depths):
                raise RuntimeError(f"{modality} depth loader paths are invalid: {depths}")
        checked += 1
    if checked == 0:
        raise RuntimeError(f"{label} loader produced no timestamp records")
    print(f"[PASS] {label} official loader paths checked: records={checked}")

random.seed(303)
for modality in ("m1", "m2", "m3", "m4"):
    config = yaml_utils.load_yaml(str(config_root / "local" / modality / "config.yaml"))
    train_dataset = build_dataset(config, visualize=False, train=True)
    assert_loader_paths(f"{modality} train", train_dataset, base_root / "train")
    validate_dataset = build_dataset(config, visualize=False, train=False)
    assert_loader_paths(f"{modality} validate", validate_dataset, base_root / "validate")

final_hypes = yaml_utils.load_yaml(str(config_root / "final_infer" / "convnext" / "config.yaml"))
final_hypes = copy.deepcopy(final_hypes)
final_hypes["validate_dir"] = final_hypes["test_dir"]
final_dataset = build_dataset(final_hypes, visualize=True, train=False)
assert_loader_paths("FINAL test", final_dataset, base_root / "test")

print(
    "[PASS] exhaustive real-file preflight "
    + " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
)
PY
}

gpu_preflight() {
    if [[ -n "${RANK:-}" || -n "${WORLD_SIZE:-}" || -n "${LOCAL_RANK:-}" || -n "${SLURM_PROCID:-}" ]]; then
        fail "distributed-launch environment variables are set; DDP is not approved"
        return 1
    fi
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[FAIL] CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"[FAIL] expected exactly one visible GPU, got {torch.cuda.device_count()}")
print(f"[PASS] single visible CUDA device: {torch.cuda.get_device_name(0)}")
PY
}

checker_preflight() {
    CUDA_VISIBLE_DEVICES="${GPU_ID}" env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u SLURM_PROCID \
        "${PYTHON_BIN}" -B opencood/tools/check_doma_protocol_stamp.py \
        --stage all --max-samples 1 --formal
}

perform_check() {
    local failures=0 protocol_ok=0 inference_ok=0 dataset_ok=0 gpu_ok=0

    log "preflight repository=${REPO_ROOT} git=$(repo_sha)"
    if static_repository_preflight; then :; else failures=$((failures + 1)); fi

    log "checking formal protocol evidence"
    if protocol_preflight; then protocol_ok=1; pass "formal range evidence is resolved"; else failures=$((failures + 1)); fi

    log "checking formal inference source compatibility"
    if inference_source_preflight; then inference_ok=1; else failures=$((failures + 1)); fi

    log "checking OPV2V and OPV2V_Hetero real files and official loader paths"
    if dataset_preflight; then dataset_ok=1; else failures=$((failures + 1)); fi

    log "checking approved single-GPU runtime"
    if gpu_preflight; then gpu_ok=1; else failures=$((failures + 1)); fi

    if (( protocol_ok && inference_ok && dataset_ok && gpu_ok )); then
        log "running formal Gate 1-5 checker"
        if checker_preflight; then
            pass "formal Gate 1-5 preflight"
        else
            failures=$((failures + 1))
        fi
    else
        warn "formal Gate 1-5 checker skipped because prerequisite blockers remain"
    fi

    if (( failures > 0 )); then
        blocker "preflight failed with ${failures} blocker group(s); no formal training was started"
        return 1
    fi
    pass "all formal preflight checks passed"
}

protocol_values() {
    protocol_preflight | awk -F= '/^(DETECTION_RANGE_ARG|COMMUNICATION_RANGE)=/ {print $2}'
}

checkpoint_validate() {
    local config="$1" checkpoint="$2"
    "${PYTHON_BIN}" - "${config}" "${checkpoint}" <<'PY'
import sys
from pathlib import Path
import torch

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils

config_path, checkpoint_path = map(Path, sys.argv[1:])
if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
    raise SystemExit(f"checkpoint is missing or empty: {checkpoint_path}")
hypes = yaml_utils.load_yaml(str(config_path))
model = train_utils.create_model(hypes)
state = torch.load(str(checkpoint_path), map_location="cpu")
if not isinstance(state, dict) or not state:
    raise SystemExit(f"checkpoint is not a nonempty state_dict: {checkpoint_path}")
expected = model.state_dict()
missing = sorted(set(expected) - set(state))
extra = sorted(set(state) - set(expected))
shape_mismatch = sorted(
    key for key in set(expected).intersection(state)
    if tuple(expected[key].shape) != tuple(state[key].shape)
)
if missing or extra or shape_mismatch:
    raise SystemExit(
        f"checkpoint contract mismatch missing={missing[:8]} extra={extra[:8]} "
        f"shape={shape_mismatch[:8]}"
    )
model.load_state_dict(state, strict=True)
print(f"[PASS] strict checkpoint load keys={len(state)} path={checkpoint_path}")
PY
}

write_checkpoint_manifest() {
    local directory="$1" manifest tmp file relative count=0
    manifest="${directory}/checkpoints.sha256"
    tmp="${manifest}.tmp"
    : >"${tmp}" || return 1
    while IFS= read -r -d '' file; do
        relative="${file#${directory}/}"
        printf '%s  %s\n' "$(sha256_file "${file}")" "${relative}" >>"${tmp}" || return 1
        count=$((count + 1))
    done < <(find "${directory}" -maxdepth 1 -type f -name '*.pth' -print0 | sort -z)
    if (( count == 0 )); then
        rm -f -- "${tmp}"
        fail "no checkpoint files found for manifest in ${directory}"
        return 1
    fi
    mv -- "${tmp}" "${manifest}" || return 1
    sha256_file "${manifest}"
}

validate_checkpoint_manifest() {
    local stage="$1" manifest expected
    manifest="$(state_value "${stage}" CHECKPOINT_MANIFEST_REL || true)"
    expected="$(state_value "${stage}" CHECKPOINT_MANIFEST_SHA256 || true)"
    [[ -f "${manifest}" && -n "${expected}" ]] || return 1
    [[ "$(sha256_file "${manifest}")" == "${expected}" ]] || return 1
    (
        cd -- "${STAGE_DIR[$stage]}"
        sha256sum --check --strict checkpoints.sha256 >/dev/null
    )
}

best_checkpoint() {
    local directory="$1"
    local matches=()
    mapfile -t matches < <(find "${directory}" -maxdepth 1 -type f -name 'net_epoch_bestval_at*.pth' -print | sort)
    if (( ${#matches[@]} != 1 )); then
        fail "expected exactly one best-validation checkpoint in ${directory}, found ${#matches[@]}"
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

initialize_experiment() {
    if [[ -e "${OUTPUT_ROOT}" ]]; then
        if [[ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
            fail "${OUTPUT_ROOT} already contains an experiment; use status or resume"
            return 1
        fi
    fi
    mkdir -p -- "${OUTPUT_ROOT}" "${STATE_ROOT}" "${ARCHIVE_ROOT}" "${RESULT_ROOT}"
    {
        printf 'FORMAT_VERSION=1\n'
        printf 'CREATED_AT=%s\n' "$(utc_now)"
        printf 'REPO_SHA=%s\n' "$(repo_sha)"
        printf 'PROTOCOL_LOCK_SHA256=%s\n' "$(sha256_file "${LOCK_FILE}")"
        printf 'RUNNER_SHA256=%s\n' "$(sha256_file "${RUNNER_PATH}")"
        printf 'GPU_ID=%s\n' "${GPU_ID}"
        printf 'RESUME_POLICY=restart_incomplete_stage_from_template\n'
    } >"${META_FILE}"
    git status --short >"${OUTPUT_ROOT}/git_status_at_start.txt"
    git rev-parse HEAD >"${OUTPUT_ROOT}/git_sha.txt"
    cp -- "${LOCK_FILE}" "${OUTPUT_ROOT}/protocol_lock.yaml"
    cp -- "${RUNNER_PATH}" "${OUTPUT_ROOT}/run_stamp_doma_full.snapshot.sh"
    pass "initialized immutable experiment root ${OUTPUT_ROOT}"
}

check_experiment_identity() {
    [[ -f "${META_FILE}" ]] || { fail "missing experiment metadata; use run for a new experiment"; return 1; }
    local recorded current lock_hash recorded_lock runner_hash recorded_runner
    recorded="$(awk -F= '$1=="REPO_SHA" {print $2}' "${META_FILE}")"
    current="$(repo_sha)"
    if [[ "${recorded}" != "${current}" ]]; then
        fail "repository SHA changed since experiment creation: recorded=${recorded} current=${current}"
        return 1
    fi
    recorded_lock="$(awk -F= '$1=="PROTOCOL_LOCK_SHA256" {print $2}' "${META_FILE}")"
    lock_hash="$(sha256_file "${LOCK_FILE}")"
    if [[ "${recorded_lock}" != "${lock_hash}" ]]; then
        fail "protocol_lock changed since experiment creation"
        return 1
    fi
    recorded_runner="$(awk -F= '$1=="RUNNER_SHA256" {print $2}' "${META_FILE}")"
    runner_hash="$(sha256_file "${RUNNER_PATH}")"
    if [[ "${recorded_runner}" != "${runner_hash}" ]]; then
        fail "runner script changed since experiment creation"
        return 1
    fi
}

prepare_clean_stage_dir() {
    local stage="$1" directory="${STAGE_DIR[$1]}" config="${STAGE_CONFIG[$1]}"
    [[ ! -e "${directory}" ]] || { fail "stage directory already exists: ${directory}"; return 1; }
    mkdir -p -- "${directory}" || return 1
    cp -- "${config}" "${directory}/config.yaml" || return 1
}

complete_training_stage() {
    local stage="$1" checkpoint config_hash checkpoint_hash manifest_hash
    checkpoint="$(best_checkpoint "${STAGE_DIR[$stage]}")" || return 1
    grep -Fq "Training Finished" "${STAGE_DIR[$stage]}/train.log" || {
        fail "${stage} process exited without the official completion marker"
        return 1
    }
    checkpoint_validate "${STAGE_DIR[$stage]}/config.yaml" "${checkpoint}" || return 1
    config_hash="$(sha256_file "${STAGE_DIR[$stage]}/config.yaml")"
    checkpoint_hash="$(sha256_file "${checkpoint}")"
    manifest_hash="$(write_checkpoint_manifest "${STAGE_DIR[$stage]}")" || return 1
    write_state "${stage}" COMPLETE "official training process completed and checkpoint passed strict load"
    append_state_fields "${stage}" \
        "CONFIG_SHA256=${config_hash}" \
        "CHECKPOINT_REL=${checkpoint}" \
        "CHECKPOINT_SHA256=${checkpoint_hash}" \
        "CHECKPOINT_MANIFEST_REL=${STAGE_DIR[$stage]}/checkpoints.sha256" \
        "CHECKPOINT_MANIFEST_SHA256=${manifest_hash}"
    pass "${stage} COMPLETE checkpoint=${checkpoint} sha256=${checkpoint_hash}"
}

run_training_stage() {
    local stage="$1" entrypoint
    if [[ "${stage}" == A* ]]; then
        entrypoint="opencood/tools/train_adapter.py"
    else
        entrypoint="opencood/tools/train.py"
    fi
    prepare_clean_stage_dir "${stage}" || return 1
    if [[ "${stage}" == A* ]]; then
        prepare_adapter_parents "${stage}" || return 1
    fi
    write_state "${stage}" RUNNING "${entrypoint} started" || return 1
    log "${stage} start: ${entrypoint} --model_dir ${STAGE_DIR[$stage]}"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u SLURM_PROCID \
        "${PYTHON_BIN}" "${entrypoint}" \
        --hypes_yaml "${STAGE_DIR[$stage]}/config.yaml" \
        --model_dir "${STAGE_DIR[$stage]}" \
        2>&1 | tee "${STAGE_DIR[$stage]}/train.log"
    local rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
        write_state "${stage}" FAILED "training command exited ${rc}"
        fail "${stage} failed with exit code ${rc}; stopping pipeline"
        return "${rc}"
    fi
    if ! complete_training_stage "${stage}"; then
        write_state "${stage}" FAILED "checkpoint or completion validation failed"
        return 1
    fi
}

stage_checkpoint_from_state() {
    local stage="$1" path expected actual
    [[ "$(state_value "${stage}" STATUS || true)" == "COMPLETE" ]] || {
        fail "dependency ${stage} is not COMPLETE"
        return 1
    }
    path="$(state_value "${stage}" CHECKPOINT_REL || true)"
    expected="$(state_value "${stage}" CHECKPOINT_SHA256 || true)"
    [[ -f "${path}" && -n "${expected}" ]] || { fail "dependency ${stage} checkpoint metadata is missing"; return 1; }
    actual="$(sha256_file "${path}")"
    [[ "${actual}" == "${expected}" ]] || { fail "dependency ${stage} checkpoint hash mismatch"; return 1; }
    printf '%s\n' "${path}"
}

prepare_adapter_parents() {
    local stage="$1" local_stage modality protocol_checkpoint ego_checkpoint target
    case "${stage}" in
        A1) local_stage=L1; modality=m1 ;;
        A2) local_stage=L2; modality=m2 ;;
        A3) local_stage=L3; modality=m3 ;;
        A4) local_stage=L4; modality=m4 ;;
        *) fail "not an adapter stage: ${stage}"; return 1 ;;
    esac
    protocol_checkpoint="$(stage_checkpoint_from_state P0)" || return 1
    ego_checkpoint="$(stage_checkpoint_from_state "${local_stage}")" || return 1
    for target in protocol.pth ego.pth; do
        [[ ! -e "${STAGE_DIR[$stage]}/${target}" ]] || {
            fail "refusing to overwrite adapter parent ${STAGE_DIR[$stage]}/${target}"
            return 1
        }
    done
    cp -- "${protocol_checkpoint}" "${STAGE_DIR[$stage]}/protocol.pth" || return 1
    cp -- "${ego_checkpoint}" "${STAGE_DIR[$stage]}/ego.pth" || return 1
    printf '%s\n' "${protocol_checkpoint}" >"${STAGE_DIR[$stage]}/protocol.source.txt" || return 1
    printf '%s\n' "${ego_checkpoint}" >"${STAGE_DIR[$stage]}/ego.source.txt" || return 1
    log "${stage} parents protocol=P0 ego=${local_stage} modality=${modality}"
}

run_merge_stage() {
    local stage=MERGE checkpoint checkpoint_hash config_hash manifest_hash
    prepare_clean_stage_dir "${stage}" || return 1
    write_state "${stage}" RUNNING "official merge_model_w_adapter.py started" || return 1
    set +e
    "${PYTHON_BIN}" opencood/tools/merge_model_w_adapter.py \
        --model_dir "${OUTPUT_ROOT}" \
        --sub_dir local_adapter \
        --adapter_dir convnext \
        --with_protocol \
        2>&1 | tee "${STAGE_DIR[$stage]}/merge.log"
    local rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
        write_state "${stage}" FAILED "merge command exited ${rc}"
        return "${rc}"
    fi
    checkpoint="${STAGE_DIR[$stage]}/net_epoch1.pth"
    checkpoint_validate "${STAGE_DIR[$stage]}/config.yaml" "${checkpoint}" || {
        write_state "${stage}" FAILED "merged checkpoint strict validation failed"
        return 1
    }
    checkpoint_hash="$(sha256_file "${checkpoint}")"
    config_hash="$(sha256_file "${STAGE_DIR[$stage]}/config.yaml")"
    manifest_hash="$(write_checkpoint_manifest "${STAGE_DIR[$stage]}")" || return 1
    write_state "${stage}" COMPLETE "official merge completed and strict checkpoint load passed"
    append_state_fields "${stage}" \
        "CONFIG_SHA256=${config_hash}" \
        "CHECKPOINT_REL=${checkpoint}" \
        "CHECKPOINT_SHA256=${checkpoint_hash}" \
        "CHECKPOINT_MANIFEST_REL=${STAGE_DIR[$stage]}/checkpoints.sha256" \
        "CHECKPOINT_MANIFEST_SHA256=${manifest_hash}"
    pass "MERGE COMPLETE checkpoint=${checkpoint} sha256=${checkpoint_hash}"
}

eval_result_file() {
    local directory="$1" cav="$2" matches=()
    mapfile -t matches < <(find "${directory}" -maxdepth 1 -type f -name "eval_*use_cav${cav}.yaml" -print | sort)
    if (( ${#matches[@]} != 1 )); then
        fail "expected exactly one CAV${cav} eval YAML in ${directory}, found ${#matches[@]}"
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

validate_eval_result() {
    local path="$1"
    "${PYTHON_BIN}" - "${path}" <<'PY'
import math
import sys
import yaml
path = sys.argv[1]
data = yaml.safe_load(open(path, "r", encoding="utf-8"))
for key in ("ap30", "ap_50", "ap_70"):
    value = data.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise SystemExit(f"invalid {key}={value!r} in {path}")
print(f"[PASS] AP result {path}: ap30={data['ap30']} ap50={data['ap_50']} ap70={data['ap_70']}")
PY
}

validate_runtime_protocol_log() {
    local log_path="$1" expected_cav="$2"
    "${PYTHON_BIN}" - "${log_path}" "${LOCK_FILE}" "${expected_cav}" <<'PY'
import json
import math
import sys
import yaml

log_path, lock_path, expected_cav = sys.argv[1], sys.argv[2], int(sys.argv[3])
marker = "STAMP_DOMA_RUNTIME_PROTOCOL="
with open(log_path, "r", encoding="utf-8", errors="replace") as stream:
    snapshots = [
        json.loads(line.split(marker, 1)[1])
        for line in stream
        if marker in line
    ]
if len(snapshots) != 1:
    raise SystemExit(f"expected exactly one runtime protocol snapshot, found {len(snapshots)}")

snapshot = snapshots[0]
lock = yaml.safe_load(open(lock_path, "r", encoding="utf-8"))
expected_range = [float(value) for value in lock["effective_detection_range"]["value"]]
actual_range = [float(value) for value in snapshot.get("detection_range", [])]
if len(actual_range) != 6 or any(
    not math.isclose(actual, expected)
    for actual, expected in zip(actual_range, expected_range)
):
    raise SystemExit(f"runtime detection range mismatch: {actual_range} != {expected_range}")
if not math.isclose(
    float(snapshot.get("comm_range")),
    float(lock["effective_communication_range"]["value"]),
):
    raise SystemExit("runtime communication range mismatch")
if snapshot.get("assignment_path") != lock["final_assignment"]["path"]:
    raise SystemExit("runtime assignment does not match final_assignment lock")
if snapshot.get("fusion_core_method") != "intermediateheteradapter":
    raise SystemExit("runtime fusion core is not intermediateheteradapter")
if snapshot.get("validate_dir") != snapshot.get("test_dir"):
    raise SystemExit("runtime validate_dir is not redirected to test_dir")
if int(snapshot.get("use_cav")) != expected_cav:
    raise SystemExit("runtime use_cav does not match requested CAV stage")
if snapshot.get("dataset_root_dir") != snapshot.get("test_dir"):
    raise SystemExit("initialized dataset root_dir is not the formal test split")
if int(snapshot.get("dataset_use_cav")) != expected_cav:
    raise SystemExit("initialized dataset use_cav does not match requested CAV stage")
print(
    "[PASS] runtime protocol "
    f"CAV{expected_cav} range={actual_range} comm={snapshot['comm_range']} "
    f"assignment={snapshot['assignment_path']} fusion={snapshot['fusion_core_method']}"
)
PY
}

run_cav_stage() {
    local stage="$1" cav="${1#CAV}" merged merged_hash values=() detection_range result result_hash manifest_hash
    merged="$(stage_checkpoint_from_state MERGE)" || return 1
    merged_hash="$(sha256_file "${merged}")"
    mapfile -t values < <(protocol_values)
    [[ ${#values[@]} -eq 2 ]] || { fail "could not read locked detection/communication ranges"; return 1; }
    detection_range="${values[0]}"

    prepare_clean_stage_dir "${stage}" || return 1
    cp -- "${merged}" "${STAGE_DIR[$stage]}/net_epoch1.pth" || return 1
    write_state "${stage}" RUNNING "formal CAV${cav} inference started" || return 1
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u SLURM_PROCID \
        "${PYTHON_BIN}" opencood/tools/inference_heter_in_order.py \
        --model_dir "${STAGE_DIR[$stage]}" \
        --fusion_method intermediate \
        --range "${detection_range}" \
        --use_cav "[${cav}]" \
        --note DOMA_V2R2_formal \
        2>&1 | tee "${STAGE_DIR[$stage]}/inference.log"
    local rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
        write_state "${stage}" FAILED "inference command exited ${rc}"
        return "${rc}"
    fi
    validate_runtime_protocol_log "${STAGE_DIR[$stage]}/inference.log" "${cav}" || {
        write_state "${stage}" FAILED "resolved inference protocol validation failed"
        return 1
    }
    result="$(eval_result_file "${STAGE_DIR[$stage]}" "${cav}")" || {
        write_state "${stage}" FAILED "AP result file validation failed"
        return 1
    }
    validate_eval_result "${result}" || {
        write_state "${stage}" FAILED "AP values are missing or invalid"
        return 1
    }
    result_hash="$(sha256_file "${result}")"
    manifest_hash="$(write_checkpoint_manifest "${STAGE_DIR[$stage]}")" || return 1
    write_state "${stage}" COMPLETE "formal CAV${cav} inference and AP validation completed"
    append_state_fields "${stage}" \
        "MERGED_CHECKPOINT_SHA256=${merged_hash}" \
        "CHECKPOINT_REL=${STAGE_DIR[$stage]}/net_epoch1.pth" \
        "CHECKPOINT_SHA256=${merged_hash}" \
        "CHECKPOINT_MANIFEST_REL=${STAGE_DIR[$stage]}/checkpoints.sha256" \
        "CHECKPOINT_MANIFEST_SHA256=${manifest_hash}" \
        "RESULT_REL=${result}" \
        "RESULT_SHA256=${result_hash}"
    pass "${stage} COMPLETE result=${result} sha256=${result_hash}"
}

aggregate_results() {
    mkdir -p -- "${RESULT_ROOT}"
    "${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${RESULT_ROOT}/ap_summary.csv" <<'PY'
import csv
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = []
for cav in range(1, 5):
    directory = root / "final_infer" / "convnext" / f"CAV{cav}"
    matches = sorted(directory.glob(f"eval_*use_cav{cav}.yaml"))
    if len(matches) != 1:
        raise SystemExit(f"expected one CAV{cav} result, found {len(matches)}")
    data = yaml.safe_load(open(matches[0], "r", encoding="utf-8"))
    rows.append([cav, data["ap30"], data["ap_50"], data["ap_70"], str(matches[0])])
with open(output, "w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["cav", "ap_0.3", "ap_0.5", "ap_0.7", "source"])
    writer.writerows(rows)
print(f"[PASS] wrote {output}")
PY
    sha256_file "${RESULT_ROOT}/ap_summary.csv" >"${RESULT_ROOT}/ap_summary.csv.sha256"
}

validate_complete_stage() {
    local stage="$1" status git_sha config_hash checkpoint expected result expected_result cav merged_expected
    status="$(state_value "${stage}" STATUS || true)"
    [[ "${status}" == "COMPLETE" ]] || return 1
    git_sha="$(state_value "${stage}" GIT_SHA || true)"
    [[ "${git_sha}" == "$(repo_sha)" ]] || { fail "${stage} COMPLETE state belongs to another Git SHA"; return 1; }

    if [[ "${stage}" == CAV* ]]; then
        cav="${stage#CAV}"
        checkpoint="$(state_value "${stage}" CHECKPOINT_REL || true)"
        expected="$(state_value "${stage}" CHECKPOINT_SHA256 || true)"
        [[ -f "${checkpoint}" && "$(sha256_file "${checkpoint}")" == "${expected}" ]] || return 1
        merged_expected="$(state_value MERGE CHECKPOINT_SHA256 || true)"
        [[ -n "${merged_expected}" && "${expected}" == "${merged_expected}" ]] || return 1
        validate_checkpoint_manifest "${stage}" || return 1
        result="$(state_value "${stage}" RESULT_REL || true)"
        expected_result="$(state_value "${stage}" RESULT_SHA256 || true)"
        [[ -f "${result}" && "$(sha256_file "${result}")" == "${expected_result}" ]] || return 1
        validate_eval_result "${result}" >/dev/null
        return
    fi

    checkpoint="$(state_value "${stage}" CHECKPOINT_REL || true)"
    expected="$(state_value "${stage}" CHECKPOINT_SHA256 || true)"
    [[ -f "${checkpoint}" && "$(sha256_file "${checkpoint}")" == "${expected}" ]] || return 1
    config_hash="$(state_value "${stage}" CONFIG_SHA256 || true)"
    [[ "${config_hash}" == "$(sha256_file "${STAGE_DIR[$stage]}/config.yaml")" ]] || return 1
    validate_checkpoint_manifest "${stage}" || return 1
    checkpoint_validate "${STAGE_DIR[$stage]}/config.yaml" "${checkpoint}" >/dev/null
}

archive_incomplete_stage() {
    local stage="$1" stamp destination directory state
    stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
    destination="${ARCHIVE_ROOT}/${stamp}/${stage}"
    directory="${STAGE_DIR[$stage]}"
    state="$(state_file "${stage}")"
    mkdir -p -- "${destination}"
    if [[ -e "${directory}" ]]; then
        mv -- "${directory}" "${destination}/stage_dir"
    fi
    if [[ -f "${state}" ]]; then
        mv -- "${state}" "${destination}/stage.state"
    fi
    warn "archived incomplete ${stage} to ${destination}; restarting from template"
}

execute_stage() {
    local stage="$1"
    case "${stage}" in
        L1|L2|L3|L4|P0|A1|A2|A3|A4) run_training_stage "${stage}" ;;
        MERGE) run_merge_stage ;;
        CAV1|CAV2|CAV3|CAV4) run_cav_stage "${stage}" ;;
        *) fail "unknown stage: ${stage}"; return 1 ;;
    esac
}

run_pipeline() {
    local mode="$1" stage stage_status rc
    for stage in "${STAGES[@]}"; do
        if [[ "${mode}" == resume ]]; then
            if validate_complete_stage "${stage}"; then
                pass "${stage} already COMPLETE and revalidated; skipping"
                continue
            fi
            if [[ -e "${STAGE_DIR[$stage]}" || -f "$(state_file "${stage}")" ]]; then
                archive_incomplete_stage "${stage}"
            fi
        fi
        if execute_stage "${stage}"; then
            :
        else
            rc=$?
            stage_status="$(state_value "${stage}" STATUS 2>/dev/null || true)"
            if [[ "${stage_status}" != FAILED ]]; then
                write_state "${stage}" FAILED "stage setup or orchestration failed; inspect the stage log and preflight output"
            fi
            fail "pipeline stopped at ${stage}"
            return "${rc}"
        fi
    done
    aggregate_results || return 1
    pass "STAMP DOMA experiment COMPLETE"
}

show_status() {
    printf 'Repository: %s\n' "${REPO_ROOT}"
    printf 'Current Git SHA: %s\n' "$(repo_sha)"
    printf 'Output root: %s\n' "${OUTPUT_ROOT}"
    if [[ -f "${META_FILE}" ]]; then
        printf '\nExperiment metadata:\n'
        sed 's/^/  /' "${META_FILE}"
    else
        printf '\nExperiment metadata: not initialized\n'
    fi
    printf '\nStages:\n'
    local stage status updated detail
    for stage in "${STAGES[@]}"; do
        status="$(state_value "${stage}" STATUS 2>/dev/null || true)"
        updated="$(state_value "${stage}" UPDATED_AT 2>/dev/null || true)"
        detail="$(state_value "${stage}" DETAIL 2>/dev/null || true)"
        printf '  %-5s %-10s %-20s %s\n' "${stage}" "${status:-PENDING}" "${updated:--}" "${detail:--}"
    done
    if [[ -f "${RESULT_ROOT}/ap_summary.csv" ]]; then
        printf '\nFinal AP summary:\n'
        cat "${RESULT_ROOT}/ap_summary.csv"
    else
        printf '\nFinal AP summary: unavailable\n'
    fi
    printf '\nProtocol lock ranges:\n'
    "${PYTHON_BIN}" - "${LOCK_FILE}" <<'PY'
import sys, yaml
lock = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
for key in ("effective_detection_range", "effective_communication_range"):
    value = lock.get(key, {})
    print(f"  {key}: status={value.get('status')} source={value.get('source')} value={value.get('value')}")
PY
}

main() {
    local command="${1:-}"
    case "${command}" in
        check)
            perform_check
            ;;
        run)
            perform_check
            initialize_experiment
            run_pipeline run
            ;;
        resume)
            perform_check
            check_experiment_identity
            run_pipeline resume
            ;;
        status)
            show_status
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            usage >&2
            return 2
            ;;
    esac
}

main "$@"
