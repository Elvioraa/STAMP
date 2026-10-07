"""Read-only Gate 1-5 checks for the isolated STAMP x DOMA configuration.

This command never trains, saves a model, writes a checkpoint, or mutates a
configuration/assignment file. Runtime checks are skipped explicitly when the
local machine lacks the dataset, CUDA, a dependency, or cached camera weights.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG_ROOT = (
    REPO_ROOT
    / "opencood"
    / "hypes_yaml"
    / "opv2v"
    / "doma_protocol_stamp_v2r2"
)

MODEL_CONFIGS = OrderedDict(
    [
        ("L1", "local/m1/config.yaml"),
        ("L2", "local/m2/config.yaml"),
        ("L3", "local/m3/config.yaml"),
        ("L4", "local/m4/config.yaml"),
        ("P0", "protocol/config.yaml"),
        ("A1", "local_adapter/convnext/m1/config.yaml"),
        ("A2", "local_adapter/convnext/m2/config.yaml"),
        ("A3", "local_adapter/convnext/m3/config.yaml"),
        ("A4", "local_adapter/convnext/m4/config.yaml"),
        ("FINAL", "final_infer/convnext/config.yaml"),
    ]
)
ALL_CONFIGS = list(MODEL_CONFIGS.values()) + ["protocol_lock.yaml"]
EXPECTED_ASSIGNMENT = "opencood/modality_assign/opv2v_4modality_in_order.json"
EXPECTED_ASSIGNMENT_GIT_BLOB = "61489d9fa23faef35548f8f73a48c7238c21a7ef"
AGENT_ORDER = ["m1", "m2", "m3", "m4"]
PHYSICAL_AGENT = {
    "m1": ("lidar", "point_pillar", None),
    "m2": ("camera", "lift_splat_shoot", "EfficientNet"),
    "m3": ("lidar", "second", None),
    "m4": ("camera", "lift_splat_shoot", "Resnet101"),
}


class CheckFailure(RuntimeError):
    """A real configuration/protocol failure, not an environment skip."""


class ResultBook(object):
    def __init__(self):
        self.failures = []
        self.pending = []

    def passed(self, message):
        print("[PASS] {}".format(message))

    def failed(self, message):
        self.failures.append(message)
        print("[FAIL] {}".format(message))

    def skipped(self, message):
        self.pending.append(message)
        print("[SKIPPED - environment] {}".format(message))

    def pending_server(self, message):
        self.pending.append(message)
        print("[PENDING SERVER] {}".format(message))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only Gate 1-5 checks for STAMP x DOMA batch 1"
    )
    parser.add_argument(
        "--config-root",
        default=str(DEFAULT_CONFIG_ROOT),
        help="isolated configuration root",
    )
    parser.add_argument(
        "--stage",
        choices=["yaml", "model", "dataset", "forward", "all"],
        default="all",
    )
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument(
        "--formal",
        action="store_true",
        help="fail closed unless the formal range and communication range are resolved",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    """Run a read-only Git query at the repository root."""
    command = [
        "git",
        "-c",
        "safe.directory={}".format(REPO_ROOT.as_posix()),
    ] + list(args)
    try:
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except OSError as exc:
        raise CheckFailure("unable to run Git: {}".format(exc))
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise CheckFailure(
            "Git query failed ({}): {}".format(" ".join(args), detail)
        )
    return result.stdout.strip()


def strict_load_yaml(path):
    """Load YAML while rejecting duplicate mapping keys."""
    try:
        import yaml
    except ImportError as exc:
        raise CheckFailure("PyYAML is unavailable: {}".format(exc))

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise CheckFailure(
                    "duplicate YAML key {!r} in {} at line {}".format(
                        key, path, key_node.start_mark.line + 1
                    )
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.load(stream, Loader=UniqueKeyLoader)
    if not isinstance(data, dict):
        raise CheckFailure("{} does not contain a YAML mapping".format(path))
    return data


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


def resolve_repo_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def config_paths(config_root):
    return OrderedDict(
        (label, config_root / relative) for label, relative in MODEL_CONFIGS.items()
    )


def _modality_setting(hypes):
    return hypes["heter"]["modality_setting"]


def _model_modalities(hypes):
    args = hypes["model"]["args"]
    return OrderedDict(
        (key, value)
        for key, value in args.items()
        if isinstance(key, str) and key.startswith("m") and key[1:].isdigit()
    )


def _assert_physical_agent(modality, setting, model_setting, label):
    sensor, core, camera_encoder = PHYSICAL_AGENT[modality]
    require(
        setting["sensor_type"] == sensor,
        "{} {} sensor_type is {}, expected {}".format(
            label, modality, setting["sensor_type"], sensor
        ),
    )
    require(
        setting["core_method"] == core,
        "{} {} core_method is {}, expected {}".format(
            label, modality, setting["core_method"], core
        ),
    )
    require(model_setting["sensor_type"] == sensor, "{} model sensor mismatch".format(label))
    require(model_setting["core_method"] == core, "{} model core mismatch".format(label))
    filters = model_setting.get("backbone_args", {}).get("num_filters")
    require(filters == [64], "{} {} backbone must output 64 channels".format(label, modality))
    if camera_encoder is not None:
        actual = model_setting["encoder_args"].get("camera_encoder")
        require(
            actual == camera_encoder,
            "{} {} camera encoder is {}, expected {}".format(
                label, modality, actual, camera_encoder
            ),
        )
        grid = setting.get("grid_conf", {})
        require(grid.get("xbound") == [-51.2, 51.2, 0.4], "{} camera xbound mismatch".format(label))
        require(grid.get("ybound") == [-51.2, 51.2, 0.4], "{} camera ybound mismatch".format(label))
        expected_dim = [384, 512] if modality == "m2" else [336, 448]
        require(
            setting["data_aug_conf"].get("final_dim") == expected_dim,
            "{} {} final_dim mismatch".format(label, modality),
        )
        if label.startswith("A") or label == "FINAL":
            require(
                model_setting.get("visible_range")
                == [-51.2, -51.2, -3, 51.2, 51.2, 1],
                "{} {} visible_range mismatch".format(label, modality),
            )
    if modality == "m1":
        voxel = setting["preprocess"]["args"].get("voxel_size")
        require(voxel == [0.4, 0.4, 4], "{} m1 PointPillar voxel mismatch".format(label))
    if modality == "m3":
        voxel = setting["preprocess"]["args"].get("voxel_size")
        require(voxel == [0.1, 0.1, 0.1], "{} m3 SECOND voxel mismatch".format(label))


def _assert_thresholds(hypes, label):
    for modality, setting in _modality_setting(hypes).items():
        post = setting.get("postprocess")
        if post is None:
            continue
        score = post["target_args"].get("score_threshold")
        nms = post.get("nms_thresh")
        require(score == 0.2, "{} {} score threshold is not 0.2".format(label, modality))
        require(nms == 0.15, "{} {} NMS threshold is not 0.15".format(label, modality))


def _assert_adapter_contract(model_setting, label, modality):
    for direction in ("adapter", "reverter"):
        block = model_setting.get(direction)
        require(block is not None, "{} {} missing {}".format(label, modality, direction))
        require(
            block.get("core_method") == "adapterconvnext",
            "{} {} {} is not adapterconvnext".format(label, modality, direction),
        )
        args = block["args"]
        require(args.get("in_channels") == 64, "{} {} {} in_channels != 64".format(label, modality, direction))
        require(args.get("out_channels") == 64, "{} {} {} out_channels != 64".format(label, modality, direction))
        require(
            args.get("in_feature_shape") == [256, 256],
            "{} {} {} in_feature_shape != [256, 256]".format(label, modality, direction),
        )
        require(
            args.get("out_feature_shape") == [256, 256],
            "{} {} {} out_feature_shape != [256, 256]".format(label, modality, direction),
        )
        submodule = args.get("submodule_args", {})
        require(submodule.get("num_of_blocks") == 3, "{} adapter block count mismatch".format(label))
        require(submodule.get("dim") == 64, "{} adapter hidden dim mismatch".format(label))


def _assert_no_doma_model(hypes, label):
    payload = json.dumps(hypes.get("model", {}), sort_keys=True).lower()
    require("doma" not in payload, "{} imports/configures a DOMA model component".format(label))


def check_protocol_lock(config_root, formal, book):
    path = config_root / "protocol_lock.yaml"
    lock = strict_load_yaml(path)
    required = [
        "stamp_commit",
        "doma_commit_or_status",
        "assignment_path",
        "assignment_git_blob",
        "assignment_sha256",
        "seed",
        "agent_order",
        "ego",
        "mapping",
        "score_threshold",
        "nms_threshold",
        "test_dir",
        "test_manifest_sha256",
        "effective_detection_range",
        "effective_communication_range",
        "gt_mode",
        "ap_sort",
        "evaluator_sha256",
        "checkpoint_manifest",
    ]
    for key in required:
        require(key in lock, "protocol_lock missing {}".format(key))
    require(lock["seed"] == 303, "protocol_lock seed must be 303")
    require(lock["agent_order"] == AGENT_ORDER, "protocol_lock agent_order mismatch")
    require(lock["ego"] == "m1", "protocol_lock ego must be m1")
    require(lock["mapping"] == dict((m, m) for m in AGENT_ORDER), "protocol_lock mapping mismatch")
    require(lock["assignment_path"] == EXPECTED_ASSIGNMENT, "protocol_lock assignment path mismatch")
    assignment_blob = lock["assignment_git_blob"]
    require(
        assignment_blob.get("status") == "confirmed",
        "protocol_lock assignment_git_blob must be confirmed",
    )
    require(
        assignment_blob.get("value", "").lower() == EXPECTED_ASSIGNMENT_GIT_BLOB,
        "protocol_lock assignment_git_blob mismatch",
    )
    assignment_sha = lock["assignment_sha256"]
    require(
        assignment_sha.get("status") == "informational"
        and assignment_sha.get("hash_scope") == "windows_worktree_raw",
        "protocol_lock assignment_sha256 must be informational Windows raw bytes",
    )
    require(
        len(assignment_sha.get("value", "")) == 64,
        "protocol_lock informational assignment_sha256 is malformed",
    )
    unresolved = []
    for key in ("effective_detection_range", "effective_communication_range"):
        value = lock[key]
        if value.get("status") == "unresolved":
            require(
                value.get("source") == "pending_server_run_artifact",
                "{} unresolved source mismatch".format(key),
            )
            require(value.get("value") is None, "{} must not contain a guessed value".format(key))
            unresolved.append(key)
    if formal and unresolved:
        raise CheckFailure(
            "formal mode is fail-closed; unresolved fields: {}".format(
                ", ".join(unresolved)
            )
        )
    if unresolved:
        book.pending_server(
            "formal protocol unresolved by design: {}".format(", ".join(unresolved))
        )
    book.passed("protocol_lock schema and unresolved-field policy")
    return lock


def check_assignment(lock, book):
    assignment_relative = lock["assignment_path"]
    assignment_path = resolve_repo_path(assignment_relative)
    require(assignment_path.is_file(), "assignment file does not exist: {}".format(assignment_path))

    tracked = git_output("ls-files", "--error-unmatch", "--", assignment_relative)
    git_status = git_output("status", "--porcelain=v1", "--", assignment_relative)
    head_blob = git_output("rev-parse", "HEAD:{}".format(assignment_relative)).lower()
    worktree_blob = git_output("hash-object", "--", assignment_relative).lower()
    raw_sha = sha256_file(assignment_path)
    locked_blob = lock["assignment_git_blob"]["value"].lower()
    locked_raw = lock["assignment_sha256"]["value"].lower()

    print("[INFO] assignment HEAD blob: {}".format(head_blob))
    print("[INFO] assignment worktree blob: {}".format(worktree_blob))
    print(
        "[INFO] assignment raw SHA256 (informational, current worktree): {}".format(
            raw_sha
        )
    )
    print(
        "[INFO] assignment raw SHA256 lock (informational, windows_worktree_raw): {}".format(
            locked_raw
        )
    )
    print("[INFO] assignment git status: {}".format(git_status or "clean"))

    tracked_paths = [line.replace("\\", "/") for line in tracked.splitlines()]
    require(
        assignment_relative in tracked_paths,
        "assignment file is not tracked by Git",
    )
    require(not git_status, "assignment working tree is modified: {}".format(git_status))
    require(head_blob == locked_blob, "assignment HEAD blob does not match protocol lock")
    require(
        worktree_blob == head_blob,
        "assignment worktree blob does not match HEAD blob",
    )
    with assignment_path.open("r", encoding="utf-8") as stream:
        assignment = json.load(stream, object_pairs_hook=OrderedDict)
    require(assignment, "assignment JSON is empty")
    checked = 0
    for scenario, cav_mapping in assignment.items():
        modalities = list(cav_mapping.values())
        prefix_len = min(4, len(modalities))
        require(
            modalities[:prefix_len] == AGENT_ORDER[:prefix_len],
            "assignment order mismatch in {}: {}".format(scenario, modalities[:prefix_len]),
        )
        checked += 1
    expected_prefixes = [AGENT_ORDER[:n] for n in range(1, 5)]
    require(
        expected_prefixes
        == [["m1"], ["m1", "m2"], ["m1", "m2", "m3"], AGENT_ORDER],
        "internal CAV prefix definition mismatch",
    )
    book.passed(
        "assignment tracked/clean Git blob identity and m1->m2->m3->m4 prefixes "
        "across {} scenarios".format(checked)
    )


def check_yaml(config_root, formal, book):
    missing = [relative for relative in ALL_CONFIGS if not (config_root / relative).is_file()]
    require(not missing, "missing YAML files: {}".format(", ".join(missing)))
    parsed = OrderedDict()
    for label, path in config_paths(config_root).items():
        parsed[label] = strict_load_yaml(path)
    lock = check_protocol_lock(config_root, formal, book)
    check_assignment(lock, book)

    for label, hypes in parsed.items():
        require(hypes.get("seed") == 303, "{} seed must be 303".format(label))
        require(
            hypes.get("train_params", {}).get("max_cav") == 5,
            "{} max_cav must be 5".format(label),
        )
        require(
            hypes.get("heter", {}).get("assignment_path") == EXPECTED_ASSIGNMENT,
            "{} must reuse the checked in-order assignment".format(label),
        )
        _assert_thresholds(hypes, label)
        _assert_no_doma_model(hypes, label)

    for index, modality in enumerate(AGENT_ORDER, start=1):
        label = "L{}".format(index)
        hypes = parsed[label]
        settings = _modality_setting(hypes)
        models = _model_modalities(hypes)
        require(list(settings.keys()) == [modality], "{} modality_setting mismatch".format(label))
        require(list(models.keys()) == [modality], "{} model modality mismatch".format(label))
        require(hypes["heter"]["ego_modality"] == modality, "{} ego mismatch".format(label))
        expected_mapping = dict((m, modality) for m in AGENT_ORDER)
        require(hypes["heter"]["mapping_dict"] == expected_mapping, "{} mapping mismatch".format(label))
        _assert_physical_agent(modality, settings[modality], models[modality], label)

    require(
        parsed["L3"]["heter"].get("lidar_channels_dict") == {"m3": 32},
        "L3 must explicitly select 32-line m3 LiDAR",
    )

    protocol = parsed["P0"]
    require(protocol["model"]["core_method"] == "homo_pyramid_collab", "P0 model changed")
    require(list(_modality_setting(protocol).keys()) == ["m0"], "P0 modality must be m0")
    require(protocol["heter"]["ego_modality"] == "m0", "P0 ego must be m0")
    require(
        protocol["heter"]["mapping_dict"] == dict((m, "m0") for m in AGENT_ORDER),
        "P0 mapping mismatch",
    )
    require(
        protocol["model"]["args"]["m0"]["backbone_args"]["num_filters"] == [64],
        "P0 protocol feature channel must be 64",
    )

    for index, modality in enumerate(AGENT_ORDER, start=1):
        label = "A{}".format(index)
        hypes = parsed[label]
        settings = _modality_setting(hypes)
        models = _model_modalities(hypes)
        require(list(settings.keys()) == ["m0", modality], "{} modalities mismatch".format(label))
        require(list(models.keys()) == ["m0", modality], "{} model modalities mismatch".format(label))
        require(hypes["model"]["core_method"] == "collab_w_adapter", "{} core model changed".format(label))
        require(
            hypes["model"]["args"].get("stage") == "train_adapter_w_output",
            "{} adapter stage mismatch".format(label),
        )
        _assert_physical_agent(modality, settings[modality], models[modality], label)
        _assert_adapter_contract(models[modality], label, modality)

    require(
        parsed["A3"]["heter"].get("lidar_channels_dict") == {"m3": 32},
        "A3 must explicitly select 32-line m3 LiDAR",
    )

    final = parsed["FINAL"]
    require(final["heter"]["ego_modality"] == "m1", "FINAL ego must be m1")
    require(
        final["heter"]["mapping_dict"] == dict((m, m) for m in AGENT_ORDER),
        "FINAL mapping must be identity",
    )
    require(
        final["heter"].get("lidar_channels_dict") == {"m3": 32},
        "FINAL must explicitly select 32-line m3 LiDAR",
    )
    final_settings = _modality_setting(final)
    final_models = _model_modalities(final)
    require(set(final_settings.keys()) == set(["m0"] + AGENT_ORDER), "FINAL settings incomplete")
    require(set(final_models.keys()) == set(["m0"] + AGENT_ORDER), "FINAL models incomplete")
    for modality in AGENT_ORDER:
        _assert_physical_agent(modality, final_settings[modality], final_models[modality], "FINAL")
        _assert_adapter_contract(final_models[modality], "FINAL", modality)

    book.passed("all 11 YAML files parse with unique keys and resolved anchors")
    book.passed("physical Agent mapping, thresholds, seed, max_cav, and adapter contracts")
    book.passed("no DOMA model/loss is configured in the isolated STAMP model YAMLs")
    return parsed, lock


def check_gt_control_flow(book):
    dataset_path = (
        REPO_ROOT
        / "opencood"
        / "data_utils"
        / "datasets"
        / "heter_infer"
        / "intermediate_heter_infer_fusion_dataset.py"
    )
    source = dataset_path.read_text(encoding="utf-8")
    gt_pos = source.find("object_stack.append")
    use_cav_pos = source.find("if _i >= self.use_cav")
    require(gt_pos >= 0 and use_cav_pos >= 0, "GT/use_cav control-flow markers missing")
    require(gt_pos < use_cav_pos, "use_cav truncates input before full GT collection")
    heter_path = REPO_ROOT / "opencood" / "utils" / "heter_utils.py"
    heter_source = heter_path.read_text(encoding="utf-8")
    require(
        "self.lidar_channels_dict.get(cav_modality, None) == 32" in heter_source
        and 'replace(".pcd", "_32.pcd")' in heter_source,
        "m3=32 cannot be expressed by the existing data adaptor",
    )
    book.passed("static GT-before-use_cav control flow and 32-line path switch")


def _camera_weight_cached():
    candidates = []
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        candidates.append(Path(torch_home) / "hub" / "checkpoints")
    candidates.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints")
    for directory in candidates:
        try:
            if directory.is_dir() and list(directory.glob("efficientnet-b0*.pth")):
                return True
        except OSError:
            continue
    return False


def _environment_exception(exc):
    text = str(exc).lower()
    markers = [
        "no module named",
        "dll load failed",
        "cannot open shared object file",
        "cuda",
        "spconv",
        "torch_scatter",
        "not compiled",
        "connection",
        "url",
    ]
    return isinstance(exc, (ImportError, ModuleNotFoundError, OSError)) or any(
        marker in text for marker in markers
    )


def _state_prefixes(model):
    prefixes = set()
    for key in model.state_dict().keys():
        parts = key.split(".")
        prefixes.add(".".join(parts[: min(2, len(parts))]))
    return sorted(prefixes)


def check_model(config_root, parsed, book):
    try:
        from opencood.tools import train_utils
    except Exception as exc:
        if _environment_exception(exc):
            book.skipped("Gate 3 imports unavailable: {}".format(exc))
            book.pending_server("Gate 3 model construction")
            return {}
        raise

    constructed = {}
    camera_cache = _camera_weight_cached()
    for label, path in config_paths(config_root).items():
        yaml_text = path.read_text(encoding="utf-8")
        if "camera_encoder: EfficientNet" in yaml_text and not camera_cache:
            book.skipped(
                "{} construction: cached EfficientNet-B0 weights absent and downloads are forbidden".format(
                    label
                )
            )
            continue
        try:
            model = train_utils.create_model(parsed[label])
        except Exception as exc:
            if _environment_exception(exc):
                book.skipped("{} construction unavailable: {}".format(label, exc))
                continue
            raise CheckFailure("{} construction failed: {}".format(label, exc))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        prefixes = _state_prefixes(model)
        require("doma" not in model.__class__.__name__.lower(), "{} constructed a DOMA class".format(label))
        print(
            "[PASS] {} class={} parameters={} state_prefixes={}".format(
                label,
                model.__class__.__name__,
                parameter_count,
                ",".join(prefixes[:16]),
            )
        )
        constructed[label] = model
    if len(constructed) == len(MODEL_CONFIGS):
        book.passed("Gate 3 constructed all ten STAMP models")
    else:
        missing = [label for label in MODEL_CONFIGS if label not in constructed]
        book.pending_server("Gate 3 remaining models: {}".format(", ".join(missing)))
    return constructed


def _dataset_runtime_imports():
    from opencood.data_utils.datasets import build_dataset
    return build_dataset


def _load_runtime_hypes(config_root, label):
    """Load one config through STAMP's official runtime YAML pipeline."""
    require(label in MODEL_CONFIGS, "unknown runtime config label: {}".format(label))
    path = config_root / MODEL_CONFIGS[label]
    require(path.is_file(), "runtime config does not exist: {}".format(path))
    try:
        from opencood.hypes_yaml import yaml_utils
        hypes = yaml_utils.load_yaml(str(path))
    except Exception as exc:
        raise CheckFailure(
            "{} official runtime YAML load failed: {}".format(label, exc)
        )
    require(isinstance(hypes, dict), "{} runtime YAML is not a mapping".format(label))
    return hypes


def _validate_final_runtime_hypes(hypes, lock, gate, book):
    """Verify official parser output and the read-only test-split override."""
    require(
        hypes.get("heter", {}).get("assignment_path") == EXPECTED_ASSIGNMENT,
        "{} FINAL runtime assignment mismatch".format(gate),
    )
    require(
        hypes.get("test_dir") == lock["test_dir"],
        "{} FINAL runtime test_dir does not match protocol lock".format(gate),
    )
    modality_setting = hypes.get("heter", {}).get("modality_setting", {})
    required_modalities = ["m0"] + AGENT_ORDER
    require(
        all(modality in modality_setting for modality in required_modalities),
        "{} FINAL runtime modality_setting is incomplete".format(gate),
    )
    resolved = []
    required_anchor_keys = {"W", "H", "D", "vw", "vh", "vd"}
    for modality in required_modalities:
        postprocess = modality_setting[modality].get("postprocess", {})
        if postprocess.get("core_method") != "VoxelPostprocessor":
            continue
        anchor_args = postprocess.get("anchor_args", {})
        missing = sorted(required_anchor_keys.difference(anchor_args))
        require(
            not missing,
            "{} FINAL runtime {} anchor_args missing {}".format(
                gate, modality, ", ".join(missing)
            ),
        )
        resolved.append(modality)
    require(resolved, "{} FINAL runtime has no VoxelPostprocessor anchors".format(gate))

    original_validate_dir = hypes.get("validate_dir")
    prepared = _prepare_infer_hypes(hypes, 1)
    require(
        prepared.get("validate_dir") == prepared.get("test_dir"),
        "{} inference runtime did not redirect validate_dir to test_dir".format(gate),
    )
    require(
        hypes.get("validate_dir") == original_validate_dir,
        "{} inference preparation mutated the loaded base config".format(gate),
    )
    book.passed(
        "{} FINAL loaded via yaml_utils.load_yaml with resolved anchors for {}".format(
            gate, ",".join(resolved)
        )
    )
    book.passed("{} runtime validate_dir redirected to test_dir".format(gate))


def _prepare_infer_hypes(hypes, use_cav):
    copied = copy.deepcopy(hypes)
    core = copied["fusion"]["core_method"]
    if core == "intermediateheter":
        copied["fusion"]["core_method"] = "intermediateheterinfer"
    copied["validate_dir"] = copied["test_dir"]
    copied["use_cav"] = use_cav
    return copied


def _sample_source_records(dataset, index):
    scenario_index = 0
    for position, end_index in enumerate(dataset.len_record):
        if index < end_index:
            scenario_index = position
            break
    scenario = dataset.scenario_database[scenario_index]
    local_index = index if scenario_index == 0 else index - dataset.len_record[scenario_index - 1]
    timestamp = dataset.return_timestamp_key(scenario, local_index)
    records = []
    for cav_id, cav_content in scenario.items():
        entry = cav_content[timestamp]
        modality = entry.get("modality_name")
        sensor_paths = [entry.get("lidar")]
        sensor_paths.extend(entry.get("cameras", []))
        records.append(
            {
                "sample_id": "{}:{}".format(scenario_index, timestamp),
                "cav_id": str(cav_id),
                "modality": modality,
                "source_paths": [str(path) for path in sensor_paths if path],
            }
        )
    return records


def _hash_records(records):
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _gt_hash(item):
    import numpy as np

    ego = item["ego"]
    boxes = np.asarray(ego["object_bbx_center"])
    mask = np.asarray(ego["object_bbx_mask"]).astype(bool)
    boxes = np.asarray(boxes[mask], dtype=np.float32)
    object_ids = [str(value) for value in ego.get("object_ids", range(len(boxes)))]
    pairs = sorted(zip(object_ids, list(boxes)), key=lambda pair: pair[0])
    digest = hashlib.sha256()
    for object_id, box in pairs:
        digest.update(object_id.encode("utf-8"))
        digest.update(np.asarray(box, dtype=np.float32).tobytes())
    return digest.hexdigest()


def check_dataset(config_root, lock, max_samples, book):
    runtime_final = _load_runtime_hypes(config_root, "FINAL")
    _validate_final_runtime_hypes(runtime_final, lock, "Gate 4", book)
    test_dir = resolve_repo_path(runtime_final["test_dir"])
    if not test_dir.is_dir():
        book.skipped("Gate 4 dataset is absent: {}".format(test_dir))
        book.pending_server("Gate 4 real single-batch source-path checks")
        book.pending_server("runtime input_hash/gt_hash equality for CAV1-4")
        return None
    try:
        build_dataset = _dataset_runtime_imports()
    except Exception as exc:
        if _environment_exception(exc):
            book.skipped("Gate 4 dataset imports unavailable: {}".format(exc))
            book.pending_server("Gate 4 real single-batch checks")
            return None
        raise

    final_datasets = {}
    prefix_records = {}
    gt_hashes = {}
    sample_count = None
    for use_cav in range(1, 5):
        try:
            hypes = _prepare_infer_hypes(runtime_final, use_cav)
            dataset = build_dataset(hypes, visualize=False, train=False)
        except Exception as exc:
            if _environment_exception(exc):
                book.skipped("Gate 4 dataset build unavailable: {}".format(exc))
                book.pending_server("Gate 4/GT runtime checks")
                return None
            raise CheckFailure("CAV{} dataset build failed: {}".format(use_cav, exc))
        final_datasets[use_cav] = dataset
        if sample_count is None:
            sample_count = min(max_samples, len(dataset))
        prefix_records[use_cav] = []
        gt_hashes[use_cav] = []
        for index in range(sample_count):
            full_records = _sample_source_records(dataset, index)
            prefix = full_records[:use_cav]
            expected = AGENT_ORDER[:use_cav]
            actual = [record["modality"] for record in prefix]
            require(actual == expected, "CAV{} sample {} order is {}".format(use_cav, index, actual))
            if use_cav >= 3:
                m3_paths = prefix[2]["source_paths"]
                require(
                    any(path.endswith("_32.pcd") for path in m3_paths),
                    "CAV{} sample {} m3 does not use _32.pcd".format(use_cav, index),
                )
            prefix_records[use_cav].append(prefix)
            item = dataset[index]
            require(item is not None, "CAV{} sample {} returned None".format(use_cav, index))
            gt_hashes[use_cav].append(_gt_hash(item))
            print(
                "[PASS] CAV{} sample={} input_hash={} gt_hash={} records={}".format(
                    use_cav,
                    index,
                    _hash_records(prefix),
                    gt_hashes[use_cav][-1],
                    json.dumps(prefix, sort_keys=True),
                )
            )
    for index in range(sample_count or 0):
        require(
            all(gt_hashes[cav][index] == gt_hashes[1][index] for cav in range(1, 5)),
            "sample {} GT hash differs across CAV1-4".format(index),
        )
        for cav in range(2, 5):
            require(
                prefix_records[cav][index][: cav - 1] == prefix_records[cav - 1][index],
                "sample {} CAV{} input is not a strict prefix extension".format(index, cav),
            )
    book.passed("Gate 4 source paths, in-order prefixes, and m3 32-line paths")
    book.passed("runtime input prefix growth and identical GT hashes for CAV1-4")
    return final_datasets


def _shape_of(value):
    if hasattr(value, "shape"):
        return tuple(int(v) for v in value.shape)
    if isinstance(value, (list, tuple)):
        for item in value:
            shape = _shape_of(item)
            if shape is not None:
                return shape
    if isinstance(value, dict):
        for item in value.values():
            shape = _shape_of(item)
            if shape is not None:
                return shape
    return None


def check_forward(config_root, lock, max_samples, book):
    runtime_final = _load_runtime_hypes(config_root, "FINAL")
    _validate_final_runtime_hypes(runtime_final, lock, "Gate 5", book)
    test_dir = resolve_repo_path(runtime_final["test_dir"])
    if not test_dir.is_dir():
        book.skipped("Gate 5 dataset is absent: {}".format(test_dir))
        book.pending_server("Gate 5 five-point feature-shape forward checks")
        return
    try:
        import torch
        from torch.utils.data import DataLoader
        from opencood.data_utils.datasets import build_dataset
        from opencood.tools import train_utils
    except Exception as exc:
        if _environment_exception(exc):
            book.skipped("Gate 5 imports unavailable: {}".format(exc))
            book.pending_server("Gate 5 forward checks")
            return
        raise
    if not torch.cuda.is_available():
        book.skipped("Gate 5 requires CUDA for real spconv/camera forward")
        book.pending_server("Gate 5 five-point feature-shape forward checks")
        return
    if not _camera_weight_cached():
        book.skipped("Gate 5 cached EfficientNet-B0 weights absent; downloads are forbidden")
        book.pending_server("Gate 5 camera-inclusive forward checks")
        return

    for use_cav, modality in enumerate(AGENT_ORDER, start=1):
        label = "FINAL-CAV{}".format(use_cav)
        hypes = _prepare_infer_hypes(runtime_final, use_cav)
        try:
            dataset = build_dataset(hypes, visualize=True, train=False)
            loader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=0,
                collate_fn=dataset.collate_batch_train,
                shuffle=False,
                drop_last=False,
            )
            batch = next(iter(loader))
            require(batch is not None, "{} first batch is None".format(label))
            model = train_utils.create_model(hypes).cuda().eval()
        except Exception as exc:
            if _environment_exception(exc):
                book.skipped("{} Gate 5 setup unavailable: {}".format(label, exc))
                continue
            raise CheckFailure("{} Gate 5 setup failed: {}".format(label, exc))
        observed = {}
        handles = []
        for stage_name, attribute in [
            ("encoder", "encoder_{}".format(modality)),
            ("backbone", "backbone_{}".format(modality)),
            ("aligner", "aligner_{}".format(modality)),
            ("adapter", "adapter_{}".format(modality)),
            ("reverter", "reverter_{}".format(modality)),
        ]:
            module = getattr(model, attribute, None)
            require(module is not None, "{} missing {}".format(label, attribute))

            def capture(_module, _inputs, output, name=stage_name):
                observed[name] = _shape_of(output)

            handles.append(module.register_forward_hook(capture))
        try:
            with torch.no_grad():
                device_batch = train_utils.to_device(batch, torch.device("cuda"))
                model(device_batch["ego"])
        finally:
            for handle in handles:
                handle.remove()
        for stage_name in ("encoder", "backbone", "aligner", "adapter", "reverter"):
            shape = observed.get(stage_name)
            require(shape is not None, "{} did not capture {} shape".format(label, stage_name))
            if len(shape) >= 4 and stage_name != "encoder":
                require(shape[1] == 64, "{} {} channel != 64: {}".format(label, stage_name, shape))
            if stage_name in ("adapter", "reverter"):
                require(shape[-2:] == (256, 256), "{} {} spatial mismatch: {}".format(label, stage_name, shape))
        print("[PASS] {} {} shapes={}".format(label, modality, observed))
    book.passed("Gate 5 completed available five-point forward checks")


def main():
    args = parse_args()
    book = ResultBook()
    if args.max_samples <= 0:
        book.failed("--max-samples must be positive")
        return 1
    config_root = Path(args.config_root).resolve()
    try:
        parsed, lock = check_yaml(config_root, args.formal, book)
        check_gt_control_flow(book)
        if args.stage in ("model", "all"):
            check_model(config_root, parsed, book)
        if args.stage in ("dataset", "all"):
            check_dataset(config_root, lock, args.max_samples, book)
        if args.stage in ("forward", "all"):
            check_forward(config_root, lock, args.max_samples, book)
    except CheckFailure as exc:
        book.failed(str(exc))
    except Exception as exc:
        book.failed("unexpected checker error: {}: {}".format(type(exc).__name__, exc))

    print(
        "SUMMARY failures={} pending_or_skipped={}".format(
            len(book.failures), len(book.pending)
        )
    )
    return 1 if book.failures else 0


if __name__ == "__main__":
    sys.exit(main())
