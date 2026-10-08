#!/usr/bin/env python3
"""Gate 6: single-GPU, one-step training smoke for STAMP DOMA protocol.

This checker is intentionally independent from the read-only Gate 1--5 checker.
It uses real training data and the official STAMP model, loss, optimizer,
scheduler, and parent-checkpoint loaders, but writes every artifact only below a
system temporary directory.

Passing this smoke gate means only that the selected training code paths can
complete one optimizer step.  It does not resolve or approve the formal DOMA
detection/communication ranges and does not make a run ready for formal
training.
"""

import argparse
import copy
import gc
import json
import math
import os
import random
import shutil
import sys
import tempfile
import traceback
from collections import OrderedDict
from pathlib import Path


SEED = 303
ALL_STAGES = ("L1", "L2", "L3", "L4", "P0", "A1", "A2", "A3", "A4")
STAGE_CONFIGS = OrderedDict(
    (
        ("L1", "local/m1/config.yaml"),
        ("L2", "local/m2/config.yaml"),
        ("L3", "local/m3/config.yaml"),
        ("L4", "local/m4/config.yaml"),
        ("P0", "protocol/config.yaml"),
        ("A1", "local_adapter/convnext/m1/config.yaml"),
        ("A2", "local_adapter/convnext/m2/config.yaml"),
        ("A3", "local_adapter/convnext/m3/config.yaml"),
        ("A4", "local_adapter/convnext/m4/config.yaml"),
    )
)
ADAPTER_TO_LOCAL = {"A1": "L1", "A2": "L2", "A3": "L3", "A4": "L4"}
ADAPTER_TO_MODALITY = {"A1": "m1", "A2": "m2", "A3": "m3", "A4": "m4"}
PARENT_MODULES = (
    "encoder",
    "backbone",
    "pyramid_backbone",
    "shrink_conv",
    "cls_head",
    "reg_head",
    "dir_head",
)

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG_ROOT = REPO_ROOT / "opencood" / "hypes_yaml" / "opv2v" / "doma_protocol_stamp_v2r2"


class Gate6Error(RuntimeError):
    """A fail-closed Gate 6 contract violation."""


def _parser():
    parser = argparse.ArgumentParser(
        description="STAMP DOMA-protocol Gate 6: single-GPU one-step training smoke"
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=DEFAULT_CONFIG_ROOT,
        help="DOMA protocol config root (default: %(default)s)",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=32,
        help="maximum number of sequential train samples inspected per stage (default: %(default)s)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        metavar="STAGE",
        help="optional subset of L1 L2 L3 L4 P0 A1 A2 A3 A4; comma-separated values are also accepted",
    )
    return parser


def _normalize_stages(values, parser):
    if not values:
        return list(ALL_STAGES)

    requested = []
    for value in values:
        requested.extend(part.strip().upper() for part in value.split(",") if part.strip())

    if not requested:
        parser.error("--stages must name at least one stage")

    invalid = sorted(set(requested) - set(ALL_STAGES))
    if invalid:
        parser.error("unknown stage(s): %s" % ", ".join(invalid))

    requested_set = set(requested)
    return [stage for stage in ALL_STAGES if stage in requested_set]


def _seed_everything(np, torch):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _is_within(path, root):
    path = os.path.normcase(os.path.realpath(str(path)))
    root = os.path.normcase(os.path.realpath(str(root)))
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _assert_safe_temp_root(temp_root, config_root):
    forbidden_roots = (
        REPO_ROOT / "opencood" / "logs",
        REPO_ROOT / "opencood" / "logs" / "STAMP_DOMA",
        config_root,
    )
    for forbidden in forbidden_roots:
        if _is_within(temp_root, forbidden):
            raise Gate6Error("temporary directory is inside forbidden path: %s" % forbidden)


def _mark_smoke_only(path, stage=None):
    path.mkdir(parents=True, exist_ok=True)
    marker = {
        "smoke_only": True,
        "do_not_reuse": True,
        "gate": "STAMP_DOMA_PROTOCOL_GATE6",
        "stage": stage,
    }
    marker_path = path / "SMOKE_ONLY_DO_NOT_REUSE.json"
    with marker_path.open("w", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _prepare_adapter_directory(stage, config_path, temp_root):
    local_stage = ADAPTER_TO_LOCAL[stage]
    protocol_parent = temp_root / "parents" / "P0" / "protocol.pth"
    ego_parent = temp_root / "parents" / local_stage / "ego.pth"
    missing = []
    if not protocol_parent.is_file():
        missing.append("P0")
    if not ego_parent.is_file():
        missing.append(local_stage)
    if missing:
        raise Gate6Error("%s requires %s smoke parents" % (stage, " and ".join(missing)))

    adapter_dir = temp_root / "adapters" / stage
    _mark_smoke_only(adapter_dir, stage)
    shutil.copy2(str(config_path), str(adapter_dir / "config.yaml"))
    shutil.copy2(str(protocol_parent), str(adapter_dir / "protocol.pth"))
    shutil.copy2(str(ego_parent), str(adapter_dir / "ego.pth"))
    return adapter_dir, adapter_dir / "config.yaml"


def _load_runtime_hypes(config_path, yaml_utils):
    loaded = yaml_utils.load_yaml(str(config_path))
    hypes = copy.deepcopy(loaded)
    hypes.setdefault("train_params", {})["batch_size"] = 1
    hypes["train_params"]["num_workers"] = 0
    return hypes


def _build_bounded_loader(dataset, max_scan, torch):
    scan_count = min(max_scan, len(dataset))
    if scan_count <= 0:
        raise Gate6Error("train dataset has no samples within max_scan")

    class IndexedPrefixDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, length):
            self.base_dataset = base_dataset
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            return index, self.base_dataset[index]

    def collate_indexed(rows):
        if len(rows) != 1:
            raise Gate6Error("Gate 6 loader requires runtime batch_size=1")
        index, item = rows[0]
        if item is None:
            return index, None
        return index, dataset.collate_batch_train([item])

    prefix_dataset = IndexedPrefixDataset(dataset, scan_count)
    return torch.utils.data.DataLoader(
        prefix_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_indexed,
        shuffle=False,
        drop_last=False,
    )


def _positive_gt(batch, torch):
    if not isinstance(batch, dict) or not isinstance(batch.get("ego"), dict):
        return False
    mask = batch["ego"].get("object_bbx_mask")
    if mask is None:
        return False
    if torch.is_tensor(mask):
        return bool(mask.detach().sum().item() > 0)
    try:
        return bool(float(mask.sum()) > 0)
    except (AttributeError, TypeError, ValueError):
        return False


def _has_required_labels(batch, adapter, supervise_single):
    ego = batch.get("ego", {}) if isinstance(batch, dict) else {}
    required = ["label_dict"]
    if adapter:
        required.append("label_dict_protocol")
    elif supervise_single:
        required.append("label_dict_single")
    return all(isinstance(ego.get(key), dict) and bool(ego[key]) for key in required)


def _select_train_batch(dataset, loader, stage, adapter, torch):
    supervise_single = bool(getattr(dataset, "supervise_single", False))
    for index, batch in loader:
        if batch is None:
            continue
        if not _positive_gt(batch, torch):
            continue
        if not _has_required_labels(batch, adapter, supervise_single):
            continue
        print("[INFO] Gate6 %s selected train sample index=%d" % (stage, index))
        return index, batch, supervise_single
    raise Gate6Error("%s found no valid real train batch within max_scan" % stage)


def _tensor_finite(value, torch):
    tensor = value.detach()
    if tensor.is_sparse:
        tensor = tensor.coalesce().values()
    return bool(torch.isfinite(tensor).all().item())


def _tensor_nonzero(value, torch):
    tensor = value.detach()
    if tensor.is_sparse:
        tensor = tensor.coalesce().values()
    return bool(torch.count_nonzero(tensor).item() > 0)


def _check_loss_tree(value, path, torch):
    checked = 0
    if torch.is_tensor(value):
        if not _tensor_finite(value, torch):
            raise Gate6Error("nonfinite loss component: %s" % path)
        return 1
    if isinstance(value, dict):
        for key, child in value.items():
            checked += _check_loss_tree(child, "%s.%s" % (path, key), torch)
        return checked
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            checked += _check_loss_tree(child, "%s[%d]" % (path, index), torch)
        return checked
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise Gate6Error("nonfinite loss component: %s" % path)
        return 1
    return 0


def _loss_dict_snapshot(criterion):
    return copy.copy(getattr(criterion, "loss_dict", {}))


def _run_detection_loss(model, batch, dataset, hypes, criterion, torch):
    output_dict = model(batch["ego"])
    detection_loss = criterion(output_dict, batch["ego"]["label_dict"])
    components = {"detection": _loss_dict_snapshot(criterion)}
    total_loss = detection_loss

    if bool(getattr(dataset, "supervise_single", False)):
        single_loss = criterion(output_dict, batch["ego"]["label_dict_single"], suffix="_single")
        components["single"] = _loss_dict_snapshot(criterion)
        total_loss = total_loss + single_loss * hypes["train_params"].get("single_weight", 1)

    if not torch.is_tensor(total_loss) or not _tensor_finite(total_loss, torch):
        raise Gate6Error("total detection loss is not a finite tensor")
    _check_loss_tree(components, "loss", torch)
    return total_loss, components


def _verify_adapter_loss_weights(hypes):
    args = hypes.get("loss_adapter", {}).get("args", {})
    expected = {"alpha_P2M": 9.0, "alpha_M2P2M": 1.0, "alpha_M2P": 10.0}
    actual = {key: args.get(key) for key in expected}
    if actual != expected:
        raise Gate6Error("AdapterLoss weights do not match formal DOMA contract: %r" % actual)


def _run_adapter_loss(model, batch, criterion_dict, criterion_adapter, member_modality, torch):
    result = model(batch["ego"])
    if not isinstance(result, tuple) or len(result) != 2:
        raise Gate6Error("adapter forward did not return (output_dict, output_feat)")
    output_dict, output_feat = result
    if output_dict is None or output_feat is None or len(output_feat) != 5:
        raise Gate6Error("adapter forward did not return detection outputs and five protocol features")

    missing_losses = {"m0", member_modality} - set(criterion_dict)
    if missing_losses:
        raise Gate6Error("missing formal detection criterion(s): %s" % sorted(missing_losses))

    FM, FP2M, FM2P2M, FP, FM2P = output_feat
    adapter_loss = criterion_adapter(FM, FP2M, FM2P2M, FP, FM2P)
    protocol_loss = criterion_dict["m0"](output_dict["m0"], batch["ego"]["label_dict_protocol"])
    member_loss = criterion_dict[member_modality](
        output_dict[member_modality], batch["ego"]["label_dict"]
    )
    total_loss = protocol_loss + member_loss + adapter_loss

    components = {
        "protocol_detection": _loss_dict_snapshot(criterion_dict["m0"]),
        "member_detection": _loss_dict_snapshot(criterion_dict[member_modality]),
        "adapter": _loss_dict_snapshot(criterion_adapter),
        "protocol_detection_total": protocol_loss,
        "member_detection_total": member_loss,
        "adapter_total": adapter_loss,
    }
    if not _tensor_finite(total_loss, torch):
        raise Gate6Error("adapter total loss is nonfinite")
    _check_loss_tree(components, "loss", torch)
    return total_loss, components


def _named_parameters(model):
    return OrderedDict(model.named_parameters())


def _assert_adapter_trainable_contract(model, stage, member_modality):
    expected_prefixes = ("adapter_%s." % member_modality, "reverter_%s." % member_modality)
    named = _named_parameters(model)
    trainable = [name for name, parameter in named.items() if parameter.requires_grad]
    unexpected = [name for name in trainable if not name.startswith(expected_prefixes)]
    expected_parameters = [name for name in named if name.startswith(expected_prefixes)]
    frozen_expected = [name for name in expected_parameters if not named[name].requires_grad]
    if not trainable:
        raise Gate6Error("%s has no trainable Adapter/Reverter parameters" % stage)
    if unexpected:
        raise Gate6Error("%s has unexpected trainable parameters: %s" % (stage, unexpected[:8]))
    if frozen_expected:
        raise Gate6Error("%s expected trainable parameters are frozen: %s" % (stage, frozen_expected[:8]))
    for prefix in expected_prefixes:
        if not any(name.startswith(prefix) for name in trainable):
            raise Gate6Error("%s has no trainable parameters below %s" % (stage, prefix))


def _check_grad_group(named, names, label, torch):
    gradients = [named[name].grad for name in names if named[name].grad is not None]
    if not gradients:
        raise Gate6Error("%s has no gradients" % label)
    if not all(_tensor_finite(gradient, torch) for gradient in gradients):
        raise Gate6Error("%s has nonfinite gradients" % label)
    if not any(_tensor_nonzero(gradient, torch) for gradient in gradients):
        raise Gate6Error("%s has no nonzero gradient" % label)


def _check_gradients(model, adapter, member_modality, torch):
    named = _named_parameters(model)
    trainable_names = [name for name, parameter in named.items() if parameter.requires_grad]
    _check_grad_group(named, trainable_names, "trainable parameter set", torch)

    if adapter:
        adapter_names = [name for name in trainable_names if name.startswith("adapter_%s." % member_modality)]
        reverter_names = [name for name in trainable_names if name.startswith("reverter_%s." % member_modality)]
        _check_grad_group(named, adapter_names, "adapter_%s" % member_modality, torch)
        _check_grad_group(named, reverter_names, "reverter_%s" % member_modality, torch)
        frozen_with_grad = [
            name for name, parameter in named.items() if not parameter.requires_grad and parameter.grad is not None
        ]
        if frozen_with_grad:
            raise Gate6Error("frozen parameters received gradients: %s" % frozen_with_grad[:8])
    return named, trainable_names


def _snapshot_parameters(named, names):
    return {name: named[name].detach().cpu().clone() for name in names}


def _changed_parameters(named, snapshot, torch):
    return [
        name
        for name, before in snapshot.items()
        if not torch.equal(before, named[name].detach().cpu())
    ]


def _check_optimizer_update(model, adapter, member_modality, trainable_names, trainable_before, frozen_before, torch):
    named = _named_parameters(model)
    changed_trainable = _changed_parameters(named, trainable_before, torch)
    if not changed_trainable:
        raise Gate6Error("optimizer step changed no trainable parameter")

    if adapter:
        changed_adapter = [
            name for name in changed_trainable if name.startswith("adapter_%s." % member_modality)
        ]
        changed_reverter = [
            name for name in changed_trainable if name.startswith("reverter_%s." % member_modality)
        ]
        if not changed_adapter:
            raise Gate6Error("optimizer step changed no adapter_%s parameter" % member_modality)
        if not changed_reverter:
            raise Gate6Error("optimizer step changed no reverter_%s parameter" % member_modality)
        changed_frozen = _changed_parameters(named, frozen_before, torch)
        if changed_frozen:
            raise Gate6Error("optimizer step changed frozen parameters: %s" % changed_frozen[:8])
    else:
        if not any(name in changed_trainable for name in trainable_names):
            raise Gate6Error("optimizer update was outside the expected trainable set")


def _checkpoint_state(path, torch):
    state = torch.load(str(path), map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise Gate6Error("checkpoint is not a nonempty state_dict: %s" % path)
    return state


def _prefix_match_keys(checkpoint, model_state, modality):
    matched = []
    missing_prefixes = []
    for module in PARENT_MODULES:
        prefix = "%s_%s." % (module, modality)
        current = [
            key
            for key, value in checkpoint.items()
            if key.startswith(prefix)
            and key in model_state
            and hasattr(value, "shape")
            and tuple(value.shape) == tuple(model_state[key].shape)
        ]
        if not current:
            missing_prefixes.append(prefix)
        matched.extend(current)
    if missing_prefixes:
        raise Gate6Error(
            "parent checkpoint has no compatible keys for prefixes: %s" % ", ".join(missing_prefixes)
        )
    return sorted(set(matched))


def _assert_loaded_values(model, checkpoint, matched_keys, label, torch):
    model_state = model.state_dict()
    mismatched = [
        key
        for key in matched_keys
        if not key.endswith("num_batches_tracked")
        and not torch.equal(model_state[key].detach().cpu(), checkpoint[key].detach().cpu())
    ]
    if mismatched:
        raise Gate6Error("%s parent keys were not loaded: %s" % (label, mismatched[:8]))


def _load_adapter_parents(model, adapter_dir, member_modality, train_utils, torch):
    protocol_checkpoint = _checkpoint_state(adapter_dir / "protocol.pth", torch)
    ego_checkpoint = _checkpoint_state(adapter_dir / "ego.pth", torch)
    model_state = model.state_dict()
    protocol_matches = _prefix_match_keys(protocol_checkpoint, model_state, "m0")
    ego_matches = _prefix_match_keys(ego_checkpoint, model_state, member_modality)

    init_epoch, model = train_utils.load_saved_model(str(adapter_dir), model)
    if init_epoch != 0:
        raise Gate6Error("fresh adapter smoke directory unexpectedly contains a resumable epoch")
    model = train_utils.load_original_model(str(adapter_dir), model)

    _assert_loaded_values(model, protocol_checkpoint, protocol_matches, "protocol", torch)
    _assert_loaded_values(model, ego_checkpoint, ego_matches, "ego", torch)
    return model, init_epoch, len(protocol_matches), len(ego_matches)


def _save_and_reload_smoke(stage, output_dir, model, optimizer, scheduler, hypes, adapter, train_utils, torch):
    net_path = output_dir / "net_epoch1.pth"
    smoke_path = output_dir / "smoke_state.pt"
    torch.save(model.state_dict(), str(net_path))
    parsed_net = _checkpoint_state(net_path, torch)
    if set(parsed_net) != set(model.state_dict()):
        raise Gate6Error("net_epoch1.pth keys do not match model state")

    torch.save(
        {
            "smoke_only": True,
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        str(smoke_path),
    )
    payload = torch.load(str(smoke_path), map_location="cpu")
    if payload.get("smoke_only") is not True or payload.get("stage") != stage:
        raise Gate6Error("smoke_state.pt marker/stage mismatch")

    reload_model = train_utils.create_model(hypes)
    reload_model.load_state_dict(payload["model_state_dict"], strict=True)
    reload_optimizer = train_utils.setup_optimizer(hypes, reload_model)
    if adapter:
        reload_scheduler = train_utils.setup_lr_schedular(hypes, reload_optimizer, init_epoch=0)
    else:
        reload_scheduler = train_utils.setup_lr_schedular(hypes, reload_optimizer)
    reload_optimizer.load_state_dict(payload["optimizer_state_dict"])
    reload_scheduler.load_state_dict(payload["scheduler_state_dict"])
    if len(reload_optimizer.param_groups) != len(optimizer.param_groups):
        raise Gate6Error("optimizer param-group count changed after reload")
    if reload_scheduler.state_dict().get("last_epoch") != scheduler.state_dict().get("last_epoch"):
        raise Gate6Error("scheduler last_epoch changed after reload")

    del payload, parsed_net, reload_scheduler, reload_optimizer, reload_model


def _save_parent(stage, output_dir, temp_root):
    if stage == "P0":
        parent_dir = temp_root / "parents" / "P0"
        parent_name = "protocol.pth"
    elif stage in ("L1", "L2", "L3", "L4"):
        parent_dir = temp_root / "parents" / stage
        parent_name = "ego.pth"
    else:
        return
    _mark_smoke_only(parent_dir, stage)
    shutil.copy2(str(output_dir / "net_epoch1.pth"), str(parent_dir / parent_name))


def _run_stage(stage, config_root, temp_root, max_scan, yaml_utils, train_utils, build_dataset, torch):
    adapter = stage in ADAPTER_TO_LOCAL
    member_modality = ADAPTER_TO_MODALITY.get(stage)
    source_config = config_root / STAGE_CONFIGS[stage]
    if not source_config.is_file():
        raise Gate6Error("missing stage config: %s" % source_config)

    if adapter:
        output_dir, runtime_config = _prepare_adapter_directory(stage, source_config, temp_root)
    else:
        output_dir = temp_root / "stages" / stage
        _mark_smoke_only(output_dir, stage)
        runtime_config = source_config

    hypes = _load_runtime_hypes(runtime_config, yaml_utils)
    if hypes["train_params"]["batch_size"] != 1 or hypes["train_params"]["num_workers"] != 0:
        raise Gate6Error("runtime-only batch_size/num_workers override was not applied")

    dataset = build_dataset(hypes, visualize=False, train=True)
    loader = _build_bounded_loader(dataset, max_scan, torch)
    _, batch, _ = _select_train_batch(dataset, loader, stage, adapter, torch)

    model = train_utils.create_model(hypes)
    if adapter:
        _verify_adapter_loss_weights(hypes)
        criterion = train_utils.create_losses_heter(hypes)
        criterion_adapter = train_utils.create_adapter_loss(hypes)
    else:
        criterion = train_utils.create_loss(hypes)
        criterion_adapter = None

    optimizer = train_utils.setup_optimizer(hypes, model)
    if adapter:
        model, init_epoch, protocol_keys, ego_keys = _load_adapter_parents(
            model, output_dir, member_modality, train_utils, torch
        )
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print("[PASS] Gate6 %s parent protocol loaded keys=%d" % (stage, protocol_keys))
        print("[PASS] Gate6 %s parent ego loaded keys=%d" % (stage, ego_keys))
    else:
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    device = torch.device("cuda:0")
    model.to(device)
    model.train()
    train_initializer = getattr(model, "model_train_init", None)
    if callable(train_initializer):
        train_initializer()
    if adapter:
        _assert_adapter_trainable_contract(model, stage, member_modality)
        print("[PASS] Gate6 %s trainable set" % stage)

    model.zero_grad()
    optimizer.zero_grad()
    batch = train_utils.to_device(batch, device)
    batch["ego"]["epoch"] = 0

    if adapter:
        total_loss, loss_components = _run_adapter_loss(
            model, batch, criterion, criterion_adapter, member_modality, torch
        )
    else:
        total_loss, loss_components = _run_detection_loss(
            model, batch, dataset, hypes, criterion, torch
        )
    print("[INFO] Gate6 %s loss=%.8f" % (stage, total_loss.detach().item()))
    print("[PASS] Gate6 %s forward" % stage)
    print("[PASS] Gate6 %s loss finite components=%d" % (
        stage, _check_loss_tree(loss_components, "loss", torch)
    ))

    total_loss.backward()
    print("[PASS] Gate6 %s backward" % stage)
    named, trainable_names = _check_gradients(model, adapter, member_modality, torch)
    print("[PASS] Gate6 %s gradients finite" % stage)

    trainable_before = _snapshot_parameters(named, trainable_names)
    frozen_names = [name for name, parameter in named.items() if not parameter.requires_grad]
    frozen_before = _snapshot_parameters(named, frozen_names) if adapter else {}
    optimizer.step()
    _check_optimizer_update(
        model,
        adapter,
        member_modality,
        trainable_names,
        trainable_before,
        frozen_before,
        torch,
    )
    print("[PASS] Gate6 %s optimizer update" % stage)
    if adapter:
        print("[PASS] Gate6 %s adapter gradients/update" % stage)
        print("[PASS] Gate6 %s reverter gradients/update" % stage)
        print("[PASS] Gate6 %s frozen parameters unchanged" % stage)

    optimizer.zero_grad()
    scheduler.step(0)
    print("[PASS] Gate6 %s scheduler" % stage)

    _save_and_reload_smoke(
        stage, output_dir, model, optimizer, scheduler, hypes, adapter, train_utils, torch
    )
    print("[PASS] Gate6 %s checkpoint reload" % stage)
    _save_parent(stage, output_dir, temp_root)

    del frozen_before, trainable_before, named, loss_components
    del total_loss, batch, model, optimizer, scheduler, criterion, criterion_adapter
    del loader, dataset, hypes


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    stages = _normalize_stages(args.stages, parser)
    if args.max_scan <= 0:
        parser.error("--max-scan must be positive")

    config_root = args.config_root.resolve()
    if not config_root.is_dir():
        print("[FAIL] Gate6 config root not found: %s" % config_root)
        print("SUMMARY failures=1")
        return 1

    # Keep --help usable on CPU-only developer machines by importing runtime
    # dependencies only after argument parsing.
    try:
        import numpy as np
        import torch
    except Exception as error:
        print("[FAIL] Gate6 runtime import failed: %s" % error)
        print("SUMMARY failures=1")
        return 1

    if not torch.cuda.is_available():
        print("[FAIL] Gate6 requires CUDA; torch.cuda.is_available() is False")
        print("SUMMARY failures=1")
        return 1

    _seed_everything(np, torch)
    torch.cuda.set_device(0)
    print("[INFO] Gate6 CUDA device=%s" % torch.cuda.get_device_name(0))
    print("[INFO] Gate6 stages=%s" % ",".join(stages))
    print("[PENDING FORMAL] ranges unresolved; allowed for Gate 6 smoke only")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os.chdir(str(REPO_ROOT))

    try:
        import opencood.hypes_yaml.yaml_utils as yaml_utils
        from opencood.data_utils.datasets import build_dataset
        from opencood.tools import train_utils
    except Exception as error:
        print("[FAIL] Gate6 STAMP import failed: %s" % error)
        print("SUMMARY failures=1")
        return 1

    failures = []
    try:
        with tempfile.TemporaryDirectory(prefix="stamp_doma_gate6_") as temporary:
            temp_root = Path(temporary).resolve()
            _assert_safe_temp_root(temp_root, config_root)
            _mark_smoke_only(temp_root)
            print("[INFO] Gate6 temporary root=%s" % temp_root)

            for stage in stages:
                print("[INFO] Gate6 %s start" % stage)
                try:
                    _run_stage(
                        stage,
                        config_root,
                        temp_root,
                        args.max_scan,
                        yaml_utils,
                        train_utils,
                        build_dataset,
                        torch,
                    )
                    print("[PASS] Gate6 %s completed one-step training smoke" % stage)
                except Exception as error:
                    failures.append((stage, str(error)))
                    print("[FAIL] Gate6 %s: %s" % (stage, error))
                    traceback.print_exc()
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
    except Exception as error:
        failures.append(("SETUP", str(error)))
        print("[FAIL] Gate6 setup/cleanup: %s" % error)
        traceback.print_exc()

    if failures:
        print("SUMMARY failures=%d" % len(failures))
        return 1
    if tuple(stages) == ALL_STAGES:
        print("[PASS] Gate 6 completed all 9 one-step training checks")
    else:
        print("[PASS] Gate 6 completed all %d requested one-step training checks" % len(stages))
    print("SUMMARY failures=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
