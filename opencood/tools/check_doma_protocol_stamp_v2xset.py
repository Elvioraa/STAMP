"""Fail-closed V2XSet preflight. --stage yaml is CPU-only; all requires real CUDA/data.

No training, checkpoint writes or configuration mutation. JSON stdout records are
captured by the runner. Assignment hashes retain JSON object insertion order.
"""
import argparse
import ast
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'opencood/hypes_yaml/v2xset/doma_protocol_stamp_v2'
STAGES = {**{'L%d' % i: 'local/m%d/config.yaml' % i for i in range(1, 5)},
          'P0': 'protocol/config.yaml',
          **{'A%d' % i: 'local_adapter/convnext/m%d/config.yaml' % i for i in range(1, 5)},
          'FINAL': 'final_infer/convnext/config.yaml'}
ORDER = ['m1', 'm2', 'm3', 'm4']
RANGE = [-204.8, -102.4, -3, 204.8, 102.4, 1]
EXCLUDED = '2021_09_09_13_20_58'
MAPPING = {'m1': ('point_pillar', None), 'm2': ('lift_splat_shoot', 'EfficientNet'),
           'm3': ('second', None), 'm4': ('lift_splat_shoot', 'Resnet101')}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def emit(event, **fields):
    print('V2XSET_CHECK=' + json.dumps(dict(event=event, **fields), sort_keys=True), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_json_hash(value):
    # sort_keys=False is intentional: CAV/scene insertion order affects loading.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf8')).hexdigest()


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        require(key not in result, 'duplicate YAML key: %s at %s' % (key, key_node.start_mark))
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def read_yaml(path):
    return yaml.load(Path(path).read_text(encoding='utf8'), Loader=UniqueLoader)


def walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def cli_contracts():
    expected = {
        'train.py': {'--hypes_yaml', '--model_dir'},
        'train_adapter.py': {'--hypes_yaml', '--model_dir'},
        'merge_model_w_adapter.py': {'--model_dir', '--adapter_dir', '--with_protocol'},
        'inference_heter_in_order.py': {'--model_dir', '--range', '--use_cav', '--fusion_method', '--note'},
    }
    for file, required in expected.items():
        tree = ast.parse((ROOT / 'opencood/tools' / file).read_text(encoding='utf8'))
        flags = {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
                 for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
        require(required <= flags, '%s CLI changed: %s' % (file, required - flags))


def check_static(config_root=CONFIG):
    lock = read_yaml(config_root / 'protocol_lock.yaml')
    require(lock['schema_version'] == 2, 'expected dual-assignment schema v2')
    require(lock['seed'] == 303 and lock['max_cav'] == 5, 'seed/max_cav lock mismatch')
    require(lock['evaluation']['detection_range'] == RANGE, 'range lock mismatch')
    require(lock['evaluation']['comm_range'] == 180, 'comm range lock mismatch')
    require(lock['camera_excluded_scenes'] == [EXCLUDED], 'camera scene exclusion lock mismatch')
    assignments = {}
    for name in ('training_assignment', 'final_assignment'):
        spec = lock[name]
        path = ROOT / spec['path']
        data = json.loads(path.read_text(encoding='utf8'))
        require(ordered_json_hash(data) == spec['sha256']['value'], name + ' content/order changed')
        require(spec['sha256']['hash_scope'] == 'ordered_json_utf8_compact', 'unknown hash scope')
        for scene, members in data.items():
            require(members and all(m in ORDER for m in members.values()), 'invalid assignment ' + scene)
            if name == 'final_assignment':
                require(int(next(iter(members))) >= 0, 'RSU cannot be ego: ' + scene)
                require(list(members.values())[:4] == ORDER[:min(4, len(members))], 'unordered ' + scene)
        assignments[name] = data
        emit('assignment', name=name, scenes=len(data), raw_sha256=sha256(path),
             ordered_sha256=ordered_json_hash(data))
    for label, relative in STAGES.items():
        h = read_yaml(config_root / relative)
        require(h['seed'] == 303 and h['v2xset_reproducibility'] is True, label + ' seed opt-in')
        require(h['fusion']['dataset'] == 'v2xset', label + ' dataset')
        for key, split in [('root_dir', 'train'), ('validate_dir', 'validate'), ('test_dir', 'test')]:
            require(h[key] == 'dataset/V2XSET/' + split, label + ' split ' + key)
        spec = 'final_assignment' if label == 'FINAL' else 'training_assignment'
        require(h['heter']['assignment_path'] == lock[spec]['path'], label + ' assignment')
        require(h['train_params']['max_cav'] == 5, label + ' max_cav')
        require('depth' not in h['input_source'] and not h.get('add_data_extension'), label + ' extra data')
        require('lidar_channels_dict' not in h['heter'], label + ' channel override')
        for key, value in walk(h):
            if key in ('depth_supervision', 'use_gt_depth', 'use_depth_gt'):
                require(not value, label + ' depth enabled')
        if label != 'FINAL':
            require(h['train_params']['epoches'] == (5 if label.startswith('A') else 30), label + ' budget')
            require(h['comm_range'] == 70, label + ' original training communication policy changed')
        else:
            require(h['comm_range'] == 180 and h['model']['args']['stage'] == 'infer', 'FINAL stage/comm')
            require(h['model']['core_method'] == 'collab_w_adapter', 'FINAL model')
            require(h['fusion']['core_method'] == 'intermediateheteradapter', 'FINAL fusion')
            require(h['heter']['mapping_dict'] == dict(zip(ORDER, ORDER)), 'FINAL mapping')
            for key, value in walk(h):
                if key in ('cav_lidar_range', 'gt_range', 'lidar_range',
                           'in_cav_lidar_range', 'out_cav_lidar_range'):
                    require(value == RANGE, 'FINAL range mismatch: ' + key)
        for modality, (encoder, camera) in MAPPING.items():
            model = h['model']['args'].get(modality)
            if model is None:
                continue
            require(model['core_method'] == encoder, label + ' encoder ' + modality)
            if camera:
                require(model['encoder_args']['camera_encoder'] == camera, label + ' camera mapping')
                aug = h['heter']['modality_setting'][modality]['data_aug_conf']
                require(aug['Ncams'] == 4 and len(aug['cams']) == 4, label + ' four cameras')
            if modality == 'm3':
                args = model['encoder_args']
                require(args['mean_vfe']['num_point_features'] == args['spconv']['num_features_in'] == 4,
                        label + ' SECOND input must be 4')
            for kind in ('adapter', 'reverter'):
                if kind in model:
                    args = model[kind]['args']
                    require(model[kind]['core_method'] == 'adapterconvnext', 'adapter architecture')
                    require(args['in_channels'] == args['out_channels'] == 64, label + ' channels')
                    if label == 'FINAL':
                        require(args['in_feature_shape'] == args['out_feature_shape'] == [256, 512],
                                'formal BEV shape')
    cli_contracts()
    emit('static_pass', configs=len(STAGES), cli_contracts=True)
    return lock, assignments


def check_data(assignments):
    """Enumerate actual files, not just dataset symlinks. Never substitute inputs."""
    import numpy as np
    from PIL import Image
    from opencood.hypes_yaml import yaml_utils
    from opencood.utils.heter_utils import Adaptor
    adaptor = Adaptor('m1', ORDER, assignments['final_assignment'], {}, dict(zip(ORDER, ORDER)), None, False)
    manifests = {}
    for split in ('train', 'validate', 'test'):
        root = ROOT / 'dataset/V2XSET' / split
        require(root.is_dir(), 'missing real dataset split: ' + str(root))
        scenes = sorted(p for p in root.iterdir() if p.is_dir())
        require(scenes, 'empty split: ' + split)
        assignment = assignments['final_assignment' if split == 'test' else 'training_assignment']
        names = {s.name for s in scenes}
        require(names <= assignment.keys(), 'unassigned scenes: ' + str(names - assignment.keys()))
        if split == 'test':
            require(names == set(assignment), 'test scene set differs from ordered assignment')
        count = images = 0
        manifest = hashlib.sha256()
        for scene in scenes:
            members = assignment[scene.name]
            actual = {p.name for p in scene.iterdir() if p.is_dir()}
            require(set(members) == actual, 'CAV membership mismatch: ' + str(scene))
            timeline = None
            for cav_id in members:
                cav = scene / cav_id
                files = sorted(p for p in cav.glob('*.yaml') if 'additional' not in p.name)
                stamps = {p.stem for p in files}
                require(stamps, 'no YAML frames: ' + str(cav))
                require(timeline is None or stamps == timeline, 'CAV timestamp mismatch: ' + str(cav))
                timeline = stamps
                for file in files:
                    pcd = file.with_suffix('.pcd')
                    require(pcd.is_file() and pcd.stat().st_size > 0, 'missing raw PCD: ' + str(pcd))
                    require(adaptor.switch_lidar_channels('m3', str(pcd)) == str(pcd), 'm3 path changed')
                    with pcd.open('rb') as stream:
                        header = stream.read(2048)
                    require(b'FIELDS' in header and b'DATA' in header, 'invalid PCD header: ' + str(pcd))
                    json_path = file.with_suffix('.json')
                    params = (json.loads(json_path.read_text()) if json_path.is_file()
                              else yaml_utils.load_yaml(str(file)))
                    pose = np.asarray(params['lidar_pose'])
                    require(pose.shape == (6,) and np.isfinite(pose).all(), 'invalid pose: ' + str(file))
                    require(isinstance(params['vehicles'], dict), 'missing object labels: ' + str(file))
                    if scene.name != EXCLUDED:
                        for camera_id in range(4):
                            camera = params['camera%d' % camera_id]
                            intr = np.asarray(camera['intrinsic'])
                            cords = np.asarray(camera['cords'])
                            require(intr.shape == (3, 3) and cords.shape == (6,), 'camera calibration shape')
                            require(np.isfinite(intr).all() and np.isfinite(cords).all(), 'nonfinite calibration')
                            require(abs(np.linalg.det(intr)) > 0, 'singular intrinsic')
                            image_path = cav / (file.stem + '_camera%d.png' % camera_id)
                            with Image.open(image_path) as image:
                                require(image.size == (800, 600), 'unexpected camera resolution: ' + str(image_path))
                                image.verify()
                            manifest.update(json.dumps([str(image_path.resolve()), image_path.stat().st_size,
                                                        image_path.stat().st_mtime_ns]).encode())
                            images += 1
                    manifest.update(json.dumps([file.relative_to(root).as_posix(), str(pcd.resolve()),
                                                pcd.stat().st_size, pcd.stat().st_mtime_ns,
                                                params], sort_keys=True, default=lambda a: a.tolist()).encode())
                    count += 1
            emit('scene_pass', split=split, scene=scene.name, camera_excluded=scene.name == EXCLUDED)
        manifests[split] = dict(scenes=sorted(names), camera_scenes=sorted(names - {EXCLUDED}),
                                frames=count, images=images, metadata_sha256=manifest.hexdigest())
        emit('data_pass', split=split, **manifests[split])
    split_sets = [set(manifests[split]['scenes']) for split in ('train', 'validate', 'test')]
    require(set.union(*split_sets) == set(assignments['training_assignment']), 'full assignment/split union mismatch')
    require(sum(map(len, split_sets)) == len(set.union(*split_sets)), 'train/validate/test scene overlap')
    return manifests


def check_database_paths(dataset):
    count = 0
    for scene in dataset.scenario_database.values():
        for member in scene.values():
            for entry in member.values():
                if not isinstance(entry, dict) or 'yaml' not in entry:
                    continue
                require(Path(entry['lidar']) == Path(entry['yaml']).with_suffix('.pcd'),
                        'actual dataset substituted point cloud: ' + str(entry['lidar']))
                if entry.get('modality_name') == 'm3':
                    count += 1
    emit('database_raw_paths', m3_frames=count)


def load_runtime(path, cav=None):
    from opencood.hypes_yaml import yaml_utils
    h = yaml_utils.load_yaml(str(path))
    if cav is not None:
        from opencood.tools.inference_heter_in_order import (
            _apply_inference_range, _prepare_inference_hypes, _inference_lidar_configs)
        h, extent = _apply_inference_range(h, '204.8,102.4')
        h = _prepare_inference_hypes(h, 'intermediate')
        require(extent == RANGE, 'resolved final range')
        require(_inference_lidar_configs(h, [cav]) == [(cav, None)], 'V2XSET raw input policy')
        h['validate_dir'] = h['test_dir']
        h['use_cav'] = cav
    for setting in h['heter']['modality_setting'].values():
        anchor = setting['postprocess']['anchor_args']
        require({'W', 'H', 'D', 'vw', 'vh', 'vd'} <= anchor.keys(), 'official YAML parsers did not resolve anchors')
    return h


def finite_tensors(node):
    import torch
    if torch.is_tensor(node):
        require(torch.isfinite(node).all().item(), 'nonfinite model output')
        return 1
    if isinstance(node, dict):
        return sum(finite_tensors(v) for v in node.values())
    if isinstance(node, (tuple, list)):
        return sum(finite_tensors(v) for v in node)
    return 0


def check_weight_contracts(signatures):
    for index in range(1, 5):
        parents = dict(signatures['P0'], **signatures['L%d' % index])
        adapter = signatures['A%d' % index]
        for key, shape in parents.items():
            require(adapter.get(key) == shape, 'parent/adapter key-shape mismatch: ' + key)
        frozen = {key for key in adapter if not key.startswith(('adapter_', 'reverter_'))}
        require(frozen <= parents.keys(), 'uninitialized adapter-stage frozen keys: ' + str(frozen - parents.keys()))
    merged = {}
    for label in ['L1', 'L2', 'L3', 'L4', 'A1', 'A2', 'A3', 'A4', 'P0']:
        for key, shape in signatures[label].items():
            require(signatures['FINAL'].get(key) == shape, 'source/final key-shape mismatch: ' + key)
            merged[key] = shape
    require(merged == signatures['FINAL'], 'final model has uncovered or incompatible weight keys')
    emit('weight_contract_pass', final_keys=len(merged), parent_keys_complete=True)


def check_runtime(config_root, manifests):
    import torch
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils, v2xset_reproducibility
    from opencood.tools.check_doma_protocol_stamp import (
        _find_gate5_progressive_sample, _gt_hash, _sample_valid_modalities, _sample_source_records)
    from opencood.tools.inference_heter_in_order import _runtime_protocol_snapshot
    require(torch.cuda.is_available(), 'CUDA/spconv is required; runtime NOT PASSED')
    from efficientnet_pytorch.utils import url_map
    cache = Path(torch.hub.get_dir()) / 'checkpoints' / url_map['efficientnet-b0'].split('/')[-1]
    require(cache.is_file(), 'cache official EfficientNet-B0 weights before run: ' + str(cache))
    emit('device', name=torch.cuda.get_device_name(0), visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
         torch=torch.__version__, cuda=torch.version.cuda)
    signatures = {}
    # All nine models, both train and validation datasets, actual loss API.
    for label, relative in STAGES.items():
        if label == 'FINAL':
            continue
        h = load_runtime(config_root / relative)
        v2xset_reproducibility.configure(h)
        model = train_utils.create_model(h).cuda().eval()
        signatures[label] = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        for training in (True, False):
            dataset = build_dataset(copy.deepcopy(h), visualize=False, train=training)
            check_database_paths(dataset)
            expected_scenes = manifests['train' if training else 'validate'][
                'camera_scenes' if 'camera' in h['input_source'] else 'scenes']
            require(sorted(Path(p).name for p in dataset.scenario_folders) == expected_scenes, label + ' filtering')
            require(len(dataset) >= h['train_params']['batch_size'], label + ' drop_last leaves no batches')
            item = dataset[0]
            require(item is not None, label + ' empty sample')
            batch = train_utils.to_device(dataset.collate_batch_train([item]), torch.device('cuda'))
            with torch.no_grad():
                output = model(batch['ego'])
                require(finite_tensors(output) > 0, label + ' empty forward output')
                if label.startswith('A'):
                    pred, feat = output
                    losses = train_utils.create_losses_heter(h)
                    values = [losses[m](pred[m], batch['ego']['label_dict_protocol' if m == 'm0' else 'label_dict'])
                              for m in losses]
                    values.append(train_utils.create_adapter_loss(h)(*feat))
                    finite_tensors(values)
                    del pred, feat, values, losses
                else:
                    finite_tensors(train_utils.create_loss(h)(output, batch['ego']['label_dict']))
            emit('stage_smoke_pass', label=label, training_data=training, length=len(dataset))
            del batch, output, item, dataset
        del model
        gc.collect()
        torch.cuda.empty_cache()
    h = load_runtime(config_root / STAGES['FINAL'], 4)
    selection = build_dataset(h, visualize=True, train=False)
    check_database_paths(selection)
    index = _find_gate5_progressive_sample(selection)
    require(index is not None, 'no valid m1,m2,m3,m4 test sample')
    # A reference with no prefix cutoff (up to the protocol max_cav=5) proves
    # the CAV1-4 hashes also match the full valid-scene GT, not merely each other.
    full_dataset = build_dataset(load_runtime(config_root / STAGES['FINAL'], 5), visualize=True, train=False)
    full_gt_hash = _gt_hash(full_dataset[index])
    del full_dataset
    gt_hash = None
    full_pairwise = full_poses = None
    model = train_utils.create_model(h).cuda().eval()
    signatures['FINAL'] = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    check_weight_contracts(signatures)
    for cav in range(1, 5):
        h = load_runtime(config_root / STAGES['FINAL'], cav)
        dataset = selection if cav == 4 else build_dataset(h, visualize=True, train=False)
        _runtime_protocol_snapshot(dataset.params, RANGE, cav, dataset)
        require(sorted(Path(p).name for p in dataset.scenario_folders) == manifests['test']['camera_scenes'], 'test filter')
        records = _sample_source_records(dataset, index)
        for record in records:
            if record['modality'] == 'm3':
                require(any(p.endswith('.pcd') and not p.endswith(('_32.pcd', '_16.pcd'))
                            for p in record['source_paths']), 'actual m3 source is not raw')
        item = dataset[index]
        ego = item['ego']
        require(ego['agent_modality_list'] == ORDER[:cav] and ego['cav_num'] == cav, 'prefix mismatch')
        require(len(ego['cav_id_list']) == cav, 'prefix CAV IDs')
        full_count = len(_sample_valid_modalities(dataset, index))
        require(len(ego['lidar_poses']) == full_count, 'poses lost full-scene semantics')
        current_hash = _gt_hash(item)
        require(current_hash == full_gt_hash, 'prefix GT differs from full valid-scene GT')
        if gt_hash is not None:
            require(current_hash == gt_hash, 'GT changed with use_cav')
            import numpy as np
            require(np.array_equal(full_pairwise, ego['pairwise_t_matrix']), 'pairwise changed with prefix')
            require(np.array_equal(full_poses, ego['lidar_poses']), 'poses changed with prefix')
        gt_hash, full_pairwise, full_poses = current_hash, ego['pairwise_t_matrix'], ego['lidar_poses']
        observed, handles = {}, []
        for modality in ORDER[:cav]:
            for kind in ('adapter', 'reverter'):
                name = kind + '_' + modality
                def capture(module, inputs, output, key=name):
                    observed[key] = tuple(output.shape)
                handles.append(getattr(model, name).register_forward_hook(capture))
        try:
            batch = train_utils.to_device(dataset.collate_batch_test([item]), torch.device('cuda'))
            require(batch['ego']['record_len'].tolist() == [cav], 'record_len mismatch')
            with torch.no_grad():
                output = model(batch['ego'])
                finite_tensors(output)
                require('cls_preds' in output['m1'], 'ego detection head missing')
        finally:
            for handle in handles:
                handle.remove()
        for modality in ORDER[:cav]:
            for kind in ('adapter', 'reverter') if cav > 1 else ('adapter',):
                require(observed.get(kind + '_' + modality, ())[1:] == (64, 256, 512), 'shape contract ' + kind + modality)
        emit('final_smoke_pass', cav=cav, sample=index, gt_hash=gt_hash, full_cavs=full_count,
             sources=records, shapes=observed, samples=len(dataset))
        del batch, output, item
    emit('runtime_pass', formal_training_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-root', type=Path, default=CONFIG)
    parser.add_argument('--stage', choices=['yaml', 'data', 'all'], default='all')
    args = parser.parse_args()
    os.chdir(ROOT)
    try:
        _, assignments = check_static(args.config_root)
        if args.stage != 'yaml':
            manifests = check_data(assignments)
            if args.stage == 'all':
                check_runtime(args.config_root, manifests)
        emit('pass', stage=args.stage, runtime_verified=args.stage == 'all')
        return 0
    except Exception as exc:
        emit('FAIL', error='%s: %s' % (type(exc).__name__, exc), stage=args.stage)
        return 1


if __name__ == '__main__':
    sys.exit(main())
