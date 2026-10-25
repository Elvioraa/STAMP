"""CPU-only contracts. Run: python -B -m unittest discover -s tests -p test_stamp_doma_v2xset.py -v"""
import ast
import copy
import io
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

from opencood.tools import check_doma_protocol_stamp_v2xset as check
from opencood.tools import run_stamp_doma_v2xset as runner
from opencood.tools import v2xset_reproducibility as seeds


def isolated_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf8'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError('function not found')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class ProtocolTests(unittest.TestCase):
    def test_static_and_cli(self):
        check.check_static()

    def test_parent_merge_shape_contracts(self):
        signatures = {'P0': {'encoder_m0.weight': (2, 2)}}
        final = dict(signatures['P0'])
        for index in range(1, 5):
            local = {'encoder_m%d.weight' % index: (2, 2)}
            signatures['L%d' % index] = local
            signatures['A%d' % index] = dict(signatures['P0'], **local,
                **{'adapter_m%d.weight' % index: (2, 2), 'reverter_m%d.weight' % index: (2, 2)})
            final.update(signatures['A%d' % index])
        signatures['FINAL'] = final
        check.check_weight_contracts(signatures)
        signatures['A2']['encoder_m2.weight'] = (3, 2)
        with self.assertRaises(RuntimeError):
            check.check_weight_contracts(signatures)

    def test_duplicate_yaml_rejected(self):
        import yaml
        with self.assertRaises(RuntimeError):
            yaml.load('a: 1\na: 2\n', Loader=check.UniqueLoader)

    def test_assignment_hash_is_newline_stable_and_order_sensitive(self):
        left = json.loads('{"1":"m1",\n"2":"m2"}')
        right = json.loads('{"1":"m1",\r\n"2":"m2"}')
        self.assertEqual(check.ordered_json_hash(left), check.ordered_json_hash(right))
        self.assertNotEqual(check.ordered_json_hash(left), check.ordered_json_hash(dict(reversed(list(left.items())))))

    def test_original_training_math_and_budgets_unchanged(self):
        original = check.ROOT / 'opencood/hypes_yaml/opv2v/doma_protocol_stamp_v2r2'
        for label, relative in check.STAGES.items():
            if label == 'FINAL':
                continue
            before, after = check.read_yaml(original / relative), check.read_yaml(check.CONFIG / relative)
            for key in ('optimizer', 'lr_scheduler', 'loss', 'loss_adapter', 'train_params'):
                self.assertEqual(before.get(key), after.get(key), (label, key))
            def disable_depth(node):
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key == 'depth_supervision':
                            node[key] = False
                        else:
                            disable_depth(value)
                elif isinstance(node, list):
                    for value in node:
                        disable_depth(value)
            disable_depth(before['model'])
            self.assertEqual(before['model'], after['model'], label)

    def test_inference_exact_mapping_and_raw_pcd(self):
        ns = isolated_functions(check.ROOT / 'opencood/tools/inference_heter_in_order.py',
                                ['_prepare_inference_hypes', '_inference_lidar_configs'],
                                {'os': os, 'FORMAL_COMM_RANGE': 180})
        for source, target in [('intermediateheter', 'intermediateheterinfer'),
                               ('intermediateheteradapter', 'intermediateheteradapter'),
                               ('intermediateheterinfer', 'intermediateheterinfer'), ('late', 'late')]:
            h = {'fusion': {'core_method': source, 'dataset': 'v2xset'},
                 'heter': {'assignment_path': 'v2xset_4modality_in_order.json'}}
            resolved = ns['_prepare_inference_hypes'](h, 'intermediate')
            self.assertEqual(resolved['fusion']['core_method'], target)
            self.assertEqual(resolved['heter']['assignment_path'], 'v2xset_4modality_in_order.json')
            self.assertEqual(resolved['comm_range'], 180)
            self.assertEqual(ns['_inference_lidar_configs'](h, [1, 2, 3, 4]), [(i, None) for i in range(1, 5)])
            with self.assertRaises(ValueError):
                ns['_inference_lidar_configs'](h, [4], True)
            h['heter']['lidar_channels_dict'] = {'m3': 32}
            with self.assertRaises(ValueError):
                ns['_inference_lidar_configs'](h, [4])
            h['fusion']['dataset'] = 'opv2v'
            self.assertEqual(ns['_inference_lidar_configs'](h, [4]), [(4, {'m3': 32})])
            self.assertEqual(ns['_inference_lidar_configs'](h, [4], True),
                             [(4, {'m1': 32, 'm3': 16}), (4, {'m1': 16, 'm3': 16})])

    def test_camera_mask_argument_forwards_without_default_change(self):
        file = check.ROOT / 'opencood/data_utils/datasets/basedataset/v2xset_basedataset.py'
        tree = ast.parse(file.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        ns = {'OPV2VBaseDataset': object}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(file), 'exec'), ns)
        instance = ns['V2XSETBaseDataset'].__new__(ns['V2XSETBaseDataset'])
        from unittest.mock import Mock
        instance.post_processor = Mock()
        instance.generate_object_center_camera([], [0] * 6, mask_outside_range=False)
        instance.post_processor.generate_object_center_v2xset_camera.assert_called_with([], [0] * 6, mask_outside_range=False)
        instance.generate_object_center_camera([], [0] * 6)
        instance.post_processor.generate_object_center_v2xset_camera.assert_called_with([], [0] * 6, mask_outside_range=True)

    def test_seed_optin(self):
        import numpy as np
        import torch
        h = {'v2xset_reproducibility': True, 'seed': 303, 'fusion': {'dataset': 'v2xset'}}
        def sample():
            return random.random(), float(np.random.rand()), float(torch.rand(1))
        seeds.configure(h)
        first = sample()
        seeds.configure(h)
        self.assertEqual(first, sample())
        before = random.getstate()
        seeds.configure({})
        self.assertEqual(before, random.getstate())
        self.assertEqual(seeds.loader_kwargs({}), {})
        self.assertEqual(seeds.loader_kwargs(h)['generator'].initial_seed(), 303)
        self.assertEqual(seeds.loader_kwargs(h, True)['generator'].initial_seed(), 304)
        torch.manual_seed(9123)
        seeds.seed_worker(0)
        pair = (random.random(), float(np.random.rand()))
        seeds.seed_worker(0)
        self.assertEqual(pair, (random.random(), float(np.random.rand())))

    def test_actual_seeded_loader_repeatability(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        h = {'v2xset_reproducibility': True, 'seed': 303}
        dataset = TensorDataset(torch.arange(12))
        def collect():
            return [x[0].tolist() for x in DataLoader(dataset, batch_size=3, shuffle=True,
                                                     num_workers=0, **seeds.loader_kwargs(h))]
        self.assertEqual(collect(), collect())

    def test_gpu_respects_visible_and_no_ddp(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-example-b,GPU-example-a', 'RANK': '1'}):
            env = runner.gpu_environment('auto')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-example-b')
            self.assertNotIn('RANK', env)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '-1'}):
            with self.assertRaises(RuntimeError):
                runner.gpu_environment('auto')

    def test_gpu_auto_discovers_uuid_not_assumed_index(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(runner.subprocess, 'check_output',
                return_value='GPU-actual-a, 2000\nGPU-actual-b, 7000\n'):
            self.assertEqual(runner.gpu_environment('auto')['CUDA_VISIBLE_DEVICES'], 'GPU-actual-b')

    def test_strict_checkpoint_rejects_missing_or_wrong_shape(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            config, checkpoint = Path(temp) / 'config.yaml', Path(temp) / 'test.pth'
            config.write_text('synthetic: true')
            fake = types.ModuleType('opencood.tools.train_utils')
            fake.create_model = lambda h: torch.nn.Linear(2, 1)
            with patch.dict('sys.modules', {'opencood.tools.train_utils': fake}), patch.object(runner, 'load_runtime', return_value={}):
                torch.save(torch.nn.Linear(2, 1).state_dict(), checkpoint)
                self.assertEqual(runner.strict_checkpoint(config, checkpoint)['keys'], 2)
                torch.save({'weight': torch.zeros(1, 2)}, checkpoint)
                with self.assertRaises(RuntimeError):
                    runner.strict_checkpoint(config, checkpoint)
                torch.save(torch.nn.Linear(3, 1).state_dict(), checkpoint)
                with self.assertRaises(RuntimeError):
                    runner.strict_checkpoint(config, checkpoint)

    def test_real_file_preflight_and_missing_camera_rejection(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assignment = {'training_assignment': {}, 'final_assignment': {}}
            for split in ('train', 'validate', 'test'):
                scene = 'scene_' + split
                assignment['training_assignment'][scene] = {'1': 'm1'}
                if split == 'test':
                    assignment['final_assignment'][scene] = {'1': 'm1'}
                directory = root / 'dataset/V2XSET' / split / scene / '1'
                directory.mkdir(parents=True)
                params = {'lidar_pose': [0] * 6, 'vehicles': {}}
                for i in range(4):
                    params['camera%d' % i] = {'cords': [0] * 6, 'intrinsic': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
                    Image.new('RGB', (800, 600)).save(directory / ('000001_camera%d.png' % i))
                (directory / '000001.yaml').write_text(json.dumps(params))
                (directory / '000001.pcd').write_bytes(b'FIELDS x y z rgb\nDATA ascii\n0 0 0 0\n')
            with patch.object(check, 'ROOT', root), patch('sys.stdout', new=io.StringIO()):
                manifests = check.check_data(assignment)
                self.assertEqual(manifests['test']['images'], 4)
                (directory / '000001_camera3.png').unlink()
                with self.assertRaises(FileNotFoundError):
                    check.check_data(assignment)

    def test_incomplete_training_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / 'train.log'
            log.write_text('At epoch 0, the validation loss is 0.1\nTraining Finished, checkpoints saved to x')
            with self.assertRaises(RuntimeError):
                runner.validate_training(log, 30)
            log.write_text(''.join('At epoch %d, the validation loss is 0.1\n' % e for e in range(5)) +
                           'Training Finished, checkpoints saved to x')
            self.assertEqual(len(runner.validate_training(log, 5)), 5)

    def test_failure_never_complete_and_resume_preserves_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            pipeline = runner.Pipeline(path, {'stages': {}, 'status': 'RUNNING'}, {})
            def fail():
                pipeline.attempt('preflight')
                raise RuntimeError('synthetic failure')
            with patch.object(pipeline, 'preflight', side_effect=fail), self.assertRaises(RuntimeError):
                pipeline.run()
            self.assertEqual(pipeline.state['status'], 'FAILED')
            self.assertEqual(pipeline.state['stages']['preflight']['status'], 'FAILED')
            old = Path(pipeline.state['stages']['preflight']['attempt'])
            new = pipeline.attempt('preflight')
            self.assertNotEqual(old, new)
            self.assertTrue(old.is_dir())

    def test_corrupt_complete_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            file = path / 'test'
            file.write_text('first')
            state = {'stages': {'x': {'status': 'COMPLETE', 'parents': {}, 'artifacts': {str(file): runner.sha256(file)}}}}
            pipeline = runner.Pipeline(path, state, {})
            self.assertTrue(pipeline.verify_complete('x', {}))
            file.write_text('tampered')
            with self.assertRaises(RuntimeError):
                pipeline.verify_complete('x', {})

    def test_evaluation_parser_keeps_precision_and_checks_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            log = directory / 'inference.log'
            evaluation = directory / 'eval_test.yaml'
            evaluation.write_text('ap30: 0.123456789\nap_50: 0.23456789\nap_70: 0.3456789\n')
            runtime = dict(detection_range=check.RANGE, comm_range=180, fusion_core_method='intermediateheteradapter',
                           assignment_path='opencood/modality_assign/v2xset_4modality_in_order.json',
                           dataset_root_dir='dataset/V2XSET/test', test_dir='dataset/V2XSET/test', dataset_use_cav=1)
            complete = dict(use_cav=1, samples=2, gt_sha256='test', eval_file='eval_test.yaml')
            log.write_text('STAMP_DOMA_RUNTIME_PROTOCOL=' + json.dumps(runtime) + '\n' +
                           'STAMP_V2XSET_EVAL_COMPLETE=' + json.dumps(complete) + '\n')
            result = runner.evaluate_record(log, directory, 1)
            self.assertEqual(result['ap'][0], 0.123456789)
            with self.assertRaises(RuntimeError):
                runner.evaluate_record(log, directory, 2)

    def test_synthetic_full_pipeline_and_completed_stage_reuse(self):
        """Exercise orchestration only: fake processes, synthetic weights, NO training."""
        import torch
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            shutil.copytree(check.CONFIG, directory / 'templates')
            state = {'stages': {}, 'status': 'RUNNING', 'git_sha': 'synthetic-test'}
            pipeline = runner.Pipeline(directory, state, {})
            calls = []
            def preflight():
                attempt = pipeline.attempt('preflight')
                log = attempt / 'synthetic.log'
                log.write_text('synthetic preflight, no GPU/data verification')
                pipeline.complete('preflight', [log])
            def strict(config, checkpoint):
                return dict(path=str(checkpoint), sha256=runner.sha256(checkpoint),
                            config=str(config), config_sha256=runner.sha256(config),
                            keys=len(runner.torch_state(checkpoint)))
            def execute(command, log, env):
                tool = Path(command[2]).name
                calls.append(tool)
                path = Path(command[command.index('--model_dir') + 1])
                if tool in ('train.py', 'train_adapter.py'):
                    h = check.read_yaml(path / 'config.yaml')
                    modality = h['heter']['ego_modality']
                    model = {'encoder_' + modality + '.weight': torch.tensor([float(modality[1])])}
                    if tool == 'train_adapter.py':
                        model = dict(runner.torch_state(path / 'protocol.pth'), **runner.torch_state(path / 'ego.pth'))
                        model.update({'adapter_' + modality + '.weight': torch.ones(1),
                                      'reverter_' + modality + '.weight': torch.ones(1)})
                    torch.save(model, path / 'net_epoch_bestval_at1.pth')
                    log.write_text(''.join('At epoch %d, the validation loss is 0.1\n' % i
                                          for i in range(0, h['train_params']['epoches'], h['train_params']['eval_freq'])) +
                                   'Training Finished, checkpoints saved to ' + str(path))
                elif tool == 'merge_model_w_adapter.py':
                    merged = {}
                    for label in ['L1', 'L2', 'L3', 'L4', 'A1', 'A2', 'A3', 'A4', 'P0']:
                        ckpt = next((path / Path(check.STAGES[label]).parent).glob('net_epoch_bestval_at*.pth'))
                        merged.update(runner.torch_state(ckpt))
                    torch.save(merged, path / 'final_infer/convnext/net_epoch1.pth')
                    log.write_text('synthetic merge')
                else:
                    self.assertEqual(tool, 'inference_heter_in_order.py')
                    cav = json.loads(command[command.index('--use_cav') + 1])[0]
                    filename = 'eval_intermediate_stamp_v2xset_204.8_102.4_epoch1_use_cav%d.yaml' % cav
                    (path / filename).write_text('ap30: 0.11111111\nap_50: 0.1\nap_70: 0.01\n')
                    runtime = dict(detection_range=check.RANGE, comm_range=180, fusion_core_method='intermediateheteradapter',
                                   assignment_path='opencood/modality_assign/v2xset_4modality_in_order.json',
                                   dataset_root_dir='dataset/V2XSET/test', test_dir='dataset/V2XSET/test', dataset_use_cav=cav)
                    complete = dict(use_cav=cav, samples=2, gt_sha256='synthetic-identical-gt', eval_file=filename)
                    log.write_text('STAMP_DOMA_RUNTIME_PROTOCOL=' + json.dumps(runtime) + '\n' +
                                   'STAMP_V2XSET_EVAL_COMPLETE=' + json.dumps(complete) + '\n')
            with patch.object(pipeline, 'preflight', side_effect=preflight), patch.object(runner, 'execute', side_effect=execute), \
                 patch.object(runner, 'strict_checkpoint', side_effect=strict), patch.object(pipeline, 'verify_adapter_parents'):
                pipeline.run()
                self.assertEqual(state['status'], 'COMPLETE')
                self.assertEqual(calls.count('train.py'), 5)
                self.assertEqual(calls.count('train_adapter.py'), 4)
                self.assertEqual(calls.count('merge_model_w_adapter.py'), 1)
                self.assertEqual(calls.count('inference_heter_in_order.py'), 4)
                first_calls = list(calls)
                pipeline.run()
                self.assertEqual(calls, first_calls)
            self.assertEqual(len((directory / 'results.csv').read_text().splitlines()), 5)
            for i in range(1, 5):
                parents = state['stages']['A%d' % i]['parents']
                self.assertEqual(parents['protocol.pth']['stage'], 'P0')
                self.assertEqual(parents['ego.pth']['stage'], 'L%d' % i)


if __name__ == '__main__':
    unittest.main()
