"""Single-GPU sequential STAMP pipeline. No weight-only optimizer-resume claims.

All writes are contained in a new run directory. Failed attempts are retained;
resume revalidates completed artifacts and restarts incomplete stages from their
original parents in a NEW attempt directory, never from partial checkpoints.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from opencood.tools.check_doma_protocol_stamp_v2xset import (
    CONFIG, STAGES, RANGE, require, sha256, read_yaml, load_runtime)

OUTPUT = ROOT / 'opencood/logs/STAMP_DOMA_V2XSET'


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf8')
    os.replace(temp, path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def source_manifest():
    files = sorted(set(ROOT.glob('opencood/**/*.py')) | set(CONFIG.glob('**/*.yaml')) |
                   set(ROOT.glob('opencood/modality_assign/v2xset_4modality*.json')) |
                   {ROOT / 'scripts/run_stamp_doma_v2xset_full.sh'})
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}


def gpu_environment(request):
    """Auto chooses an actually visible device, respecting scheduler visibility."""
    env = os.environ.copy()
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'SLURM_PROCID'):
        env.pop(key, None)  # this runner never opts into DDP
    if request == 'auto':
        visible = env.get('CUDA_VISIBLE_DEVICES')
        if visible is not None:
            devices = [v.strip() for v in visible.split(',') if v.strip() and v.strip() != '-1']
            require(devices, 'CUDA_VISIBLE_DEVICES exposes no GPUs')
            selected = devices[0]
        else:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
            devices = [line.split(',') for line in output.splitlines() if line.strip()]
            require(devices, 'nvidia-smi found no GPU')
            selected = max(devices, key=lambda row: int(row[1].strip()))[0].strip()
    else:
        require(',' not in request and request not in ('', '-1'), 'provide exactly one GPU index/UUID')
        selected = request
    env.update(CUDA_VISIBLE_DEVICES=selected, PYTHONHASHSEED='303', PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    print('Serial execution; CUDA_VISIBLE_DEVICES=' + selected, flush=True)
    return env


def execute(command, logfile, env):
    require(not logfile.exists(), 'refuse to overwrite log: ' + str(logfile))
    print('COMMAND ' + json.dumps(command), flush=True)
    with logfile.open('x', encoding='utf8') as log:
        log.write('COMMAND=' + json.dumps(command) + '\n')
        log.flush()
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, errors='replace',
                                 start_new_session=os.name == 'posix')
        try:
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
        except BaseException:
            if os.name == 'posix':
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
                child.wait()
            raise
    require(code == 0, 'exit %d: %s; log=%s' % (code, command, logfile))


def torch_state(path):
    import torch
    # Own generated files only. Older STAMP environments predate weights_only.
    try:
        state = torch.load(str(path), map_location='cpu', weights_only=True)
    except TypeError:
        state = torch.load(str(path), map_location='cpu')
    require(isinstance(state, dict) and state, 'not a state_dict: ' + str(path))
    for key, value in state.items():
        require(torch.is_tensor(value), 'non-tensor checkpoint key: ' + key)
        require(torch.isfinite(value).all().item(), 'nonfinite checkpoint tensor: ' + key)
    return state


def strict_checkpoint(config, checkpoint):
    from opencood.tools import train_utils
    model = train_utils.create_model(load_runtime(config))
    state = torch_state(checkpoint)
    model.load_state_dict(state, strict=True)
    result = dict(path=str(checkpoint), sha256=sha256(checkpoint), keys=len(state),
                  config=str(config), config_sha256=sha256(config))
    del model, state
    return result


def validate_training(log, epochs):
    text = log.read_text(encoding='utf8')
    require('Training Finished, checkpoints saved to ' in text, 'training did not finish')
    evaluations = re.findall(r'At epoch (\d+), the validation loss is ([^\s]+)', text)
    require(evaluations and all(math.isfinite(float(value)) for _, value in evaluations), 'invalid validation loss')
    # Original eval_freq=2 for 30 epochs; eval_freq=1 for 5 epochs.
    expected = list(range(0, epochs, 1 if epochs == 5 else 2))
    require([int(epoch) for epoch, _ in evaluations] == expected, 'incomplete/duplicated validation epochs')
    return {int(epoch) + 1: float(value) for epoch, value in evaluations}


def export_file(source, destination):
    if destination.exists():
        require(sha256(source) == sha256(destination), 'existing export differs: ' + str(destination))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def records(log, marker):
    return [json.loads(line.split(marker, 1)[1]) for line in log.read_text(encoding='utf8').splitlines()
            if line.startswith(marker)]


def evaluate_record(log, directory, cav):
    runtime = records(log, 'STAMP_DOMA_RUNTIME_PROTOCOL=')
    complete = records(log, 'STAMP_V2XSET_EVAL_COMPLETE=')
    require(len(runtime) == len(complete) == 1, 'missing/duplicate runtime completion evidence')
    runtime, complete = runtime[0], complete[0]
    require(runtime['detection_range'] == RANGE and runtime['comm_range'] == 180, 'runtime protocol mismatch')
    require(runtime['fusion_core_method'] == 'intermediateheteradapter', 'runtime fusion mismatch')
    require(runtime['assignment_path'] == 'opencood/modality_assign/v2xset_4modality_in_order.json', 'runtime assignment')
    require(runtime['dataset_root_dir'] == runtime['test_dir'] == 'dataset/V2XSET/test', 'runtime test split')
    require(runtime['dataset_use_cav'] == complete['use_cav'] == cav and complete['samples'] > 0, 'CAV runtime')
    require(Path(complete['eval_file']).name == complete['eval_file'], 'invalid evaluation output name')
    file = directory / complete['eval_file']
    ap = read_yaml(file)
    scores = [float(ap[k]) for k in ('ap30', 'ap_50', 'ap_70')]
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in scores), 'invalid AP result')
    return dict(cav=cav, ap=scores, samples=complete['samples'], gt_sha256=complete['gt_sha256'],
                eval_file=str(file), eval_sha256=sha256(file), runtime=runtime, log=str(log))


class Pipeline:
    def __init__(self, directory, state, env):
        self.directory, self.state, self.env = directory, state, env
        self.config = directory / 'templates'

    def save(self):
        self.state['updated_at'] = now()
        atomic_json(self.directory / 'state.json', self.state)

    def attempt(self, label):
        base = self.directory / 'stages' / label
        base.mkdir(parents=True, exist_ok=True)
        index = 1
        while (base / ('attempt_%03d' % index)).exists():
            index += 1
        attempt = base / ('attempt_%03d' % index)
        attempt.mkdir()
        self.state['active'] = label
        self.state['stages'][label] = dict(status='RUNNING', attempt=str(attempt), started_at=now())
        self.save()
        return attempt

    def verify_complete(self, label, parents):
        entry = self.state['stages'].get(label, {})
        if entry.get('status') != 'COMPLETE':
            return False
        require(entry.get('parents', {}) == parents, label + ' parent provenance mismatch')
        for file, digest in entry['artifacts'].items():
            require(Path(file).is_file() and sha256(file) == digest, label + ' artifact changed: ' + file)
        if 'checkpoint' in entry:
            strict_checkpoint(Path(entry['config']), Path(entry['checkpoint']))
        print('Verified COMPLETE: ' + label, flush=True)
        return True

    def complete(self, label, files, parents=None, **fields):
        entry = self.state['stages'][label]
        entry.update(status='COMPLETE', finished_at=now(), parents=parents or {},
                     artifacts={str(p): sha256(p) for p in files}, **fields)
        self.state['active'] = None
        self.save()

    def preflight(self):
        previous_data = self.state.get('data_manifest', [])
        attempt = self.attempt('preflight')
        log = attempt / 'preflight.log'
        execute([sys.executable, '-B', 'opencood/tools/check_doma_protocol_stamp_v2xset.py',
                 '--config-root', str(self.config), '--stage', 'all'], log, self.env)
        checks = records(log, 'V2XSET_CHECK=')
        require(checks[-1].get('runtime_verified') is True and checks[-1]['event'] == 'pass', 'preflight incomplete')
        current_data = [v for v in checks if v['event'] == 'data_pass']
        require(not previous_data or current_data == previous_data, 'dataset metadata/file manifest changed; start a new run')
        self.state['data_manifest'] = current_data
        self.complete('preflight', [log], checks=checks)

    def train(self, label):
        parents = {}
        if label.startswith('A'):
            for filename, source_label in [('protocol.pth', 'P0'), ('ego.pth', 'L' + label[1])]:
                source = self.state['stages'][source_label]
                require(source['status'] == 'COMPLETE', 'parent not complete: ' + source_label)
                parents[filename] = dict(stage=source_label, path=source['checkpoint'], sha256=sha256(source['checkpoint']))
        if self.verify_complete(label, parents):
            return
        attempt = self.attempt(label)
        config = attempt / 'config.yaml'
        shutil.copy2(self.config / STAGES[label], config)
        for filename, parent in parents.items():
            shutil.copy2(parent['path'], attempt / filename)
            require(sha256(attempt / filename) == parent['sha256'], 'parent copy mismatch')
        if parents:
            self.verify_adapter_parents(config, attempt, parents)
            atomic_json(attempt / 'parents.json', parents)
        tool = 'train_adapter.py' if label.startswith('A') else 'train.py'
        log = attempt / 'train.log'
        execute([sys.executable, '-B', 'opencood/tools/' + tool, '--hypes_yaml', str(config),
                 '--model_dir', str(attempt)], log, self.env)
        evaluations = validate_training(log, 5 if label.startswith('A') else 30)
        best = list(attempt.glob('net_epoch_bestval_at*.pth'))
        require(len(best) == 1, label + ' must have exactly one validation-best checkpoint')
        best_epoch = int(re.fullmatch(r'net_epoch_bestval_at(\d+)\.pth', best[0].name).group(1))
        require(best_epoch in evaluations and evaluations[best_epoch] == min(evaluations.values()),
                'checkpoint does not match minimum validation loss in official log')
        audit = strict_checkpoint(config, best[0])
        if parents:
            import torch
            trained = torch_state(best[0])
            for filename in parents:
                for key, tensor in torch_state(attempt / filename).items():
                    require(torch.equal(trained[key], tensor), 'frozen parent changed during adapter training: ' + key)
            del trained
        atomic_json(attempt / 'checkpoint.json', audit)
        files = [config, log, best[0], attempt / 'checkpoint.json']
        files += [attempt / name for name in parents]
        if parents:
            files.append(attempt / 'parents.json')
        self.complete(label, files, parents, checkpoint=str(best[0]), config=str(config), audit=audit)

    @staticmethod
    def verify_adapter_parents(config, directory, parents):
        import torch
        from opencood.tools import train_utils
        model = train_utils.create_model(load_runtime(config))
        expected = model.state_dict()
        provided = {}
        for filename in ('protocol.pth', 'ego.pth'):
            state = torch_state(directory / filename)
            require(set(state) <= set(expected), 'unexpected parent keys: ' + filename)
            for key, tensor in state.items():
                require(tensor.shape == expected[key].shape, 'parent shape mismatch: ' + key)
            provided.update(state)
        frozen = {key for key in expected if not key.startswith(('adapter_', 'reverter_'))}
        require(frozen <= provided.keys(), 'missing pretrained non-adapter keys: ' + str(frozen - provided.keys()))
        train_utils.load_original_model(str(directory), model)
        require(all(torch.equal(model.state_dict()[key], provided[key]) for key in frozen), 'official parent loader mismatch')
        del model, provided

    def merge(self):
        parents = {label: sha256(self.state['stages'][label]['checkpoint']) for label in STAGES if label != 'FINAL'}
        if self.verify_complete('merge', parents):
            return
        attempt = self.attempt('merge')
        # Stock merger directory layout, containing only validated best checkpoints.
        for label, relative in STAGES.items():
            if label == 'FINAL':
                continue
            entry = self.state['stages'][label]
            destination = attempt / Path(relative).parent
            export_file(Path(entry['config']), destination / 'config.yaml')
            export_file(Path(entry['checkpoint']), destination / Path(entry['checkpoint']).name)
        final = attempt / Path(STAGES['FINAL']).parent
        final.mkdir(parents=True)
        config = final / 'config.yaml'
        shutil.copy2(self.config / STAGES['FINAL'], config)
        log = attempt / 'merge.log'
        execute([sys.executable, '-B', 'opencood/tools/merge_model_w_adapter.py', '--model_dir', str(attempt),
                 '--adapter_dir', 'convnext', '--with_protocol'], log, self.env)
        checkpoint = final / 'net_epoch1.pth'
        audit = strict_checkpoint(config, checkpoint)
        import torch
        merged = torch_state(checkpoint)
        expected, owners = {}, {}
        for label in ['L1', 'L2', 'L3', 'L4', 'A1', 'A2', 'A3', 'A4', 'P0']:
            source = torch_state(self.state['stages'][label]['checkpoint'])
            for key, tensor in source.items():
                expected[key], owners[key] = tensor, label
        require(set(merged) == set(expected), 'merge missing/extra keys (random initialization forbidden)')
        require(all(torch.equal(merged[key], expected[key]) for key in merged), 'merge tensor/source mismatch')
        manifest = attempt / 'merge_manifest.json'
        atomic_json(manifest, dict(checkpoint=audit, source_checkpoints={k: self.state['stages'][k]['audit'] for k in parents},
                                   key_owners=owners, strict_load=True, exact_source_values=True))
        self.complete('merge', [checkpoint, config, log, manifest], parents,
                      checkpoint=str(checkpoint), config=str(config), audit=audit)

    def inference(self, cav):
        merge = self.state['stages']['merge']
        parents = dict(checkpoint=sha256(merge['checkpoint']), config=sha256(merge['config']))
        label = 'CAV%d' % cav
        if self.verify_complete(label, parents):
            return
        attempt = self.attempt(label)
        shutil.copy2(merge['config'], attempt / 'config.yaml')
        shutil.copy2(merge['checkpoint'], attempt / 'net_epoch1.pth')
        log = attempt / 'inference.log'
        execute([sys.executable, '-B', 'opencood/tools/inference_heter_in_order.py', '--model_dir', str(attempt),
                 '--fusion_method', 'intermediate', '--range', '204.8,102.4', '--use_cav', '[%d]' % cav,
                 '--note', '_stamp_v2xset', '--save_vis_interval', '1000000000'], log, self.env)
        require(sha256(attempt / 'config.yaml') == parents['config'] and
                sha256(attempt / 'net_epoch1.pth') == parents['checkpoint'], 'inference inputs changed')
        result = evaluate_record(log, attempt, cav)
        atomic_json(attempt / 'result.json', result)
        self.complete(label, [attempt / 'config.yaml', attempt / 'net_epoch1.pth', log,
                              Path(result['eval_file']), attempt / 'result.json'], parents, result=result)

    def summarize(self):
        results = [self.state['stages']['CAV%d' % cav]['result'] for cav in range(1, 5)]
        require(len({r['samples'] for r in results}) == 1, 'CAV test counts differ')
        require(len({r['gt_sha256'] for r in results}) == 1, 'evaluator GT differs across CAV runs')
        atomic_json(self.directory / 'results.json', dict(method='STAMP', results=results,
                                                        git_sha=self.state['git_sha'], lock=read_yaml(self.config / 'protocol_lock.yaml')))
        path = self.directory / 'results.csv'
        if not path.exists():
            with path.open('x', newline='', encoding='utf8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['Method', 'CAV', 'AP@0.3', 'AP@0.5', 'AP@0.7', 'eval_file'])
                for result in results:
                    writer.writerow(['STAMP', result['cav'], *result['ap'], result['eval_file']])
        self.state.update(status='COMPLETE', finished_at=now(), active=None)
        self.save()
        print('COMPLETE: ' + str(path))

    def run(self):
        try:
            self.preflight()
            for label in STAGES:
                if label != 'FINAL':
                    self.train(label)
            self.merge()
            for cav in range(1, 5):
                self.inference(cav)
            self.summarize()
        except BaseException as exc:
            active = self.state.get('active')
            if active:
                self.state['stages'][active].update(status='FAILED', error=str(exc), traceback=traceback.format_exc())
            self.state.update(status='FAILED', error=str(exc))
            self.save()
            raise


@contextmanager
def run_lock(directory):
    import fcntl
    with (directory / '.run.lock').open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run', 'resume', 'status'])
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--gpu', default='auto', help='one visible GPU index/UUID; default discovers a real device')
    parser.add_argument('--static', action='store_true', help='check only: no dataset/CUDA imports')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == 'status':
        if args.run_dir:
            state = json.loads((args.run_dir / 'state.json').read_text(encoding='utf8'))
            print(json.dumps({k: state.get(k) for k in ('status', 'git_sha', 'active', 'error', 'updated_at')}, indent=2))
            for label, entry in state['stages'].items():
                print(label, entry['status'], entry.get('attempt'), entry.get('result', {}).get('ap', ''))
            if (args.run_dir / 'results.csv').is_file():
                print((args.run_dir / 'results.csv').read_text(encoding='utf8'))
        else:
            for file in sorted(OUTPUT.glob('*/state.json')):
                state = json.loads(file.read_text(encoding='utf8'))
                print(file.parent, state['status'], state.get('active'))
        return 0
    require(not args.static or args.command == 'check', '--static is only allowed for check')
    if args.command == 'check':
        env = os.environ.copy() if args.static else gpu_environment(args.gpu)
        return subprocess.call([sys.executable, '-B', 'opencood/tools/check_doma_protocol_stamp_v2xset.py',
                                '--stage', 'yaml' if args.static else 'all'], cwd=ROOT, env=env)
    require(sys.platform == 'linux', 'formal run/resume is supported on the Linux CUDA server only')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('received signal %d' % signum)
    signal.signal(signal.SIGTERM, interrupted)
    require(not git('diff', '--name-only') and not git('diff', '--cached', '--name-only'),
            'publish tracked source edits before a formal run; no automatic commit')
    for path in [*CONFIG.glob('**/*.yaml'), Path(__file__),
                 ROOT / 'opencood/tools/check_doma_protocol_stamp_v2xset.py',
                 ROOT / 'opencood/tools/v2xset_reproducibility.py', ROOT / 'scripts/run_stamp_doma_v2xset_full.sh']:
        git('ls-files', '--error-unmatch', path.relative_to(ROOT).as_posix())
    env = gpu_environment(args.gpu)
    # CPU checkpoint audits may import model code: use the same explicit device/env.
    os.environ.update(env)
    if args.command == 'run':
        require(args.run_dir is None, 'run always creates a new unique directory; use resume for an existing run')
        identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + git('rev-parse', '--short=12', 'HEAD') + '_' + uuid.uuid4().hex[:6]
        directory = OUTPUT / identity
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copytree(CONFIG, directory / 'templates')
        state = dict(status='RUNNING', created_at=now(), git_sha=git('rev-parse', 'HEAD'),
                     source_manifest=source_manifest(), stages={}, env={k: env[k] for k in
                     ('CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED')}, resume_policy='restart_incomplete_stage_no_optimizer_resume')
        atomic_json(directory / 'state.json', state)
    else:
        require(args.run_dir is not None, 'resume requires --run-dir')
        directory = args.run_dir.resolve()
        require(directory.parent == OUTPUT.resolve(), 'resume path must be an existing V2XSet run directory')
        state = json.loads((directory / 'state.json').read_text(encoding='utf8'))
        require(state['git_sha'] == git('rev-parse', 'HEAD') and state['source_manifest'] == source_manifest(),
                'source changed; resume refused (use a new run)')
        for relative in [*STAGES.values(), 'protocol_lock.yaml']:
            require(sha256(directory / 'templates' / relative) == sha256(CONFIG / relative), 'template snapshot changed')
        if state['status'] == 'COMPLETE':
            for label, entry in state['stages'].items():
                require(entry['status'] == 'COMPLETE', 'inconsistent COMPLETE state: ' + label)
                for path, digest in entry['artifacts'].items():
                    require(Path(path).is_file() and sha256(path) == digest, 'completed artifact changed: ' + path)
                if 'checkpoint' in entry:
                    strict_checkpoint(Path(entry['config']), Path(entry['checkpoint']))
            print('Verified already COMPLETE; no files changed: ' + str(directory))
            return 0
    print('RUN_DIR=' + str(directory), flush=True)
    with run_lock(directory):
        Pipeline(directory, state, env).run()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('BLOCKER: %s: %s' % (type(exc).__name__, exc), file=sys.stderr)
        sys.exit(1)
