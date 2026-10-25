# STAMP x DOMA V2XSet (isolated protocol)

Run from the repository root in the existing STAMP CUDA environment. No script
creates screen/tmux, connects to a server, commits source, or enables DDP.

```bash
bash scripts/run_stamp_doma_v2xset_full.sh check --static  # CPU config/CLI checks
bash scripts/run_stamp_doma_v2xset_full.sh check           # real data + CUDA preflight only
bash scripts/run_stamp_doma_v2xset_full.sh run --gpu auto  # preflight then the complete experiment
bash scripts/run_stamp_doma_v2xset_full.sh status
bash scripts/run_stamp_doma_v2xset_full.sh status --run-dir opencood/logs/STAMP_DOMA_V2XSET/<RUN_ID>
bash scripts/run_stamp_doma_v2xset_full.sh resume --run-dir opencood/logs/STAMP_DOMA_V2XSET/<RUN_ID> --gpu auto
```

`auto` respects an existing CUDA_VISIBLE_DEVICES allocation, otherwise selects the
GPU UUID with the most free memory reported by nvidia-smi. Exactly ONE process/GPU
is used at a time. `--gpu <index-or-UUID>` explicitly selects one device. The runner
clears inherited distributed launch variables. Activate the right Python environment
before execution, or set `PYTHON=/absolute/path/to/python` for the shell wrapper.

## Locked method and protocol

- IDs are DOMA IDs, not original STAMP IDs: m1=official m4 PointPillar,
  m2=official m1 EfficientNet LSS, m3=official m2 SECOND, m4=official m3 ResNet101 LSS.
- m1 and m3 read the same original V2XSET PCD source. No claim of separate 32-line
  sensors. No `_32.pcd`, `_16.pcd`, depth image or depth-supervision requirement.
- `L1 -> L2 -> L3 -> L4 -> P0 -> A1 -> A2 -> A3 -> A4 -> merge -> CAV1..4`.
  Each Ai loads the shared P0 as `protocol.pth` and the matching Li as `ego.pth`
  through STAMP's `load_original_model`; all frozen keys and shapes are verified.
- Original STAMP budgets: Li/P0=30 epochs, Ai=5, minimum validation-loss checkpoint.
  Original losses, optimizers, schedulers, batch sizes and stage training ranges
  remain unchanged. Training comm_range=70; final inference comm_range=180.
  This training budget is NOT equalized to HEAL/DOMA.
- Seed303 is explicitly opt-in, including Python/NumPy/torch/CUDA, worker seeds,
  DataLoader generators and launch-time PYTHONHASHSEED. This does not claim bitwise
  determinism for all CUDA/spconv kernels.
- Final: test split, max_cav=5, CAV input prefix=1/2/3/4, full valid-scene GT,
  detection range [-204.8,-102.4,-3,204.8,102.4,1], AP IoU=0.3/0.5/0.7.
- V2XSet's inherited camera scene exclusion `2021_09_09_13_20_58` is verified
  against the actual dataset scene lists. Camera local supervision retains the
  V2XSet postprocessor's original range filtering. Formal inference uses LiDAR GT.
- Assignment SHA256 is computed over compact UTF-8 JSON preserving insertion
  order (CAV order matters); raw-byte SHA256 is informational, not a CRLF gate.

## What prevents training

The formal runner requires committed tracked source, a usable GPU, the official
cached EfficientNet-B0 pretrained weights, all real splits, exact scene/CAV
assignment coverage, aligned timestamps, valid raw PCD headers, four readable RGB
PNGs and finite poses/intrinsics/camera poses. It validates actual database PCD
paths, official runtime YAML parsing, all nine training-stage model/loss forwards
on train and validation samples, and a progressive CAV1..4 test sample including
Adapter/Reverter shapes and GT equality with all valid CAVs. Missing prerequisites
are blockers, never automatic PASS or fallback data. The first file preflight
scans every assigned scene/frame; this can take time on network storage.

During final inference, an empty batch fails rather than silently skipping a test
frame. Each process records the initialized dataset's range, split, assignment,
CAV count and fusion method. Completed evaluations record total samples and a
per-frame, order-independent evaluator GT hash; all four runs must agree.

## Checkpoints and recovery

Only successful official training with a complete finite validation history and a
strictly loadable validation-best state_dict is COMPLETE. Adapter parents are
validated before use; their frozen state is checked again after training. The
official merger is used unmodified. The merged model must strictly load, have
exactly the union of source keys, and match every expected source tensor; random
initialization filling a missing key is rejected.

Stock checkpoints do not store a complete optimizer/scheduler/RNG state. Resume
therefore does NOT perform weight-only continuation: verified COMPLETE stages are
reused; a failed/incomplete stage restarts from its original parents in a new
attempt directory. Prior attempts are retained. Code/config/assignment changes or
changed dataset metadata/file-stat manifests block resume. Dataset manifests are
not advertised as full byte hashes of every image/point cloud. A per-run OS lock
prevents two runners from writing one experiment. A completed run is not rerun.

## Exact output layout

`R=opencood/logs/STAMP_DOMA_V2XSET/<UTC_TIMESTAMP>_<GIT_SHA12>_<UNIQUE_SUFFIX>`.
The runner prints the absolute `RUN_DIR` before preflight. Each retry increments
`attempt_001` to `attempt_002`, etc.; `state.json` always records exact active paths.

```text
R/
  state.json                              # status, SHA, source hashes, stage artifacts
  templates/                              # all 10 YAMLs + protocol_lock snapshot
  stages/preflight/attempt_001/preflight.log
  stages/L1/attempt_001/{config.yaml,train.log,net_epoch_bestval_atN.pth,checkpoint.json}
  stages/L2..L4,P0/attempt_001/...
  stages/A1..A4/attempt_001/{config.yaml,protocol.pth,ego.pth,parents.json,train.log,...}
  stages/merge/attempt_001/merge.log
  stages/merge/attempt_001/merge_manifest.json
  stages/merge/attempt_001/final_infer/convnext/{config.yaml,net_epoch1.pth}
  stages/CAV1/attempt_001/{config.yaml,net_epoch1.pth,inference.log,result.json}
  stages/CAV1/attempt_001/eval_intermediate_stamp_v2xset_204.8_102.4_epoch1_use_cav1.yaml
  stages/CAV2/attempt_001/eval_intermediate_stamp_v2xset_204.8_102.4_epoch1_use_cav2.yaml
  stages/CAV3/attempt_001/eval_intermediate_stamp_v2xset_204.8_102.4_epoch1_use_cav3.yaml
  stages/CAV4/attempt_001/eval_intermediate_stamp_v2xset_204.8_102.4_epoch1_use_cav4.yaml
  results.json                            # full-precision AP and source evidence
  results.csv                             # Method,CAV,AP@0.3,AP@0.5,AP@0.7,eval_file
```

`results.csv` is produced only after all four evaluations succeed and match their
GT/sample contracts. No pre-existing OPV2V-H output, checkpoint or config is used.

## Local tests (no training or inference)

```bash
bash -n scripts/run_stamp_doma_v2xset_full.sh
python -B -m unittest discover -s tests -p test_stamp_doma_v2xset.py -v
python -B opencood/tools/run_stamp_doma_v2xset.py check --static
git diff --check
```

The local CPU suite uses synthetic files/state_dicts where needed; it cannot certify
real V2XSet data, CUDA forwards, training convergence or AP. These remain mandatory
server-time checks and real experiment outputs.
