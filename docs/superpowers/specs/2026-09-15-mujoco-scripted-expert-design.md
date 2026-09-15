# MuJoCo Table Scene and Scripted Expert

**Date:** 2026-09-15  
**Status:** Draft for review  
**Slice:** first RecoverVLA sub-project (simulation + demonstration collection)

## Goal

Add a real MuJoCo bimanual table-setting scene and a scripted expert that records LeRobot-compatible demonstrations. Keep the existing dry-run control plane unchanged and honest.

This slice does not train ACT, export OpenVINO, or claim Intel hardware results.

## Why this slice first

The L4 GPU is only useful after we can generate demonstrations. The current tree is a stdlib planner plus `DryRunExecutor`. Collection, not training, is the blocker.

## Non-goals

- Learned policy, ACT training, or SmolVLA
- OpenVINO IR, ONNX, or Core Ultra benchmarks
- OMPL, MoveIt, ROS, or a general IK solver
- True CFD / MuJoCo flex fluid
- Grasping the fork, knife, or napkin
- Replacing `DryRunExecutor` as the default `evaluate()` path
- Committing datasets, videos, checkpoints, or binary weights

## Architecture

Keep `recovervla.planner`, `recovervla.types`, and `evaluate()` as they are. Add an optional `recovervla.sim` package and a separate collection CLI.

```text
instruction -> TableSettingPlanner -> TaskPlan
                                      |
                                      v
SceneVariation(seed) -> TableScene (MuJoCo) -> ScriptedExpert
                                      |              |
                                      v              v
                               cameras/qpos     ctrl waypoints
                                      \              /
                                       v            v
                                    Recorder -> LeRobot v2 dataset
                                       |
                                       v
                                 MujocoExecutor -> RunResult
                                 executor="mujoco"
```

`evaluate()` remains dry-run unless a later slice wires `--executor mujoco`. This slice’s runnable command is `python3 -m recovervla.collect`.

## Components

| Unit | Responsibility | Depends on |
|---|---|---|
| `assets/so101/` | Vendored SO-101 MJCF + meshes | none |
| `assets/table_scene.xml` | Dual-arm table world, objects, cameras, keyframes, particles | SO-101 include |
| `sim/scene.py` | Load model, apply seed randomization, step, render | MuJoCo |
| `sim/success.py` | Skill and task predicates, including particle-in-mug | scene |
| `sim/expert.py` | Interpolate named keyframes for each `TaskPlan` skill | scene, success, planner types |
| `sim/recorder.py` | Write LeRobot v2 parquet + mp4 + meta | numpy, pyarrow, imageio |
| `sim/executor.py` | Run one episode, return `RunResult` | expert, success, recorder optional |
| `collect.py` | CLI for N seeded episodes | executor, recorder |

Each unit has a single purpose and a function-level interface. Scene loading does not write datasets. The recorder does not know about skills. The expert does not know about parquet.

## Scene

### Robots

Vendor [MuJoCo Menagerie `robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) at commit `ac6b2b09983786f3036cab1000221017fa2193b4`. License: Apache 2.0. Requires MuJoCo >= 3.1.3.

Include the arm twice with body/joint/actuator/camera prefixes `left_` and `right_`. Place the bases on opposite sides of a table so both grippers can reach the workspace without a base collision.

Per arm, 6 actuators already exist in the upstream MJCF:

`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`

Compiler angle is radian. Keep radians everywhere in this slice.

### Table objects

All non-robot geometry is MuJoCo primitives (no extra meshes):

- Table: box
- Top drawer: sliding box with a handle site, left arm can pull it open
- Plate: short cylinder, starts in the drawer
- Mug: solid cylinder with a named site `mug_inside` at the cavity center. Catch volume is a cylinder of radius 0.028 m and height 0.055 m aligned with that site. The catch test is geometric, not “contact with mug geom”
- Bottle: open-top cylinder that initially contains the water particles
- Fork: thin capsule, on the table, not grasped
- Knife: thin capsule, on the table, not grasped
- Napkin: thin box, on the table, not grasped

### Particle water

Exactly 60 sphere particles named `water_00` … `water_59`, radius 0.006 m, each a body with a `freejoint`, starting inside the bottle above the bottle floor. No cohesion constraint.

Pour success: at least 20 particles have XY inside the mug inner radius and Z between mug floor and mug rim. Count is taken after a settle window at the end of the pour skill.

If any qpos/qvel is non-finite, the episode is a failure. Do not count it as a successful pour.

### Cameras

Named exactly:

- `scene` — fixed, three-quarter view of the table and both arms
- `left_wrist` — the prefixed upstream `wrist_cam` on the left gripper
- `right_wrist` — the prefixed upstream `wrist_cam` on the right gripper

Recorded resolution is 128×128 RGB. Control and recording rate is 20 FPS. Physics may run faster (default menagerie timestep 0.005 s); the recorder subsamples to 20 FPS.

### Lighting

One main light whose diffuse intensity is scaled by `SceneVariation.light_intensity`.

## State and action

14-D, float32, radians, this order, these names:

```text
left_shoulder_pan, left_shoulder_lift, left_elbow_flex,
left_wrist_flex, left_wrist_roll, left_gripper,
right_shoulder_pan, right_shoulder_lift, right_elbow_flex,
right_wrist_flex, right_wrist_roll, right_gripper
```

- `observation.state` = current actuator joint positions in that order
- `action` = position-actuator targets in that order (absolute targets, not deltas)

Do not convert to degrees in this slice. Real-robot SO-101 LeRobot datasets often use degrees; document the unit mismatch in `meta/info.json` (`"angles": "radian"`).

## Scripted expert

The expert executes the existing `TaskPlan` from `TableSettingPlanner`. It does not invent a different task sequence.

Motion is joint-space linear interpolation of actuator `ctrl` between named MuJoCo keyframes stored in `table_scene.xml`. One keyframe per skill boundary, including pre-grasp, grasp-close, lift, place, pre-pour, pour-tilt, and retract.

No Jacobian IK, no OMPL, no MoveIt. Gripper open/close is the `gripper` hinge target.

Approximate skill mapping:

| Skill | Arm | Expert behavior | Success predicate |
|---|---|---|---|
| `open_drawer` | left | Pull drawer handle to open pose | drawer joint above threshold |
| `grasp` plate | left | Approach, close gripper | plate contact with left gripper and lifted |
| `place` plate | left | Move to table place zone, open | plate on table, not in drawer |
| `grasp` mug | right | Approach, close | mug in right gripper |
| `grasp` bottle | left | Approach, close | bottle in left gripper |
| `pour` | both | Tilt bottle over mug, settle | >= 20 particles in mug |
| `place` mug | right | Set mug in mug zone, open | mug upright in zone |
| `verify` | both | Hold | all of the above still true |

Fork, knife, and napkin stay on the table. They are obstacles and visual clutter only.

Keyframe numeric values are part of the scene asset. Seed 0 must succeed after those values are tuned. Other seeds may fail; that is acceptable and is recorded as `success=false`.

## Randomization

Reuse `make_variation(seed)` from `recovervla.evaluation`. Apply it to the live model before episode start:

| Field | Effect |
|---|---|
| `position_offset_m` | XY jitter of plate, mug, bottle, fork, knife, napkin (clamped so objects stay on the table and the plate stays in the drawer) |
| `friction` | friction of manipulable geoms |
| `object_mass_scale` | mass of plate, mug, bottle, particles |
| `light_intensity` | main light diffuse scale |

Geometry of the table and arm bases does not change. Randomization must not put objects inside the table volume or outside the reachable workspace. Invalid samples are regenerated from `Random(seed)` with a bounded retry; if still invalid, the episode fails closed with a recorded reason.

## Dataset format

LeRobot v2 layout, written by us. Do not import `lerobot` at runtime in this slice.

```text
<output>/
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    stats.json
  data/chunk-000/episode_XXXXXX.parquet
  videos/chunk-000/observation.images.scene/episode_XXXXXX.mp4
  videos/chunk-000/observation.images.left_wrist/episode_XXXXXX.mp4
  videos/chunk-000/observation.images.right_wrist/episode_XXXXXX.mp4
```

`meta/info.json` required fields:

- `codebase_version`: `"v2.1"`
- `robot_type`: `"so101_bimanual"`
- `fps`: `20`
- `angles`: `"radian"`
- `features` for `observation.state` and `action` with shape `[14]` and the motor names above
- `features` for each camera with `dtype: video`, shape `[128, 128, 3]`, names `height, width, channels`

Parquet columns per frame:

`observation.state`, `action`, `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`, `next.done`

`meta/tasks.jsonl` contains the reference instruction from `REFERENCE_INSTRUCTION`.

`meta/stats.json` stores per-dimension min/max/mean/std for state and action over the written episodes. JSON rather than `stats.safetensors` so collection does not need PyTorch.

Videos are RGB mp4, no audio. If the renderer cannot start, collection aborts with an error; it does not write a numeric-only dataset that pretends to have cameras.

## RunResult and honesty

`MujocoExecutor.execute(plan, scene) -> RunResult` uses the existing dataclass. Collection JSON sidecar (optional `report.json` next to the dataset) sets:

- `executor`: `"mujoco"`
- `disclaimer`: `"MuJoCo scripted-expert demonstration, not a learned policy or Intel hardware result"`
- `intel_benchmark` device/precision/latency remain `null`

Do not report these runs as task-success of a VLA.

## Error handling

| Condition | Behavior |
|---|---|
| `mujoco` not installed | `collect` exits with a message to install `recovervla[sim]`; `evaluate` dry-run still works |
| XML / mesh missing | Fail at load with the asset path |
| Renderer missing | Fail collection; do not emit fake videos |
| Non-finite physics | Mark episode failed and delete that episode’s parquet/mp4 if already opened. Do not keep a partial episode in the dataset |
| Skill timeout | Fail that skill, stop the episode, record `failures` like the dry-run (`"{skill}:{target}"`) |
| Pour below 20 particles | Fail `pour` even if the bottle tilted |
| Output path under a gitignored `datasets/` directory | Required; refuse to write into the source tree root |

## Dependencies

Keep default install stdlib-only so the current milestone still runs without MuJoCo.

```toml
[project.optional-dependencies]
sim = [
  "mujoco>=3.1.3",
  "numpy",
  "pyarrow",
  "imageio",
  "imageio-ffmpeg",
]
```

## File map

- Create: `src/recovervla/assets/NOTICE`
- Create: `src/recovervla/assets/so101/` (vendored MJCF + STL from menagerie)
- Create: `src/recovervla/assets/table_scene.xml`
- Create: `src/recovervla/sim/__init__.py`
- Create: `src/recovervla/sim/scene.py`
- Create: `src/recovervla/sim/success.py`
- Create: `src/recovervla/sim/expert.py`
- Create: `src/recovervla/sim/recorder.py`
- Create: `src/recovervla/sim/executor.py`
- Create: `src/recovervla/collect.py`
- Create: `tests/test_sim_scene.py`
- Create: `tests/test_sim_success.py`
- Create: `tests/test_lerobot_recorder.py`
- Modify: `pyproject.toml` (optional `sim` extra and `recovervla-collect` script)
- Modify: `README.md` (install extra, collect command, provenance, honesty)
- Modify: `.gitignore` only if a new artifact pattern appears; `datasets/` is already ignored

Do not move or rewrite `planner.py` / `DryRunExecutor` except to import shared constants if needed. `REFERENCE_INSTRUCTION` and `make_variation` stay the single source of task text and randomization.

## Commands

Dry-run, unchanged:

```bash
PYTHONPATH=src python3 -m recovervla.evaluation --seeds 10
python3 -m unittest discover -s tests -v
```

Simulation extra:

```bash
pip install -e ".[sim]"
PYTHONPATH=src python3 -m recovervla.collect --seeds 10 --output datasets/table_set_v0
```

`--seeds` is the count of integer seeds `0 .. N-1`. Default output is `datasets/table_set_v0`. Headless rendering should use `MUJOCO_GL=egl` on Linux and the platform default on macOS.

## Testing

MuJoCo tests skip with `unittest.skipUnless` when `mujoco` cannot be imported, so the stdlib suite stays green on a bare machine.

Required tests:

1. `table_scene.xml` loads and contains prefixes `left_` / `right_`, cameras `scene`, `left_wrist`, `right_wrist`, 60 water bodies, and fork/knife/napkin geoms.
2. `make_variation(7)` applied twice yields the same object poses, masses, friction, and light; seed 8 differs.
3. Particle-in-mug predicate is true for a constructed state with 20 spheres in the mug volume and false for 19.
4. Recorder writes a two-frame fake episode whose `info.json`, parquet columns, and video paths match the schema. This test must not need MuJoCo.
5. With MuJoCo installed, `MujocoExecutor` on seed 0 returns `success=True` and `executor` is not `"dry-run"`. Keyframes are tuned as part of implementation until this test passes. A crash or NaN is a test failure.
6. Existing `tests/test_control_plane.py` still passes.

Do not assert a 10-seed MuJoCo success rate in this slice.

## Provenance

Document in `src/recovervla/assets/NOTICE` and README:

- SO-101 MJCF/STL: Google DeepMind MuJoCo Menagerie, Apache 2.0, commit `ac6b2b09983786f3036cab1000221017fa2193b4`, upstream The Robot Studio / I2RT
- Table objects and particles: original RecoverVLA primitives, no third-party meshes

Do not vendor files under a conflicting license.

## Success criteria for this slice

- `pip install -e ".[sim]"` plus collect writes a schema-valid LeRobot v2 dataset for seed 0
- Seed 0 expert completes drawer → plate → mug → particle pour → mug place
- Fork, knife, napkin are visible and randomized, not manipulated
- Dry-run evaluation still runs with only the stdlib
- README states that collect output is scripted-expert simulation data, not a learned policy and not an Intel benchmark

## Next slices (out of scope)

1. ACT training on the L4 from this dataset
2. Swap `MujocoExecutor` from expert to policy
3. OpenVINO export and Core Ultra metrics
