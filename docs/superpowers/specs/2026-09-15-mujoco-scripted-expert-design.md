# MuJoCo Table Scene and Scripted Expert

**Date:** 2026-09-15  
**Status:** Revised specification; implementation and integration validation pending
**Slice:** first RecoverVLA sub-project (simulation + demonstration collection)

## Goal

Implementation note: the initial code uses the official LeRobot 0.4.4 v3 writer
on the remote host instead of a custom v2.1 writer. See `docs/REMOTE.md` for actual
implemented scope and remaining remote acceptance checks. No local execution is
authorized. The source revision used for assets is recorded in `scripts/fetch_assets.py`.

Add a real MuJoCo bimanual table-setting scene and a scripted expert that records LeRobot-compatible demonstrations. Keep the existing dry-run control plane unchanged and honest.

This slice does not train ACT, export OpenVINO, or claim Intel hardware results.

## Why this slice first

The L4 GPU is only useful after we can generate demonstrations. The current tree is a stdlib planner plus `DryRunExecutor`. Collection, not training, is the blocker.

## Non-goals

- Learned policy, ACT training, or SmolVLA
- OpenVINO IR, ONNX, or Core Ultra benchmarks
- OMPL, MoveIt, ROS, or a general motion-planning framework (small MuJoCo Jacobian IK is in scope for demonstration generation)
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
| `sim/expert.py` | Execute object-relative waypoints for each `TaskPlan` skill | scene, IK, success, planner types |
| `sim/recorder.py` | Write LeRobot v2 parquet + mp4 + meta | numpy, pyarrow, imageio |
| `sim/executor.py` | Run one episode, return `RunResult` | expert, success, recorder optional |
| `collect.py` | CLI for N seeded episodes | executor, recorder |

Each unit has a single purpose and a function-level interface. Scene loading does not write datasets. The recorder does not know about skills. The expert does not know about parquet.

## Scene

### Robots

Vendor [MuJoCo Menagerie `robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) at commit `ac6b2b09983786f3036cab1000221017fa2193b4`. License: Apache 2.0. Requires MuJoCo >= 3.1.3.

Verify the upstream commit, model files, minimum MuJoCo version, and asset license before vendoring; the identifiers above are draft inputs until checked. Preserve upstream license files.

Generate two namespaced arm definitions with a deterministic XML transformation script. Prefix all named entities and their references, including bodies, joints, actuators, sites, cameras, defaults/classes, tendons, sensors, contact exclusions, and assets. Shared meshes/materials may instead be declared once with verified shared references. Do not rely on a plain duplicate include to apply prefixes. Validate uniqueness and reference resolution by loading the combined model. Place the bases so both grippers can reach the shared workspace without base collision.

Per arm, 6 actuators already exist in the upstream MJCF:

`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`

Compiler angle is radian. Keep radians everywhere in this slice.

### Table objects

All non-robot geometry is MuJoCo primitives (no extra meshes):

- Table: box
- Top drawer: cavity made from a floor and separate walls, with a sliding joint and physical graspable handle plus handle site
- Plate: short cylinder, starts in the drawer
- Mug: hollow compound geometry with a bottom and overlapping wall segments, with a named site `mug_inside`. Inner radius is 0.028 m and inner height is 0.055 m. No solid collision geom may fill the cavity.
- Bottle: hollow compound geometry with a floor and wall segments, an open mouth, and dimensions that allow non-overlapping particle placement. Validate retention upright and release when tilted before robot integration.
- Fork: thin capsule, on the table, not grasped
- Knife: thin capsule, on the table, not grasped
- Napkin: thin box, on the table, not grasped

### Particle water

Use 60 sphere particles named `water_00` … `water_59`, radius 0.006 m, each a body with a `freejoint`. Spawn without overlaps with other particles or vessel walls. Enable particle-vessel and particle-particle collision. These represent granular transfer, not realistic fluid; label demonstrations accordingly. First validate this subsystem independently before integrating the full task.

Pour success: at least 20 particles remain contained for a configured settle window. Transform their positions into the mug's local frame; include particle radius in radial and floor/rim margins. Check containment again after mug placement. Record the count, settle duration, and thresholds. Bottle tilt alone is not success. Liquid realism is optional; if transfer is deferred, the complete reference-task milestone remains incomplete.

If any qpos/qvel is non-finite, the episode is a failure. Do not count it as a successful pour.

### Cameras

Named exactly:

- `scene` — fixed, three-quarter view of the table and both arms
- `left_wrist` — explicitly mounted on the left gripper, reusing an upstream camera if verified present
- `right_wrist` — explicitly mounted on the right gripper, reusing an upstream camera if verified present

Recorded resolution is 128×128 RGB. Control and recording rate is 20 FPS. Physics may run faster (default menagerie timestep 0.005 s); the recorder subsamples to 20 FPS.

### Lighting

One main light whose diffuse intensity is scaled by `SceneVariation.light_intensity`.

## State and action

12-D (6 actuators per arm), float32, radians, this order, these names:

```text
left_shoulder_pan, left_shoulder_lift, left_elbow_flex,
left_wrist_flex, left_wrist_roll, left_gripper,
right_shoulder_pan, right_shoulder_lift, right_elbow_flex,
right_wrist_flex, right_wrist_roll, right_gripper
```

- `observation.state` = current actuator joint positions in that order
- `action` = position-actuator targets in that order (absolute targets, not deltas)

Resolve named joints through MuJoCo address mappings; never slice the first 12 entries of global `qpos`, which also contains object and particle coordinates. Assert 12 unique controlled joints and 12 matching actuators. At time t, capture state and images, compute and record the action applied for the next control interval, then step physics. Record identical timestamps across cameras and numeric observations. Store action limits and normalization conventions alongside the dataset.

Do not convert to degrees in this slice. Document units in the dataset manifest and, if supported, `meta/info.json` (`"angles": "radian"`). Check each reused checkpoint's units and normalization explicitly before adapting it.

## Scripted expert

The expert executes the existing `TaskPlan` from `TableSettingPlanner`. It does not invent a different task sequence.

Define object-relative waypoints for approach, grasp, lift, place, tilt, and retract. Solve them using bounded damped-least-squares MuJoCo Jacobian IK, respecting joint limits and reachable orientation constraints of each arm. Use joint-space interpolation between solved waypoints with bounded velocity and contact checks. Fixed keyframes may define home poses and initial guesses only. Abort on unreachable targets, unsafe contact, or timeout.

This expert uses privileged simulator poses only for demonstration generation. The later learned policy receives the declared camera/state/language inputs. Gripper open/close uses the verified actuator limits. Object lifting and drawer opening must result from physical contact; do not teleport objects, overwrite their poses during execution, or attach them using hidden welds.

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
| `verify` | both | Hold | plate and mug placed and stable, mug upright, transfer retained, no unsafe contact; completed grasp events recorded in history |

Fork, knife, and napkin stay on the table. They are obstacles and visual clutter only.

Seed 0 must succeed after physical integration. Report every attempted seed, including failures. Only complete successful episodes enter imitation-learning data; keep failures in a separate diagnostic report. Never require historical grasp predicates to remain true after objects have been released.

## Randomization

Reuse `make_variation(seed)` from `recovervla.evaluation`. Apply it to the live model before episode start:

| Field | Effect |
|---|---|
| `position_offset_m` | XY jitter of plate, mug, bottle, fork, knife, napkin (clamped so objects stay on the table and the plate stays in the drawer) |
| `friction` | friction of manipulable geoms |
| `object_mass_scale` | mass of plate, mug, bottle, particles |
| `light_intensity` | main light diffuse scale |

Geometry of the table and arm bases does not change. Randomization must not put objects inside the table volume or outside the reachable workspace. Invalid samples are regenerated from `Random(seed)` with a bounded retry; if still invalid, the episode fails closed with a recorded reason.

Derive independent per-object XY offsets from a single seeded RNG, using `abs(position_offset_m)` as the jitter bound; record all sampled values and retry counts. Do not reinitialize the RNG inside the retry loop. Scale inertia consistently with mass and refresh derived model constants as required. Start each episode from pristine model/data state to avoid cumulative randomization. Background and geometry variation are deferred to a later robustness slice and must not be claimed here.

## Dataset format

Pin a LeRobot release/commit that reads v2.1 and the planned ACT training code before implementing the writer. Keep LeRobot out of the lightweight collection runtime, but require it in a separate dataset-validation environment. The layout below is illustrative, not a complete compatibility contract: generate every field, path template, counter, episode statistic, and video metadata item required by that pinned loader. A self-authored schema test alone does not establish compatibility.

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
- `features` for `observation.state` and `action` with shape `[12]` and the motor names above
- `features` for each camera with `dtype: video`, shape `[128, 128, 3]`, names `height, width, channels`

Parquet columns per frame:

`observation.state`, `action`, `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`, `next.done`

`meta/tasks.jsonl` contains the reference instruction from `REFERENCE_INSTRUCTION`.

`meta/stats.json` stores per-dimension min/max/mean/std for state and action over the written episodes. JSON rather than `stats.safetensors` so collection does not need PyTorch.

Add any per-episode statistics required by the pinned loader. Derive training normalization from the training split only, including image statistics if required. Preserve episode-to-task mapping, episode lengths, video frame counts, and contiguous frame/episode indices after failed attempts are excluded. Keep simulation seeds in a separate manifest; seed number is not dataset episode index.

Write each attempt into a temporary staging directory. Publish episode files and update metadata only after success and video/numeric validation. Refuse to overwrite an existing output dataset. Reports retain failed attempt reasons without adding failed trajectories to training data.

The 10-seed command is a smoke collection. Initial training target is 50 successful episodes, split deterministically at episode/seed level into 40 training and 10 validation episodes. Reserve a separate fixed list of at least 10 evaluation seeds that is never used for training or tuning. Set a maximum attempt budget and report a shortfall instead of silently looping until enough successes occur. Store split membership in a manifest, with actual task text per episode. The current planner supports one reference workflow; paraphrases alone do not establish multi-task language understanding.

Acceptance requires loading the dataset with the pinned `LeRobotDataset`, decoding all three cameras, indexing first/last frames of each episode, and sampling an ACT-shaped batch with state/action width 12. Run validation on the remote training host; heavy dependencies and datasets need not be installed or stored on the user's Mac.

Videos are RGB mp4, no audio. If the renderer cannot start, collection aborts with an error; it does not write a numeric-only dataset that pretends to have cameras.

## RunResult and honesty

`MujocoExecutor.execute(plan, scene) -> RunResult` uses the existing dataclass. A required `report.json` next to the dataset wraps per-attempt results and sets:

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
| Non-finite physics | Mark episode failed, discard only this attempt's staged files, and leave published episodes intact |
| Skill timeout | Fail that skill, stop the episode, record `failures` like the dry-run (`"{skill}:{target}"`) |
| Pour below 20 particles | Fail `pour` even if the bottle tilted |
| Output path | Allow external directories or verified gitignored artifact directories; reject source paths, tracked paths, and existing datasets |

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
- Create: `src/recovervla/sim/ik.py` (bounded expert-only IK)
- Create: `scripts/vendor_so101.py` (deterministic namespace transformation and provenance checks)
- Create: `scripts/validate_dataset.py` (pinned LeRobot loader integration)
- Create: `src/recovervla/sim/recorder.py`
- Create: `src/recovervla/sim/executor.py`
- Create: `src/recovervla/collect.py`
- Create: `tests/test_sim_scene.py`
- Create: `tests/test_sim_success.py`
- Create: `tests/test_lerobot_recorder.py`
- Modify: `pyproject.toml` (optional `sim` extra, `recovervla-collect` script, and package data for XML/meshes/licenses)
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

Gate recorder tests on their optional dependencies as well. A required remote integration job installs all extras and the pinned dataset validator; skipped tests cannot satisfy this slice's acceptance. Verify a built wheel contains the scene, meshes, and licenses and can load the model outside the source checkout.

Required tests:

1. `table_scene.xml` loads and contains prefixes `left_` / `right_`, cameras `scene`, `left_wrist`, `right_wrist`, 60 water bodies, and fork/knife/napkin geoms.
2. `make_variation(7)` applied twice yields the same object poses, masses, friction, and light; seed 8 differs.
3. Particle containment distinguishes 20 from 19, handles translated/rotated mugs, and rejects boundary overlaps; physical tests demonstrate vessel retention and transfer.
4. Recorder writes a two-frame fake episode whose `info.json`, parquet columns, and video paths match the schema. This test must not need MuJoCo.
5. With MuJoCo installed, seed 0 succeeds through physical contacts and the report wrapper has `executor="mujoco"` (`RunResult` itself has no executor field). A crash or NaN fails the test. Verify the expert adapts to at least one distinct reachable object placement.
6. Existing `tests/test_control_plane.py` still passes.
7. A failed attempt produces a report entry but no training episode; a published dataset passes the pinned LeRobot loader and ACT batch check.
8. Final-state verification succeeds after releasing the mug and plate while retaining recorded historical skill completion.

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
