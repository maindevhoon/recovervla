# RecoverVLA

RecoverVLA is a recovery-aware control plane for the Intel Physical AI online
challenge. It translates a table-setting instruction into explicit bimanual
skills, evaluates the plan across reproducible randomized scenes, and emits
machine-readable results that can later include MuJoCo and OpenVINO metrics.

This first milestone is deliberately lightweight: it has no model downloads and
uses only the Python standard library. The current executor is a deterministic
dry-run model, not a claim of completed robot simulation.

## Run

Run commands on the remote Linux host. The Mac is used only for editing files.
See [remote setup, collection, and L4 training](docs/REMOTE.md).
For the enhanced dining-room scene, see [Colab rendering instructions](docs/DINING_PREVIEW.md).
For the time-boxed Kaggle path, see [Kaggle fast path](docs/KAGGLE_QUICKSTART.md).

MuJoCo rendering and contact rollouts have been executed on Kaggle. The drawer
opened through physical contact in 10/10 randomized seeds (100–109). A central
plate tab was lifted by a two-jaw pinch in 10/10 of the same seeds. Seed 100
then placed the plate on the table and lifted the mug. Bottle lift and pouring
remain unvalidated; ACT training has not started. The default evaluation below
remains a dry run.

```bash
PYTHONPATH=src python3 -m recovervla.evaluation --seeds 10
python3 -m unittest discover -s tests -v
```

## Next integration points

- Replace `DryRunExecutor` with a MuJoCo executor using dual SO-101 models.
- Connect visual verification to camera observations.
- Compile supported perception/policy components with OpenVINO.
- Populate the existing benchmark fields on the Intel Core Ultra machine.
