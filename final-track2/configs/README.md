# Configuration guide

Run the documented commands from `handout/`. This folder contains reusable
algorithm/evaluation settings, not paths to a particular development run.

- `baselines/perception_v2.yaml`: the supplied perception settings.
- `baselines/fire3d_recon.yaml`: reconstruction settings. The student launcher
  injects `source_run`, `output`, `evaluation_output`, and `input_identity`
  from `--scene`, `--algorithm-name`, and `--output-root`; these do not belong
  in the reusable template.
- `baselines/frozen/`: the supplied model/settings identity and its checksum.
  It does not point to a local reference result. To change model settings,
  follow the open-method guidance instead of overwriting a completed run.
- Metric settings are defined once by `PROTOCOL` in
  `src/evaluation/portable_metric.py`. CPU and GPU local rendering use the same
  canonical cameras/lighting. Do not edit these settings for scored runs.
- `baseline_reference.json`: contextual aggregate measurements and their
  measurement status. These are not an eight-point attainment requirement.
- `schemas/`: the prediction-manifest schema.

Portable storage paths such as `data/scenes`, `cache/models`, `third_party`,
`env`, and `outputs` are relative to `handout/`. Setup/download scripts create
them; no previous local run is needed. The two upstream configuration keys
are exceptions: `inference_protocol` is relative to the installed Fire3D
repository, and `sam2_config` is an upstream SAM2 configuration identifier.
They are not missing files in this handout.

Public launchers accept `--data-root` and `--output-root` for scene/output
storage. Runtime executables may be overridden with
`TRACK2_PERCEPTION_PYTHON`, `TRACK2_RECONSTRUCTION_PYTHON`, and
`BLENDER_EXECUTABLE`. Use documented setup scripts for the default layout.
