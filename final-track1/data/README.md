# Multi-View 3D Semantic Occupancy Prediction data

Use `python run.py download --split train|val|test` from the project root.
You may add `--houses train_house_000` (with the matching split) for a subset.
Data extract into `data/triview/`; do not generate scenes or install a simulator.

Public repository: https://huggingface.co/datasets/hongchi/cs498
The downloader pins an immutable revision and verifies each archive's SHA-256.
Repeated subset downloads merge their receipts at that revision; re-downloading
one house does not discard the receipt for other houses. A revision change resets
that split's receipt rather than claiming data from different revisions match.
All public archives total about 4.23 GiB; budget at least 12 GiB for HF cache,
extraction, checkpoints and outputs. Complete Train/Val targets are prepared on
the canonical 25 cm grid. Test has no targets or annotations.

Each `samples/HOUSE/sample_NNNNN/` contains:

- `rgb/{front_left,front,front_right}.png` (224 x 224);
- matching uint16 millimetre `depth/` images;
- `K.npy`, `T_world_camera.npy` (3 x 4 x 4), `T_world_anchor.npy` (4 x 4);
- a sanitized `manifest.json`;
- Train/Val only: `target.npz` with labels, completion_mask, output_mask.

Output labels and masks have shape (40,14,80), ordered z,y,x. Labels are -1
(ignore), 0 (free), or 1-9 occupied semantics. Predictions must use 0-9 only.
The source 10 cm world labels have already been converted; no conversion or
preprocessing task is assigned to students. See the handout for exact aggregation.
