# Further improve with your own algorithm

This runnable template predicts free everywhere. It is interface scaffolding,
not a competitive baseline or an implemented alternative method.

Copy `submission.py` and `method.json` into `student_submission/`, implement
your algorithm, declare dependencies in `student_submission/requirements.txt`,
and place any weights under `student_submission/weights/`. You may remove
unused baseline modules. Keep `run.py` and `cs498_track1/` unchanged.
Place all custom training and inference code under `student_submission/` so the
final ZIP contains everything needed to reproduce your method. The unchanged
provided trainer and common tools do not need to be copied into that directory.

The generator handles all data loading, ordered sample IDs and CSV encoding.
Your method implements only `load_model` and `predict_sample`. Predictions must
be independent across samples and have the fixed `(40,14,80)` z,y,x grid.
Any internal representation is permitted; resample to the canonical grid yourself.

`load_model(model_dir, device)` receives the `weights/` directory, not the method
root. `inputs` contains NumPy float32 arrays: RGB in [0,1], optical-axis depth in
metres (zero invalid, or `None` when absent), intrinsics, and camera/anchor poses.
See the root README for exact shapes, coordinate conventions and helper imports.
The adapter must support the CPU interface probe. Put helper code in your own
package, e.g. `my_method/`, and import `from my_method.network import Model`.

Run `python run.py validate`, full `python run.py evaluate`, and the
CSV generator before packaging. You can first smoke-test the template with
`--submission-dir alternative_method_template`. Use the final
`student_submission/` directory for reported metrics and packaging.
Use `python run.py figures` for the provided figures, or record your own four
verified images with `python run.py figures --record-existing`, then rebuild
the report before packaging.
