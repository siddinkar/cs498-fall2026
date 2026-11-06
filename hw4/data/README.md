# HW4 data

From `hw4/handout`, run:

```bash
python data/download_data.py
```

The script downloads the public `hw4_data.zip` release from
<https://huggingface.co/datasets/hongchi/cs498>, verifies the archive and every
extracted file against `SHA256SUMS`, and installs:

```text
data/nerf_synthetic/lego/
  train/                 50 RGB/RGBA images
  val/                    5 RGB/RGBA images
  test/                  12 RGB/RGBA images
  transforms_train.json
  transforms_val.json
  transforms_test.json
```

All images are 200 by 200 pixels. The complete archive is about 2.7 MB. It is
the only dataset required by HW4; no authentication or manual extraction is
needed.

To recheck an existing installation without downloading:

```bash
python data/download_data.py --verify-only
```

Downloaded data are intentionally ignored by Git.
