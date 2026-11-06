"""Florence-2 OD + region proposals and independent SAM2 masks on every frame."""
import sys
from pathlib import Path
import time
from unittest.mock import patch
from contextlib import contextmanager
import numpy as np
from PIL import Image
from .common import context, read_json, resolve, sha256, stage_args, write_json, write_masks


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = wh.prod()
    return float(intersection / max(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection, 1e-8))


def mask_iou(a, b):
    return float(np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1))


def load_models(cfg):
    import torch
    import transformers.dynamic_module_utils as dynamic
    from transformers import AutoModelForCausalLM, AutoProcessor
    original = dynamic.get_imports
    def imports(path):
        # Florence imports FlashAttention conditionally, but Transformers 4.41
        # scans it as a hard dependency. Explicit eager attention works on CUDA.
        return [x for x in original(path) if x != 'flash_attn']
    with patch.object(dynamic, 'get_imports', imports):
        florence = AutoModelForCausalLM.from_pretrained(str(resolve(cfg['florence_path'])),
            trust_remote_code=True, torch_dtype=torch.float16, attn_implementation='eager',
            local_files_only=True).eval().to('cuda')
        processor = AutoProcessor.from_pretrained(str(resolve(cfg['florence_path'])),
            trust_remote_code=True, local_files_only=True)
    sys.path.insert(0, str(resolve(cfg['sam2_repo'])))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam = build_sam2(cfg['sam2_config'], str(resolve(cfg['sam2_checkpoint'])), device='cuda',
                     apply_postprocessing=False)
    return florence, processor, SAM2ImagePredictor(sam)



@contextmanager
def cached_image_encoding(model):
    """Four tasks share one frame; keep native token/beam generation unchanged."""
    native=model._encode_image;cached=[]
    def encode(pixels):
        if not cached:cached.append(native(pixels))
        return cached[0]
    model._encode_image=encode
    try:yield
    finally:model._encode_image=native


def main(args=None):
    import torch
    torch.set_num_threads(4)
    args = stage_args() if args is None else args
    run, config, manifest, frames = context(args)
    cfg = config['proposals']
    out = run / 'proposals'
    out.mkdir(exist_ok=True)
    signature = {'settings': cfg, 'source_sha256': sha256(__file__),
        'loading_sha256':sha256(Path(__file__).parents[1]/'loading.py')}
    if (out / 'settings.json').exists() and read_json(out / 'settings.json') != signature:
        raise ValueError('Proposal settings changed; use another run ID')
    write_json(out / 'settings.json', signature)
    start = time.monotonic()
    florence, processor, sam = load_models(cfg)
    records = []
    with torch.inference_mode():
        for frame in frames:
            name = f"frame_{frame['frame_index']:04d}"
            saved, masks_path = out / (name + '.json'), out / (name + '.npz')
            if saved.exists() and masks_path.exists():
                records.append(read_json(saved))
                continue
            image = Image.open(frame['rgb_path']).convert('RGB')
            rgb = np.asarray(image)
            candidates, raw = [], {}
            caption = ''
            with cached_image_encoding(florence):
                for task in cfg['tasks']:
                    prompt = task + caption if task == '<CAPTION_TO_PHRASE_GROUNDING>' else task
                    inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda')
                    inputs['pixel_values'] = inputs['pixel_values'].half()
                    tokens = florence.generate(**inputs, max_new_tokens=cfg['max_new_tokens'],
                        num_beams=cfg['num_beams'], do_sample=False)
                    response = processor.batch_decode(tokens, skip_special_tokens=False)[0]
                    parsed = processor.post_process_generation(response, task=task, image_size=image.size)[task]
                    raw[task] = parsed
                    if task == '<MORE_DETAILED_CAPTION>':
                        caption = parsed
                        continue
                    boxes = parsed.get('bboxes', [])
                    labels = parsed.get('labels') or ['unknown'] * len(boxes)
                    for box, label in zip(boxes, labels):
                        box = np.asarray(box, float)
                        box[[0, 2]] = np.clip(box[[0, 2]], 0, image.width - 1)
                        box[[1, 3]] = np.clip(box[[1, 3]], 0, image.height - 1)
                        if np.any(box[2:] - box[:2] < 2):
                            continue
                        if any(box_iou(box, old['bbox_xyxy']) >= cfg['box_dedup_iou'] for old in candidates):
                            continue
                        candidates.append({'source': task, 'label': label.strip() or 'unknown',
                                           'bbox_xyxy': box.tolist()})
            kept, masks = [], []
            with torch.autocast('cuda', dtype=torch.bfloat16):
                sam.set_image(rgb)
                for candidate in candidates:
                    proposed, quality, _ = sam.predict(box=np.asarray(candidate['bbox_xyxy']), multimask_output=True)
                    best = int(np.argmax(quality))
                    mask = proposed[best].astype(bool)
                    count = int(mask.sum())
                    if count < cfg['min_mask_pixels'] or count > cfg['max_mask_fraction'] * mask.size:
                        continue
                    if any(mask_iou(mask, old) >= cfg['mask_dedup_iou'] for old in masks):
                        continue
                    candidate.update(proposal_id=len(kept), sam_quality=float(quality[best]), mask_pixels=count)
                    kept.append(candidate)
                    masks.append(mask)
            if masks:
                write_masks(masks_path, masks)
            else:
                write_masks(masks_path, np.empty((0, rgb.shape[0], rgb.shape[1]), bool))
            record = {'frame_index': frame['frame_index'], 'objects': kept, 'raw_florence': raw,
                      'mask_file': masks_path.name, 'rgb_sha256': frame['rgb_sha256']}
            write_json(saved, record)
            records.append(record)
            print(f'PROPOSALS {name} boxes={len(candidates)} masks={len(kept)} labels={[x["label"] for x in kept]}', flush=True)
    write_json(out / 'summary.json', {'frames': len(records), 'proposals': sum(len(x['objects']) for x in records),
               'seconds': time.monotonic() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'tasks': cfg['tasks'], 'input_contract': 'RGB only, no GT object labels or masks'})


if __name__ == '__main__':
    main()
