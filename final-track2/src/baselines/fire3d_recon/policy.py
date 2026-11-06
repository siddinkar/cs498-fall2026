"""Low-support instances are skipped before model inference and scene export."""
def select_objects(objects, config):
    minimum = config.get('minimum_fused_points', 100)
    skip = config.get('skip_low_support', True)
    kept, skipped = [], []
    for obj in objects:
        points = obj.get('source_points', obj.get('points', 0))
        weak = obj.get('low_support', False) or points < minimum
        if skip and weak:
            skipped.append({'object_id': obj['object_id'], 'name': obj['name'], 'family': obj['family'],
                            'source_points': points, 'reason': f'low_support flag or fewer than {minimum} fused points',
                            'status': 'skipped_before_inference'})
        else:
            kept.append(obj)
    return kept, skipped


def select_conditioned(manifest, config):
    """Keep conditioning-stage skip provenance in every downstream stage."""
    kept, skipped = select_objects(manifest['objects'], config)
    merged = {o['object_id']: o for o in manifest.get('skipped_objects', [])}
    merged.update({o['object_id']: o for o in skipped})
    return kept, [merged[i] for i in sorted(merged)]
