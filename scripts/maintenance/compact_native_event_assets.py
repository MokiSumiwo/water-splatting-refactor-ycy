"""Losslessly compact completed event model snapshots while training continues."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.native_responsibility_controller import load_event_asset


def compact(path):
    original = torch.load(path, map_location='cpu', weights_only=False)
    if original.get('base'):
        return None
    if path.stem.endswith('_capped'):
        base_name = path.stem.replace('_capped', '_S0') + '.pt'
    elif path.stem.endswith('_M1'):
        base_name = path.stem.replace('_M1', '_capped') + '.pt'
    else:
        base_name = 'reference.pt'
    base = load_event_asset(path.parent / base_name)
    # Compare all tensors, including buffers; never assume a parameter stayed fixed.
    delta = {k: v for k, v in original['model'].items()
             if k not in base['model'] or not torch.equal(v, base['model'][k])}
    reconstructed = {**base['model'], **delta}
    if set(reconstructed) != set(original['model']) or not all(
            torch.equal(v, reconstructed[k]) for k, v in original['model'].items()):
        raise RuntimeError(f'nonexact reconstruction: {path}')
    payload = dict(original, model=delta, base=base_name)
    temporary = path.with_suffix('.compacting')
    before = path.stat().st_size
    torch.save(payload, temporary)
    checked = load_event_asset(temporary)
    if not all(torch.equal(v, checked['model'][k]) for k, v in original['model'].items()):
        raise RuntimeError(f'disk reconstruction mismatch: {path}')
    after = temporary.stat().st_size
    if after >= before:
        temporary.unlink()
        return None
    os.replace(temporary, path)
    return dict(path=str(path), before_bytes=before, after_bytes=after,
                exact_model_reconstruction=True, state='preserved unchanged')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--watch-pid', type=int)
    args = parser.parse_args()
    ledger = args.run_dir / 'checks/event_compaction.jsonl'
    seen = set()
    while True:
        for path in sorted(args.run_dir.glob('*/*/events/event_*.pt')):
            if path in seen:
                continue
            if time.time() - path.stat().st_mtime < 60:
                continue
            result = compact(path)
            seen.add(path)
            if result:
                with ledger.open('a') as stream:
                    stream.write(json.dumps(result) + '\n')
                print(json.dumps(result), flush=True)
        proc = Path('/proc') / str(args.watch_pid) / 'stat'
        if not args.watch_pid or not proc.exists() or proc.read_text().split()[2] == 'Z':
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
