"""Verify external inputs without loading model or touching training state."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[2]
manifest = json.loads((Path(__file__).parent/'data_sha256.json').read_text())
failures = []
for scene, entries in manifest.items():
    for name, expected in entries.items():
        # Depth assets are historical context, not required by this protocol.
        if name.startswith('depthAnything_u16/'):
            continue
        p = root/'undistorted_data'/('undistorted_'+scene)/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            failures.append(str(p))
if failures:
    raise SystemExit('Missing or changed inputs:\n'+'\n'.join(failures))
print('All registered training images, COLMAP files and split lists match.')
