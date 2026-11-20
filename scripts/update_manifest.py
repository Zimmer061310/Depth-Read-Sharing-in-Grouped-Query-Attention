#!/usr/bin/env python3
"""Refresh existing release-file hashes after deliberate documentation changes."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parents[1]
p=root/'MANIFEST.json';manifest=json.loads(p.read_text())
source=json.loads((root/'SOURCE-MAP.json').read_text())
for name,spec in source['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==spec['sha256'],name
for name in manifest['files']:
    raw=(root/name).read_bytes()
    manifest['files'][name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
p.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print('Refreshed existing release-file hashes; original source mappings still match.')
