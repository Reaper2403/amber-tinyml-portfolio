#!/usr/bin/env python3
"""Verify the curated snapshot without third-party dependencies."""
import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    checked_sources = 0
    source_manifest = json.loads((ROOT / 'evidence/source_manifest.json').read_text())
    for item in source_manifest['files']:
        data = (ROOT / item['published_path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['published_sha256'], item['published_path']
        checked_sources += 1
    models = json.loads((ROOT / 'models/manifest.json').read_text())['models']
    for item in models:
        data = (ROOT / item['path']).read_bytes()
        assert len(data) == item['bytes'], item['name']
        assert data[4:8] == b'TFL3', item['name']
        assert hashlib.sha256(data).hexdigest() == item['sha256'], item['name']
        assert len(set(item['labels'])) == len(item['labels'])

    counts = dict(python_files=0, json_files=0, svg_files=0, local_links=0)
    patterns = [
        r'/(?:Users|home)/[A-Za-z][^\s"\']*/',
        r'gh[pousr]_[A-Za-z0-9]{30,}',
        r'github_pat_[A-Za-z0-9_]{30,}',
        r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    ]
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ('.git', '.venv', '__pycache__')):
            continue
        if p.suffix == '.tflite':
            continue
        text = p.read_text()
        for pattern in patterns:
            assert not re.search(pattern, text), f'Publication scan: {p.relative_to(ROOT)}'
        if p.suffix == '.py':
            ast.parse(text)
            counts['python_files'] += 1
        if p.suffix == '.json':
            json.loads(text)
            counts['json_files'] += 1
        if p.suffix == '.svg':
            ET.fromstring(text)
            counts['svg_files'] += 1
        if p.suffix == '.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
                if target.startswith(('https://', 'http://', '#', 'mailto:')):
                    continue
                path = unquote(target.split('#')[0].strip('<>'))
                assert (p.parent / path).exists(), f'Broken link in {p}: {target}'
                counts['local_links'] += 1
    print(json.dumps({'status': 'passed', 'source_hashes': checked_sources,
                      'model_artifacts': len(models), **counts}, indent=2))


if __name__ == '__main__':
    main()
