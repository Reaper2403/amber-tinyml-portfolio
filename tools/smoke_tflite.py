#!/usr/bin/env python3
"""Invoke both artifacts on deterministic synthetic tensors, not labeled data."""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    results = []
    rng = np.random.default_rng(42)
    for item in manifest['models']:
        interpreter = tf.lite.Interpreter(model_path=str(ROOT / item['path']), num_threads=1)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        assert inp['shape'].tolist() == item['input_shape']
        assert inp['dtype'] == np.int8 and out['dtype'] == np.int8
        assert out['shape'].tolist() == [1, len(item['labels'])]
        scale, zero = inp['quantization']
        out_scale, out_zero = out['quantization']
        assert scale > 0 and out_scale > 0
        shape = item['input_shape']
        probes = [np.full(shape, zero, dtype=np.int8),
                  rng.integers(-128, 128, size=shape, dtype=np.int16).astype(np.int8),
                  np.full(shape, -128, dtype=np.int8), np.full(shape, 127, dtype=np.int8)]
        for sample in probes:
            interpreter.set_tensor(inp['index'], sample)
            interpreter.invoke()
            logits = (interpreter.get_tensor(out['index']).astype(np.float32) - out_zero) * out_scale
            assert np.isfinite(logits).all()
        results.append({'model': item['name'], 'input_shape': shape,
                        'output_shape': out['shape'].tolist(), 'input_dtype': 'int8',
                        'output_dtype': 'int8', 'synthetic_invocations': len(probes), 'status': 'passed'})
    print(json.dumps({'tensorflow_version': tf.__version__, 'status': 'passed',
                      'scope': 'Host compatibility smoke check only; no accuracy, latency or hardware claim.',
                      'models': results}, indent=2))


if __name__ == '__main__':
    main()
