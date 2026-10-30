#!/usr/bin/env python3
"""CPU-only budget checks for all four views; never enters the submission branch."""
import argparse
from pathlib import Path
import sys

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R / 'multiview/scripts'))
from run_model_pipeline import Pipeline
from run_vllm_omni_transfer import submit

parser = argparse.ArgumentParser()
parser.add_argument('case', choices=['fog', 'rain', 'night', 'sand'])
case = parser.parse_args().case
pipeline = Pipeline(R / 'multiview/configs' / (case + '.json'))
pipeline.check()
output = pipeline.experiment / 'runs' / pipeline.run_name
assert not any(output.rglob('async_submit.json')), 'Already submitted'
args = argparse.Namespace(output_dir=output, resolution='720', edge_weight=1.0,
                          edge_threshold='medium', gateway_url='http://127.0.0.1:8002',
                          upstream_url='http://127.0.0.1:8000')
failures = []
specs = sorted((pipeline.experiment / 'specs' / pipeline.run_name).glob('*.json'))
assert len(specs) == 4
for spec in specs:
    try:
        submit(args, spec, preflight_only=True)
    except Exception as exc:
        failures.append((spec.name, type(exc).__name__))
print('PREFLIGHT', {'cases': len(specs), 'failures': failures}, flush=True)
raise SystemExit(bool(failures))
