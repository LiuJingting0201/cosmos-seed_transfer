#!/usr/bin/env python3
"""Run an isolated weather case, with one shared generation admission lock."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R / 'multiview/scripts'))
from run_model_pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('case', choices=['fog', 'rain', 'night', 'sand', 'snow'])
parser.add_argument('--from-stage', default='init')
args = parser.parse_args()
cfg = R / 'multiview/configs' / (args.case + '.json')
state_file = R / 'states' / (args.case + '.json')

def state(stage, **extra):
    data = {'case': args.case, 'stage': stage, 'updated_unix': time.time(), **extra}
    temp = state_file.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(state_file)
    print(json.dumps(data, ensure_ascii=False), flush=True)

def protected():
    manifest = json.loads((R / 'protected_files.before.json').read_text())
    for name, expected in manifest.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Protected file changed: ' + name)
    return len(manifest)

stages = ['init', 'survey', 'author', 'reasoner', 'finalize', 'edit', 'audit', 'check', 'build-specs', 'transfer', 'verify-api', 'package']
if args.from_stage in ('repair-budget', 'repair-composition'):
    stages = [args.from_stage, *stages[stages.index('audit'):]]
if args.from_stage not in stages:
    raise ValueError('Unknown stage')
p = Pipeline(cfg)
active = args.from_stage
try:
    protected()
    for active in stages[stages.index(args.from_stage):]:
        if active == 'transfer':
            state('waiting_gpu')
            with (R / 'gpu-admission.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                protected()
                state('generating')
                p.execute('transfer')
        elif active == 'verify-api':
            state('verifying')
            subprocess.run([sys.executable, str(R / 'multiview/scripts/verify_api_outputs.py'), '--config', str(cfg)], check=True)
            state('verified', protected_files=protected())
        elif active == 'package':
            state('packaging')
            subprocess.run([sys.executable, str(R / 'package_case.py'), args.case], check=True)
            state('completed', protected_files=protected())
        elif active == 'repair-budget':
            state('edit', repair='token_budget')
            subprocess.run([sys.executable, str(R / 'repair_budget.py'), args.case], check=True)
        elif active == 'repair-composition':
            assert args.case == 'sand'
            state('edit', repair='composition_color')
            subprocess.run([sys.executable, str(R / 'repair_sand_composition.py')], check=True)
        else:
            state(active)
            p.execute(active)
except BaseException as exc:
    state('failed', failed_stage=active, error_type=type(exc).__name__, error=str(exc)[:1000])
    raise
