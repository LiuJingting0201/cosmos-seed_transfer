#!/usr/bin/env python3
"""Make a synchronized single-view preview from an already completed job."""
import argparse
import hashlib
import json
from pathlib import Path
import comparison_base as media

R = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('case', choices=['fog', 'night', 'rain', 'sand'])
parser.add_argument('view', choices=['LEFT', 'FRONT', 'REAR', 'UP'])
args = parser.parse_args()
name = dict(media.VIEWS)[args.view]
source = Path('/root/sam31/data/dji_weather_rwy_standard/rwy_split2') / (name + '.mp4')
run = R / 'multiview/outputs' / args.case / 'runs/gpt6_vllm_edge_v1' / ('rwy_split2__' + name)
assert json.loads((run / 'status.json').read_text())['status'] == 'completed'
out = R / 'evaluation' / (args.case + '_partial')
out.mkdir(parents=True, exist_ok=True)
video = out / (args.case + '_' + args.view.lower() + '_comparison.mp4')
if not video.exists():
    filters = [media.filt(0, args.view + ' - SOURCE', 640),
               media.filt(1, args.view + ' - ' + args.case.upper(), 640),
               '[v0][v1]hstack=inputs=2:shortest=1[out]']
    media.run(media.encode_inputs([source, run / 'vision.mp4']) +
              ['-filter_complex', ';'.join(filters)] + media.tail(video, 80))
report = media.probe(video, 80, 1280, 640)
report['sha256'] = hashlib.sha256(video.read_bytes()).hexdigest()
report['video'] = str(video)
video.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
