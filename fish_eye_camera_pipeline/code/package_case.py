#!/usr/bin/env python3
"""Package a completed case without modifying source or previous experiments."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
import comparison_base as media

R = Path(__file__).resolve().parent
case = sys.argv[1]
NAMES = {'fog': '雾天', 'rain': '雨天', 'night': '夜晚', 'sand': '沙尘', 'snow': '雪天'}
label = NAMES[case]
cfg_file = R / 'multiview/configs' / (case + '.json')
cfg = json.loads(cfg_file.read_text())
exp = Path(cfg['experiment_root'])
run = exp / 'runs' / cfg['run_name']
out = R / 'evaluation' / case
out.mkdir(parents=True, exist_ok=True)
verification = json.loads((exp / 'evaluations' / cfg['run_name'] / 'api_video_verification.json').read_text())
assert verification['video_pass_count'] == verification['expected_count'] == 4

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

manifest = json.loads((R / 'protected_files.before.json').read_text())
assert all(sha(Path(p)) == expected for p, expected in manifest.items())
budgets = {}
for p in run.glob('*/prompt_token_audit.json'):
    x = json.loads(p.read_text())
    budgets[p.parent.name] = {side: x['counts'][side]['before_cap_tokens'] for side in ('positive', 'negative')}
    assert x['safe'] and all(x['counts'][s]['tokens_removed'] == 0 for s in ('positive', 'negative'))
assert len(budgets) == 4
calls = []
for p in (exp / 'provenance').rglob('call.json'):
    x = json.loads(p.read_text())
    calls.append({'path': str(p.relative_to(exp)), **x})
report = {'case': case, 'label': label, 'config': cfg, 'verification': verification,
          'token_budgets': budgets, 'protected_files': {'count': len(manifest), 'unchanged': True},
          'model_calls': calls, 'visual_verdict': 'pending independent review'}
(out / 'technical_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

# Configure a copied helper; outputs remain wholly inside this new case root.
media.ROOT = R
media.OUT = out
media.SOURCE = Path(cfg['views_root'])
media.RUN = run
media.WEATHER_LABEL = label
media.WEATHER_ENGLISH = case.upper()
media.main()

with zipfile.ZipFile(out / (label + '_四路生成视频.zip'), 'w', zipfile.ZIP_STORED) as z:
    sums = {}
    for view, name in media.VIEWS:
        p = run / ('rwy_split2__' + name) / 'vision.mp4'
        n = view.lower() + '.mp4'
        z.write(p, n)
        sums[n] = sha(p)
    z.writestr('SHA256SUMS.json', json.dumps(sums, indent=2))
files = {'config.json': cfg_file, 'technical_report.json': out / 'technical_report.json'}
for sub in ('survey', 'prompt_sets', 'specs', 'evaluations', 'repair_history'):
    for p in (exp / sub).rglob('*'):
        if p.is_file() and p.name not in {'events.jsonl', 'request.json', 'final.json'}:
            files[str(p.relative_to(exp))] = p
for p in run.rglob('*'):
    if p.is_file() and p.suffix in {'.json', '.txt', '.headers'}:
        files[str(p.relative_to(exp))] = p
for p in (exp / 'provenance').rglob('*'):
    if p.is_file() and p.name in {'parsed.json', 'raw_text.txt', 'response.json', 'failure.json', 'call.json'}:
        files[str(p.relative_to(exp))] = p
for p in (exp / 'provenance').glob('*/*.json'):
    files[str(p.relative_to(exp))] = p
archive = out / (label + '_Prompt与中间产物.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for n, p in sorted(files.items()):
        z.write(p, n)
    z.writestr('SHA256SUMS.json', json.dumps({n: sha(p) for n, p in files.items()}, ensure_ascii=False, indent=2))
    z.writestr('README.txt', '输入：' + cfg['user_prompt'] + '。本机 GPT-6 Author/Editor/Auditor，远端 Cosmos Reasoner 与 Cosmos3-Nano 视频生成。每路5秒；四路独立生成，拼接仅为展示，技术通过不等于天气或跨视角一致性通过。')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
delivery = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in out.iterdir()
            if p.is_file() and p.name != 'delivery_manifest.json'}
(out / 'delivery_manifest.json').write_text(json.dumps(delivery, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'case': case, 'output': str(out), 'files': len(delivery)}, ensure_ascii=False), flush=True)
