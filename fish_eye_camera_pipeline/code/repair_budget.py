#!/usr/bin/env python3
"""Model-authored compaction before submission, preserving real editor provenance."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

R = Path(__file__).resolve().parent
sys.path[:0] = [str(R), str(R / 'multiview/scripts')]
from run_model_pipeline import Pipeline, validate_prompt
from model_api import read_json, write_json, validate_receipt, receipt
from prompt_budget import audit

parser = argparse.ArgumentParser()
parser.add_argument('case', choices=['fog', 'rain', 'night', 'sand'])
args = parser.parse_args()
P = Pipeline(R / 'multiview/configs' / (args.case + '.json'))
scene = 'rwy_split2'
run_root = P.experiment / 'runs' / P.run_name
assert not any(run_root.rglob('async_submit.json')), 'Already submitted; refuse prompt changes'
old = validate_receipt(P.record_path(scene, 'edit'))
archive = P.experiment / 'repair_history' / ('token_budget_' + str(time.time_ns()))
archive.mkdir(parents=True, exist_ok=False)
for stage in ('edit', 'audit'):
    shutil.copy2(P.record_path(scene, stage), archive / (stage + '.json'))
for view in P.scenes[scene]:
    dest = archive / view
    dest.mkdir()
    for name in ('modified_prompt.json', 'compliance_report.json'):
        shutil.copy2(P.prompt_root / scene / view / name, dest / name)
extra = read_json(next(run_root.rglob('extra_params.json')))
allowed = {'subjects', 'cinematography', 'actions', 'segments', 'transitions',
           'temporal_caption', 'audio_description', 'text_and_signage_elements'}

def measure(prompt):
    return audit({'prompt': json.dumps(prompt, ensure_ascii=False, indent=2),
                  'negative_prompt': '', 'extra_params': extra,
                  'video_meta': {'width': 1088, 'height': 1088, 'fps': 16, 'num_frames': 80}})

calls = [Path(c) for c in old['calls']]
inputs = [Path(p) for p in old['inputs']]
inputs.extend([archive / 'edit.json', archive / 'audit.json', Path(__file__)])
candidates = {}
for index, view in enumerate(P.scenes[scene]):
    current_file = archive / view / 'modified_prompt.json'
    current = read_json(current_file)
    before = measure(current)
    write_json(archive / view / 'token_before.json', before)
    count = before['counts']['positive']['before_cap_tokens']
    print(f'{view}: tokens={count}', flush=True)
    if before['safe']:
        continue
    result, call, source_inputs = P.visual_model(scene, 'edit/' + view, """
The CURRENT already-edited prompt exceeds the deployed Cosmos3 token budget. Compact
redundant wording in the eight permitted per-view fields by about 20%, preserving every
visible subject and all meaningful facts, materials, colors, image-space trajectories,
occlusions and timing. Retain the requested weather effect and its visible cues. Inspect
the source contact sheets to retain grounded corrections. Do not add new scene content.
Preserve subject order and count, field names and numeric values. Keep the complete
0:00-0:05 timeline. Shared fields and media metadata must remain untouched.
Return exactly {"view":"<current_view>","replacements":{...},"findings":[...]}.
replacements MUST contain complete values for ALL eight fields: subjects, cinematography,
actions, segments, transitions, temporal_caption, audio_description, text_and_signage_elements.
These replace the original finalized base, so retain previous editor corrections while
shortening repetition. Return compact valid JSON without surrounding prose.
""", {'user_prompt': P.config['user_prompt'],
       'plan': read_json(P.prompt_root / scene / 'transfer_plan.json'),
       'current_view': view, 'current_view_prompt': current,
       'observed_tokens_with_system_and_metadata': count,
       'hard_token_cap': 4096, 'target_tokens': 3800})
    assert result.get('view') == view
    updates = result.get('replacements')
    assert isinstance(updates, dict) and set(updates) == allowed
    candidate = {**read_json(P.prompt_root / scene / view / 'finalized_prompt.json'), **updates}
    validate_prompt(candidate)
    assert len(candidate['subjects']) == len(current['subjects'])
    for key in set(current) - allowed:
        assert candidate[key] == current[key], ('Forbidden field change', key)
    after = measure(candidate)
    write_json(archive / view / 'token_after.json', after)
    write_json(archive / view / 'candidate_prompt.json', candidate)
    assert after['safe'], 'Compacted model output still exceeds token budget'
    assert after['counts']['positive']['before_cap_tokens'] < count
    candidates[view] = candidate
    calls[index] = call
    inputs.extend([current_file, archive / view / 'token_before.json', *source_inputs])
    print(f"{view}: compacted {count} -> {after['counts']['positive']['before_cap_tokens']}", flush=True)

assert candidates, 'No over-budget prompt; nothing to repair'
P.invalidate()
for view, candidate in candidates.items():
    write_json(P.prompt_root / scene / view / 'modified_prompt.json', candidate)
receipt(P.record_path(scene, 'edit'), inputs, P.scene_files(scene, 'modified_prompt.json'), calls)
print('Compacted prompts recorded; fresh audit still required', flush=True)
