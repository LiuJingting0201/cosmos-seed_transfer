#!/usr/bin/env python3
"""Correct one per-view color phrase using a fresh, recorded GPT editor call."""
import copy
from pathlib import Path
import shutil
import sys
import time

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R / 'multiview/scripts'))
from run_model_pipeline import Pipeline, validate_prompt
from model_api import read_json, write_json, validate_receipt, receipt

P = Pipeline(R / 'multiview/configs/sand.json')
scene = 'rwy_split2'
view = 'rwy_split2_v_up_left_original'
assert not any((P.experiment / 'runs').rglob('async_submit.json'))
old = validate_receipt(P.record_path(scene, 'edit'))
archive = P.experiment / 'repair_history' / ('composition_color_' + str(time.time_ns()))
archive.mkdir(parents=True, exist_ok=False)
for stage in ('edit', 'audit'):
    shutil.copy2(P.record_path(scene, stage), archive / (stage + '.json'))
for v in P.scenes[scene]:
    (archive / v).mkdir()
    for name in ('modified_prompt.json', 'compliance_report.json'):
        shutil.copy2(P.prompt_root / scene / v / name, archive / v / name)
current = read_json(archive / view / 'modified_prompt.json')
fields = {'subjects', 'cinematography', 'actions', 'segments', 'transitions',
          'temporal_caption', 'audio_description', 'text_and_signage_elements', 'aesthetics'}
result, call, source_inputs = P.visual_model(scene, 'edit/' + view, """
Make one precise correction to the CURRENT already-edited upward-view prompt. The
independent auditor found that aesthetics.composition abbreviates the preserved housing
color as 'teal', whereas its subject, palette and other fields use 'dark teal-blue'.
Inspect the supplied source sheets and feedback. Align this occurrence in composition
with the existing exact object-color phrase. Change no other content or punctuation.
Composition is per-view under the schema; the shared color_scheme, mood_atmosphere and
patterns inside aesthetics MUST remain byte-identical. All other fields stay untouched.
Return exactly {"view":"<current_view>","replacements":{...},"findings":[...]}.
Return the complete CURRENT values for all eight normal per-view fields (subjects,
cinematography, actions, segments, transitions, temporal_caption, audio_description,
text_and_signage_elements), plus the complete aesthetics object with only this one
composition phrase corrected. This preserves previous grounded edits when replacing
fields in the finalized base. Return JSON only.
""", {'user_prompt': P.config['user_prompt'], 'current_view': view,
       'current_view_prompt': current,
       'independent_audit_feedback': read_json(archive / view / 'compliance_report.json')})
assert result.get('view') == view
updates = result.get('replacements')
assert isinstance(updates, dict) and set(updates) == fields
candidate = {**read_json(P.prompt_root / scene / view / 'finalized_prompt.json'), **updates}
validate_prompt(candidate)
expected = copy.deepcopy(current)
old_composition = current['aesthetics']['composition']
assert 'teal housing' in old_composition and 'dark teal-blue housing' not in old_composition
expected['aesthetics']['composition'] = old_composition.replace('teal housing', 'dark teal-blue housing')
assert candidate == expected, 'Reject changes beyond the reviewed color phrase'
write_json(archive / 'repair_diff.json', {
    'path': 'aesthetics.composition', 'before': old_composition,
    'after': candidate['aesthetics']['composition'], 'editor_call': str(call)})
P.invalidate()
write_json(P.prompt_root / scene / view / 'modified_prompt.json', candidate)
calls = [Path(c) for c in old['calls']]
calls[list(P.scenes[scene]).index(view)] = call
inputs = [Path(p) for p in old['inputs']]
inputs.extend([archive / 'edit.json', archive / 'audit.json',
               archive / view / 'modified_prompt.json', archive / view / 'compliance_report.json',
               Path(__file__), *source_inputs])
receipt(P.record_path(scene, 'edit'), inputs, P.scene_files(scene, 'modified_prompt.json'), calls)
print('Composition phrase repaired with model evidence; fresh audit still required', flush=True)
