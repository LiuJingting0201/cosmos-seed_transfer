#!/usr/bin/env python3
"""CPU-only preflight using installed Cosmos3 formatter/tokenizer; never loads weights."""
from __future__ import annotations
import ast
import hashlib
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

SOURCE = Path('/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py')
MODEL = Path('/vemlp-demo-models/models/nvidia/Cosmos3-Nano/Cosmos3-Nano')


def load_formatters():
    import torch
    src = SOURCE.read_text()
    tree = ast.parse(src)
    methods = {'_get_sp_param', '_apply_metadata_templates', '_normalize_token_ids',
               '_tokenize_prompt', '_format_and_tokenize_prompts'}
    ns = {'json': json, 'torch': torch, 'logger': logging.getLogger('prompt_budget'),
          '_is_rank_zero': lambda: True}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith('COSMOS3_'):
                    try: ns[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError): pass
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Cosmos3OmniDiffusersPipeline')
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    assert {n.name for n in selected} == methods
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_format_json_object_prompt']
    slim = ast.Module(body=[*ast.parse('from __future__ import annotations').body, *funcs,
                          ast.ClassDef(name='TextOnly', bases=[], keywords=[], body=selected, decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(slim), str(SOURCE), 'exec'), ns)
    action = SOURCE.with_name('action.py').read_text()
    nodes = ast.parse(action).body
    ns['VIDEO_RES_SIZE_INFO'] = ast.literal_eval(next(n.value for n in nodes if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == 'VIDEO_RES_SIZE_INFO'))
    fn = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == 'find_closest_target_size')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE.with_name('action.py')), 'exec'), ns)
    return ns


def audit(req):
    from transformers import AutoTokenizer
    ns = load_formatters()
    tok = AutoTokenizer.from_pretrained(str(MODEL), subfolder='text_tokenizer', local_files_only=True)
    pipe = ns['TextOnly']()
    records = []
    class Recorder:
        def __getattr__(self, key): return getattr(tok, key)
        def apply_chat_template(self, messages, **kwargs):
            result = tok.apply_chat_template(messages, **kwargs)
            ids = pipe._normalize_token_ids(result)
            records.append({'before_cap_tokens': len(ids), 'formatted_text': messages[-1]['content'],
                            'chat_template_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
            return result
    pipe.tokenizer = Recorder()
    pipe.device = 'cpu'
    extra = req.get('extra_params', {})
    # These copies use the transfer path, whose system prompt is always enabled.
    if not any(extra.get(k) for k in ('edge', 'seg', 'depth', 'blur', 'wsm')):
        raise ValueError('This guard supports Cosmos3 transfer requests only')
    raw = req['prompt']
    negative = req.get('negative_prompt', '')
    cap = int(extra.get('max_sequence_length') or ns['COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH'])
    if cap > ns['COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH']:
        raise ValueError('Do not raise the model token cap to bypass the guard')
    video = req.get('video_meta')
    if video is None:
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                'stream=width,height,avg_frame_rate,nb_frames', '-of', 'json', req['source_video']],
                               check=True, text=True, capture_output=True)
        stream = json.loads(probe.stdout)['streams'][0]
        n,d = stream['avg_frame_rate'].split('/')
        video = {'width':stream['width'], 'height':stream['height'], 'fps':float(n)/float(d),
                 'num_frames':int(stream['nb_frames'])}
    width,height = ns['find_closest_target_size'](video['height'], video['width'], extra.get('resolution',720))
    total = min(video['num_frames'],int(extra.get('max_frames',video['num_frames'])))
    chunk = int(extra.get('num_video_frames_per_chunk',121)) if total > 1 else 1
    temporal = 4  # Verified against the deployed VAE config; fail closed if changed.
    vae = json.loads((MODEL/'vae/config.json').read_text())
    if int(vae['scale_factor_temporal']) != temporal:
        raise ValueError('VAE temporal compression changed; update budget preflight')
    chunk = math.ceil((chunk-1)/temporal)*temporal+1
    pipe._format_and_tokenize_prompts(raw, negative, chunk, float(video['fps']), height, width,
        cap, SimpleNamespace(extra_args=extra), True, is_t2i=False,
        default_system_prompt=ns['COSMOS3_TRANSFER_SYSTEM_PROMPT'])
    assert len(records)==2
    result={'pipeline_source':str(SOURCE),'pipeline_source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'tokenizer':str(MODEL/'text_tokenizer'), 'tokenizer_class':type(tok).__name__,
            'cap_before_two_terminal_tokens':cap,'terminal_tokens_after_cap':2,
            'metadata':{'height':height,'width':width,'chunk_frames':chunk,'fps':video['fps']},
            'counts':{},'safe':True}
    for label,text,rec in zip(('positive','negative'),(raw,negative),records):
        count=rec.pop('before_cap_tokens')
        formatted=rec.pop('formatted_text')
        rec.update(raw_chars=len(text), raw_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   formatted_chars=len(formatted), formatted_sha256=hashlib.sha256(formatted.encode()).hexdigest(),
                   before_cap_tokens=count, actual_encoded_tokens=min(count,cap)+2,
                   tokens_removed=max(0,count-cap), headroom=cap-count,
                   formatted_tail=formatted[-200:])
        result['counts'][label]=rec
        result['safe'] &= count <= cap
    return result


def main():
    req=json.load(sys.stdin)
    report=audit(req)
    if req.get('report_path'):
        Path(req['report_path']).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
    if not report['safe']:raise SystemExit(2)

if __name__=='__main__':main()
