"""Shared guards for the new pipeline copies; standard library only."""
from pathlib import Path
import hashlib
import json
import subprocess


def check_budget(run_dir, prompt, negative_prompt, extra_params, source_video):
    root=Path(__file__).resolve().parents[1]
    req={'prompt':prompt,'negative_prompt':negative_prompt,'extra_params':extra_params,
         'source_video':str(source_video),'report_path':str(run_dir/'prompt_token_audit.json')}
    subprocess.run(['/usr/bin/python3',str(root/'prompt_budget.py')],input=json.dumps(req),text=True,check=True,
                   stdout=subprocess.DEVNULL)


def check_echo(run_dir, response, expected):
    actual=response.get('prompt')
    report={'request_id':response.get('id'),'expected_chars':len(expected),
            'actual_chars':len(actual) if isinstance(actual,str) else None,
            'expected_sha256':hashlib.sha256(expected.encode()).hexdigest(),
            'actual_sha256':hashlib.sha256(actual.encode()).hexdigest() if isinstance(actual,str) else None,
            'exact_match':actual==expected}
    (run_dir/'prompt_transport_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['exact_match']:
        raise RuntimeError('Cosmos API prompt echo differs from submitted text; request recorded, stop pipeline')
