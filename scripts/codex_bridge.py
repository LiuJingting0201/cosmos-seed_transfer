"""File-based Codex provider. Runs on the GPU host; never connects to GPT."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def response_text(response):
    if response.get('provider') != 'codex-local' or response.get('status') != 'completed':
        raise ValueError('Codex call did not complete')
    if response.get('requested_model') != 'gpt-6-astra':
        raise ValueError('Unexpected Codex model')
    text = response.get('output_text')
    if not isinstance(text, str) or not isinstance(json.loads(text), dict):
        raise ValueError('Codex must return a JSON object')
    return text


def unwrap_final(final, request):
    if request.get('business_output_schema') is not None:
        if set(final) != {'result'} or not isinstance(final['result'], dict):
            raise ValueError('Expected schema-constrained result object')
        return json.dumps(final['result'], ensure_ascii=False)
    if set(final) != {'result_json'} or not isinstance(final['result_json'], str):
        raise ValueError('Expected result_json string envelope')
    return final['result_json']


def validate_evidence(directory):
    directory = Path(directory)
    request = read_json(directory / 'request.json')
    response = read_json(directory / 'response.json')
    text = response_text(response)
    if response['request_sha256'] != digest(directory / 'request.json'):
        raise ValueError('Codex result belongs to a different request')
    if response['requested_model'] != request['model']:
        raise ValueError('Requested model mismatch')
    for name in ('events.jsonl', 'final.json'):
        if digest(directory / name) != response['artifacts'][name]:
            raise ValueError('Codex raw evidence was modified: ' + name)
    final = read_json(directory / 'final.json')
    if unwrap_final(final, request) != text:
        raise ValueError('Final output differs from recorded Codex response')
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines() if line.strip()]
    if not any(e.get('type') == 'turn.completed' for e in events):
        raise ValueError('No completed Codex turn')
    if any(e.get('type') in ('turn.failed', 'error') for e in events):
        raise ValueError('Failed Codex events')
    outputs = [e['item'].get('text') for e in events
               if e.get('type') == 'item.completed' and e.get('item', {}).get('type') == 'agent_message']
    if not outputs or json.loads(outputs[-1]) != final:
        raise ValueError('Final JSON does not match raw Codex event')
    forbidden = {'command_execution', 'mcp_tool_call', 'web_search', 'file_change'}
    if any(e.get('item', {}).get('type') in forbidden for e in events):
        raise ValueError('Pure model stage unexpectedly used a tool')
    if response.get('exit_code') != 0:
        raise ValueError('Codex process failed')
    return json.loads(text), response


def call_codex(payload, evidence, timeout=2400):
    """Export request, await locally computed evidence, verify before importing."""
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    request_path = evidence / 'request.json'
    if request_path.exists():
        if read_json(request_path) != payload:
            raise ValueError('Cannot reuse evidence for changed input')
    else:
        write_json(request_path, payload)
    if (evidence / 'response.json').exists():
        return validate_evidence(evidence)
    root = Path(os.environ['CODEX_BRIDGE_QUEUE'])
    root.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    staging = root / ('.' + job_id)
    staging.mkdir()
    shutil.copyfile(request_path, staging / 'request.json')
    write_json(staging / 'job.json', {'id': job_id, 'evidence': str(evidence.resolve()),
                                     'request_sha256': digest(request_path)})
    job = root / job_id
    staging.rename(job)
    print('CODEX_JOB=' + job_id, flush=True)
    start = time.monotonic()
    while not (job / 'response' / 'DONE').exists():
        if time.monotonic() - start > timeout:
            write_json(job / 'expired.json', {'reason': 'timeout'})
            raise TimeoutError('Local Codex worker did not answer ' + job_id)
        time.sleep(1)
    response_dir = job / 'response'
    if (response_dir / 'failure.json').exists():
        shutil.copyfile(response_dir / 'failure.json', evidence / 'failure.json')
        raise RuntimeError('Local Codex worker failed; see ' + str(evidence / 'failure.json'))
    for name in ('response.json', 'events.jsonl', 'final.json'):
        shutil.copyfile(response_dir / name, evidence / name)
    parsed, response = validate_evidence(evidence)
    (evidence / 'raw_text.txt').write_text(response['output_text'])
    write_json(evidence / 'parsed.json', parsed)
    write_json(evidence / 'call.json', {
        'provider': 'codex-local', 'endpoint': 'local-codex-over-ssh',
        'requested_model': payload['model'], 'response_model': response.get('response_model'),
        'response_id': response.get('thread_id'), 'elapsed_s': response['elapsed_s'],
        'usage': response.get('usage'), 'status': 'completed',
    })
    write_json(job / 'consumed.json', {'evidence': str(evidence)})
    return parsed, response
