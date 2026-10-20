#!/usr/bin/env python3
"""Recorded model calls and content-addressed stage receipts (no credential logging)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid
from codex_bridge import call_codex, validate_evidence as validate_codex_evidence, response_text as codex_response_text


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fingerprints(paths) -> dict[str, str]:
    return {str(Path(p).resolve()): digest(Path(p)) for p in paths}


def verify_hashes(hashes: dict[str, str]) -> None:
    if not hashes:
        raise ValueError("Empty provenance manifest")
    for name, expected in hashes.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Artifact changed since model call/audit: {name}")


def parse_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Model must return one JSON object")
    return value


def response_text(response: dict, provider: str) -> str:
    if provider == "codex-local":
        return codex_response_text(response)
    if response.get("error"):
        raise ValueError("Model response contains an error")
    if provider == "ark":
        if response.get("status") != "completed" or response.get("incomplete_details"):
            raise ValueError(f"Incomplete Ark response: {response.get('status')}")
        parts = [
            part["text"] for item in response.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text" and isinstance(part.get("text"), str)
        ]
        text = "\n".join(parts)
    else:
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError(f"Incomplete Reasoner response: {choice.get('finish_reason')}")
        text = choice["message"]["content"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Model returned no final text")
    return text


def call_model(url: str, payload: dict, provider: str, evidence_root: Path,
               key_env: str, timeout: float = 600, format_retry: bool = True,
               rate_limit_retries: int = 6, rate_limit_wait: int = 120) -> tuple[dict, Path]:
    if provider == "codex-local":
        evidence = evidence_root / uuid.uuid4().hex
        parsed, response = call_codex(payload, evidence)
        return parsed, evidence
    key = os.environ.get(key_env, "")
    if provider == "ark" and not key:
        raise ValueError(f"{key_env} is missing; no template fallback is allowed")
    evidence = evidence_root / uuid.uuid4().hex
    evidence.mkdir(parents=True, exist_ok=False)
    write_json(evidence / "request.json", payload)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as result:
            raw = result.read()
        (evidence / "response.json").write_bytes(raw)
        response = json.loads(raw)
        text = response_text(response, provider)
        (evidence / "raw_text.txt").write_text(text, encoding="utf-8")
        parsed = parse_object(text)
        write_json(evidence / "parsed.json", parsed)
        write_json(evidence / "call.json", {
            "provider": provider, "endpoint": url, "requested_model": payload["model"],
            "response_model": response.get("model"), "response_id": response.get("id"),
            "elapsed_s": round(time.monotonic() - start, 3),
            "usage": response.get("usage"), "status": "completed",
        })
        return parsed, evidence
    except Exception as exc:
        # Do not persist exception bodies: gateways may echo authentication headers.
        write_json(evidence / "failure.json", {
            "type": type(exc).__name__, "http_status": getattr(exc, "code", None),
            "elapsed_s": round(time.monotonic() - start, 3),
        })
        if getattr(exc, "code", None) == 429 and rate_limit_retries > 0:
            print(
                f"Model rate limited; retrying this call in {rate_limit_wait}s, "
                f"preserved failure at {evidence}",
                flush=True,
            )
            time.sleep(rate_limit_wait)
            parsed, retry_evidence = call_model(
                url, payload, provider, evidence_root, key_env, timeout, format_retry,
                rate_limit_retries=rate_limit_retries - 1,
                rate_limit_wait=rate_limit_wait,
            )
            write_json(retry_evidence / "retry_of.json", {
                "call": str(evidence), "reason": "http_429",
            })
            return parsed, retry_evidence
        if format_retry and isinstance(exc, json.JSONDecodeError) and (evidence / "raw_text.txt").exists():
            retry_payload = json.loads(json.dumps(payload))
            feedback = (
                f"The previous response was invalid JSON: {exc.msg} at character {exc.pos}. "
                "Repeat the requested task and return exactly one valid JSON object. "
                "Check balanced braces and commas. No trailing prose or extra root objects."
            )
            if provider == "ark":
                retry_payload["input"].append({
                    "role": "user", "content": [{"type": "input_text", "text": feedback}],
                })
            else:
                retry_payload["messages"].append({"role": "user", "content": feedback})
            print(f"Invalid model JSON; one model retry, preserved failure at {evidence}", flush=True)
            parsed, retry_evidence = call_model(
                url, retry_payload, provider, evidence_root, key_env, timeout,
                format_retry=False, rate_limit_retries=rate_limit_retries,
                rate_limit_wait=rate_limit_wait,
            )
            write_json(retry_evidence / "retry_of.json", {"call": str(evidence), "reason": "invalid_json"})
            return parsed, retry_evidence
        raise RuntimeError(f"Model call failed ({type(exc).__name__}); evidence: {evidence}") from exc


def receipt(path: Path, inputs, outputs, calls=()) -> None:
    evidence_files = []
    for call in calls:
        evidence_files.extend(Path(call) / name for name in (
            "request.json", "response.json", "raw_text.txt", "parsed.json", "call.json",
        ))
        if read_json(Path(call) / "call.json")["provider"] == "codex-local":
            evidence_files.extend(Path(call) / name for name in ("events.jsonl", "final.json"))
    write_json(path, {
        "inputs": fingerprints(inputs), "outputs": fingerprints(outputs),
        "evidence": fingerprints(evidence_files), "calls": [str(p) for p in calls],
    })


def validate_receipt(path: Path, require_model: bool = True) -> dict:
    record = read_json(path)
    verify_hashes(record["inputs"])
    verify_hashes(record["outputs"])
    if require_model:
        if not record.get("calls"):
            raise ValueError(f"No model evidence in {path}")
        verify_hashes(record["evidence"])
        for directory in record["calls"]:
            directory = Path(directory)
            call = read_json(directory / "call.json")
            if call["provider"] == "codex-local":
                validate_codex_evidence(directory)
            response = read_json(directory / "response.json")
            parsed = parse_object(response_text(response, call["provider"]))
            if call["status"] != "completed" or parsed != read_json(directory / "parsed.json"):
                raise ValueError(f"Invalid model evidence: {directory}")
    return record
