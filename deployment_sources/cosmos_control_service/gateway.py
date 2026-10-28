"""Multipart proxy that materializes automatic Cosmos3 depth controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from starlette.datastructures import UploadFile

from .paths import control_cache_dir

logger = logging.getLogger(__name__)

HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
DEPTH_GATEWAY_FIELDS = {
    "auto_generate",
    "encoder",
    "input_size",
    "max_frames",
    "target_width",
    "target_height",
}


@dataclass(frozen=True)
class GatewaySettings:
    upstream_url: str = "http://127.0.0.1:8000"
    depth_worker_url: str = "http://127.0.0.1:8101"
    cache_dir: str = field(default_factory=lambda: str(control_cache_dir()))
    request_timeout_s: float = 3600.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _parse_extra_params(raw: str | None) -> dict[str, Any]:
    if raw is None or not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("extra_params must be a JSON object.")
    return parsed


def _target_dimensions(fields: dict[str, str], depth: dict[str, Any]) -> tuple[int | None, int | None]:
    if depth.get("target_width") is not None or depth.get("target_height") is not None:
        width = int(depth["target_width"]) if depth.get("target_width") is not None else None
        height = int(depth["target_height"]) if depth.get("target_height") is not None else None
        return width, height
    if fields.get("width") and fields.get("height"):
        return int(fields["width"]), int(fields["height"])
    if fields.get("size") and "x" in fields["size"].lower():
        width, height = fields["size"].lower().split("x", maxsplit=1)
        return int(width), int(height)
    return None, None


def materialize_depth_extra_params(
    extra_params: dict[str, Any],
    *,
    control_path: str,
) -> dict[str, Any]:
    rewritten = dict(extra_params)
    raw_depth = rewritten.get("depth")
    if not isinstance(raw_depth, dict):
        raise ValueError("extra_params.depth must be a JSON object.")
    depth = {key: value for key, value in raw_depth.items() if key not in DEPTH_GATEWAY_FIELDS}
    depth["control_path"] = control_path
    rewritten["depth"] = depth
    return rewritten


async def _persist_input(cache_dir: str, filename: str, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    suffix = Path(filename).suffix.lower()
    if len(suffix) > 12 or not suffix:
        suffix = ".mp4"
    output = Path(cache_dir).expanduser().resolve() / "inputs" / f"{digest}{suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_bytes(content)
        temporary.replace(output)
    return str(output)


def _forward_headers(request: Request) -> dict[str, str]:
    allowed: dict[str, str] = {}
    for name in ("authorization", "x-request-id"):
        value = request.headers.get(name)
        if value:
            allowed[name] = value
    return allowed


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {key: value for key, value in response.headers.items() if key.lower() not in HOP_BY_HOP_HEADERS}


def create_app(
    settings: GatewaySettings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    app = FastAPI(title="Cosmos3 Control Gateway")

    async def proxy_video(request: Request, endpoint: str) -> Response:
        try:
            form = await request.form()
            fields: dict[str, str] = {}
            files: dict[str, tuple[str, bytes, str]] = {}
            for key, value in form.multi_items():
                if isinstance(value, UploadFile):
                    content = await value.read()
                    files[key] = (
                        value.filename or "input.mp4",
                        content,
                        value.content_type or "application/octet-stream",
                    )
                else:
                    fields[key] = str(value)

            extra_params = _parse_extra_params(fields.get("extra_params"))
            raw_depth = extra_params.get("depth")
            auto_generate = isinstance(raw_depth, dict) and _as_bool(raw_depth.get("auto_generate"))
            depth_metadata: dict[str, Any] | None = None
            if auto_generate:
                assert isinstance(raw_depth, dict)
                if raw_depth.get("control_path"):
                    fields["extra_params"] = json.dumps(
                        materialize_depth_extra_params(
                            extra_params,
                            control_path=str(raw_depth["control_path"]),
                        ),
                        separators=(",", ":"),
                    )
                else:
                    input_file = files.get("input_reference")
                    if input_file is None:
                        raise ValueError("depth.auto_generate currently requires an uploaded input_reference video.")
                    input_path = await _persist_input(
                        settings.cache_dir,
                        input_file[0],
                        input_file[1],
                    )
                    target_width, target_height = _target_dimensions(fields, raw_depth)
                    depth_request: dict[str, Any] = {
                        "input_video": input_path,
                        "encoder": str(raw_depth.get("encoder") or "vitl"),
                    }
                    if target_width is not None:
                        depth_request["target_width"] = target_width
                    if target_height is not None:
                        depth_request["target_height"] = target_height
                    if raw_depth.get("max_frames") is not None:
                        depth_request["max_frames"] = int(raw_depth["max_frames"])
                    elif fields.get("num_frames"):
                        depth_request["max_frames"] = int(fields["num_frames"])

                    async with httpx.AsyncClient(
                        transport=transport,
                        timeout=settings.request_timeout_s,
                    ) as client:
                        depth_response = await client.post(
                            f"{settings.depth_worker_url.rstrip('/')}/v1/depth-controls",
                            json=depth_request,
                        )
                    if depth_response.is_error:
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                f"Depth worker failed with HTTP {depth_response.status_code}: {depth_response.text}"
                            ),
                        )
                    depth_metadata = depth_response.json()
                    fields["extra_params"] = json.dumps(
                        materialize_depth_extra_params(
                            extra_params,
                            control_path=str(depth_metadata["control_path"]),
                        ),
                        separators=(",", ":"),
                    )

            async with httpx.AsyncClient(
                transport=transport,
                timeout=settings.request_timeout_s,
            ) as client:
                upstream_response = await client.post(
                    f"{settings.upstream_url.rstrip('/')}{endpoint}",
                    data=fields,
                    files=files or None,
                    headers=_forward_headers(request),
                )
            headers = _response_headers(upstream_response)
            if depth_metadata is not None:
                headers["X-Cosmos-Depth-Control-Path"] = str(depth_metadata["control_path"])
                headers["X-Cosmos-Depth-Cache-Hit"] = str(bool(depth_metadata.get("cache_hit"))).lower()
            return Response(
                content=upstream_response.content,
                status_code=upstream_response.status_code,
                media_type=upstream_response.headers.get("content-type"),
                headers=headers,
            )
        except HTTPException:
            raise
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            logger.exception("Control gateway upstream request failed")
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    @app.get("/health")
    async def health() -> Response:
        details: dict[str, Any] = {}
        ok = True
        async with httpx.AsyncClient(transport=transport, timeout=5.0) as client:
            for name, url in (
                ("upstream", f"{settings.upstream_url.rstrip('/')}/health"),
                ("depth_worker", f"{settings.depth_worker_url.rstrip('/')}/health"),
            ):
                try:
                    response = await client.get(url)
                    details[name] = {"status_code": response.status_code, "ok": response.is_success}
                    ok = ok and response.is_success
                except httpx.HTTPError as exc:
                    details[name] = {"ok": False, "error": str(exc)}
                    ok = False
        return Response(
            content=json.dumps({"status": "ok" if ok else "unhealthy", "services": details}),
            status_code=200 if ok else 503,
            media_type="application/json",
        )

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
            response = await client.get(
                f"{settings.upstream_url.rstrip('/')}/v1/models",
                headers=_forward_headers(request),
            )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
            headers=_response_headers(response),
        )

    @app.post("/v1/videos/sync")
    async def videos_sync(request: Request) -> Response:
        return await proxy_video(request, "/v1/videos/sync")

    @app.post("/v1/videos")
    async def videos(request: Request) -> Response:
        return await proxy_video(request, "/v1/videos")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the non-invasive Cosmos3 control gateway.")
    parser.add_argument("--upstream-url", default="http://127.0.0.1:8000")
    parser.add_argument("--depth-worker-url", default="http://127.0.0.1:8101")
    parser.add_argument("--cache-dir", default=str(control_cache_dir()))
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = GatewaySettings(
        upstream_url=args.upstream_url,
        depth_worker_url=args.depth_worker_url,
        cache_dir=args.cache_dir,
        request_timeout_s=args.request_timeout,
    )
    logging.basicConfig(level=args.log_level.upper())
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
