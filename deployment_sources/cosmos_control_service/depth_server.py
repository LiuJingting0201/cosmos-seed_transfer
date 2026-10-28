"""Persistent Video-Depth-Anything HTTP worker."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from .depth_model import VideoDepthGenerator, validate_control_video
from .paths import control_cache_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthServerSettings:
    weights_path: str
    encoder: str = "vitl"
    device: str = "cuda"
    input_size: int = 518
    fp32: bool = False
    cache_dir: str = field(default_factory=lambda: str(control_cache_dir()))
    allowed_input_root: str = "/"
    max_frames: int = 5000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_input_path(path: str, allowed_root: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = Path(allowed_root).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Input video does not exist: {resolved}")
    if root != Path("/") and resolved != root and root not in resolved.parents:
        raise ValueError(f"Input video {resolved} is outside allowed root {root}")
    return resolved


class DepthControlService:
    def __init__(self, settings: DepthServerSettings, generator: VideoDepthGenerator) -> None:
        self.settings = settings
        self.generator = generator
        self.inference_lock = asyncio.Lock()
        self.cache_root = Path(settings.cache_dir).expanduser().resolve()

    @staticmethod
    def _read_cached_metadata(
        output_path: Path,
        metadata_path: Path,
        cache_key: str,
    ) -> dict[str, Any] | None:
        if not output_path.is_file() or not metadata_path.is_file():
            return None
        try:
            validate_control_video(output_path)
            metadata = json.loads(metadata_path.read_text())
        except (OSError, RuntimeError, json.JSONDecodeError):
            output_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            return None
        metadata.update({"cache_hit": True, "cache_key": cache_key})
        return metadata

    def _cache_key(
        self,
        input_path: Path,
        *,
        target_width: int | None,
        target_height: int | None,
        max_frames: int,
    ) -> str:
        weights = Path(self.settings.weights_path)
        payload = {
            "video_sha256": _sha256_file(input_path),
            "encoder": self.settings.encoder,
            "weights_size": weights.stat().st_size,
            "weights_mtime_ns": weights.stat().st_mtime_ns,
            "input_size": self.settings.input_size,
            "fp32": self.settings.fp32,
            "target_width": target_width,
            "target_height": target_height,
            "max_frames": max_frames,
            "normalization": "global-minmax-v1",
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        requested_encoder = str(payload.get("encoder") or self.settings.encoder)
        if requested_encoder != self.settings.encoder:
            raise ValueError(
                f"Depth worker is serving encoder={self.settings.encoder!r}, "
                f"but request asked for {requested_encoder!r}."
            )
        input_path = _resolve_input_path(str(payload.get("input_video") or ""), self.settings.allowed_input_root)
        target_width = int(payload["target_width"]) if payload.get("target_width") is not None else None
        target_height = int(payload["target_height"]) if payload.get("target_height") is not None else None
        max_frames = min(int(payload.get("max_frames") or self.settings.max_frames), self.settings.max_frames)
        if max_frames <= 0:
            raise ValueError("max_frames must be positive.")

        cache_key = self._cache_key(
            input_path,
            target_width=target_width,
            target_height=target_height,
            max_frames=max_frames,
        )
        output_dir = self.cache_root / "depth" / cache_key
        output_path = output_dir / "control_depth.mp4"
        metadata_path = output_dir / "metadata.json"
        if metadata := self._read_cached_metadata(output_path, metadata_path, cache_key):
            return metadata

        async with self.inference_lock:
            if metadata := self._read_cached_metadata(output_path, metadata_path, cache_key):
                return metadata
            output_dir.mkdir(parents=True, exist_ok=True)
            result = await asyncio.to_thread(
                self.generator.generate,
                input_video=str(input_path),
                output_video=str(output_path),
                target_width=target_width,
                target_height=target_height,
                max_frames=max_frames,
            )
            metadata = {
                "control_path": result.output_path,
                "fps": result.fps,
                "num_frames": result.num_frames,
                "width": result.width,
                "height": result.height,
                "encoder": self.settings.encoder,
                "cache_key": cache_key,
                "cache_hit": False,
                "source_path": str(input_path),
            }
            temporary = metadata_path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True))
            os.replace(temporary, metadata_path)
            return metadata


def create_app(
    settings: DepthServerSettings,
    *,
    generator: VideoDepthGenerator | None = None,
) -> FastAPI:
    resolved_generator = generator or VideoDepthGenerator(
        weights_path=settings.weights_path,
        encoder=settings.encoder,
        device=settings.device,
        input_size=settings.input_size,
        fp32=settings.fp32,
    )
    service = DepthControlService(settings, resolved_generator)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("Loading persistent Video-Depth-Anything model from %s", settings.weights_path)
        await asyncio.to_thread(resolved_generator.setup)
        logger.info("Video-Depth-Anything model is ready on %s", settings.device)
        yield

    app = FastAPI(title="Cosmos3 Depth Control Worker", lifespan=lifespan)
    app.state.depth_service = service

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": resolved_generator.model is not None,
            "encoder": settings.encoder,
            "device": settings.device,
            "weights_path": str(Path(settings.weights_path).resolve()),
        }

    @app.post("/v1/depth-controls")
    async def create_depth_control(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            return await service.generate(payload)
        except (TypeError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Depth control generation failed")
            raise HTTPException(status_code=500, detail=f"Depth generation failed: {exc}") from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the persistent Cosmos3 depth-control worker.")
    parser.add_argument("--weights", required=True, help="Path to Video-Depth-Anything weights.")
    parser.add_argument("--encoder", choices=["vits", "vitl"], default="vitl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--cache-dir", default=str(control_cache_dir()))
    parser.add_argument("--allowed-input-root", default="/")
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = DepthServerSettings(
        weights_path=args.weights,
        encoder=args.encoder,
        device=args.device,
        input_size=args.input_size,
        fp32=args.fp32,
        cache_dir=args.cache_dir,
        allowed_input_root=args.allowed_input_root,
        max_frames=args.max_frames,
    )
    logging.basicConfig(level=args.log_level.upper())
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
