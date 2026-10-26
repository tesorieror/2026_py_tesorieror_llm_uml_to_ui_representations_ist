import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import requests

from es_uclm_iseresearch_uml_to_ui.config import Config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(args: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, check=False, timeout=30)
        return {
            "command": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"command": args, "error": repr(exc)}


def _ollama_show(model_name: str, base_url: str) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/show",
            json={"model": model_name},
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        digest = None
        try:
            tags_response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=60)
            tags_response.raise_for_status()
            for item in tags_response.json().get("models", []):
                if item.get("name") == model_name or item.get("model") == model_name:
                    digest = item.get("digest")
                    break
        except Exception:
            pass
        return {
            "requested_name": model_name,
            "digest": digest,
            "details": body.get("details"),
            "model_info": body.get("model_info"),
            "parameters": body.get("parameters"),
            "template_sha256": hashlib.sha256(body.get("template", "").encode()).hexdigest(),
            "system_sha256": hashlib.sha256(body.get("system", "").encode()).hexdigest(),
        }
    except Exception as exc:
        return {"requested_name": model_name, "error": repr(exc)}


def build_manifest(metadata_file: str = "experiment/metadata.json") -> dict[str, Any]:
    config = Config.instance()
    project_root = Path.cwd().resolve()
    files = []
    for relative_root in (config.input_path, config.prompt_path):
        root = relative_root.resolve()
        if root.exists():
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                files.append({
                    "path": str(path.relative_to(project_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                })

    base_url = config.metadata.get("ollama_base_url", "http://localhost:11434")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_file": metadata_file,
        "metadata_sha256": _sha256(Path(metadata_file)),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "python_executable": sys.executable,
            "cpu_count": os.cpu_count(),
        },
        "software": {
            "pip_packages": sorted(
                ({"name": d.metadata["Name"], "version": d.version} for d in distributions()),
                key=lambda item: item["name"].lower(),
            ),
            "ollama_version": _command(["ollama", "--version"]),
            "git_commit": _command(["git", "rev-parse", "HEAD"]),
            "git_status": _command(["git", "status", "--porcelain"]),
            "gpu": _command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        },
        "ollama_models": [
            _ollama_show(model["name"], base_url) for model in config.models
        ],
        "artifacts": files,
    }


def save_manifest(filename: str = "experiment/experiment_manifest.json") -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(build_manifest(), f, indent=2, ensure_ascii=False)
    return path
