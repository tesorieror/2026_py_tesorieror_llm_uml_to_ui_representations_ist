import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from es_uclm_iseresearch_uml_to_ui.config import Config


OLLAMA_BASE_URL = "http://localhost:11434"


def sha256_file(filename):
    filename = Path(filename)

    sha256 = hashlib.sha256()

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def command_output(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return None

        return result.stdout.strip()

    except (FileNotFoundError, OSError):
        return None


def get_ollama_version():
    response = requests.get(
        f"{OLLAMA_BASE_URL}/api/version",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()["version"]


def get_ollama_models():
    response = requests.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=30,
    )

    response.raise_for_status()

    return response.json().get("models", [])


def get_installed_packages():
    packages = {}

    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]

        if name:
            packages[name] = distribution.version

    return dict(
        sorted(
            packages.items(),
            key=lambda item: item[0].lower(),
        )
    )


def get_gpu_info():
    output = command_output(
        [
            "nvidia-smi",
            "--query-gpu="
            "name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )

    if output is None:
        return None

    return [
        line.strip()
        for line in output.splitlines()
        if line.strip()
    ]


def collect_files(path, pattern="*"):
    path = Path(path)

    result = {}

    if not path.exists():
        return result

    for filename in sorted(path.rglob(pattern)):
        if filename.is_file():
            result[str(filename)] = {
                "sha256": sha256_file(filename),
                "size_bytes": filename.stat().st_size,
            }

    return result


def collect_model_information(config):
    installed_models = get_ollama_models()

    installed_by_name = {
        item["name"]: item
        for item in installed_models
    }

    result = []

    for configured_model in config.models:
        model_name = configured_model["name"]

        if model_name not in installed_by_name:
            raise RuntimeError(
                f"Configured Ollama model is not installed: "
                f"{model_name}"
            )

        installed = installed_by_name[model_name]

        result.append(
            {
                "name": model_name,
                "configured": configured_model,
                "digest": installed.get("digest"),
                "size_bytes": installed.get("size"),
                "modified_at": installed.get("modified_at"),
                "details": installed.get("details"),
            }
        )

    return result


def create_manifest(metadata_filename):
    metadata_filename = Path(metadata_filename)

    config = Config.instance()

    repo_root = Path.cwd()

    git_commit = command_output(
        ["git", "rev-parse", "HEAD"]
    )

    git_status = command_output(
        ["git", "status", "--porcelain"]
    )

    manifest = {
        "manifest_version": 1,

        "created_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),

        "experiment_configuration": {
            "metadata_file": str(metadata_filename),
            "metadata_sha256": sha256_file(
                metadata_filename
            ),
            "models": config.models,
            "iterations": config.iterations,
            "uml_models": config.uml_models,
            "representations": config.representations,
        },

        "ollama": {
            "version": get_ollama_version(),
            "models": collect_model_information(
                config
            ),
        },

        "system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "gpu": get_gpu_info(),
        },

        "git": {
            "commit": git_commit,
            "dirty": (
                bool(git_status)
                if git_status is not None
                else None
            ),
            "status_porcelain": git_status,
        },

        "python_packages": get_installed_packages(),

        "inputs": collect_files(
            config.input_path
        ),

        "prompts": collect_files(
            config.prompt_path,
            "*.txt",
        ),

        "source_code": collect_files(
            repo_root
            / "src"
            / "es_uclm_iseresearch_uml_to_ui",
            "*.py",
        ),
    }

    return manifest


def save_manifest(
    metadata_filename,
    output_filename=None,
):
    config = Config.instance()

    if output_filename is None:
        output_filename = (
            config.results_path
            / "experiment_manifest.json"
        )

    output_filename = Path(output_filename)

    output_filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = create_manifest(
        metadata_filename
    )

    with open(
        output_filename,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Experiment manifest generated: "
        f"{output_filename}"
    )

    return manifest