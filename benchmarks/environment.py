import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def git_output(*args: str):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def build_environment_metadata():
    import torch

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpus = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
        properties = torch.cuda.get_device_properties(0)
        gpu_name = gpus[0]["name"]
        gpu_memory_bytes = properties.total_memory
        peer_access = {
            f"{source}->{target}": bool(
                torch.cuda.can_device_access_peer(source, target)
            )
            for source in range(torch.cuda.device_count())
            for target in range(torch.cuda.device_count())
            if source != target
        }
    else:
        gpu_name = None
        gpu_memory_bytes = None
        gpus = []
        peer_access = {}
    try:
        topology = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        topology = None
    commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_memory_bytes": gpu_memory_bytes,
        "gpus": gpus,
        "cuda_peer_access": peer_access,
        "nvidia_smi_topology": topology,
    }


def discover_model_revision(model_path: str | None):
    if model_path is None:
        return None
    metadata_path = (
        Path(model_path)
        / ".cache"
        / "huggingface"
        / "download"
        / "config.json.metadata"
    )
    try:
        revision = metadata_path.read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    if len(revision) != 40:
        return None
    try:
        int(revision, 16)
    except ValueError:
        return None
    return revision


def atomic_write_json(path: Path, value: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary_path, path)
