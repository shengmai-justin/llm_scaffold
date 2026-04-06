"""Ray-based parallel evaluation workers for multi-GPU RL pipeline.

Each worker owns an isolated repo copy and a GPU.
Workers receive edited code strings, run train.py, and return parsed metrics.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import ray


def create_worker_repo(base_repo: str, worker_id: int) -> str:
    """Create isolated repo copy for a worker."""
    repo_name = os.path.basename(base_repo)
    worker_dir = os.path.join(os.path.dirname(base_repo), f"{repo_name}_worker_{worker_id}")
    if not os.path.exists(worker_dir):
        shutil.copytree(base_repo, worker_dir)
    return worker_dir


def parse_metrics_from_output(output: str) -> tuple[float | None, int | None]:
    """Parse val_bpb and peak_vram_mb from train.py stdout."""
    val_bpb = None
    peak_vram_mb = None
    m = re.search(r"val_bpb:\s+([\d.]+)", output)
    if m:
        val_bpb = float(m.group(1))
    m = re.search(r"peak_vram_mb:\s+([\d.]+)", output)
    if m:
        peak_vram_mb = int(float(m.group(1)))
    return val_bpb, peak_vram_mb


@ray.remote
class EvalWorker:
    """Each worker owns an isolated repo copy and a GPU."""

    def __init__(self, gpu_id: int, base_repo: str, worker_id: int,
                 gpu_mem_limit_mb: int = 0):
        self.gpu_id = gpu_id
        self.gpu_mem_limit_mb = gpu_mem_limit_mb
        self.repo_path = create_worker_repo(base_repo, worker_id)

    def evaluate(self, parent_code: str, edited_code: str, step: int) -> dict:
        """Write edited code, run train.py, parse metrics, reset."""
        train_path = os.path.join(self.repo_path, "train.py")

        # Write edited code
        Path(train_path).write_text(edited_code)

        try:
            # Run train.py on assigned GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            if self.gpu_mem_limit_mb > 0:
                env["GPU_MEM_LIMIT_MB"] = str(self.gpu_mem_limit_mb)
                lib = os.path.abspath(os.path.join(
                    os.path.dirname(__file__),
                    "..", "gpu_mem_limit", "libgpumemlimit.so"))
                if not os.path.exists(lib):
                    raise FileNotFoundError(
                        f"gpu_mem_limit not compiled: {lib}\n"
                        f"Run: make -C {os.path.dirname(lib)}")
                env["LD_PRELOAD"] = lib
            try:
                r = subprocess.run(
                    ["uv", "run", "train.py"],
                    cwd=self.repo_path,
                    capture_output=True, text=True,
                    timeout=600, env=env,
                )
                output = r.stdout + r.stderr
                timed_out = False
            except subprocess.TimeoutExpired:
                output = "timeout"
                timed_out = True
                r = None
        finally:
            # Always reset train.py to parent code
            Path(train_path).write_text(parent_code)

        if timed_out:
            return {"val_bpb": None, "peak_vram_mb": None,
                    "output": "timeout", "success": False}
        if r.returncode != 0:
            return {"val_bpb": None, "peak_vram_mb": None,
                    "output": output[-2000:], "success": False}

        val_bpb, peak_vram_mb = parse_metrics_from_output(output)
        return {
            "val_bpb": val_bpb,
            "peak_vram_mb": peak_vram_mb,
            "output": output[-2000:],
            "success": val_bpb is not None,
        }
