# -*- coding: utf-8 -*-
"""
Filename: non_llm_dual_gpu_inference.py
Description: Helpers for dual-GPU device selection, task sharding, and parallel
             execution for non-LLM HuggingFace baselines.
Keywords: non-LLM, dual GPU, HuggingFace, batch inference, task sharding

Workflow:
1. Resolve the locally available CUDA devices and use at most two GPUs by default.
2. Split sentence-, segment-, or prompt-level tasks evenly across devices.
3. Execute the per-device inference loops concurrently to improve non-LLM throughput.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def resolve_worker_devices(
    preferred_devices: Sequence[str] | None = None,
    max_devices: int = 2,
) -> List[str]:
    """Resolve the device list for the current inference round."""
    if preferred_devices:
        devices = [str(device).strip() for device in preferred_devices if str(device).strip()]
        return devices or ["cpu"]

    try:
        import torch
    except Exception:
        return ["cpu"]

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        use_count = max(1, min(int(max_devices or 1), device_count))
        return [f"cuda:{idx}" for idx in range(use_count)]
    return ["cpu"]


def split_tasks_evenly(tasks: Sequence[T], num_shards: int) -> List[List[T]]:
    """Split tasks as evenly as possible to avoid long-tail imbalance."""
    if num_shards <= 1 or not tasks:
        return [list(tasks)]

    total = len(tasks)
    base = total // num_shards
    remainder = total % num_shards
    shards: List[List[T]] = []
    start = 0
    for shard_idx in range(num_shards):
        shard_size = base + (1 if shard_idx < remainder else 0)
        end = start + shard_size
        shards.append(list(tasks[start:end]))
        start = end
    return shards


def parallel_run_on_devices(
    *,
    tasks: Sequence[T],
    device_names: Sequence[str],
    handler: Callable[[str, List[T]], List[R]],
    task_label: str,
) -> List[R]:
    """Run task shards concurrently across devices."""
    if not tasks:
        return []

    devices = list(device_names) or ["cpu"]
    if len(devices) == 1:
        return handler(devices[0], list(tasks))

    shards = split_tasks_evenly(tasks, len(devices))
    non_empty = [(device, shard) for device, shard in zip(devices, shards) if shard]
    if len(non_empty) <= 1:
        device, shard = non_empty[0] if non_empty else (devices[0], list(tasks))
        return handler(device, shard)

    print(
        f"  Launching non-LLM dual-GPU execution: devices={','.join(device for device, _ in non_empty)}, "
        f"{task_label}={len(tasks)}, shards={len(non_empty)}"
    )

    merged: List[R] = []
    with ThreadPoolExecutor(max_workers=len(non_empty), thread_name_prefix="nonllm_dualgpu") as executor:
        futures = {
            executor.submit(handler, device, shard): (device, len(shard))
            for device, shard in non_empty
        }
        for future in as_completed(futures):
            device, shard_size = futures[future]
            shard_results = future.result()
            merged.extend(shard_results)
            print(f"  Device {device} finished: shard={shard_size}, returned={len(shard_results)}")
    return merged
