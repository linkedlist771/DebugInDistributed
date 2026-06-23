#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CPU_CATEGORIES = {"cpu_op", "python_function", "user_annotation"}
LAUNCH_NAMES = {"cudaLaunchKernel", "cudaGraphLaunch"}


def as_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("traceEvents"), list):
        return payload["traceEvents"]
    if isinstance(payload, list):
        return payload
    raise ValueError("unsupported chrome trace JSON shape")


def event_end(event: dict[str, Any]) -> float:
    return float(event.get("ts", 0.0)) + float(event.get("dur", 0.0))


def find_window(events: list[dict[str, Any]], label: str) -> tuple[float, float]:
    matches = [
        event
        for event in events
        if event.get("name") == label and event.get("ph") == "X" and "ts" in event and "dur" in event
    ]
    if not matches:
        raise ValueError(f"could not find profiling window {label!r}")
    event = max(matches, key=lambda item: float(item.get("dur", 0.0)))
    return float(event["ts"]), event_end(event)


def inside_window(event: dict[str, Any], start: float, end: float) -> bool:
    if "ts" not in event:
        return False
    ts = float(event["ts"])
    return start <= ts <= end


def external_id(event: dict[str, Any]) -> Any:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    return args.get("External id") or args.get("external id") or args.get("External ID")


def correlation_id(event: dict[str, Any]) -> Any:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    return args.get("correlation") or args.get("Correlation") or args.get("correlation id")


def parent_external_id(event: dict[str, Any]) -> Any:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    return args.get("Ev Idx") or args.get("Parent id") or args.get("parent")


def summarize(events: list[dict[str, Any]], window_label: str) -> dict[str, Any]:
    start, end = find_window(events, window_label)
    window_events = [event for event in events if inside_window(event, start, end)]
    complete_events = [event for event in window_events if event.get("ph") == "X"]

    categories = Counter(str(event.get("cat", "")) for event in complete_events)
    names = Counter(str(event.get("name", "")) for event in complete_events)

    aten_events = [
        event
        for event in complete_events
        if str(event.get("name", "")).startswith("aten::")
        and (str(event.get("cat", "")) in CPU_CATEGORIES or str(event.get("cat", "")) == "")
    ]
    launch_events = [
        event
        for event in complete_events
        if event.get("name") in LAUNCH_NAMES or str(event.get("name", "")).startswith("cudaGraphLaunch")
    ]
    cuda_launch_kernel_events = [
        event for event in complete_events if event.get("name") == "cudaLaunchKernel"
    ]
    cuda_graph_launch_events = [
        event for event in complete_events if str(event.get("name", "")).startswith("cudaGraphLaunch")
    ]
    kernel_events = [
        event
        for event in complete_events
        if str(event.get("cat", "")) == "kernel" or str(event.get("cat", "")).startswith("kernel")
    ]

    launch_by_correlation = {
        correlation_id(event): event
        for event in launch_events
        if correlation_id(event) is not None
    }
    launch_to_aten_parent: dict[Any, dict[str, Any]] = {}
    for launch in launch_events:
        corr = correlation_id(launch)
        if corr is None or "ts" not in launch:
            continue
        launch_ts = float(launch["ts"])
        parents = [
            event
            for event in aten_events
            if float(event.get("ts", 0.0)) <= launch_ts <= event_end(event)
        ]
        if parents:
            launch_to_aten_parent[corr] = min(
                parents,
                key=lambda event: float(event.get("dur", 0.0)),
            )

    kernels_with_launch_correlation = 0
    kernels_with_launch_inside_aten = 0
    kernel_to_aten_names: list[str] = []
    for kernel in kernel_events:
        corr = correlation_id(kernel)
        if corr in launch_by_correlation:
            kernels_with_launch_correlation += 1
        parent = launch_to_aten_parent.get(corr)
        if parent is not None:
            kernels_with_launch_inside_aten += 1
            kernel_to_aten_names.append(str(parent.get("name", "")))

    cpu_external_ids = {external_id(event) for event in aten_events if external_id(event) is not None}
    kernel_external_ids = [external_id(event) for event in kernel_events]
    kernel_parent_ids = [parent_external_id(event) for event in kernel_events]
    kernel_correlation_ids = [correlation_id(event) for event in kernel_events]

    kernels_with_external_id = sum(item is not None for item in kernel_external_ids)
    kernels_with_parent_external_id = sum(item in cpu_external_ids for item in kernel_parent_ids)
    kernels_with_correlation_id = sum(item is not None for item in kernel_correlation_ids)

    return {
        "window_label": window_label,
        "window_ts": start,
        "window_end": end,
        "window_duration_us": end - start,
        "total_complete_events_in_window": len(complete_events),
        "aten_op_count": len(aten_events),
        "cudaLaunchKernel_count": len(cuda_launch_kernel_events),
        "cudaGraphLaunch_count": len(cuda_graph_launch_events),
        "launch_api_count": len(launch_events),
        "device_kernel_count": len(kernel_events),
        "unique_cpu_external_id_count": len(cpu_external_ids),
        "kernels_with_external_id": kernels_with_external_id,
        "kernels_with_correlation_id": kernels_with_correlation_id,
        "kernels_with_parent_external_id": kernels_with_parent_external_id,
        "kernels_with_launch_correlation": kernels_with_launch_correlation,
        "kernels_with_launch_inside_aten": kernels_with_launch_inside_aten,
        "kernel_parent_link_coverage": (
            kernels_with_parent_external_id / len(kernel_events) if kernel_events else None
        ),
        "kernel_external_id_coverage": (
            kernels_with_external_id / len(kernel_events) if kernel_events else None
        ),
        "kernel_correlation_id_coverage": (
            kernels_with_correlation_id / len(kernel_events) if kernel_events else None
        ),
        "kernel_launch_correlation_coverage": (
            kernels_with_launch_correlation / len(kernel_events) if kernel_events else None
        ),
        "kernel_launch_inside_aten_coverage": (
            kernels_with_launch_inside_aten / len(kernel_events) if kernel_events else None
        ),
        "top_categories": categories.most_common(20),
        "top_names": names.most_common(30),
        "kernel_names": Counter(str(event.get("name", "")) for event in kernel_events).most_common(),
        "launch_names": Counter(str(event.get("name", "")) for event in launch_events).most_common(),
        "aten_names": Counter(str(event.get("name", "")) for event in aten_events).most_common(),
        "kernel_to_aten_names": Counter(kernel_to_aten_names).most_common(),
        "sample_kernel_args": [
            {
                "name": event.get("name"),
                "cat": event.get("cat"),
                "args": event.get("args", {}),
            }
            for event in kernel_events[:5]
        ],
        "sample_launch_args": [
            {
                "name": event.get("name"),
                "cat": event.get("cat"),
                "args": event.get("args", {}),
            }
            for event in launch_events[:5]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PyTorch chrome trace window")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--window-label", default="PROFILE_ITER_0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.trace.read_text())
    result = summarize(as_events(payload), args.window_label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
