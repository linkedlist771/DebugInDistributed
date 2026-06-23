#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def load_nvtx_window(conn: sqlite3.Connection, label: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text=? ORDER BY start LIMIT 1",
        (label,),
    ).fetchone()
    if row is None:
        raise ValueError(f"NVTX window {label!r} not found")
    return int(row[0]), int(row[1])


def load_strings(conn: sqlite3.Connection) -> dict[int, str]:
    if not table_exists(conn, "StringIds"):
        return {}
    return {
        int(row[0]): str(row[1])
        for row in conn.execute("SELECT id, value FROM StringIds").fetchall()
    }


def decode_name(raw: Any, strings: dict[int, str]) -> str:
    if raw is None:
        return ""
    if isinstance(raw, int):
        return strings.get(raw, str(raw))
    return str(raw)


def runtime_name_counts(
    conn: sqlite3.Connection,
    start: int,
    end: int,
    strings: dict[int, str],
) -> Counter[str]:
    if not table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return Counter()
    rows = conn.execute(
        """
        SELECT nameId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE start >= ? AND start <= ?
        """,
        (start, end),
    ).fetchall()
    return Counter(decode_name(row[0], strings) for row in rows)


def count_runtime_prefix(counter: Counter[str], prefix: str) -> int:
    return sum(count for name, count in counter.items() if name.startswith(prefix))


def table_count_in_window(
    conn: sqlite3.Connection,
    table: str,
    start: int,
    end: int,
    overlap: bool = False,
) -> int | None:
    if not table_exists(conn, table):
        return None
    if overlap:
        query = f"SELECT COUNT(*) FROM {table} WHERE start <= ? AND end >= ?"
        params = (end, start)
    else:
        query = f"SELECT COUNT(*) FROM {table} WHERE start >= ? AND start <= ?"
        params = (start, end)
    return int(conn.execute(query, params).fetchone()[0])


def kernel_summary_from_csv(csv_path: Path, start: int, end: int) -> dict[str, Any]:
    kernel_rows: list[dict[str, str]] = []
    all_rows = 0
    in_window_rows = 0
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            all_rows += 1
            try:
                row_start = int(row["Start (ns)"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (start <= row_start <= end):
                continue
            in_window_rows += 1
            name = row.get("Name", "")
            if name and not name.startswith("[CUDA memcpy"):
                kernel_rows.append(row)
    return {
        "csv_total_rows": all_rows,
        "csv_rows_in_window": in_window_rows,
        "kernel_count_in_window": len(kernel_rows),
        "kernel_top_names_in_window": Counter(
            row.get("Name", "") for row in kernel_rows
        ).most_common(30),
        "kernel_rows_in_window": kernel_rows,
    }


def api_summary_from_csv(csv_path: Path, start: int, end: int) -> dict[str, Any]:
    api_rows: list[dict[str, str]] = []
    graph_launch_corr_ids: set[str] = set()
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                row_start = int(row["Start (ns)"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (start <= row_start <= end):
                continue
            api_rows.append(row)
            if row.get("Name", "").startswith("cudaGraphLaunch"):
                corr_id = row.get("CorrID")
                if corr_id:
                    graph_launch_corr_ids.add(corr_id)
    return {
        "api_rows_in_window": api_rows,
        "graph_launch_corr_ids": sorted(graph_launch_corr_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse an NVTX-filtered nsys window")
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--gpu-trace-csv", type=Path, required=True)
    parser.add_argument("--api-trace-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-label", default="PLAIN_ITER_0")
    args = parser.parse_args()

    conn = sqlite3.connect(str(args.sqlite))
    try:
        strings = load_strings(conn)
        start, end = load_nvtx_window(conn, args.window_label)
        runtime_counts = runtime_name_counts(conn, start, end, strings)
        gpu_summary = kernel_summary_from_csv(args.gpu_trace_csv, start, end)
        api_summary = api_summary_from_csv(args.api_trace_csv, start, end)
        graph_launch_corr_ids = set(api_summary["graph_launch_corr_ids"])
        graph_internal_kernel_names = [
            row.get("Name", "")
            for row in gpu_summary["kernel_rows_in_window"]
            if row.get("CorrId") in graph_launch_corr_ids
        ]
        host_kernel_names = [
            row.get("Name", "")
            for row in gpu_summary["kernel_rows_in_window"]
            if row.get("CorrId") not in graph_launch_corr_ids
        ]
        result = {
            "window_label": args.window_label,
            "window_start_ns": start,
            "window_end_ns": end,
            "window_duration_ns": end - start,
            "csv_total_rows": gpu_summary["csv_total_rows"],
            "csv_rows_in_window": gpu_summary["csv_rows_in_window"],
            "kernel_count_in_window": gpu_summary["kernel_count_in_window"],
            "kernel_top_names_in_window": gpu_summary["kernel_top_names_in_window"],
            "graph_launch_corr_ids": api_summary["graph_launch_corr_ids"],
            "graph_internal_kernel_count_in_window": len(graph_internal_kernel_names),
            "graph_internal_kernel_top_names_in_window": Counter(
                graph_internal_kernel_names
            ).most_common(30),
            "host_kernel_count_in_window": len(host_kernel_names),
            "host_kernel_top_names_in_window": Counter(host_kernel_names).most_common(30),
            "cudaLaunchKernel_runtime_count": count_runtime_prefix(
                runtime_counts,
                "cudaLaunchKernel",
            ),
            "cudaGraphLaunch_runtime_count": count_runtime_prefix(
                runtime_counts,
                "cudaGraphLaunch",
            ),
            "runtime_top_names_in_window": runtime_counts.most_common(30),
            "cuda_graph_node_events_start_in_window": table_count_in_window(
                conn,
                "CUDA_GRAPH_NODE_EVENTS",
                start,
                end,
            ),
            "cuda_graph_node_events_overlap_window": table_count_in_window(
                conn,
                "CUDA_GRAPH_NODE_EVENTS",
                start,
                end,
                overlap=True,
            ),
            "cuda_graph_events_start_in_window": table_count_in_window(
                conn,
                "CUDA_GRAPH_EVENTS",
                start,
                end,
            ),
            "cuda_graph_trace_events_start_in_window": table_count_in_window(
                conn,
                "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
                start,
                end,
            ),
            "cuda_graph_trace_events_overlap_window": table_count_in_window(
                conn,
                "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
                start,
                end,
                overlap=True,
            ),
        }
    finally:
        conn.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
