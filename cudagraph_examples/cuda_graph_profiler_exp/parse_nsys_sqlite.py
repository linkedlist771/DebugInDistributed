#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {row[0] for row in rows}


def column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def decode_name(conn: sqlite3.Connection, raw: Any, strings: dict[int, str]) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, int):
        if raw in strings:
            return strings[raw]
        try:
            row = conn.execute("SELECT value FROM StringIds WHERE id = ?", (raw,)).fetchone()
            if row:
                return str(row[0])
        except sqlite3.Error:
            pass
        return str(raw)
    return str(raw)


def load_strings(conn: sqlite3.Connection, tables: set[str]) -> dict[int, str]:
    if "StringIds" not in tables:
        return {}
    cols = column_names(conn, "StringIds")
    value_col = "value" if "value" in cols else "string"
    return {
        int(row[0]): str(row[1])
        for row in conn.execute(f"SELECT id, {value_col} FROM StringIds").fetchall()
    }


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def summarize_kernel_table(
    conn: sqlite3.Connection,
    table: str,
    strings: dict[int, str],
) -> dict[str, Any]:
    cols = column_names(conn, table)
    name_col = "shortName" if "shortName" in cols else "mangledName" if "mangledName" in cols else None
    if name_col is None:
        return {"table": table, "count": count_rows(conn, table), "top_names": []}
    rows = conn.execute(f"SELECT {name_col} FROM {table}").fetchall()
    names = Counter(decode_name(conn, row[0], strings) for row in rows)
    return {"table": table, "count": len(rows), "top_names": names.most_common(30)}


def summarize_runtime_table(
    conn: sqlite3.Connection,
    table: str,
    strings: dict[int, str],
) -> dict[str, Any]:
    cols = column_names(conn, table)
    name_col = "nameId" if "nameId" in cols else "name" if "name" in cols else None
    if name_col is None:
        return {"table": table, "count": count_rows(conn, table), "top_names": []}
    rows = conn.execute(f"SELECT {name_col} FROM {table}").fetchall()
    names = Counter(decode_name(conn, row[0], strings) for row in rows)
    return {"table": table, "count": len(rows), "top_names": names.most_common(30)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Nsight Systems exported SQLite")
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(str(args.sqlite))
    try:
        tables = table_names(conn)
        strings = load_strings(conn, tables)

        kernel_tables = [
            table
            for table in sorted(tables)
            if "KERNEL" in table.upper() or "CUDA_GPU_KERNEL" in table.upper()
        ]
        runtime_tables = [
            table
            for table in sorted(tables)
            if "RUNTIME" in table.upper() and "CUDA" in table.upper()
        ]
        graph_tables = [table for table in sorted(tables) if "GRAPH" in table.upper()]

        result = {
            "sqlite": str(args.sqlite),
            "tables": sorted(tables),
            "kernel_tables": [
                summarize_kernel_table(conn, table, strings) for table in kernel_tables
            ],
            "runtime_tables": [
                summarize_runtime_table(conn, table, strings) for table in runtime_tables
            ],
            "graph_tables": [
                {"table": table, "count": count_rows(conn, table)}
                for table in graph_tables
            ],
        }
    finally:
        conn.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
