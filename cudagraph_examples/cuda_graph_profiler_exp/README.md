# CUDA Graph Profiler Experiment

This directory contains a local RTX 4090 experiment for checking what profiler
visibility remains after PyTorch CUDA Graph capture/replay.

Main entry points:

- `harness.py`: runs eager or CUDA Graph replay with optional `torch.profiler`.
- `parse_torch_trace.py`: parses PyTorch chrome traces for a marked iteration.
- `parse_nsys_sqlite.py`: summarizes exported Nsight Systems SQLite reports.
- `report.md`: final measured report.
