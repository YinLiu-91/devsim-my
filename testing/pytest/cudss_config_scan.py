#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import statistics
import tempfile


def _load_last_jsonl(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def _f(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "-"


def _mode_env(mode: str) -> dict[str, str]:
    if mode == "native":
        return {
            "DEVSIM_CUDSS_DIRECT_SOLVER": "cudss",
            "DEVSIM_CUDSS_BACKEND_POLICY": "native",
            "DEVSIM_CUDSS_MT_MODE": "0",
            "DEVSIM_CUDSS_USE_STREAM": "0",
        }
    if mode == "native_mt":
        return {
            "DEVSIM_CUDSS_DIRECT_SOLVER": "cudss",
            "DEVSIM_CUDSS_BACKEND_POLICY": "native",
            "DEVSIM_CUDSS_MT_MODE": "1",
            "DEVSIM_CUDSS_USE_STREAM": "0",
        }
    if mode == "native_mt_stream":
        return {
            "DEVSIM_CUDSS_DIRECT_SOLVER": "cudss",
            "DEVSIM_CUDSS_BACKEND_POLICY": "native",
            "DEVSIM_CUDSS_MT_MODE": "1",
            "DEVSIM_CUDSS_USE_STREAM": "1",
        }
    raise ValueError(f"unsupported mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan native cuDSS ConfigSet combinations on one benchmark case.")
    parser.add_argument(
        "--case-filter",
        default="examples/capacitance/cap2d_large",
        help="pytest case filter passed to test_cudss_compare.py",
    )
    parser.add_argument(
        "--modes",
        default="native,native_mt,native_mt_stream",
        help="comma-separated run modes: native,native_mt,native_mt_stream",
    )
    parser.add_argument("--repeats", type=int, default=3, help="runs per variant (median reported)")
    parser.add_argument(
        "--devsim-so",
        default="/devsim/linux_x86_64_release/src/main/devsim_py3.so",
        help="devsim_py3 shared object to load during pytest runs",
    )
    args = parser.parse_args()

    base_env = {
        "DEVSIM_CUDSS_PROFILE": "1",
        "DEVSIM_CUDSS_RESULT_MODE": os.environ.get("DEVSIM_CUDSS_RESULT_MODE", "device_experimental"),
        "DEVSIM_CUDSS_PINNED_STAGING": os.environ.get("DEVSIM_CUDSS_PINNED_STAGING", "1"),
    }
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    variants: list[tuple[str, dict[str, str]]] = [
        ("default", {}),
        ("reorder_0", {"DEVSIM_CUDSS_REORDERING_ALG": "0"}),
        ("reorder_1", {"DEVSIM_CUDSS_REORDERING_ALG": "1"}),
        ("reorder_2", {"DEVSIM_CUDSS_REORDERING_ALG": "2"}),
        ("hybrid_1", {"DEVSIM_CUDSS_HYBRID_MODE": "1"}),
        ("hybrid_1_exec_1", {"DEVSIM_CUDSS_HYBRID_MODE": "1", "DEVSIM_CUDSS_HYBRID_EXECUTE_MODE": "1"}),
        ("host_threads_8", {"DEVSIM_CUDSS_HOST_NTHREADS": "8"}),
    ]

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="cudss_scan_") as td:
        timing_json = Path(td) / "timing.jsonl"
        for mode in modes:
            for name, extra in variants:
                env = os.environ.copy()
                env.update(base_env)
                env.update(_mode_env(mode))
                env.update(extra)

                run_records: list[dict[str, object]] = []
                run_rcs: list[int] = []
                errors: list[str] = []
                for _ in range(max(1, args.repeats)):
                    if timing_json.exists():
                        timing_json.unlink()
                    cmd = [
                        "python3",
                        "-m",
                        "pytest",
                        "testing/pytest/test_cudss_compare.py",
                        f"--case-filter={args.case_filter}",
                        "--solver-mode=both",
                        "--print-timing",
                        f"--timing-json={timing_json}",
                        "-q",
                    ]
                    if args.devsim_so:
                        cmd.extend(["--devsim-so", args.devsim_so])
                    cp = subprocess.run(cmd, env=env, text=True, capture_output=True, cwd="/devsim")
                    run_rcs.append(cp.returncode)
                    record = _load_last_jsonl(timing_json)
                    if record:
                        run_records.append(record)
                    else:
                        errors.append("no timing record (likely xfail/skip)")

                row: dict[str, object] = {"mode": mode, "name": name, "rc": max(run_rcs) if run_rcs else 1}
                if run_records:
                    numeric_keys = (
                        "speedup",
                        "solve_only_speedup",
                        "baseline_total_s",
                        "cudss_total_s",
                        "baseline_solver_s",
                        "cudss_solver_s",
                        "analysis_seconds",
                        "factor_seconds_profile",
                        "solve_seconds_profile",
                        "cudss_solver_wrapper_s",
                        "h2d_bytes",
                        "d2h_bytes",
                        "config_set_applied",
                        "mt_mode",
                        "stream_mode",
                    )
                    for key in numeric_keys:
                        vals = [float(r[key]) for r in run_records if isinstance(r.get(key), (int, float))]
                        if vals:
                            row[key] = statistics.median(vals)
                    row["samples"] = len(run_records)
                    if "speedup" in row:
                        sp_vals = [float(r["speedup"]) for r in run_records if isinstance(r.get("speedup"), (int, float))]
                        if len(sp_vals) > 1:
                            row["speedup_stdev"] = statistics.pstdev(sp_vals)
                else:
                    row["error"] = errors[-1] if errors else "no timing record"
                rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            str(r.get("mode", "")),
            -(float(r.get("speedup", 0.0)) if isinstance(r.get("speedup"), (int, float)) else -1.0),
        ),
    )

    print(f"case-filter: {args.case_filter}")
    print("| mode | variant | rc | samples | note | speedup | speedup_stdev | solve_only | baseline_solver | cudss_solver | baseline_total | cudss_total | analysis_s | factor_s | solve_s(profile) | solver_wrap | h2d | d2h | config_set_applied | mt_mode | stream_mode |")
    print("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows_sorted:
        print(
            "| {mode} | {name} | {rc} | {samples} | {note} | {speedup} | {speedup_stdev} | {solve_only_speedup} | {baseline_solver_s} | {cudss_solver_s} | {baseline_total_s} | {cudss_total_s} | "
            "{analysis_seconds} | {factor_seconds_profile} | {solve_seconds_profile} | {cudss_solver_wrapper_s} | "
            "{h2d_bytes} | {d2h_bytes} | {config_set_applied} | {mt_mode} | {stream_mode} |".format(
                mode=r.get("mode", "-"),
                name=r.get("name", "-"),
                rc=r.get("rc", "-"),
                samples=r.get("samples", 0),
                note=str(r.get("error", "")),
                speedup=_f(r.get("speedup")),
                speedup_stdev=_f(r.get("speedup_stdev")),
                solve_only_speedup=_f(r.get("solve_only_speedup")),
                baseline_solver_s=_f(r.get("baseline_solver_s")),
                cudss_solver_s=_f(r.get("cudss_solver_s")),
                baseline_total_s=_f(r.get("baseline_total_s")),
                cudss_total_s=_f(r.get("cudss_total_s")),
                analysis_seconds=_f(r.get("analysis_seconds")),
                factor_seconds_profile=_f(r.get("factor_seconds_profile")),
                solve_seconds_profile=_f(r.get("solve_seconds_profile")),
                cudss_solver_wrapper_s=_f(r.get("cudss_solver_wrapper_s")),
                h2d_bytes=_f(r.get("h2d_bytes")),
                d2h_bytes=_f(r.get("d2h_bytes")),
                config_set_applied=_f(r.get("config_set_applied")),
                mt_mode=_f(r.get("mt_mode")),
                stream_mode=_f(r.get("stream_mode")),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
